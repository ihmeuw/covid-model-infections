import sys
import os
from typing import Dict
from pathlib import Path
import dill as pickle
import functools
import multiprocessing
import sys
import tqdm
from loguru import logger

import pandas as pd
import numpy as np

from covid_model_infections.model import data, support, mr_spline, plotter
from covid_model_infections.utils import OMP_NUM_THREADS

LOG_OFFSET = 1
FLOOR = 1e-4
CONSTRAINT_POINTS = 50
NUM_SUBMODELS = 50


def model_measure(measure: str, model_type: str,
                  data: pd.Series, ratio: pd.Series, population: float,
                  n_draws: int, lag: int,
                  log: bool, knot_days: int,) -> Dict:
    logger.info(f'{measure.capitalize()}:')
    data = data.rename(measure)
    
    dep_trans_in, dep_se_trans_in, dep_trans_out = support.get_rate_transformations(log)
    
    n_knots = support.determine_n_knots(data, knot_days)
    
    spline_options = {'spline_knots_type':'domain',
                      'spline_degree':3,}

    if model_type == 'cumul':
        spline_options.update({'prior_spline_monotonicity':'increasing',})
    else:
        spline_options = {'spline_l_linear':True,
                          'spline_r_linear':True,}

    if not log:
        spline_options.update({'prior_spline_funval_uniform':np.array([0, np.inf]),})
        
    if model_type == 'cumul' or not log:
        spline_options.update({'prior_spline_num_constraint_points':CONSTRAINT_POINTS,})
        
    logger.info('Generating smooth past curve.')
    data = data.clip(FLOOR, np.inf)
    if log:
        data += LOG_OFFSET
    model_data, smooth_data, mr_model = mr_spline.estimate_time_series(
        data=data.reset_index(),
        dep_var=measure,
        spline_options=spline_options,
        n_knots=n_knots,
        dep_trans_in=dep_trans_in,
        #dep_var_se='y',
        #dep_se_trans_in=dep_se_trans_in,
        dep_trans_out=dep_trans_out,
        num_submodels=NUM_SUBMODELS,
    )
    
    logger.info('Converting to infections.')
    if log:
        data -= LOG_OFFSET
        smooth_data -= LOG_OFFSET
    data = data.clip(FLOOR, np.inf)
    smooth_data = smooth_data.clip(FLOOR, np.inf)
    
    if model_type == 'cumul':
        data = data.diff().fillna(data)
        smooth_data = smooth_data.diff().fillna(smooth_data)
    raw_infections = (data / ratio[smooth_data.index]).rename('infections')
    raw_infections.index -= pd.Timedelta(days=lag)
    smooth_infections = (smooth_data / ratio[smooth_data.index]).rename('infections')
    smooth_infections.index -= pd.Timedelta(days=lag)

    return {'cumul':smooth_data.cumsum(), 'daily':smooth_data,
            'infections_cumul_raw':raw_infections.cumsum(), 'infections_daily_raw':raw_infections,
            'infections_cumul':smooth_infections.cumsum(), 'infections_daily':smooth_infections}


def model_infections(inputs: pd.Series, log: bool, knot_days: int, diff: bool, refit: bool = False,
                     **spline_kwargs) -> pd.Series:
    n_knots = support.determine_n_knots(inputs, knot_days)
    
    if diff and not log:
        raise ValueError('Must do ln(diff) to prevent from going negative.')
    
    dep_trans_in, dep_se_trans_in, dep_trans_out = support.get_rate_transformations(log)
    if log and refit:
        _, _, dep_trans_out = support.get_rate_transformations(log=False)
    
    inputs = inputs.clip(FLOOR, np.inf)
    spline_options = {'spline_knots_type':'domain',
                      'spline_degree':3,}
    if log:
        inputs += LOG_OFFSET
    elif not diff:
        spline_options.update({'prior_spline_funval_uniform':np.array([0, np.inf]),
                               'prior_spline_num_constraint_points':CONSTRAINT_POINTS,})
    spline_options.update(spline_kwargs)
    
    _, outputs, _ = mr_spline.estimate_time_series(
        data=inputs.reset_index(),
        dep_var=inputs.columns.unique().item(),
        spline_options=spline_options,
        n_knots=n_knots,
        dep_trans_in=dep_trans_in,
        diff=diff,
        #dep_var_se='y',
        #dep_se_trans_in=lambda x: 0.1,
        dep_trans_out=dep_trans_out,
        num_submodels=NUM_SUBMODELS,
    )
    if diff:
        outputs = mr_spline.model_intercept(data=inputs.reset_index(),
                                            prediction=outputs,
                                            dep_var=inputs.columns.unique().item(),
                                            dep_trans_in=dep_trans_in,
                                            dep_trans_out=dep_trans_out,)
    
    return outputs


def sample_infections_residuals(smooth_infections: pd.Series, raw_infections: pd.DataFrame,
                                n_draws: int, rmse_radius: int = 60):
    dep_trans_in, _, dep_trans_out = support.get_rate_transformations(log=True)
    
    logger.info('Calculating residuals.')
    smooth_infections = dep_trans_in(smooth_infections.copy() + LOG_OFFSET)    
    residuals = dep_trans_in(raw_infections.copy() + LOG_OFFSET)
    
    residuals['infections'] = smooth_infections.to_frame().values - residuals.values
    residuals = mr_spline.reshape_data_long(residuals.reset_index(), 'infections', None)
    residuals = residuals.dropna().sort_values('date').rename(columns={'infections':'residuals'})
    
    dates = smooth_infections.index
    dates = dates[rmse_radius:-rmse_radius]
    
    logger.info(f'Getting MAD (using rolling {int(rmse_radius*2)} day window), translating to SD.')
    sigmas = []
    for date in dates:
        avg_dates = pd.date_range(date - pd.Timedelta(days=rmse_radius), date + pd.Timedelta(days=rmse_radius), freq=None)
        mad = np.abs(residuals.loc[residuals['date'].isin(avg_dates), 'residuals']).median()
        sigma = mad * 1.4826
        sigmas.append(pd.Series(sigma, index=pd.Index([date], name='date'), name='sigma'))
    sigma = pd.concat(sigmas)
    
    smooth_infections = pd.concat([smooth_infections, sigma], axis=1).sort_index()
    smooth_infections['sigma'] = smooth_infections['sigma'].fillna(method='bfill')
    smooth_infections['sigma'] = smooth_infections['sigma'].fillna(method='ffill')
    
    logger.info('Sampling residuals.')
    draws = np.random.normal(smooth_infections['infections'].values, smooth_infections['sigma'].values,
                             (n_draws, len(smooth_infections)))
    draws = [pd.DataFrame({f'draw_{d}':dep_trans_out(draw)}, index=smooth_infections.index) for d, draw in enumerate(draws)]
    draws -= LOG_OFFSET
    data = data.clip(FLOOR, np.inf)
    
    return draws


def get_infected(location_id: int,
                 n_draws: int,
                 model_in_dir: str,
                 model_out_dir: str,
                 plot_dir: str,
                 measure_type: str = 'cumul',
                 measure_log: bool = False, measure_knot_days: int = 21,
                 infection_log: bool = True, infection_knot_days: int = 28,):
    logger.info('Loading data.')
    input_data, population, location_name = data.load_model_inputs(location_id, Path(model_in_dir))    
    
    logger.info(f'Running measure-specific models.')
    output_data = {measure: model_measure(measure,
                                          measure_type,
                                          measure_data[measure_type].copy(), measure_data['ratio'].copy(),
                                          population, n_draws, measure_data['lag'],
                                          measure_log, measure_knot_days,)
                   for measure, measure_data in input_data.items()}
    
    logger.info('Fitting infection curves to draws of all available input measures.')
    smooth_infections = pd.concat([v['infections_daily'] for k, v in output_data.items()], axis=1).sort_index()
    smooth_infections = model_infections(smooth_infections, infection_log, infection_knot_days, diff=True)
    raw_infections = pd.concat([v['infections_daily_raw'] for k, v in output_data.items()], axis=1).sort_index()
    input_draws = sample_infections_residuals(smooth_infections, raw_infections, n_draws)
    
    _estimator = functools.partial(
        model_infections,
        log=infection_log, knot_days=infection_knot_days,
        diff=False, refit=True, #spline_r_linear=True, spline_l_linear=True
    )
    with multiprocessing.Pool(25) as p:
        output_draws = list(tqdm.tqdm(p.imap(_estimator, input_draws), total=n_draws, file=sys.stdout))
    output_draws = pd.concat(output_draws, axis=1)
    _, _, dep_trans_out = support.get_rate_transformations(infection_log)
    if infection_log:
        output_draws -= np.var(output_draws.values, axis=1, keepdims=True) / 2
    output_draws = dep_trans_out(output_draws)
    
    logger.info('Plot data.')
    sero_data, test_data = data.load_extra_plot_inputs(location_id, Path(model_in_dir))
    test_data = (test_data['daily_tests'] / population).rename('testing_rate')
    plotter.plotter(
        Path(plot_dir), location_id, location_name,
        input_data, test_data, sero_data,
        output_data, output_draws, population
    )
    
    logger.info('Writing outputs.')
    data_path = Path(model_out_dir) / f'{location_id}_data.pkl'
    with data_path.open('wb') as file:
        pickle.dump(output_data, file, -1)
    draw_path = Path(model_out_dir) / f'{location_id}_draws.h5'
    output_draws.to_hdf(draw_path, key='data', mode='w')


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = OMP_NUM_THREADS
    get_infected(location_id=int(sys.argv[1]),
                 n_draws=int(sys.argv[2]),
                 model_in_dir=sys.argv[3],
                 model_out_dir=sys.argv[4],
                 plot_dir=sys.argv[5],)
