import sys
import os
from typing import Dict
from pathlib import Path
import dill as pickle
import sys
from tqdm import tqdm
from loguru import logger

import pandas as pd
import numpy as np

from covid_shared.cli_tools.logging import configure_logging_to_terminal

from covid_model_infections.model import data, mr_spline, plotter
from covid_model_infections.utils import OMP_NUM_THREADS
from covid_model_infections.cluster import F_THREAD

LOG_OFFSET = 1
FLOOR = 1e-4
CONSTRAINT_POINTS = 40


def model_measure(measure: str, model_type: str,
                  input_data: pd.Series, ratio: pd.Series, population: float,
                  n_draws: int, lag: int,
                  log: bool, knot_days: int,
                  num_submodels: int,) -> Dict:
    logger.info(f'{measure.capitalize()}:')
    input_data = input_data.rename(measure)
    
    dep_trans_in, dep_se_trans_in, dep_trans_out = get_rate_transformations(log)
    
    n_knots = determine_n_knots(input_data, knot_days)
    
    spline_options = {'spline_knots_type':'domain',
                      'spline_degree':3,}

    if model_type == 'cumul':
        spline_options.update({'prior_spline_monotonicity':'increasing',})
        prior_spline_maxder_gaussian = np.array([[0, 5e-3]] * (n_knots - 1))
        prior_spline_maxder_gaussian[:4] = [0, 1e-2]
        prior_spline_maxder_gaussian[-4:] = [0, 1e-2]
        spline_options.update({'prior_spline_maxder_gaussian':prior_spline_maxder_gaussian.T,})
    else:
        spline_options = {'spline_l_linear':True,
                          'spline_r_linear':True,}

    if not log:
        spline_options.update({'prior_spline_funval_uniform':np.array([0, np.inf]),})
        
    if model_type == 'cumul' or not log:
        spline_options.update({'prior_spline_num_constraint_points':CONSTRAINT_POINTS,})
        
    logger.info('Generating smooth past curve.')
    input_data = input_data.clip(FLOOR, np.inf)
    if log:
        input_data += LOG_OFFSET
    _, smooth_data, _ = mr_spline.estimate_time_series(
        data=input_data.reset_index(),
        dep_var=measure,
        spline_options=spline_options,
        n_knots=n_knots,
        dep_trans_in=dep_trans_in,
        #dep_var_se='y',
        #dep_se_trans_in=dep_se_trans_in,
        dep_trans_out=dep_trans_out,
        num_submodels=num_submodels,
    )
    
    logger.info('Converting to infections.')
    if log:
        input_data -= LOG_OFFSET
        smooth_data -= LOG_OFFSET
    if model_type == 'cumul':
        input_data = input_data.diff().fillna(input_data)
        smooth_data = smooth_data.diff().fillna(smooth_data)
    input_data = input_data.clip(FLOOR, np.inf)
    smooth_data = smooth_data.clip(FLOOR, np.inf)
    raw_infections = (input_data / ratio[input_data.index]).rename('infections')
    raw_infections.index -= pd.Timedelta(days=lag)
    smooth_infections = (smooth_data / ratio[smooth_data.index]).rename('infections')
    smooth_infections.index -= pd.Timedelta(days=lag)

    return {'cumul':smooth_data.cumsum(), 'daily':smooth_data,
            'infections_cumul_raw':raw_infections.cumsum(), 'infections_daily_raw':raw_infections,
            'infections_cumul':smooth_infections.cumsum(), 'infections_daily':smooth_infections}


def model_infections(inputs: pd.Series, log: bool, knot_days: int, diff: bool,
                     refit: bool, num_submodels: int,
                     **spline_kwargs) -> pd.Series:
    if refit:
        if isinstance(inputs, pd.DataFrame):
            draw = int(inputs.columns.unique().item().split('_')[-1])
        else:
            draw = int(inputs.name.split('_')[-1])
        np.random.seed(draw)
    
    n_knots = determine_n_knots(inputs, knot_days)
    
    if diff and not log:
        raise ValueError('Must do ln(diff) to prevent from going negative.')
    
    dep_trans_in, dep_se_trans_in, dep_trans_out = get_rate_transformations(log)
    if log and refit:
        _, _, dep_trans_out = get_rate_transformations(log=False)
    
    inputs = inputs.clip(FLOOR, np.inf)
    spline_options = {'spline_knots_type':'domain',
                      'spline_degree':3,}
    if log:
        inputs += LOG_OFFSET
        prior_spline_maxder_gaussian = np.array([[0, np.inf]] * (n_knots - 1))
        prior_spline_maxder_gaussian[0] = [0, 1e-3]
        prior_spline_maxder_gaussian[-1] = [0, 1e-3]
        spline_options.update({'prior_spline_maxder_gaussian':prior_spline_maxder_gaussian.T,})
        # spline_options.update({'spline_l_linear':True,
        #                        'spline_r_linear':True,})
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
        #dep_se_trans_in=dep_se_trans_in,
        dep_trans_out=dep_trans_out,
        num_submodels=num_submodels,
        single_random_knot=refit,
    )
    if diff:
        int_inputs = inputs[inputs.diff().notnull()]
        outputs = mr_spline.model_intercept(data=int_inputs.reset_index(),
                                            prediction=outputs,
                                            dep_var=int_inputs.columns.unique().item(),
                                            dep_trans_in=dep_trans_in,
                                            dep_trans_out=dep_trans_out,)
    
    if not refit:
        if log:
            outputs -= LOG_OFFSET
        outputs = outputs.clip(FLOOR, np.inf)
    
    return outputs


def sample_infections_residuals(smooth_infections: pd.Series, raw_infections: pd.DataFrame,
                                n_draws: int, rmse_radius: int = 90):
    dep_trans_in, _, dep_trans_out = get_rate_transformations(log=True)
    
    logger.info('Calculating residuals.')
    smooth_infections = dep_trans_in(smooth_infections.copy().clip(FLOOR, np.inf) + LOG_OFFSET)    
    residuals = dep_trans_in(raw_infections.copy().clip(FLOOR, np.inf) + LOG_OFFSET)
    
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
    draws = [pd.DataFrame({f'draw_{d}':(dep_trans_out(draw) - LOG_OFFSET).clip(FLOOR, np.inf)},
                          index=smooth_infections.index) for d, draw in enumerate(draws)]
    
    return draws


def splice_ratios(ratio_data: pd.Series,
                  smooth_data: pd.Series,
                  infections: pd.Series,
                  lag: int,) -> pd.Series:    
    infections.index += pd.Timedelta(days=lag)
    new_ratio = (smooth_data / infections).dropna().rename('new_ratio')
    start_date = new_ratio.index.min()
    end_date = new_ratio.index.max()
    new_ratio = pd.concat([ratio_data, new_ratio], axis=1)
    new_ratio.loc[new_ratio.index < start_date - pd.Timedelta(days=30), 'new_ratio'] = new_ratio[ratio_data.name]
    new_ratio.loc[new_ratio.index > end_date + pd.Timedelta(days=60), 'new_ratio'] = new_ratio[ratio_data.name]
    new_ratio = new_ratio['new_ratio'].rename(ratio_data.name)
    new_ratio = new_ratio.interpolate()
    
    return new_ratio

    
def determine_n_knots(data: pd.Series, knot_days: int, min_k: int = 4) -> int:
    n_days = (data.reset_index()['date'].max() - data.reset_index()['date'].min()).days
    n_knots = int(np.ceil(n_days / knot_days))
    
    return max(min_k, n_knots)


def get_rate_transformations(log: bool):
    if log:
        dep_trans_in = lambda x: np.log(x)
        dep_se_trans_in = lambda x: 1. / np.exp(x)
        dep_trans_out = lambda x: np.exp(x)
    else:
        dep_trans_in = lambda x: x
        dep_se_trans_in = lambda x: 1.
        dep_trans_out = lambda x: x
        
    return dep_trans_in, dep_se_trans_in, dep_trans_out


def get_infected(location_id: int,
                 n_draws: int,
                 model_in_dir: str,
                 model_out_dir: str,
                 plot_dir: str,
                 measure_type: str = 'cumul',
                 measure_log: bool = True, measure_knot_days: int = 7,
                 infection_log: bool = True, infection_knot_days: int = 28,):
    np.random.seed(location_id)
    logger.info('Loading data.')
    input_data, population, location_name = data.load_model_inputs(location_id, Path(model_in_dir))    
    
    logger.info(f'Running measure-specific smoothing splines.')
    output_data = {measure: model_measure(measure,
                                          measure_type,
                                          measure_data[measure_type].copy(), measure_data['ratio'].copy(),
                                          population, n_draws, measure_data['lag'],
                                          measure_log, measure_knot_days, num_submodels=1,)
                   for measure, measure_data in input_data.items()}
    
    logger.info('Fitting infection curve (w/ random knots) based on all available input measures.')
    infections_inputs = pd.concat([v['infections_daily'] for k, v in output_data.items()], axis=1).sort_index()
    smooth_infections = model_infections(infections_inputs, infection_log, infection_knot_days,
                                         diff=True, refit=False, num_submodels=100)
    raw_infections = pd.concat([v['infections_daily_raw'] for k, v in output_data.items()], axis=1).sort_index()
    input_draws = sample_infections_residuals(smooth_infections, raw_infections, n_draws)
    
    logger.info('Fitting infection curves to draws of all available input measures.')
    output_draws = []
    for input_draw in tqdm(input_draws, total=n_draws, file=sys.stdout):
        output_draws.append(model_infections(
            input_draw,
            log=infection_log, knot_days=infection_knot_days, num_submodels=1,
            diff=False, refit=True, #spline_r_linear=True, spline_l_linear=True
        ))
    output_draws = pd.concat(output_draws, axis=1)
    _, _, dep_trans_out = get_rate_transformations(infection_log)
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
    
    if 'deaths' in input_data.keys():
        logger.info('Create and storint new ratios (should do w/ draws!!!).')
        ifr = splice_ratios(input_data['deaths']['ratio'].copy(),
                            output_data['deaths']['daily'].copy(),
                            output_draws.mean(axis=1).rename('infections'),
                            input_data['deaths']['lag'],)
        ifr_path = Path(model_out_dir) / f'{location_id}_ifr.h5'
        ifr.to_hdf(ifr_path, key='data', mode='w')
    
    logger.info('Writing outputs.')
    data_path = Path(model_out_dir) / f'{location_id}_data.pkl'
    with data_path.open('wb') as file:
        pickle.dump({location_id:output_data}, file, -1)
    output_draws['location_id'] = location_id
    output_draws = (output_draws
                    .reset_index()
                    .set_index(['location_id', 'date'])
                    .sort_index())
    draw_path = Path(model_out_dir) / f'{location_id}_draws.h5'
    output_draws.to_hdf(draw_path, key='data', mode='w')


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = OMP_NUM_THREADS
    configure_logging_to_terminal(verbose=2)

    get_infected(location_id=int(sys.argv[1]),
                 n_draws=int(sys.argv[2]),
                 model_in_dir=sys.argv[3],
                 model_out_dir=sys.argv[4],
                 plot_dir=sys.argv[5],)
