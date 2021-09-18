import sys
import os
from typing import Dict, List
from pathlib import Path
import dill as pickle
import sys
from tqdm import tqdm
from loguru import logger
import multiprocessing
import functools
import itertools

import pandas as pd
import numpy as np

from covid_shared.cli_tools.logging import configure_logging_to_terminal

from covid_model_infections.model import data, mr_spline, plotter
from covid_model_infections.utils import CEILINGS
from covid_model_infections.cluster import TYPE_SPECS

LOG_OFFSET = 1
FLOOR = 1e-4
CONSTRAINT_POINTS = 40


def model_measure(measure: str, measure_type: str,
                  input_data: pd.Series, ratio: pd.Series, population: float,
                  n_draws: int, lag: int,
                  log: bool, knot_days: int,
                  num_submodels: int,
                  split_l_interval: bool,
                  split_r_interval: bool,) -> Dict:
    logger.info(f'{measure.capitalize()}:')
    input_data = input_data.rename(measure)
    
    if measure_type not in ['cumul', 'daily']:
        raise ValueError(f'Invalid measure_type (must be `cumul` or `daily`): {measure_type}')
    
    logger.info('Doing 7-day rolling average to help eliminate day-of-week reporting bias.')
    if measure_type == 'daily':
        day0_value = max(0, input_data[0])
        input_data = input_data[1:]
    input_data = (input_data
                  .clip(0, np.inf)
                  .rolling(window=7,
                           min_periods=7,
                           center=True).mean()
                  .dropna())
    
    dep_trans_in, _, dep_trans_out = get_rate_transformations(log)
    
    n_knots = determine_n_knots(input_data, knot_days)
    
    spline_options = {'spline_knots_type':'domain',
                      'spline_degree':3 + (measure_type=='cumul'),}
    
    if measure_type == 'cumul':
        spline_options.update({'prior_spline_monotonicity':'increasing',})
        prior_spline_maxder_gaussian = np.array([[0, 1.]] * (n_knots + split_l_interval + split_r_interval - 1))
        spline_options.update({'prior_spline_maxder_gaussian':prior_spline_maxder_gaussian.T,})
    # else:
    #     spline_options = {'spline_l_linear':False,
    #                       'spline_r_linear':True,}

    if not log:
        spline_options.update({'prior_spline_funval_uniform':np.array([0, np.inf]),})
        
    if measure_type == 'cumul' or not log:
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
        dep_trans_out=dep_trans_out,
        num_submodels=num_submodels,
        split_l_interval=split_l_interval,
        split_r_interval=split_r_interval,
    )
    
    logger.info('Converting to infections.')
    if log:
        input_data -= LOG_OFFSET
        smooth_data -= LOG_OFFSET
    if measure_type == 'cumul':
        input_data = input_data.diff().fillna(input_data)
        smooth_data = smooth_data.diff().fillna(smooth_data)
    else:
        input_data[0] += day0_value
        smooth_data[0] += day0_value
    input_data = input_data.clip(FLOOR, np.inf)
    smooth_data = smooth_data.clip(FLOOR, np.inf)
    
    raw_infections = []
    smooth_infections = []
    for draw in range(n_draws):
        # raw_infections = pd.concat([input_data, ratio.loc[draw]], axis=1)
        # raw_infections = (raw_infections[input_data.name] / raw_infections[ratio.name])
        _raw_infections = (input_data / ratio.loc[draw]).rename('infections').dropna()
        _raw_infections.index -= pd.Timedelta(days=lag)
        raw_infections.append(_raw_infections)
        # smooth_infections = pd.concat([smooth_data, ratio.loc[draw]], axis=1)
        # smooth_infections = (smooth_infections[smooth_data.name] / smooth_infections[ratio.name]).rename('infections').dropna()
        _smooth_infections = (smooth_data / ratio.loc[draw]).rename('infections').dropna()
        _smooth_infections.index -= pd.Timedelta(days=lag)
        smooth_infections.append(_smooth_infections)

    return {'daily': smooth_data, 'cumul': smooth_data.cumsum(),
            'infections_daily_raw': raw_infections,
            'infections_cumul_raw': [ri.cumsum() for ri in raw_infections],
            'infections_daily': smooth_infections,
            'infections_cumul': [si.cumsum() for si in smooth_infections],}


def model_infections(inputs: pd.DataFrame,
                     log: bool, knot_days: int, diff: bool,
                     refit: bool, num_submodels: int,
                     weights: pd.DataFrame = None,
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
    
    dep_trans_in, _, dep_trans_out = get_rate_transformations(log)
    if log and refit:
        _, _, dep_trans_out = get_rate_transformations(log=False)
    
    inputs = inputs.clip(FLOOR, np.inf)
    spline_options = {'spline_knots_type':'domain',
                      'spline_degree':3 - diff,}
    if log:
        inputs += LOG_OFFSET
        # prior_spline_maxder_gaussian = np.array([[0, np.inf]] * (n_knots - 1))
        # prior_spline_maxder_gaussian[-1] = [0, 1e-2]
        # spline_options.update({'prior_spline_maxder_gaussian':prior_spline_maxder_gaussian.T,})
        # spline_options.update({'spline_l_linear':True,
        #                        'spline_r_linear':True,})
    elif not diff:
        spline_options.update({'prior_spline_funval_uniform':np.array([0, np.inf]),
                               'prior_spline_num_constraint_points':CONSTRAINT_POINTS,})
    
    if diff:
        # force start to be increasing
        spline_options.update({'prior_spline_funval_uniform': np.array([0, np.inf]),
                               'prior_spline_funval_uniform_domain': (0, 7 / (len(inputs) - 2))})
    ## ONLY CONTROL IN DIFF MODEL
    # else:
    #     # force start to be increasing
    #     spline_options.update({'prior_spline_monotonicity': 'increasing',
    #                            'prior_spline_monotonicity_domain': (0, 7 / (len(inputs) - 2))})
        
    spline_options.update(spline_kwargs)
    
    if not refit:
        dep_trans_se = {
            'weight_data':weights.reset_index(),
            'dep_var_se':weights.columns.unique().item(),
        }
    else:
        dep_trans_se = {}
    
    _, outputs, _ = mr_spline.estimate_time_series(
        data=inputs.reset_index(),
        dep_var=inputs.columns.unique().item(),
        spline_options=spline_options,
        n_knots=n_knots,
        dep_trans_in=dep_trans_in,
        diff=diff,
        dep_trans_out=dep_trans_out,
        num_submodels=num_submodels,
        single_random_knot=refit,
        **dep_trans_se
    )
    if diff:
        int_inputs = inputs[inputs.diff().notnull()]
        outputs = mr_spline.model_intercept(data=int_inputs.reset_index(),
                                            dep_var=int_inputs.columns.unique().item(),
                                            prediction=outputs,
                                            weight_data=weights.reset_index(),
                                            dep_var_se=weights.columns.unique().item(),
                                            dep_trans_in=dep_trans_in,
                                            dep_trans_out=dep_trans_out,)
    
    if not refit:
        if log:
            outputs -= LOG_OFFSET
        outputs = outputs.clip(FLOOR, np.inf)
    
    return outputs


def sample_infections_residuals(smooth_infections: pd.Series, raw_infections: pd.DataFrame,
                                rmse_radius: int = 180):
    dep_trans_in, _, dep_trans_out = get_rate_transformations(log=True)
    
    # logger.info('Calculating residuals.')
    smooth_infections = dep_trans_in(smooth_infections.copy().clip(FLOOR, np.inf) + LOG_OFFSET)    
    residuals = dep_trans_in(raw_infections.copy().clip(FLOOR, np.inf) + LOG_OFFSET)
    
    residuals = smooth_infections.to_frame() - residuals
    residuals = mr_spline.reshape_data_long(residuals.reset_index(), 'infections')
    residuals = residuals.dropna().sort_values('date').rename(columns={'infections':'residuals'})
    
    dates = smooth_infections.index
    if len(dates) < rmse_radius * 3:
        rmse_radius = int(len(dates) / 3)
    dates = dates[rmse_radius:-rmse_radius]
    
    # logger.info(f'Getting MAD (using rolling {int(rmse_radius*2)} day window), translating to SD.')
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
    
    # logger.info('Sampling residuals.')
    draw = np.random.normal(smooth_infections['infections'].values, smooth_infections['sigma'].values,
                            len(smooth_infections))
    draw = pd.Series((dep_trans_out(draw) - LOG_OFFSET).clip(FLOOR, np.inf),
                     name='noisy_infections',
                     index=smooth_infections.index)
    
    return draw


def splice_ratios(ratio_data: pd.Series,
                  smooth_data: pd.Series,
                  infections: pd.Series,
                  lag: int,
                  trans_period_past: int = 60,
                  trans_period_future: int = 60,) -> pd.Series:
    col_name = infections.name
    infections.index += pd.Timedelta(days=lag)
    new_ratio = (smooth_data / infections).dropna().rename('new_ratio')
    new_ratio = new_ratio[1:]  # don't start at day0, is problematic for delayed cumul series
    start_date = new_ratio.index.min()
    end_date = new_ratio.index.max()
    pre = new_ratio[:trans_period_past].mean()
    post = new_ratio[-trans_period_future:].mean()
    new_ratio = pd.concat([ratio_data, new_ratio], axis=1)
    new_ratio.loc[new_ratio.index < start_date - pd.Timedelta(days=trans_period_past), 'new_ratio'] = pre  # new_ratio[ratio_data.name]
    new_ratio.loc[new_ratio.index > end_date + pd.Timedelta(days=trans_period_future), 'new_ratio'] = post  # new_ratio[ratio_data.name]
    new_ratio = new_ratio['new_ratio'].rename(ratio_data.name)
    new_ratio = new_ratio.interpolate(limit_area='inside').rename(col_name)
    
    return new_ratio


def enforce_ratio_ceiling(output_measure: str,
                          input_measure: str,
                          obs_data: pd.Series,
                          infections_data_list: List[pd.Series],
                          lag: int,
                          ceiling: float,):
    adj_infections_data_list = []
    for infections_data in infections_data_list:
        infections_floor = obs_data / ceiling
        infections_floor.index -= pd.Timedelta(days=lag)
        infections_scalar = (infections_floor / infections_data)[infections_data.index]
        infections_scalar = infections_scalar.clip(1, np.inf)

        # backfill 1s, forward taper off of terminal scalar over a week to reduce noise
        leading_nas = infections_scalar.ffill().isnull()
        infections_scalar.loc[leading_nas] = 1
        trailing_nas = infections_scalar.bfill().isnull()
        if trailing_nas.sum():
            # transition over either a week or until inf exceed last floor point
            n_days_below = ((infections_data.loc[trailing_nas] > infections_floor[-1]).cumsum() == 0).sum()
            n_transition_days = min(n_days_below, 7)
            if n_transition_days == 0:
                infections_scalar.loc[trailing_nas] = 1
            else:
                terminal_scalar = infections_scalar.dropna()[-1]
                transition_scalar = infections_scalar.dropna()[-14:].median()
                delta = (terminal_scalar - transition_scalar) / n_transition_days
                transition = trailing_nas.cumsum().clip(0, n_transition_days).replace(0, np.nan).dropna()
                transition *= -delta
                transition += terminal_scalar
                infections_scalar.loc[trailing_nas] = transition

        needs_correction = infections_scalar.max() > 1
        if needs_correction:
            logger.info(f'Adjusting infections from {output_measure} so they are not fewer than observed {input_measure}.')
        infections_data *= infections_scalar
        adj_infections_data_list.append(infections_data)

    return adj_infections_data_list


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


def triangulate_infections(infections_inputs: pd.DataFrame, output_data: Dict, is_us: bool,
                           infection_log: bool, infection_knot_days: int,):
    infections_inputs = infections_inputs.copy()
    n = infections_inputs['draw'].unique().item()
    del infections_inputs['draw']
    
    if is_us:
        infections_weights = pd.concat([v['infections_daily'][n] ** 0 - (k == 'hospitalizations') * (1 - 0.1) for k, v in output_data.items()],
                                       axis=1).sort_index()
    else:
        infections_weights = pd.concat([v['infections_daily'][n] ** 0 for k, v in output_data.items()], axis=1).sort_index()
    infections_weights = np.sqrt(infections_weights)
    smooth_infections = model_infections(inputs=infections_inputs, weights=infections_weights,
                                         log=infection_log, knot_days=infection_knot_days,
                                         diff=True, refit=False, num_submodels=5,)
    raw_infections = pd.concat([v['infections_daily_raw'][n] for k, v in output_data.items()], axis=1).sort_index()
    input_draw = sample_infections_residuals(smooth_infections, raw_infections,)
    
    return smooth_infections, raw_infections, input_draw


def run_model(location_id: int,
              n_draws: int,
              model_in_dir: str,
              model_out_dir: str,
              plot_dir: str,
              measure_type: str = 'daily',
              measure_log: bool = True, measure_knot_days: int = 7,
              infection_log: bool = True, infection_knot_days: int = 28,
              mp: bool = True,):
    np.random.seed(15243 + location_id)
    logger.info('Loading data.')
    input_data, modeled_location, population, location_name, is_us = data.load_model_inputs(location_id, Path(model_in_dir))
    if not modeled_location:
        raise ValueError(f'Location does not have sufficient data to model ({location_id}).')
    
    logger.info('Running measure-specific smoothing splines.')
    output_data = {measure: model_measure(measure,
                                          measure_type,
                                          measure_data[measure_type].copy(),
                                          measure_data['ratio']['ratio'].copy(),
                                          population, n_draws, measure_data['lag'],
                                          measure_log, measure_knot_days, num_submodels=1,
                                          split_l_interval=False, split_r_interval=False,)
                   for measure, measure_data in input_data.items()}
    for input_measure in input_data.keys():
        infections_inputs = [enforce_ratio_ceiling(output_measure,
                                                   input_measure,
                                                   output_data[input_measure]['daily'][1:].copy(),
                                                   output_data[output_measure]['infections_daily'].copy(),
                                                   input_data[input_measure]['lag'],
                                                   CEILINGS[input_measure],)
                             for output_measure in output_data.keys()]
        for measure, new_infections in zip(output_data.keys(), infections_inputs):
            output_data[measure]['infections_daily'] = new_infections
            output_data[measure]['infections_cumul'] = [ni.cumsum() for ni in new_infections]
    infections_inputs = list(map(list, itertools.zip_longest(*infections_inputs, fillvalue=None)))
    infections_inputs = [pd.concat(ii, axis=1).sort_index() for ii in infections_inputs]
    infections_inputs = [pd.concat([ii, pd.DataFrame({'draw': n}, index=ii.index)], axis=1)
                         for n, ii in enumerate(infections_inputs)]
    
    logger.info('Fitting infection curve (w/ random knots) based on all available input measures.')
    if mp:
        _triangulate_infections = functools.partial(
            triangulate_infections,
            output_data=output_data, is_us=is_us,
            infection_log=infection_log, infection_knot_days=infection_knot_days,
        )
        with multiprocessing.Pool(int(os.environ['OMP_NUM_THREADS'])) as p:
            si_ri_id = list(tqdm(p.imap(_triangulate_infections, infections_inputs),
                                 total=n_draws, file=sys.stdout))
    else:
        si_ri_id = []
        for ii in tqdm(infections_inputs, total=n_draws, file=sys.stdout):
            si_ri_id.append(triangulate_infections(
                ii,
                output_data=output_data, is_us=is_us,
                infection_log=infection_log, infection_knot_days=infection_knot_days,
            ))

    smooth_infections = [i[0] for i in si_ri_id]
    raw_infections = [i[1] for i in si_ri_id]
    input_draws = [i[2] for i in si_ri_id]
    del si_ri_id

    smooth_infections = pd.concat(smooth_infections).groupby(level=0).mean()
    raw_infections = pd.concat(raw_infections).groupby(level=0).mean()
    input_draws = [i.rename(f'draw_{n}').to_frame() for n, i in enumerate(input_draws)]
    
    draw_args = {
        'log': infection_log,
        'knot_days': infection_knot_days,
        'num_submodels': 5,
        'diff': False,
        'refit': True,
    }
    
    logger.info('Fitting infection curves to draws of all available input measures.')
    if mp:
        _model_infections = functools.partial(
            model_infections,
            **draw_args
        )
        with multiprocessing.Pool(int(os.environ['OMP_NUM_THREADS'])) as p:
            output_draws = list(tqdm(p.imap(_model_infections, input_draws), total=n_draws, file=sys.stdout))
    else:
        output_draws = []
        for input_draw in tqdm(input_draws, total=n_draws, file=sys.stdout):
            output_draws.append(model_infections(
                input_draw,
                **draw_args
            ))
    input_draws = pd.concat(input_draws, axis=1)
    output_draws = pd.concat(output_draws, axis=1)
    _, _, dep_trans_out = get_rate_transformations(draw_args['log'])
    if draw_args['log']:
        output_draws -= np.var(output_draws.values, axis=1, keepdims=True) / 2
    output_draws = dep_trans_out(output_draws)

    logger.warning('Droppping last three days of infections for stability.')
    output_draws = output_draws[:-3]
    
    logger.info('Plot data.')
    input_data, _, population, location_name, _ = data.load_model_inputs(location_id, Path(model_in_dir))
    sero_data, daily_reinfection_rr, ratio_model_inputs = data.load_extra_plot_inputs(location_id, Path(model_in_dir))
    plotter.plotter(
        Path(plot_dir),
        location_id, location_name,
        input_data, sero_data, ratio_model_inputs, daily_reinfection_rr,
        output_data.copy(), smooth_infections.copy(), output_draws.copy(), population
    )
    
    logger.info('Writing intermediate datasets.')
    input_draws['location_id'] = location_id
    input_draws = (input_draws
                   .reset_index()
                   .set_index(['location_id', 'date'])
                   .sort_index())
    input_draws_path = Path(model_out_dir) / f'{location_id}_input_draws.parquet'
    input_draws.to_parquet(input_draws_path)
    input_data_path = Path(model_out_dir) / f'{location_id}_input_data.pkl'
    with input_data_path.open('wb') as file:
        pickle.dump({location_id: input_data}, file, -1)
    output_data_path = Path(model_out_dir) / f'{location_id}_output_data.pkl'
    with output_data_path.open('wb') as file:
        pickle.dump({location_id: output_data}, file, -1)
    input_infections_path = Path(model_out_dir) / f'{location_id}_input_infections.pkl'
    with input_infections_path.open('wb') as file:
        pickle.dump((raw_infections, smooth_infections), file, -1)

    logger.info('Creating and writing ratios.')
    ratio_measure_map = {
        'cases':'idr', 'hospitalizations':'ihr', 'deaths':'ifr'
    }
    output_draws_list = [output_draws[c].copy() for c in output_draws.columns]
    for measure in input_data.keys():
        ratio_draws = [splice_ratios(ratio_data=input_data[measure]['ratio']['ratio'].loc[n].copy(),
                                     smooth_data=output_data[measure]['daily'].copy(),
                                     infections=output_draws_list[n].copy(),
                                     lag=input_data[measure]['lag'],) for n in range(n_draws)]
        ratio_draws = pd.concat(ratio_draws, axis=1)
        ratio_draws['location_id'] = location_id
        ratio_draws = (ratio_draws
                       .reset_index()
                       .set_index(['location_id', 'date'])
                       .sort_index())
        ratio_path = Path(model_out_dir) / f'{location_id}_{ratio_measure_map[measure]}_draws.parquet'
        ratio_draws.to_parquet(ratio_path)
    
    logger.info('Writing output draws.')
    output_draws['location_id'] = location_id
    output_draws = (output_draws
                    .reset_index()
                    .set_index(['location_id', 'date'])
                    .sort_index())
    draw_path = Path(model_out_dir) / f'{location_id}_infections_draws.parquet'
    output_draws.to_parquet(draw_path)


if __name__ == '__main__':
    configure_logging_to_terminal(verbose=2)

    os.environ['OMP_NUM_THREADS'] = TYPE_SPECS['covid_loc_inf']['OMP_NUM_THREADS']
    
    run_model(location_id=int(sys.argv[1]),
              n_draws=int(sys.argv[2]),
              model_in_dir=sys.argv[3],
              model_out_dir=sys.argv[4],
              plot_dir=sys.argv[5],)
