from typing import Callable, List, Tuple, Dict
from loguru import logger

import pandas as pd
import numpy as np

from mrtool import MRData, LinearCovModel, MRBRT, MRBeRT
from mrtool.core.utils import sample_knots


def estimate_time_series(data: pd.DataFrame,
                         spline_options: Dict,
                         n_knots: int,
                         dep_var: str,
                         dep_trans_in: Callable[[pd.Series], pd.Series] = lambda x: x,
                         diff: bool = False,
                         dep_var_se: str = None,
                         dep_se_trans_in: Callable[[pd.Series], pd.Series] = lambda x: x,
                         num_submodels: int = 25,
                         min_interval_days: int = 7,
                         dep_trans_out: Callable[[pd.Series], pd.Series] = lambda x: x,
                         verbose: bool = False) -> Tuple[pd.DataFrame, pd.Series, MRBeRT]:
    if verbose: logger.info('Formatting data.')
    data = data.copy()
    data[dep_var] = dep_trans_in(data[dep_var])
    if diff:
        data[dep_var] = data[dep_var].diff()
    if data[[dep_var]].shape[1] > 1:
        reshape = True
        data = reshape_data_long(data, dep_var, dep_var_se)
    else:
        reshape = False
    data = data.rename(columns={dep_var:'y'})
    day0 = data['date'].min()
    keep_vars = ['date', 'y']
    if dep_var_se:
        keep_vars = list(set(keep_vars + [dep_var_se]))
    data = data.loc[:, keep_vars]
    start_len = len(data)
    data = data.dropna()
    end_len = len(data)
    if start_len != end_len and not reshape:
        if verbose: logger.debug('NAs in data')
    data['t'] = (data['date'] - day0).dt.days
    
    if dep_var_se:
        data['se'] = dep_se_trans_in(data[dep_var_se])
    else:
        data['se'] = 1.
    col_args = {
        'col_obs':'y',
        'col_obs_se':'se',
        'col_covs':['t'],
        #'col_study_id':'date',
    }
    
    if verbose: logger.info('Creating model data.')
    mr_data = MRData()
    mr_data.load_df(data, **col_args)
    spline_model = LinearCovModel('t',
                                  use_re=False,
                                  use_spline=True,
                                  use_spline_intercept=True,
                                  spline_knots=np.linspace(0., 1., n_knots),
                                  **spline_options)
    if num_submodels > 1:
        if verbose: logger.info('Sampling knots.')
        min_interval = min_interval_days / data['t'].max()
        ensemble_knots = get_ensemble_knots(n_knots, min_interval, num_submodels)
        
        if verbose: logger.info('Initializing model.')
        mr_model = MRBeRT(mr_data, spline_model, ensemble_knots)
    else:
        if verbose: logger.info('Initializing model.')
        mr_model = MRBRT(mr_data, [spline_model])
    
    if verbose: logger.info('Fitting model.')
    mr_model.fit_model()

    if num_submodels > 1:
        if verbose: logger.info('Scoring submodels.')
        mr_model.score_model()
    
    data = data.set_index('date')[['y', 'se']]
    
    if verbose: logger.info('Making prediction.')
    smooth_data = predict_time_series(
        day0=day0,
        dep_var=dep_var,
        mr_model=mr_model,
        dep_trans_out=dep_trans_out,
        diff=diff,
    )
    
    return data, smooth_data, mr_model


def model_intercept(data: pd.DataFrame,
                    prediction: pd.Series,
                    dep_var: str,
                    dep_trans_in: Callable[[pd.Series], pd.Series] = lambda x: x,
                    dep_trans_out: Callable[[pd.Series], pd.Series] = lambda x: x,
                    verbose: bool = True):
    data = data.copy()
    data[dep_var] = dep_trans_in(data[dep_var])
    prediction = dep_trans_in(prediction)
    data = reshape_data_long(data, dep_var, None)
    data = data.set_index('date').sort_index()
    data = data[dep_var] - prediction
    data = data.reset_index().dropna()
    data['intercept'] = 1
    
    mr_data = MRData()
    mr_data.load_df(data, 
        col_obs=dep_var,
        col_covs=['intercept'],
        col_study_id='date',)
    intercept_model = LinearCovModel('intercept', use_re=False,)
    mr_model = MRBRT(mr_data, [intercept_model])
    mr_model.fit_model()
    
    intercept = mr_model.beta_soln
    
    prediction += intercept
    prediction = pd.concat([
        pd.Series(intercept, index=pd.Index([prediction.index.min() - pd.Timedelta(days=1)], name='date')),
        prediction
    ]).rename(dep_var)
    prediction = dep_trans_out(prediction)
    
    return prediction
    
    
def reshape_data_long(data: pd.DataFrame, dep_var: str, dep_var_se: str) -> pd.DataFrame:
    if dep_var_se != 'y' and dep_var_se is not None:
        raise ValueError('Additional work required to use SE with reshape.')
    data = data.loc[:, ['date', dep_var]]
    data.columns = ['date'] + [f'{dep_var}_{i}' for i in range(data.shape[1] - 1)]
    data = pd.melt(data, id_vars='date', value_name=dep_var)
    data = data.loc[:, ['date', dep_var]]
    
    return data


def get_ensemble_knots(n_knots: int, min_interval: float, num_samples: int):
    num_intervals = n_knots - 1
    knot_bounds = np.array([[0, 1]] * (num_intervals - 1))
    interval_sizes = np.array([[min_interval, 1]] * num_intervals)
    #interval_sizes[0] = [1e-4, min_interval]
    #interval_sizes[-1] = [1e-4, min_interval]
    
    ensemble_knots = sample_knots(num_intervals, knot_bounds=knot_bounds, interval_sizes=interval_sizes, num_samples=num_samples)
    
    return ensemble_knots

    
def predict_time_series(day0: pd.Timestamp,
                        dep_var: str, 
                        mr_model: MRBRT,
                        dep_trans_out: Callable[[pd.Series], pd.Series],
                        diff: bool,) -> pd.DataFrame:
    data = mr_model.data.to_df()
    
    pred_data = MRData()
    t = np.arange(data['t'].min(), data['t'].max() + 1)
    pred_data.load_df(pd.DataFrame({'t':t}), col_covs='t')
    pred_data_value = mr_model.predict(pred_data)
    if diff:
        pred_data_value = pred_data_value.cumsum()
    pred_data_value = dep_trans_out(pred_data_value)
    pred_data = pd.DataFrame({'t':t,
                              dep_var:pred_data_value,})
    pred_data['t'] -= data['t'].min()
    pred_data['date'] = pred_data['t'].apply(lambda x: day0 + pd.Timedelta(days=x))
    pred_data = pred_data.set_index('date')[dep_var]

    return pred_data
        
