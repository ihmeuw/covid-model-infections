from typing import Callable, List, Tuple, Dict
import pandas as pd
import numpy as np

from mrtool import MRData, LinearCovModel, MRBRT, MRBeRT
from mrtool.core.utils import sample_knots


def estimate_time_series(data: pd.DataFrame,
                         spline_options: Dict,
                         n_knots: int,
                         dep_var: str,
                         dep_trans_in: Callable[[pd.Series], pd.Series] = lambda x: x,
                         dep_var_se: str = None,
                         dep_se_trans_in: Callable[[pd.Series], pd.Series] = lambda x: x,
                         num_submodels: int = 100,
                         min_interval_days: int = 7,
                         dep_trans_out: Callable[[pd.Series], pd.Series] = lambda x: x,) -> Tuple[pd.DataFrame, pd.Series, MRBeRT]:
    data = data.copy()
    data['y'] = dep_trans_in(data[dep_var])
    day0 = data['date'].min()
    data['t'] = (data['date'] - day0).dt.days
    
    if dep_var_se:
        data['se'] = dep_se_trans_in(data[dep_var_se])
    else:
        data['se'] = np.abs(data['y'].mean())
    col_args = {
        'col_obs':'y',
        'col_obs_se':'se',
        'col_covs':['t'],
        'col_study_id':'date'
    }
    
    mr_data = MRData()
    mr_data.load_df(data, **col_args)
    spline_model = LinearCovModel('t',
                                  use_re=False,
                                  use_spline=True,
                                  use_spline_intercept=True,
                                  spline_knots=np.linspace(0., 1., n_knots),
                                  **spline_options)
    if num_submodels > 1:
        min_interval = min_interval_days / data['t'].max()
        ensemble_knots = get_ensemble_knots(n_knots, min_interval, num_submodels)
        mr_model = MRBeRT(mr_data, spline_model, ensemble_knots)
    else:
        mr_model = MRBRT(mr_data, [spline_model])
        
    mr_model.fit_model()

    if num_submodels > 1:
        mr_model.score_model()
    
    data = data.set_index('date')[['y', 'se']]
    
    smooth_data = predict_time_series(
        dep_var=dep_var,
        mr_model=mr_model,
        dep_trans_out=dep_trans_out,
    )
    
    return data, smooth_data, mr_model


def get_ensemble_knots(n_knots: int, min_interval: float, num_samples: int):
    num_intervals = n_knots - 1
    knot_bounds = np.array([[0, 1]] * (num_intervals - 1))
    interval_sizes = np.array([[min_interval, 1]] * num_intervals)
    #interval_sizes[0] = [1e-4, min_interval]
    #interval_sizes[-1] = [1e-4, min_interval]
    
    ensemble_knots = sample_knots(num_intervals, knot_bounds=knot_bounds, interval_sizes=interval_sizes, num_samples=num_samples)
    
    return ensemble_knots

    
def predict_time_series(dep_var: str, 
                        mr_model: MRBeRT,
                        dep_trans_out: Callable[[pd.Series], pd.Series]) -> pd.DataFrame:
    data = mr_model.data.to_df()
    day0 = data['study_id'].min()
    
    pred_data = MRData()
    pred_data.load_df(pd.DataFrame({'t':np.arange(data['t'].min(), data['t'].max() + 1)}),
                      col_covs='t')
    pred_data_value = mr_model.predict(pred_data)
    pred_data = pred_data.to_df()
    pred_data['date'] = pred_data['t'].apply(lambda x: day0 + pd.Timedelta(days=x))
    del pred_data['t']
    pred_data[dep_var] = dep_trans_out(pred_data_value)
    pred_data = pred_data.set_index('date')[dep_var]

    return pred_data
        
