import pandas as pd
import numpy as np


def get_wrmse(y: pd.Series, y_hat: pd.Series, sigma: pd.Series, rmse_window: int, scale_tail: bool):
    
    residuals = (y_hat - y).rename('residuals')
    
    def weighted_avg(x: pd.Series, w: pd.Series) -> pd.Series:
        _w = w[x.index]
        return (x * (_w / _w.sum())).sum()
    
    if residuals.isnull().any():
        raise ValueError('NAs in residual dataset.')
    
    wrmse = residuals ** 2
    wrmse = wrmse.rolling(window=rmse_window, min_periods=rmse_window, center=True).apply(lambda x: weighted_avg(x, 1 / (sigma ** 2)))
    wrmse = wrmse.fillna(method='bfill')
    if scale_tail:
        wrmse_tail_scaler = (wrmse.isnull().cumsum() + 1)
    wrmse = wrmse.fillna(method='ffill')
    if scale_tail:
        wrmse *= wrmse_tail_scaler
    wrmse = np.sqrt(wrmse)
    wrmse = wrmse.rename('wrmse')

    return wrmse


def determine_n_knots(data: pd.Series, knot_days: int, min_k: int = 4) -> int:
    n_days = (data.reset_index()['date'].max() - data.reset_index()['date'].min()).days
    n_knots = int(np.ceil(n_days / knot_days))
    
    return max(min_k, n_knots)


def get_rate_transformations(log: bool):
    if log:
        dep_trans_in = lambda x: np.log(x)
        dep_se_trans_in = lambda x: 1.
        dep_trans_out = lambda x: np.exp(x)
    else:
        dep_trans_in = lambda x: x
        dep_se_trans_in = lambda x: 1.
        dep_trans_out = lambda x: x
        
    return dep_trans_in, dep_se_trans_in, dep_trans_out
