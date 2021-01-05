import os
from typing import Tuple, Dict
from pathlib import Path
import argparse
import dill as pickle
import functools
import multiprocessing
import sys
import tqdm

import pandas as pd
import numpy as np

from covid_model_infections.model import processing, mr_spline
from covid_model_infections.utils import KNOT_DAYS

'''
data = pd.read_csv('/ihme/covid-19/snapshot-data/best/covid_onedrive/Serological studies/global_serology_summary.csv',
                   encoding='latin1')
data = data.sort_values(['location_id', 'date']).reset_index(drop=True)

data = data.loc[data['survey_series'].isin(['cdc_series', 'react2'])]

# date formatting
data['date'] = data['date'].str.replace('.202$|.2021$', '.2020')
data.loc[data['date'] == '05.21.2020', 'date'] = '21.05.2020'
data['date'] = pd.to_datetime(data['date'], format='%d.%m.%Y')
data['start'] = data.groupby('location_id')['date'].transform(min)
data = data.loc[(data['start'] < '2020-07-01') | (data['survey_series'] == 'react2')]

for var in ['value', 'lower', 'upper']:
    data[var] = data[var].astype(float)

for location in data['location'].unique():
    plt.errorbar(data.loc[(data['location'] == location) & (data['survey_series'] == 'cdc_series'), 'date'],
                 data.loc[(data['location'] == location) & (data['survey_series'] == 'cdc_series'), 'value'],
                 yerr=(data.loc[(data['location'] == location) & (data['survey_series'] == 'cdc_series'), 'value'] - data.loc[(data['location'] == location) & (data['survey_series'] == 'cdc_series'), 'lower'],
                   data.loc[(data['location'] == location) & (data['survey_series'] == 'cdc_series'), 'upper'] - data.loc[(data['location'] == location) & (data['survey_series'] == 'cdc_series'), 'value']),
                 fmt='o', color='black', ecolor='lightgray', elinewidth=3, capsize=0)
    plt.title(location)
    plt.xticks(rotation=60)
    plt.show()

'''

def get_infected(location_id: int,
                 n_draws: int,
                 model_in_dir: str,
                 model_out_dir: str):
    model_data, population = load_inputs(location_id, Path(model_in_dir))
    
    measure = 'deaths'
    
    data = model_data[measure]['daily'].copy()
    data = data.rename(measure)
    ratio = model_data[measure]['ratio'].copy()
    
    dep_trans_in = lambda x: np.log(x / population)
    dep_se_trans_in = lambda x: 1. / (x / population) ** 0.23
    dep_trans_out = lambda x: np.exp(x) * population - 1e-4
    
    spline_options = {'spline_knots_type':'domain',
                      'spline_degree':3,
                      'spline_l_linear':True,
                      'spline_r_linear':True}
    
    n_days = (data.reset_index()['date'].max() - data.reset_index()['date'].min()).days
    n_knots = int(np.ceil(n_days / KNOT_DAYS))
    
    data = data.clip(0, np.inf)
    data += 0.1
    data, smooth_data, mr_model = mr_spline.estimate_time_series(
        data=data.reset_index(),
        dep_var=measure,
        spline_options=spline_options,
        n_knots=n_knots,
        dep_trans_in=dep_trans_in,
        dep_var_se=measure,
        dep_se_trans_in=dep_se_trans_in,
    )
    
    # fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    # ax[0].plot(dep_trans_out(data['y']), label='observed')
    # ax[0].plot(dep_trans_out(smooth_data), label='smoothed')
    # ax[0].legend()
    # ax[1].plot(dep_trans_out(data['y']).cumsum(), label='observed')
    # ax[1].plot(dep_trans_out(smooth_data).cumsum(), label='smoothed')
    # ax[1].legend()
    # fig.show()
    
    def weighted_avg(x: pd.Series, w: pd.Series):
        _w = w[x.index]
        return (x * (_w / _w.sum())).sum()
    
    rmse_window = 28
    residuals = (data['y'] - smooth_data).rename('residuals')
    if residuals.isnull().any():
        raise ValueError('NAs in residual dataset.')
    wrmse = residuals ** 2
    wrmse = wrmse.rolling(window=rmse_window, min_periods=rmse_window, center=True).apply(lambda x: weighted_avg(x, 1 / (data['se'] ** 2)))
    wrmse = wrmse.fillna(method='bfill')
    wrmse_tail_scaler = (wrmse.isnull().cumsum() + 1)
    wrmse = wrmse.fillna(method='ffill')
    wrmse *= wrmse_tail_scaler
    wrmse = np.sqrt(wrmse)
    wrmse = wrmse.rename('wrmse')
    
    noisy_draws = np.random.normal(smooth_data.values, wrmse.values, (n_draws, smooth_data.size))
    noisy_draws = dep_trans_out(noisy_draws.T)
    noisy_draws /= ratio[smooth_data.index].to_frame().values
    # plt.scatter(data.index, data['y'], alpha=0.25)
    # plt.plot(smooth_data.index, smooth_draws.mean(axis=0), color='red')
    # plt.fill_between(smooth_data.index, np.percentile(smooth_draws, 2.5, axis=0), np.percentile(smooth_draws, 97.5, axis=0),
    #                  color='red', alpha=0.5)
    noisy_draws = [pd.DataFrame({'date':smooth_data.index, 'infections':noisy_draw}) for noisy_draw in noisy_draws.T]

    return {measure: noisy_draws}

"""
    _estimator = functools.partial(
        mr_spline.estimate_time_series,
        dep_var=measure,
        spline_options=spline_options,
        n_knots=n_knots,
        dep_var_se=measure,
    )
    with multiprocessing.Pool(25) as p:
        smooth_draws = list(tqdm.tqdm(p.imap(_estimator, noisy_draws), total=n_draws, file=sys.stdout))
    smooth_draws = np.vstack([sd[1] for sd in smooth_draws]).T
"""

def load_inputs(location_id: int, model_in_dir: Path) -> Tuple[Dict, float]:
    data_path = model_in_dir / 'model_data.pkl'
    with data_path.open('rb') as file:
        model_data = pickle.load(file)
    model_data = model_data[location_id]
    
    pop_path = model_in_dir / 'pop_data.h5'
    population = pd.read_hdf(pop_path)
    population = population[location_id]
    
    return model_data, population


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--location_id', help='Location being modeled.', type=int
    )
    parser.add_argument(
        '--n_draws', help='How many samples to take.', type=int
    )
    parser.add_argument(
        '--model_in_dir', help='Directory from which model inputs are read.', type=str
    )
    parser.add_argument(
        '--model_out_dir', help='Directory to which model outputs are written.', type=str
    )
    args = parser.parse_args()
    
    get_infected(**args)


if __name__ == '__main__':
    os.environ['OMP_NUM_THREADS'] = '6'
    main()
