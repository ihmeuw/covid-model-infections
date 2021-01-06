import os
from typing import Tuple, Dict
from pathlib import Path
import argparse
import dill as pickle
import functools
import multiprocessing
import sys
import tqdm
from loguru import logger

import pandas as pd
import numpy as np

from covid_model_infections.model import support, mr_spline
from covid_model_infections.utils import OMP_NUM_THREADS

MEASURE_KNOT_DAYS = 21
INFECTION_KNOT_DAYS = 28
SPLINE_OPTIONS = {'spline_knots_type':'domain',
                  'spline_degree':3,
                  'spline_l_linear':True,
                  'spline_r_linear':True}
LOG = True
RMSE_WINDOW = 28


def model_measure(measure: str, data: pd.Series, ratio: pd.Series, population: float, n_draws: int, lag: int) -> Dict:
    logger.info(f'{measure.capitalize()}:')
    data = data.rename(measure)
    
    dep_trans_in, dep_se_trans_in, dep_trans_out = support.get_rate_transformations(LOG, population)
    
    n_knots = support.determine_n_knots(data, MEASURE_KNOT_DAYS)
    
    logger.info('Generating smooth past curve.')
    data = data.clip(0, np.inf)
    data += 0.1
    data, smooth_data, mr_model = mr_spline.estimate_time_series(
        data=data.reset_index(),
        dep_var=measure,
        spline_options=SPLINE_OPTIONS,
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
    
    logger.info('Getting weighted RMSE.')
    wrmse = support.get_wrmse(data['y'], smooth_data, data['se'], RMSE_WINDOW)
    
    logger.info('Sampling residuals.')
    draws = np.random.normal(smooth_data.values, wrmse.values, (n_draws, smooth_data.size))
    # plt.scatter(data.index, data['y'], alpha=0.25)
    # plt.plot(smooth_data.index, draws.mean(axis=0), color='red')
    # plt.fill_between(smooth_data.index, np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0),
    #                  color='red', alpha=0.5)
    
    logger.info('Converting draws to infections.')
    smooth_data = dep_trans_out(smooth_data)
    smooth_data -= 0.1
    smooth_data = smooth_data.clip(1e-4, np.inf)
    if LOG:
        draws -= np.var(draws, axis=0, keepdims=True) / 2
    draws = dep_trans_out(draws.T)
    draws -= 0.1
    draws = draws.clip(1e-4, np.inf)
    infections = (smooth_data / ratio[smooth_data.index]).rename('infections')
    draws /= ratio[infections.index].to_frame().values
    infections.index = infections.index - pd.Timedelta(days=lag)
    draws = pd.DataFrame(draws,
                         columns=[f'draw_{d}' for d in range(n_draws)],
                         index=infections.index)

    return {'cumul':smooth_data.cumsum(), 'daily':smooth_data, 'infections_mean':infections, 'infections_draws':draws}


def model_infection_draw(input_draw: pd.Series, population: float) -> pd.Series:
    n_knots = support.determine_n_knots(input_draw, INFECTION_KNOT_DAYS)
    
    dep_trans_in, dep_se_trans_in, dep_trans_out = support.get_rate_transformations(LOG, population)
    
    input_draw, output_draw, mr_model = mr_spline.estimate_time_series(
        data=input_draw.reset_index(),
        dep_var=input_draw.name,
        spline_options=SPLINE_OPTIONS,
        n_knots=n_knots,
        dep_trans_in=dep_trans_in,
        #dep_var_se=input_draw.name,
        #dep_se_trans_in=dep_se_trans_in,
        num_submodels=10,
    )
    
    return output_draw


def get_infected(location_id: int,
                 n_draws: int,
                 model_in_dir: str,
                 model_out_dir: str):
    logger.info('Loading data.')
    input_data, population = load_inputs(location_id, Path(model_in_dir))
    
    logger.info('Running measure-specific models.')
    output_data = {measure: model_measure(measure,
                                          measure_data['daily'].copy(), measure_data['ratio'].copy(),
                                          population, n_draws, measure_data['lag']) 
                   for measure, measure_data in input_data.items()}
    
    logger.info('Fitting infection curves to draws of all available input measures.')
    input_draws = pd.concat([v['infections_draws'] for k, v in output_data.items()]).sort_index()
    input_draws = [input_draws[draw] for draw in input_draws.columns]
    _estimator = functools.partial(
        model_infection_draw,
        population=population,
    )
    with multiprocessing.Pool(25) as p:
        output_draws = list(tqdm.tqdm(p.imap(_estimator, input_draws), total=n_draws, file=sys.stdout))
    output_draws = pd.concat(output_draws, axis=1)
    _, _, dep_trans_out = support.get_rate_transformations(LOG, population)
    if LOG:
        output_draws -= np.var(output_draws.values, axis=1, keepdims=True) / 2
    output_draws = dep_trans_out(output_draws)
    

def load_inputs(location_id: int, model_in_dir: Path) -> Tuple[Dict, float]:
    hierarchy_path = model_in_dir / 'hierarchy.h5'
    hierarchy = pd.read_hdf(hierarchy_path)
    location_name = hierarchy.loc[hierarchy['location_id'] == location_id, 'location_name'].item()
    logger.info(f'Model location: {location_name}')
    
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
    os.environ['OMP_NUM_THREADS'] = OMP_NUM_THREADS
    main()
