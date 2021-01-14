from typing import Tuple, Dict
from pathlib import Path
from loguru import logger
import dill as pickle

import pandas as pd

def load_model_inputs(location_id: int, model_in_dir: Path) -> Tuple[Dict, float]:
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
    
    return model_data, population, location_name


def load_extra_plot_inputs(location_id: int, model_in_dir: Path):
    sero_path = model_in_dir / 'sero_data.h5'
    sero_data = pd.read_hdf(sero_path)
    sero_data = (sero_data
                 .loc[sero_data['location_id'] == location_id]
                 .set_index('date'))
    del sero_data['location_id']

    test_path = model_in_dir / 'test_data.h5'
    test_data = pd.read_hdf(test_path)
    test_data = test_data.loc[location_id]
    
    return sero_data, test_data
