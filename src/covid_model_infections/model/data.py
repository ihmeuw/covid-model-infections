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
    sero_data = sero_data.reset_index()
    sero_data = (sero_data
                 .loc[sero_data['location_id'] == location_id]
                 .drop('location_id', axis=1)
                 .set_index('date'))
    
    ifr_model_data_path = model_in_dir / 'ifr_model_data.h5'
    ifr_model_data = pd.read_hdf(ifr_model_data_path)
    ifr_model_data = ifr_model_data.reset_index()
    ifr_model_data = (ifr_model_data
                      .loc[ifr_model_data['location_id'] == location_id]
                      .drop('location_id', axis=1)
                      .set_index('date'))
    
    ihr_model_data_path = model_in_dir / 'ihr_model_data.h5'
    ihr_model_data = pd.read_hdf(ihr_model_data_path)
    ihr_model_data = ihr_model_data.reset_index()
    ihr_model_data = (ihr_model_data
                      .loc[ihr_model_data['location_id'] == location_id]
                      .drop('location_id', axis=1)
                      .set_index('date'))
    
    idr_model_data_path = model_in_dir / 'idr_model_data.h5'
    idr_model_data = pd.read_hdf(idr_model_data_path)
    idr_model_data = idr_model_data.reset_index()
    idr_model_data = (idr_model_data
                      .loc[idr_model_data['location_id'] == location_id]
                      .drop('location_id', axis=1)
                      .set_index('date'))
    ratio_model_inputs = {
        'deaths':ifr_model_data,
        'hospitalizations':ihr_model_data,
        'cases':idr_model_data
    }
    
    return sero_data, ratio_model_inputs
