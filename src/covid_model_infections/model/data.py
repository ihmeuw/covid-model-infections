from typing import Tuple, Dict
from pathlib import Path
from loguru import logger
import dill as pickle

import pandas as pd


def load_model_inputs(location_id: int, model_in_dir: Path) -> Tuple[Dict, float]:
    hierarchy_path = model_in_dir / 'hierarchy.parquet'
    hierarchy = pd.read_parquet(hierarchy_path)
    location_name = hierarchy.loc[hierarchy['location_id'] == location_id, 'location_name'].item()
    path_to_top_parent = hierarchy.loc[hierarchy['location_id'] == location_id, 'path_to_top_parent'].item()
    is_us = '102' in path_to_top_parent.split(',')
    logger.info(f'Model location: {location_name}')
    
    data_path = model_in_dir / 'model_data.pkl'
    with data_path.open('rb') as file:
        model_data = pickle.load(file)
    model_data = model_data[location_id]
    
    pop_path = model_in_dir / 'pop_data.parquet'
    population = pd.read_parquet(pop_path)
    population = population.loc[location_id].item()
    
    return model_data, population, location_name, is_us


def load_extra_plot_inputs(location_id: int, model_in_dir: Path):
    sero_path = model_in_dir / 'sero_data.parquet'
    sero_data = pd.read_parquet(sero_path)
    sero_data = sero_data.reset_index()
    sero_data = (sero_data
                 .loc[sero_data['location_id'] == location_id]
                 .drop('location_id', axis=1)
                 .set_index('date'))
    
    reinfection_path = model_in_dir / 'reinfection_data.parquet'
    reinfection_data = pd.read_parquet(reinfection_path)
    if location_id in reinfection_data.reset_index()['location_id'].to_list():
        reinfection_data = reinfection_data.loc[location_id]
    else:
        reinfection_data = pd.DataFrame()
    
    ifr_model_data_path = model_in_dir / 'ifr_model_data.parquet'
    ifr_model_data = pd.read_parquet(ifr_model_data_path)
    ifr_model_data = ifr_model_data.reset_index()
    ifr_model_data = (ifr_model_data
                      .loc[ifr_model_data['location_id'] == location_id]
                      .drop('location_id', axis=1)
                      .set_index('date'))
    
    ihr_model_data_path = model_in_dir / 'ihr_model_data.parquet'
    ihr_model_data = pd.read_parquet(ihr_model_data_path)
    ihr_model_data = ihr_model_data.reset_index()
    ihr_model_data = (ihr_model_data
                      .loc[ihr_model_data['location_id'] == location_id]
                      .drop('location_id', axis=1)
                      .set_index('date'))
    
    idr_model_data_path = model_in_dir / 'idr_model_data.parquet'
    idr_model_data = pd.read_parquet(idr_model_data_path)
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
    
    return sero_data, reinfection_data, ratio_model_inputs
