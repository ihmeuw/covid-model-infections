from typing import Tuple, Dict
from pathlib import Path
from loguru import logger
import dill as pickle

import pandas as pd


def compile_input_data_object(location_id: int,
                              timeline: Dict,
                              daily_deaths: pd.Series, cumul_deaths: pd.Series, ifr: pd.Series,
                              daily_hospital: pd.Series, cumul_hospital: pd.Series, ihr: pd.Series,
                              daily_cases: pd.Series, cumul_cases: pd.Series, idr: pd.Series,):
    location_model_data = {}
    modeled_location = False
    # DEATHS
    if location_id in daily_deaths.reset_index()['location_id'].values:
        modeled_location = True
        location_model_data.update({'deaths':{'daily': daily_deaths.loc[location_id],
                                              'cumul': cumul_deaths.loc[location_id],
                                              'ratio': ifr.loc[location_id],
                                              'lag': timeline['deaths'],},})
    # HOSPITAL ADMISSIONS
    if location_id in daily_hospital.reset_index()['location_id'].values:
        modeled_location = True
        location_model_data.update({'hospitalizations':{'daily': daily_hospital.loc[location_id],
                                                        'cumul': cumul_hospital.loc[location_id],
                                                        'ratio': ihr.loc[location_id],
                                                        'lag': timeline['hospitalizations'],},})
    # CASES
    if location_id in daily_cases.reset_index()['location_id'].values:
        modeled_location = True
        location_model_data.update({'cases':{'daily': daily_cases.loc[location_id],
                                             'cumul': cumul_cases.loc[location_id],
                                             'ratio': idr.loc[location_id],
                                             'lag': timeline['cases'],},})
        
    return location_model_data, modeled_location


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
    model_data, modeled_location = compile_input_data_object(location_id=location_id,
                                                             **model_data)
    
    pop_path = model_in_dir / 'pop_data.parquet'
    population = pd.read_parquet(pop_path)
    population = population.loc[location_id].item()
    
    return model_data, modeled_location, population, location_name, is_us


def load_extra_plot_inputs(location_id: int, model_in_dir: Path):
    sero_path = model_in_dir / 'sero_data.parquet'
    sero_data = pd.read_parquet(sero_path)
    sero_data = sero_data.reset_index()
    sero_data = (sero_data
                 .loc[sero_data['location_id'] == location_id]
                 .drop('location_id', axis=1)
                 .set_index('date'))
    
    daily_reinfection_rr_path = model_in_dir / 'daily_reinfection_rr.parquet'
    daily_reinfection_rr = pd.read_parquet(daily_reinfection_rr_path)
    if location_id in daily_reinfection_rr.reset_index()['location_id'].to_list():
        daily_reinfection_rr = daily_reinfection_rr.loc[location_id]
    else:
        daily_reinfection_rr = pd.DataFrame()
    
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
        'deaths': ifr_model_data,
        'hospitalizations': ihr_model_data,
        'cases': idr_model_data,
    }
    
    return sero_data, daily_reinfection_rr, ratio_model_inputs
