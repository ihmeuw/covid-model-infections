from pathlib import Path
import dill as pickle
from loguru import logger

import pandas as pd

from covid_shared import shell_tools, cli_tools

from covid_model_infections import data, model
from covid_model_infections.utils import TIMELINE

## TODO:
##     - holdout
##     - ratio draws
##     - other hospital data (lowest hanging fruit is "admissions", maybe population as well)
##     - 
##     - logging!
##     - do we need to use cumulative? see if we are biased downward
##     - throw error if we have case/death/admission data but not ratio (should use parent)
##     - modularize data object creation
##     - make consistent timeline with IDR, IFR, IHR models


def make_infections(app_metadata: cli_tools.Metadata,
                    model_inputs_root: Path,
                    infection_fatality_root: Path,
                    infection_hospitalization_root: Path,
                    infection_detection_root: Path,
                    output_root: Path,
                    holdout_days: int,
                    n_draws: int):
    logger.info('Creating directories.')
    model_in_dir = output_root / 'model_inputs'
    model_out_dir = output_root / 'model_outputs'
    shell_tools.mkdir(model_in_dir)
    shell_tools.mkdir(model_out_dir)
    
    logger.info('Loading epi report data.')
    cumul_deaths, daily_deaths = data.load_model_inputs(model_inputs_root, 'deaths')
    cumul_hospital, daily_hospital = data.load_model_inputs(model_inputs_root, 'hospitalizations')
    cumul_cases, daily_cases = data.load_model_inputs(model_inputs_root, 'cases')
    
    logger.info('Loading estimated ratios.')
    ifr_data = data.load_ifr(infection_fatality_root)
    ihr_data = data.load_ihr(infection_hospitalization_root)
    idr_data = data.load_idr(infection_detection_root)
    

    logger.info('Loading extra data for plotting.')
    sero_data = data.load_sero_data(infection_detection_root)
    test_data = data.load_testing_data(infection_detection_root)
    
    logger.info('Loading supplemental data.')
    hierarchy = data.load_hierarchy(model_inputs_root)
    pop_data = data.load_population(model_inputs_root)
    
    logger.info('Creating model input data structure.')
    most_detailed = hierarchy['most_detailed'] == 1
    location_ids = hierarchy.loc[most_detailed, 'location_id'].to_list()
    model_data = {}
    for location_id in location_ids:
        location_model_data = {}
        if location_id in daily_deaths.reset_index()['location_id'].values and location_id in ifr_data.reset_index()['location_id'].values:
            location_model_data.update({'deaths':{'daily':daily_deaths.loc[location_id],
                                                  'cumul':cumul_deaths.loc[location_id],
                                                  'ratio':ifr_data.loc[location_id],
                                                  'lag': TIMELINE['deaths'],},})
        if location_id in daily_hospital.reset_index()['location_id'].values and location_id in ihr_data.reset_index()['location_id'].values:
            location_model_data.update({'hospitalizations':{'daily':daily_hospital.loc[location_id],
                                                            'cumul':cumul_hospital.loc[location_id],
                                                            'ratio':ihr_data.loc[location_id],
                                                            'lag': TIMELINE['hospitalizations'],},})
        if location_id in daily_cases.reset_index()['location_id'].values and location_id in idr_data.reset_index()['location_id'].values:
            location_model_data.update({'cases':{'daily':daily_cases.loc[location_id],
                                                 'cumul':cumul_cases.loc[location_id],
                                                 'ratio':idr_data.loc[location_id],
                                                 'lag': TIMELINE['cases'],},})
        model_data.update({
            location_id:location_model_data
        })
    
    logger.info('Writing intermediate files.')
    data_path = model_in_dir / 'model_data.pkl'
    with data_path.open('wb') as file:
        pickle.dump(model_data, file, -1)
    hierarchy.to_hdf(model_in_dir / 'hierarchy.h5', key='data', mode='w')
    pop_data.to_hdf(model_in_dir / 'pop_data.h5', key='data', mode='w')
    sero_data.to_hdf(model_in_dir / 'sero_data.h5', key='data', mode='w')
    test_data.to_hdf(model_in_dir / 'test_data.h5', key='data', mode='w')
    
    logger.info('TBD')
    model.runner.__file__
