from pathlib import Path
import dill as pickle
from loguru import logger

import pandas as pd

from covid_shared import shell_tools, cli_tools

from covid_model_infections import data, cluster, model
from covid_model_infections.utils import TIMELINE
from covid_model_infections.pdf_merger import pdf_merger

## TODO:
##     - holdout
##     - ratio draws
##     - hospital census?
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
    if holdout_days > 0:
        raise ValueError('Holdout not yet implemented.')
    
    logger.info('Creating directories.')
    model_in_dir = output_root / 'model_inputs'
    model_out_dir = output_root / 'model_outputs'
    plot_dir = output_root / 'plots'
    shell_tools.mkdir(model_in_dir)
    shell_tools.mkdir(model_out_dir)
    shell_tools.mkdir(plot_dir)
    
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
    parent_ids = hierarchy.loc[most_detailed, 'parent_id'].to_list()
    location_names = hierarchy.loc[most_detailed, 'location_name'].to_list()
    model_data = {}
    for location_id, parent_id, location_name in zip(location_ids, parent_ids, location_names):
        location_model_data = {}
        if location_id in daily_deaths.reset_index()['location_id'].values:
            if location_id in ifr_data.reset_index()['location_id'].values:
                ratio_location_id = location_id
            else:
                logger.info(f'Using parent IFR for {location_name}.')
                ratio_location_id = parent_id
            location_model_data.update({'deaths':{'daily':daily_deaths.loc[location_id],
                                                  'cumul':cumul_deaths.loc[location_id],
                                                  'ratio':ifr_data.loc[ratio_location_id],
                                                  'lag': TIMELINE['deaths'],},})
        if location_id in daily_hospital.reset_index()['location_id'].values:
            if location_id in ihr_data.reset_index()['location_id'].values:
                ratio_location_id = location_id
            else:
                logger.info(f'Using parent IHR for {location_name}.')
                ratio_location_id = parent_id
            location_model_data.update({'hospitalizations':{'daily':daily_hospital.loc[location_id],
                                                            'cumul':cumul_hospital.loc[location_id],
                                                            'ratio':ihr_data.loc[ratio_location_id],
                                                            'lag': TIMELINE['hospitalizations'],},})
        if location_id in daily_cases.reset_index()['location_id'].values:
            if location_id in idr_data.reset_index()['location_id'].values:
                ratio_location_id = location_id
            else:
                logger.info(f'Using parent IDR for {location_name}.')
                ratio_location_id = parent_id
            location_model_data.update({'cases':{'daily':daily_cases.loc[location_id],
                                                 'cumul':cumul_cases.loc[location_id],
                                                 'ratio':idr_data.loc[ratio_location_id],
                                                 'lag': TIMELINE['cases'],},})
        model_data.update({
            location_id:location_model_data
        })
    
    logger.info('Writing intermediate files.')
    data_path = model_in_dir / 'model_data.pkl'
    with data_path.open('wb') as file:
        pickle.dump(model_data, file, -1)
    hierarchy_path = model_in_dir / 'hierarchy.h5'
    hierarchy.to_hdf(hierarchy_path, key='data', mode='w')
    pop_path = model_in_dir / 'pop_data.h5'
    pop_data.to_hdf(pop_path, key='data', mode='w')
    sero_path = model_in_dir / 'sero_data.h5'
    sero_data.to_hdf(sero_path, key='data', mode='w')
    test_path = model_in_dir / 'test_data.h5'
    test_data.to_hdf(test_path, key='data', mode='w')
    
    logger.info('Launching models.')
    job_args_map = {
        location_id: [model.runner.__file__,
                      location_id, n_draws, str(model_in_dir), str(model_out_dir), str(plot_dir)]
        for location_id in location_ids
    }
    cluster.run_cluster_jobs('covid_infection_model', output_root, job_args_map)
    
    logger.debug('Merging PDFs.')
    possible_pdfs = [f'{l}.pdf' for l in hierarchy['location_id']]
    existing_pdfs = [str(x).split('/')[-1] for x in plot_dir.iterdir() if x.is_file()]
    pdf_paths = [pdf for pdf in possible_pdfs if pdf in existing_pdfs]
    pdf_location_ids = [int(pdf_path[:-4]) for pdf_path in pdf_paths]
    pdf_location_names = [hierarchy.loc[hierarchy['location_id'] == location_id, 'location_name'].item() for location_id in pdf_location_ids]
    pdf_parent_ids = [hierarchy.loc[hierarchy['location_id'] == location_id, 'parent_id'].item() for location_id in pdf_location_ids]
    pdf_parent_names = [hierarchy.loc[hierarchy['location_id'] == parent_id, 'location_name'].item() for parent_id in pdf_parent_ids]
    pdf_paths = [str(plot_dir / pdf_path) for pdf_path in pdf_paths]
    pdf_out_path = output_root / f'past_infections_{str(output_root).split("/")[-1]}.pdf'
    pdf_merger(pdf_paths, pdf_location_names, pdf_parent_names, str(pdf_out_path))
    
    logger.debug('Compiling infection draws.')
    draws = []
    for draws_path in [result_path for result_path in model_out_dir.iterdir() if str(result_path).endswith('draws.h5')]:
        draws.append(pd.read_hdf(draws_path))
    draws = pd.concat(draws)
    draw_path = output_root / 'infection_draws.h5'
    draws.to_hdf(draw_path, key='data', mode='w')
    draws = [draws[c] for c in draws.columns]
    
    logger.debug('Compiling other model outputs.')
    outputs = {}
    for outputs_path in [result_path for result_path in model_out_dir.iterdir() if str(result_path).endswith('_data.pkl')]:
        with outputs_path.open('rb') as outputs_file:
            outputs.update(pickle.load(outputs_file))
    output_path = output_root / 'output_data.pkl'
    with output_path.open('wb') as file:
        pickle.dump(output_path, file, -1)
    deaths = {k:v['deaths']['daily'] for k, v in outputs.items() if 'deaths' in list(outputs[k].keys())}
    deaths = [pd.concat([v, pd.DataFrame({'location_id':k}, index=v.index)], axis=1).reset_index() for k, v in deaths.items()]
    deaths = pd.concat(deaths)
    deaths = (deaths
              .set_index(['location_id', 'date'])
              .sort_index()
              .rename(columns={'deaths':'deaths_draw'}))
    draws = [pd.concat([draw, deaths], axis=1) for draw in draws]
    
    logger.debug('Saving draws.')
    
        
    logger.info(f'Model run complete -- {str(output_root)}.')
    