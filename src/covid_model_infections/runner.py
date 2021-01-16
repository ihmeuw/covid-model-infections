import sys
from pathlib import Path
import dill as pickle
from loguru import logger
import functools
import multiprocessing
from tqdm import tqdm

import pandas as pd

from covid_shared import shell_tools, cli_tools

from covid_model_infections import data, cluster, model
from covid_model_infections.utils import TIMELINE, IDR_LIMITS
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
    infections_draws_dir = output_root / 'infections_draws'
    ratio_draws_dir = output_root / 'ratio_draws'
    shell_tools.mkdir(model_in_dir)
    shell_tools.mkdir(model_out_dir)
    shell_tools.mkdir(plot_dir)
    shell_tools.mkdir(infections_draws_dir)
    shell_tools.mkdir(ratio_draws_dir)
    
    logger.info('Loading epi report data.')
    cumul_deaths, daily_deaths = data.load_model_inputs(model_inputs_root, 'deaths')
    cumul_hospital, daily_hospital = data.load_model_inputs(model_inputs_root, 'hospitalizations')
    cumul_cases, daily_cases = data.load_model_inputs(model_inputs_root, 'cases')
    
    logger.info('Loading estimated ratios.')
    ifr_data = data.load_ifr(infection_fatality_root)
    ihr_data = data.load_ihr(infection_hospitalization_root)
    idr_data = data.load_idr(infection_detection_root, IDR_LIMITS)

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
    unmodeled_location_ids = []
    modeled_location_ids = []
    for location_id, parent_id, location_name in zip(location_ids, parent_ids, location_names):
        location_model_data = {}
        modeled_location = False
        if location_id in daily_deaths.reset_index()['location_id'].values:
            if location_id in ifr_data.reset_index()['location_id'].values:
                ratio_location_id = location_id
            else:
                logger.info(f'Using parent IFR for {location_name}.')
                ratio_location_id = parent_id
            modeled_location = True
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
            modeled_location = True
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
            modeled_location = True
            location_model_data.update({'cases':{'daily':daily_cases.loc[location_id],
                                                 'cumul':cumul_cases.loc[location_id],
                                                 'ratio':idr_data.loc[ratio_location_id],
                                                 'lag': TIMELINE['cases'],},})
        if modeled_location:
            modeled_location_ids.append(location_id)
            model_data.update({
                location_id:location_model_data
            })
        else:
            unmodeled_location_ids.append(unmodeled_location_ids)
            
    logger.info('Identifying unmodeled locations.')
    app_metadata.update({'unmodeled_location_ids': unmodeled_location_ids})
    if unmodeled_location_ids:
        logger.debug(f'Insufficent data exists for the following location_ids: {", ".join(unmodeled_location_ids)}')
    
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
        for location_id in modeled_location_ids
    }
    cluster.run_cluster_jobs('covid_infection_model', output_root, job_args_map)
    
    logger.info('Merging PDFs.')
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
    
    logger.info('Compiling infection draws.')
    infections_draws = []
    for draws_path in [result_path for result_path in model_out_dir.iterdir() if str(result_path).endswith('_infections_draws.h5')]:
        infections_draws.append(pd.read_hdf(draws_path))
    infections_draws = pd.concat(infections_draws)
    completed_modeled_location_ids = infections_draws.reset_index()['location_id'].unique().to_list()
    draw_path = output_root / 'infections_draws.h5'
    infections_draws.to_hdf(draw_path, key='data', mode='w')
    infections_mean = infections_draws.mean(axis=1).rename('infections_mean')
    infections_draws = [infections_draws[c] for c in infections_draws.columns]
    
    logger.info('Identifying failed models.')
    failed_model_location_ids = set(modeled_location_ids) - set(completed_modeled_location_ids)
    app_metadata.update({'failed_model_location_ids': failed_model_location_ids})
    if failed_model_location_ids:
        logger.debug(f'Models failed for the following location_ids: {", ".join(failed_model_location_ids)}')
    
    logger.info('Compiling IFR draws.')
    ifr_draws = []
    for draws_path in [result_path for result_path in model_out_dir.iterdir() if str(result_path).endswith('_ifr_draws.h5')]:
        ifr_draws.append(pd.read_hdf(draws_path))
    ifr_draws = pd.concat(ifr_draws)
    draw_path = output_root / 'ifr_draws.h5'
    ifr_draws.to_hdf(draw_path, key='data', mode='w')
    ifr_mean = ifr_draws.mean(axis=1).rename('infections_mean')
    ifr_draws = [pd.concat([ifr_draws[c], ifr_mean], axis=1) for c in ifr_draws.columns]
    
    logger.info('Compiling other model outputs.')
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
              .sort_index())
    infections_draws = [pd.concat([draw, infections_mean, deaths], axis=1) for draw in infections_draws]
    
    logger.info('Writing SEIR inputs - infections draw files.')
    _inf_writer = functools.partial(
        data.write_infections_draws,
        infections_draws_dir=infections_draws_dir,
        inf_to_death=TIMELINE['deaths'],
    )
    with multiprocessing.Pool(int(cluster.F_THREAD) - 2) as p:
        infections_draws_paths = list(tqdm(p.imap(_inf_writer, infections_draws), total=n_draws, file=sys.stdout))
    
    logger.info('Writing SEIR inputs - IFR.')
    _ifr_writer = functools.partial(
        data.write_ratio_draws,
        ratio_draws_dir=ratio_draws_dir,
    )
    with multiprocessing.Pool(int(cluster.F_THREAD) - 2) as p:
        ratio_draws_paths = list(tqdm(p.imap(_ifr_writer, ifr_draws), total=n_draws, file=sys.stdout))
        
    logger.info(f'Model run complete -- {str(output_root)}.')
    