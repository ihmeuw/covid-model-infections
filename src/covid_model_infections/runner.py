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
from covid_model_infections.utils import TIMELINE, IDR_UPPER_LIMIT  # , IDR_LIMITS
from covid_model_infections.pdf_merger import pdf_merger

MP_THREADS = 25

## TODO:
##     - holdouts
##     - modularize data object creation
##     - make shared source for timeline with IDR, IFR, IHR models


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
    shell_tools.mkdir(model_in_dir)
    shell_tools.mkdir(model_out_dir)
    shell_tools.mkdir(plot_dir)
    shell_tools.mkdir(infections_draws_dir)
    
    logger.info('Loading supplemental data.')
    hierarchy = data.load_hierarchy(model_inputs_root)
    pop_data = data.load_population(model_inputs_root)
    
    logger.info('Loading epi report data.')
    cumul_deaths, daily_deaths, deaths_manipulation_metadata = data.load_model_inputs(model_inputs_root, hierarchy, 'deaths')
    cumul_hospital, daily_hospital, hospital_manipulation_metadata = data.load_model_inputs(model_inputs_root, hierarchy, 'hospitalizations')
    cumul_cases, daily_cases, cases_manipulation_metadata = data.load_model_inputs(model_inputs_root, hierarchy, 'cases')
    app_metadata.update({'data_manipulation': {
        'deaths':deaths_manipulation_metadata,
        'hospital':hospital_manipulation_metadata,
        'cases':cases_manipulation_metadata,
    }})
    
    logger.info('Loading estimated ratios and adding draw directories.')
    ifr_data = data.load_ifr(infection_fatality_root)
    ifr_model_data = data.load_ifr_data(infection_fatality_root)
    ifr_risk_data = data.load_ifr_risk_adjustment(infection_fatality_root)
    ihr_data = data.load_ihr(infection_hospitalization_root)
    ihr_model_data = data.load_ihr_data(infection_hospitalization_root)
    # Assumes IDR has estimated floor already applied
    idr_data = data.load_idr(infection_detection_root, (0, IDR_UPPER_LIMIT))
    idr_model_data = data.load_idr_data(infection_detection_root)
    # TODO: centralize this information, is used elsewhere...
    estimated_ratios = {'deaths':('ifr', ifr_data.copy()),
                        'hospitalizations':('ihr', ihr_data.copy()),
                        'cases':('idr', idr_data.copy()),}

    logger.info('Loading extra data for plotting.')
    sero_data = data.load_sero_data(infection_detection_root)
    test_data = data.load_testing_data(infection_detection_root)
        
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
            unmodeled_location_ids.append(location_id)
            
    logger.info('Identifying unmodeled locations.')
    app_metadata.update({'unmodeled_location_ids': unmodeled_location_ids})
    if unmodeled_location_ids:
        logger.debug(f'Insufficent data exists for the following location_ids: {", ".join([str(l) for l in unmodeled_location_ids])}')
    
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
    ifr_data_path = model_in_dir / 'ifr_model_data.h5'
    ifr_model_data.to_hdf(ifr_data_path, key='data', mode='w')
    ihr_data_path = model_in_dir / 'ihr_model_data.h5'
    ihr_model_data.to_hdf(ihr_data_path, key='data', mode='w')
    idr_data_path = model_in_dir / 'idr_model_data.h5'
    idr_model_data.to_hdf(idr_data_path, key='data', mode='w')
    
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
    completed_modeled_location_ids = infections_draws.reset_index()['location_id'].unique().tolist()
    
    logger.info('Identifying failed models.')
    failed_model_location_ids = list(set(modeled_location_ids) - set(completed_modeled_location_ids))
    app_metadata.update({'failed_model_location_ids': failed_model_location_ids})
    if failed_model_location_ids:
        logger.debug(f'Models failed for the following location_ids: {", ".join([str(l) for l in failed_model_location_ids])}')
    
    logger.info('Compiling other model outputs.')
    outputs = {}
    for outputs_path in [result_path for result_path in model_out_dir.iterdir() if str(result_path).endswith('_data.pkl')]:
        with outputs_path.open('rb') as outputs_file:
            outputs.update(pickle.load(outputs_file))
    output_path = output_root / 'output_data.pkl'
    with output_path.open('wb') as file:
        pickle.dump(outputs, file, -1)
    deaths = {k:v['deaths']['daily'] for k, v in outputs.items() if 'deaths' in list(outputs[k].keys())}
    deaths = [pd.concat([v, pd.DataFrame({'location_id':k}, index=v.index)], axis=1).reset_index() for k, v in deaths.items()]
    deaths = pd.concat(deaths)
    deaths = (deaths
              .set_index(['location_id', 'date'])
              .sort_index()
              .loc[:, 'deaths'])
    
    logger.info('Writing SEIR inputs - infections draw files.')
    infections_draws_cols = infections_draws.columns
    infections_draws = pd.concat([infections_draws, deaths], axis=1)
    infections_draws = infections_draws.sort_index()
    deaths = infections_draws['deaths'].copy()
    infections_draws = [infections_draws[infections_draws_col].copy() for infections_draws_col in infections_draws_cols]
    infections_draws = [pd.concat([infections_draw, deaths], axis=1) for infections_draw in infections_draws]
    _inf_writer = functools.partial(
        data.write_infections_draws,
        infections_draws_dir=infections_draws_dir,
    )
    with multiprocessing.Pool(MP_THREADS) as p:
        infections_draws_paths = list(tqdm(p.imap(_inf_writer, infections_draws), total=n_draws, file=sys.stdout))
    
    for measure, (estimated_ratio, ratio_prior_data) in estimated_ratios.items():
        logger.info(f'Compiling {estimated_ratio.upper()} draws.')
        ratio_draws = []
        for draws_path in [result_path for result_path in model_out_dir.iterdir() if str(result_path).endswith(f'_{estimated_ratio}_draws.h5')]:
            ratio_draws.append(pd.read_hdf(draws_path))
        ratio_draws = pd.concat(ratio_draws)
        
        logger.info(f'Filling {estimated_ratio.upper()} with original model estimate where we do not have a posterior.')
        ratio_draws_cols = ratio_draws.columns
        ratio_prior_data = ratio_prior_data.reset_index()
        is_missing = ~ratio_prior_data['location_id'].isin(ratio_draws.reset_index()['location_id'].unique())
        is_model_loc = ratio_prior_data['location_id'].isin(completed_modeled_location_ids)
        ratio_prior_data = ratio_prior_data.loc[is_missing & is_model_loc]
        ratio_prior_data = (ratio_prior_data
                            .set_index(['location_id', 'date'])
                            .loc[:, 'ratio'])
        ratio_draws = ratio_draws.join(ratio_prior_data, how='outer')
        ratio_draws[ratio_draws_cols] = (ratio_draws[ratio_draws_cols]
                                         .apply(lambda x: x.fillna(ratio_draws['ratio'])))
        del ratio_draws['ratio']

        logger.info(f'Writing SEIR inputs - {estimated_ratio.upper()} draw files.')
        if estimated_ratio == 'ifr':
            ratio_draws = ratio_draws.join(ifr_risk_data, on='location_id')
            ratio_draws = ratio_draws.sort_index()
            ifr_risk_data = ratio_draws[['lr_adj', 'hr_adj']].copy()
        else:
            ratio_draws = ratio_draws.sort_index()
        ratio_draws = [ratio_draws[[ratio_draws_col]].copy() for ratio_draws_col in ratio_draws_cols]
        if estimated_ratio == 'ifr':
            ratio_draws = [pd.concat([ratio_draw, ifr_risk_data], axis=1) for ratio_draw in ratio_draws]
        ratio_draws_dir = output_root / f'{estimated_ratio}_draws'
        shell_tools.mkdir(ratio_draws_dir)
        _ratio_writer = functools.partial(
            data.write_ratio_draws,
            estimated_ratio=estimated_ratio,
            duration=TIMELINE[measure],
            ratio_draws_dir=ratio_draws_dir,
        )
        with multiprocessing.Pool(MP_THREADS) as p:
            ratio_draws_paths = list(tqdm(p.imap(_ratio_writer, ratio_draws), total=n_draws, file=sys.stdout))
            
    logger.info('Writing serology data for grid plots.')
    sero_data['geo_accordance'] = 1 - sero_data['geo_accordance']
    sero_data['included'] = 1 - sero_data[['geo_accordance', 'manual_outlier']].max(axis=1)
    sero_data = sero_data.rename(columns={'seroprev_mean':'value'})
    sero_data = sero_data.loc[:, ['included', 'value']]
    sero_path = output_root / 'sero_data.csv'
    sero_data.reset_index().to_csv(sero_path, index=False)
        
    logger.info(f'Model run complete -- {str(output_root)}.')
    