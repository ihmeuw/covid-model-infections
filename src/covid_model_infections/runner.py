import sys
from pathlib import Path
import dill as pickle
from loguru import logger
import functools
import multiprocessing
from tqdm import tqdm

import pandas as pd

from covid_shared import shell_tools, cli_tools

from covid_model_infections import data, cluster, model, aggregation
from covid_model_infections.utils import TIMELINE
from covid_model_infections.pdf_merger import pdf_merger

MP_THREADS = 25

## TODO:
##     - holdouts
##     - modularize data object creation
##     - make shared source for timeline with IDR, IFR, IHR models


def make_infections(app_metadata: cli_tools.Metadata,
                    model_inputs_root: Path,
                    rates_root: Path,
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
    em_data = data.load_em_scalars(rates_root)
    ## DO THIS DIFFERENTLY ##
    excess_mortality = em_data['scaled'].unique().item()
    del em_data['scaled']
    ## ## ## ## ## ## ## ## ##
    cumul_deaths, daily_deaths, deaths_manipulation_metadata = data.load_model_inputs(
        model_inputs_root, hierarchy, 'deaths', excess_mortality
    )
    cumul_hospital, daily_hospital, hospital_manipulation_metadata = data.load_model_inputs(
        model_inputs_root, hierarchy, 'hospitalizations'
    )
    cumul_cases, daily_cases, cases_manipulation_metadata = data.load_model_inputs(
        model_inputs_root, hierarchy, 'cases'
    )
    
    cumul_deaths, cumul_hospital, cumul_cases,\
    daily_deaths, daily_hospital, daily_cases = data.trim_leading_zeros(
        [cumul_deaths, cumul_hospital, cumul_cases],
        [daily_deaths, daily_hospital, daily_cases],
    )
    
    app_metadata.update({'data_manipulation': {
        'deaths':deaths_manipulation_metadata,
        'hospitalizations':hospital_manipulation_metadata,
        'cases':cases_manipulation_metadata,
    }})
    measures = ['deaths', 'hospitalizations', 'cases']
    
    cumul_deaths, cumul_hospital, cumul_cases,\
    daily_deaths, daily_hospital, daily_cases = data.trim_leading_zeros(
        [cumul_deaths, cumul_hospital, cumul_cases],
        [daily_deaths, daily_hospital, daily_cases],
    )
    
    logger.info('Loading estimated ratios and adding draw directories.')
    ifr_data = data.load_ifr(rates_root)
    ifr_model_data = data.load_ifr_data(rates_root)
    ifr_risk_data = data.load_ifr_risk_adjustment(rates_root)
    reinfection_data = data.load_reinfection_data(rates_root)
    ihr_data = data.load_ihr(rates_root)
    ihr_model_data = data.load_ihr_data(rates_root)
    # Assumes IDR has estimated floor already applied
    idr_data = data.load_idr(rates_root, (0., 1.))
    idr_model_data = data.load_idr_data(rates_root)
    
    logger.info('Loading extra data for plotting.')
    sero_data = data.load_sero_data(rates_root)
    test_data = data.load_testing_data(rates_root)
        
    logger.info('Creating model input data structure.')
    most_detailed = hierarchy['most_detailed'] == 1
    location_ids = hierarchy.loc[most_detailed, 'location_id'].to_list()
    path_to_top_parents = hierarchy.loc[most_detailed, 'path_to_top_parent'].to_list()
    location_names = hierarchy.loc[most_detailed, 'location_name'].to_list()
    model_data = {}
    unmodeled_location_ids = []
    modeled_location_ids = []
    for location_id, path_to_top_parent, location_name in zip(location_ids, path_to_top_parents, location_names):
        location_model_data = {}
        modeled_location = False
        # DEATHS
        if location_id not in ifr_data.reset_index()['location_id'].values:
            for parent_id in reversed(path_to_top_parent.split(',')[:-1]):
                if int(parent_id) in ifr_data.reset_index()['location_id'].values:
                    logger.info(f'Using parent IFR for {location_name}.')
                    ifr_data = ifr_data.append(
                        pd.concat({location_id: ifr_data.loc[int(parent_id)]}, names=['location_id'])
                    )
                    ifr_risk_data = ifr_risk_data.append(
                        pd.concat({location_id: ifr_risk_data.loc[int(parent_id)]}, names=['location_id'])
                    )
                    break
                else:
                    pass
        if location_id in daily_deaths.reset_index()['location_id'].values:
            modeled_location = True
            location_model_data.update({'deaths':{'daily': daily_deaths.loc[location_id],
                                                  'cumul': cumul_deaths.loc[location_id],
                                                  'ratio': ifr_data.loc[location_id],
                                                  'lag': TIMELINE['deaths'],},})
        # HOSPITAL ADMISSIONS
        if location_id not in ihr_data.reset_index()['location_id'].values:
            for parent_id in reversed(path_to_top_parent.split(',')[:-1]):
                if int(parent_id) in ihr_data.reset_index()['location_id'].values:
                    logger.info(f'Using parent IHR for {location_name}.')
                    ihr_data = ihr_data.append(
                        pd.concat({location_id: ihr_data.loc[int(parent_id)]}, names=['location_id'])
                    )
                    break
                else:
                    pass
        if location_id in daily_hospital.reset_index()['location_id'].values:
            modeled_location = True
            location_model_data.update({'hospitalizations':{'daily': daily_hospital.loc[location_id],
                                                            'cumul': cumul_hospital.loc[location_id],
                                                            'ratio': ihr_data.loc[location_id],
                                                            'lag': TIMELINE['hospitalizations'],},})
        # CASES
        if location_id not in idr_data.reset_index()['location_id'].values:
            for parent_id in reversed(path_to_top_parent.split(',')[:-1]):
                if int(parent_id) in idr_data.reset_index()['location_id'].values:
                    logger.info(f'Using parent IDR for {location_name}.')
                    idr_data = idr_data.append(
                        pd.concat({location_id: idr_data.loc[int(parent_id)]}, names=['location_id'])
                    )
                    break
                else:
                    pass
        if location_id in daily_cases.reset_index()['location_id'].values:
            modeled_location = True
            location_model_data.update({'cases':{'daily': daily_cases.loc[location_id],
                                                 'cumul': cumul_cases.loc[location_id],
                                                 'ratio': idr_data.loc[location_id],
                                                 'lag': TIMELINE['cases'],},})
        if modeled_location:
            modeled_location_ids.append(location_id)
            model_data.update({
                location_id:location_model_data
            })
        else:
            unmodeled_location_ids.append(location_id)
    # TODO: centralize this information, is used elsewhere...
    estimated_ratios = {'deaths':('ifr', ifr_data.copy()),
                        'hospitalizations':('ihr', ihr_data.copy()),
                        'cases':('idr', idr_data.copy()),}
    
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
    reinfection_data_path = model_in_dir / 'reinfection_data.h5'
    reinfection_data.to_hdf(reinfection_data_path, key='data', mode='w')
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
        
    logger.info('Aggregating data.')
    agg_model_data = aggregation.aggregate_md_data_dict(model_data.copy(), hierarchy, measures)
    agg_outputs = aggregation.aggregate_md_data_dict(outputs.copy(), hierarchy, measures)
    agg_infections_draws = aggregation.aggregate_md_draws(infections_draws.copy(), hierarchy, MP_THREADS)
    
    logger.info('Plotting aggregates.')
    plot_parent_ids = agg_infections_draws.reset_index()['location_id'].unique().tolist()
    for plot_parent_id in tqdm(plot_parent_ids, total=len(plot_parent_ids), file=sys.stdout):
        aggregation.plot_aggregate(
            plot_parent_id,
            agg_model_data[plot_parent_id],
            agg_outputs[plot_parent_id],
            agg_infections_draws.loc[plot_parent_id],
            hierarchy,
            pop_data,
            sero_data,
            reinfection_data,
            ifr_model_data,
            ihr_model_data,
            idr_model_data,
            plot_dir
        )
    
    logger.info('Merging PDFs.')
    possible_pdfs = [f'{l}.pdf' for l in hierarchy['location_id']]
    existing_pdfs = [str(x).split('/')[-1] for x in plot_dir.iterdir() if x.is_file()]
    pdf_paths = [pdf for pdf in possible_pdfs if pdf in existing_pdfs]
    pdf_location_ids = [int(pdf_path[:-4]) for pdf_path in pdf_paths]
    pdf_location_names = [hierarchy.loc[hierarchy['location_id'] == location_id, 'location_name'].item() for location_id in pdf_location_ids]
    pdf_parent_ids = [hierarchy.loc[hierarchy['location_id'] == location_id, 'parent_id'].item() for location_id in pdf_location_ids]
    pdf_parent_names = [hierarchy.loc[hierarchy['location_id'] == parent_id, 'location_name'].item() for parent_id in pdf_parent_ids]
    pdf_levels = [hierarchy.loc[hierarchy['location_id'] == location_id, 'level'].item() for location_id in pdf_location_ids]
    pdf_paths = [str(plot_dir / pdf_path) for pdf_path in pdf_paths]
    pdf_out_path = output_root / f'past_infections_{str(output_root).split("/")[-1]}.pdf'
    pdf_merger(pdf_paths, pdf_location_names, pdf_parent_names, pdf_levels, str(pdf_out_path))
    
    logger.info('Processing mean deaths.')
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
            ratio_draws = ratio_draws.join(ifr_risk_data)
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
            
    logger.info('Writing serology data and EM scaling factor data.')
    em_path = output_root / 'em_data.csv'
    em_data.to_csv(em_path, index=False)
    sero_data['included'] = 1 - sero_data['manual_outlier']
    sero_data = sero_data.rename(columns={'seroprev_mean_no_vacc_waning':'value'})
    sero_data = sero_data.loc[:, ['included', 'value']]
    sero_path = output_root / 'sero_data.csv'
    sero_data.reset_index().to_csv(sero_path, index=False)
        
    logger.info(f'Model run complete -- {str(output_root)}.')
