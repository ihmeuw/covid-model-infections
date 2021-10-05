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
from covid_model_infections.pdf_merger import pdf_merger

MP_THREADS = 25

## TODO:
##     - holdouts
##     - modularize data object creation
##     - maybe job holds
##     - maybe delete refit_draws contents at the end, to save space


def make_infections(app_metadata: cli_tools.Metadata,
                    model_inputs_root: Path,
                    rates_root: Path,
                    output_root: Path,
                    holdout_days: int,
                    # n_draws: int,
                   ):
    if holdout_days > 0:
        raise ValueError('Holdout not yet implemented.')
    
    logger.info('Creating directories.')
    model_in_dir = output_root / 'model_inputs'
    model_out_dir = output_root / 'model_outputs'
    refit_dir = model_out_dir / 'refit_draws'
    plot_dir = output_root / 'plots'
    infections_draws_dir = output_root / 'infections_draws'
    shell_tools.mkdir(model_in_dir)
    shell_tools.mkdir(model_out_dir)
    shell_tools.mkdir(refit_dir)
    shell_tools.mkdir(plot_dir)
    shell_tools.mkdir(infections_draws_dir)
    
    logger.info('Loading supplemental data.')
    hierarchy = data.load_hierarchy(model_inputs_root)
    pop_data = data.load_population(model_inputs_root)
    
    logger.info('Loading epi report data.')
    em_scalar_data = data.load_em_scalars(rates_root)
    cumul_deaths, daily_deaths, deaths_manipulation_metadata = data.load_model_inputs(
        model_inputs_root, hierarchy, 'deaths', em_scalar_data
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
        'deaths': deaths_manipulation_metadata,
        'hospitalizations': hospital_manipulation_metadata,
        'cases': cases_manipulation_metadata,
    }})
    measures = ['deaths', 'hospitalizations', 'cases']
    
    logger.info('Loading estimated ratios and adding draw directories.')
    ifr, n_draws = data.load_ifr(rates_root)
    ifr_rr = data.load_ifr_rr(rates_root)
    ifr_model_data = data.load_ifr_data(rates_root)
    cross_variant_immunity = data.load_cross_variant_immunity(rates_root)
    variant_risk_ratio = data.load_variant_risk_ratio(rates_root)
    escape_variant_prevalence = data.load_escape_variant_prevalence(rates_root)
    ihr = data.load_ihr(rates_root)
    ihr_model_data = data.load_ihr_data(rates_root)
    # Assumes IDR has estimated floor already applied
    idr = data.load_idr(rates_root, (0., 1.))
    idr_model_data = data.load_idr_data(rates_root)
    
    logger.info('Loading durations for each draw.')
    durations = data.load_durations(rates_root)
    
    logger.info('Loading extra data for plotting.')
    sero_data = data.load_sero_data(rates_root)
    test_data = data.load_test_data(rates_root)
    vaccine_data = data.load_vaccine_data(rates_root)

    logger.info('Filling missing locations and creating model input data structure.')
    most_detailed = hierarchy['most_detailed'] == 1
    location_ids = hierarchy.loc[most_detailed, 'location_id'].to_list()
    path_to_top_parents = hierarchy.loc[most_detailed, 'path_to_top_parent'].to_list()
    location_names = hierarchy.loc[most_detailed, 'location_name'].to_list()
    baseline_ifr_locations = ifr.reset_index()['location_id'].unique().tolist()
    baseline_ihr_locations = ihr.reset_index()['location_id'].unique().tolist()
    baseline_idr_locations = idr.reset_index()['location_id'].unique().tolist()
    baseline_evp_locations = escape_variant_prevalence.reset_index()['location_id'].unique().tolist()
    for location_id, path_to_top_parent, location_name in zip(location_ids, path_to_top_parents, location_names):
        # DEATHS
        if location_id not in baseline_ifr_locations:
            for parent_id in reversed(path_to_top_parent.split(',')[:-1]):
                if int(parent_id) in baseline_ifr_locations:
                    logger.info(f'Using parent IFR for {location_name}.')
                    ifr = ifr.append(
                        pd.concat({location_id: ifr.loc[int(parent_id)]}, names=['location_id'])
                    )
                    ifr_rr = ifr_rr.append(
                        pd.concat({location_id: ifr_rr.loc[int(parent_id)]}, names=['location_id'])
                    )
                    break
                else:
                    pass
        # HOSPITAL ADMISSIONS
        if location_id not in baseline_ihr_locations:
            for parent_id in reversed(path_to_top_parent.split(',')[:-1]):
                if int(parent_id) in baseline_ihr_locations:
                    logger.info(f'Using parent IHR for {location_name}.')
                    ihr = ihr.append(
                        pd.concat({location_id: ihr.loc[int(parent_id)]}, names=['location_id'])
                    )
                    break
                else:
                    pass
        # CASES
        if location_id not in baseline_idr_locations:
            for parent_id in reversed(path_to_top_parent.split(',')[:-1]):
                if int(parent_id) in baseline_idr_locations:
                    logger.info(f'Using parent IDR for {location_name}.')
                    idr = idr.append(
                        pd.concat({location_id: idr.loc[int(parent_id)]}, names=['location_id'])
                    )
                    break
                else:
                    pass
        # ESCAPE VARIANT PREVALENCE
        if location_id not in baseline_evp_locations:
            for parent_id in reversed(path_to_top_parent.split(',')[:-1]):
                if int(parent_id) in baseline_evp_locations:
                    logger.info(f'Using parent escape variant prevalence for {location_name}.')
                    escape_variant_prevalence = escape_variant_prevalence.append(
                        pd.concat({location_id: escape_variant_prevalence.loc[int(parent_id)]}, names=['location_id'])
                    )
                    break
                else:
                    pass
    model_data = {
        'durations': durations,
        'daily_deaths': daily_deaths, 'cumul_deaths': cumul_deaths, 'ifr': ifr,
        'daily_hospital': daily_hospital, 'cumul_hospital': cumul_hospital, 'ihr': ihr,
        'daily_cases': daily_cases, 'cumul_cases': cumul_cases, 'idr': idr,
    }
    # TODO: centralize this information, is used elsewhere...
    estimated_ratios = {'deaths': ('ifr', [d['exposure_to_death'] for d in durations], ifr.copy()),
                        'hospitalizations': ('ihr', [d['exposure_to_admission'] for d in durations], ihr.copy()),
                        'cases': ('idr', [d['exposure_to_case'] for d in durations], idr.copy()),}
    
    logger.info('Writing intermediate files.')
    model_data_path = model_in_dir / 'model_data.pkl'
    with model_data_path.open('wb') as file:
        pickle.dump(model_data, file, -1)
    hierarchy_path = model_in_dir / 'hierarchy.parquet'
    hierarchy.to_parquet(hierarchy_path)
    pop_path = model_in_dir / 'pop_data.parquet'
    pop_data.to_parquet(pop_path)
    sero_path = model_in_dir / 'sero_data.parquet'
    sero_data.to_parquet(sero_path)
    test_path = model_in_dir / 'test_data.parquet'
    test_data.to_parquet(test_path)
    vaccine_path = model_in_dir / 'vaccine_data.parquet'
    vaccine_data.to_parquet(vaccine_path)
    ifr_model_data_path = model_in_dir / 'ifr_model_data.parquet'
    ifr_model_data.to_parquet(ifr_model_data_path)
    cross_variant_immunity_path = model_in_dir / 'cross_variant_immunity.pkl'
    with cross_variant_immunity_path.open('wb') as file:
        pickle.dump(cross_variant_immunity, file, -1)
    escape_variant_prevalence_path = model_in_dir / 'escape_variant_prevalence.parquet'
    escape_variant_prevalence.to_parquet(escape_variant_prevalence_path)
    ihr_model_data_path = model_in_dir / 'ihr_model_data.parquet'
    ihr_model_data.to_parquet(ihr_model_data_path)
    idr_model_data_path = model_in_dir / 'idr_model_data.parquet'
    idr_model_data.to_parquet(idr_model_data_path)
    
    logger.info('Launching location-specific models.')
    job_args_map = {
        location_id: [model.runner.__file__,
                      location_id, n_draws, str(model_in_dir), str(model_out_dir), str(plot_dir),]
        for location_id in location_ids
    }
    cluster.run_cluster_jobs('covid_loc_inf', output_root, job_args_map)
    
    logger.info('Compiling infection draws.')
    infections_draws = []
    for draws_path in [result_path for result_path in model_out_dir.iterdir() if str(result_path).endswith('_infections_draws.parquet')]:
        infections_draws.append(pd.read_parquet(draws_path))
    infections_draws = pd.concat(infections_draws)
    completed_modeled_location_ids = infections_draws.reset_index()['location_id'].unique().tolist()
    
    logger.info('Identifying failed models.')
    failed_model_location_ids = list(set(location_ids) - set(completed_modeled_location_ids))
    app_metadata.update({'failed_model_location_ids': failed_model_location_ids})
    if failed_model_location_ids:
        logger.debug(f'Models failed for the following location_ids: {", ".join([str(l) for l in failed_model_location_ids])}')
    
    logger.info('Compiling other model data.')
    inputs = {}
    for inputs_path in [result_path for result_path in model_out_dir.iterdir() if str(result_path).endswith('_input_data.pkl')]:
        with inputs_path.open('rb') as inputs_file:
            inputs.update(pickle.load(inputs_file))
    outputs = {}
    for outputs_path in [result_path for result_path in model_out_dir.iterdir() if str(result_path).endswith('_output_data.pkl')]:
        with outputs_path.open('rb') as outputs_file:
            outputs.update(pickle.load(outputs_file))
        
    logger.info('Aggregating inputs.')
    agg_inputs = aggregation.aggregate_md_data_dict(inputs.copy(), hierarchy, measures, 1)

    logger.info('Aggregating outputs.')
    agg_outputs = aggregation.aggregate_md_data_dict(outputs.copy(), hierarchy, measures, 1)

    logger.info('Aggregating final infecions draws.')
    agg_infections_draws = aggregation.aggregate_md_draws(infections_draws.copy(), hierarchy, MP_THREADS)
    
    logger.info('Plotting aggregates.')
    plot_parent_ids = agg_infections_draws.reset_index()['location_id'].unique().tolist()
    for plot_parent_id in tqdm(plot_parent_ids, total=len(plot_parent_ids), file=sys.stdout):
        aggregation.plot_aggregate(
            plot_parent_id,
            agg_inputs[plot_parent_id].copy(),
            agg_outputs[plot_parent_id].copy(),
            agg_infections_draws.loc[plot_parent_id].copy(),
            hierarchy,
            pop_data,
            sero_data,
            cross_variant_immunity,
            escape_variant_prevalence,
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
        
    for measure, (estimated_ratio, measure_durations, ratio_prior_estimates) in estimated_ratios.items():
        logger.info(f'Compiling {estimated_ratio.upper()} draws.')
        ratio_draws = []
        for draws_path in [result_path for result_path in model_out_dir.iterdir() if str(result_path).endswith(f'_{estimated_ratio}_draws.parquet')]:
            ratio_draws.append(pd.read_parquet(draws_path))
        ratio_draws = pd.concat(ratio_draws)
        
        logger.info(f'Filling {estimated_ratio.upper()} with original model estimate where we do not have a posterior.')
        ratio_draws_cols = ratio_draws.columns
        ratio_prior_estimates = (ratio_prior_estimates
                                 .reset_index()
                                 .loc[:, ['location_id', 'draw', 'date', 'ratio']])
        ratio_prior_estimates = pd.pivot_table(ratio_prior_estimates,
                                               index=['location_id', 'date'],
                                               columns='draw',
                                               values='ratio',)
        ratio_prior_estimates.columns = ratio_draws_cols
        ratio_locations = ratio_draws.reset_index()['location_id'].unique()
        ratio_prior_locations = ratio_prior_estimates.reset_index()['location_id'].unique()
        missing_locations = [l for l in ratio_prior_locations if l not in ratio_locations]
        ratio_prior_estimates = ratio_prior_estimates.loc[missing_locations]
        ratio_draws = ratio_draws.append(ratio_prior_estimates)

        logger.info(f'Writing SEIR inputs - {estimated_ratio.upper()} draw files.')
        ratio_draws = ratio_draws.sort_index()
        if estimated_ratio == 'ifr':
            ifr_lr_rr = (ifr_rr
                        .reset_index()
                        .loc[:, ['location_id', 'draw', 'date', 'ifr_lr_rr']])
            ifr_lr_rr = pd.pivot_table(ifr_lr_rr,
                                       index=['location_id', 'date'],
                                       columns='draw',
                                       values='ifr_lr_rr',)
            ifr_lr_rr.columns = ratio_draws_cols
            ifr_hr_rr = (ifr_rr
                        .reset_index()
                        .loc[:, ['location_id', 'draw', 'date', 'ifr_hr_rr']])
            ifr_hr_rr = pd.pivot_table(ifr_hr_rr,
                                       index=['location_id', 'date'],
                                       columns='draw',
                                       values='ifr_hr_rr',)
            ifr_hr_rr.columns = ratio_draws_cols
            ratio_draws = [(ratio_draws[ratio_draws_col].copy(),
                            ifr_lr_rr[ratio_draws_col].copy(),
                            ifr_hr_rr[ratio_draws_col].copy(),)
                           for ratio_draws_col in ratio_draws_cols]
        else:
            ratio_draws = [[ratio_draws[ratio_draws_col].copy()] for ratio_draws_col in ratio_draws_cols]
        ratio_draws_dir = output_root / f'{estimated_ratio}_draws'
        shell_tools.mkdir(ratio_draws_dir)
        _ratio_writer = functools.partial(
            data.write_ratio_draws,
            estimated_ratio=estimated_ratio,
            durations=measure_durations,
            variant_risk_ratio=variant_risk_ratio,
            ratio_draws_dir=ratio_draws_dir,
        )
        with multiprocessing.Pool(MP_THREADS) as p:
            ratio_draws_paths = list(tqdm(p.imap(_ratio_writer, ratio_draws), total=n_draws, file=sys.stdout))

    logger.info('Writing serology data and EM scaling factor data.')
    em_path = output_root / 'em_data.csv'
    em_scalar_data = (infections_draws[0]
                      .reset_index()
                      .loc[:, ['location_id', 'date']]
                      .merge(em_scalar_data, how='left'))
    em_scalar_data['em_scalar'] = em_scalar_data['em_scalar'].fillna(1)
    em_scalar_data.to_csv(em_path, index=False)
    # em_scalar_data['date'] = em_scalar_data['date'].astype(str)
    # em_path = output_root / 'em_data.parquet'
    # em_scalar_data.to_parquet(em_path, engine='fastparquet', compression='gzip')
    sero_data['included'] = 1 - sero_data['is_outlier']
    sero_data = sero_data.rename(columns={'sero_sample_mean': 'value'})
    sero_data = sero_data.loc[:, ['included', 'value']]
    sero_path = output_root / 'sero_data.csv'
    sero_data.reset_index().to_csv(sero_path, index=False)
    # sero_data = sero_data.reset_index()
    # sero_data['date'] = sero_data['date'].astype(str)
    # sero_path = output_root / 'sero_data.parquet'
    # sero_data.to_parquet(sero_path, engine='fastparquet', compression='gzip')
        
    logger.info(f'Model run complete -- {str(output_root)}.')
