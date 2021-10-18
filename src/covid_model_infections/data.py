from typing import List, Tuple, Dict
from pathlib import Path
import dill as pickle
from loguru import logger

import pandas as pd
import numpy as np


def evil_doings(data: pd.DataFrame, hierarchy: pd.DataFrame, input_measure: str) -> Tuple[pd.DataFrame, Dict]:
    manipulation_metadata = {}
    if input_measure == 'cases':
        pass

    elif input_measure == 'hospitalizations':
        ## hosp/IHR == admissions too low
        is_argentina = data['location_id'] == 97
        data = data.loc[~is_argentina].reset_index(drop=True)
        manipulation_metadata['argentina'] = 'dropped all hospitalizations'
                
        ## is just march-june 2020
        is_vietnam = data['location_id'] == 20
        data = data.loc[~is_vietnam].reset_index(drop=True)
        manipulation_metadata['vietnam'] = 'dropped all hospitalizations'

        ## is just march-june 2020
        is_murcia = data['location_id'] == 60366
        data = data.loc[~is_murcia].reset_index(drop=True)
        manipulation_metadata['murcia'] = 'dropped all hospitalizations'

        ## under-repored except for Islamabad, which we will keep
        pakistan_location_ids = hierarchy.loc[hierarchy['path_to_top_parent'].apply(lambda x: '165' in x.split(',')),
                                              'location_id'].to_list()
        pakistan_location_ids = [l for l in pakistan_location_ids if l != 53618]
        is_pakistan = data['location_id'].isin(pakistan_location_ids)
        data = data.loc[~is_pakistan].reset_index(drop=True)
        manipulation_metadata['pakistan'] = 'dropped all hospitalizations'
        
        ## ECDC is garbage
        ecdc_location_ids = [77, 82, 83, 59, 60, 88, 91, 52, 55]
        is_ecdc = data['location_id'].isin(ecdc_location_ids)
        data = data.loc[~is_ecdc].reset_index(drop=True)
        manipulation_metadata['ecdc_countries'] = 'dropped all hospitalizations'
        
        ## CLOSE, but seems a little low... check w/ new data
        is_goa = data['location_id'] == 4850
        data = data.loc[~is_goa].reset_index(drop=True)
        manipulation_metadata['goa'] = 'dropped all hospitalizations'

        ## too late, starts March 2021
        is_haiti = data['location_id'] == 114
        data = data.loc[~is_haiti].reset_index(drop=True)
        manipulation_metadata['haiti'] = 'dropped all hospitalizations'

        ## late, starts Jan/Feb 2021 (and is a little low, should check w/ new data)
        is_jordan = data['location_id'] == 144
        data = data.loc[~is_jordan].reset_index(drop=True)
        manipulation_metadata['jordan'] = 'dropped all hospitalizations'
        
        ## too low then too high? odd series
        is_andorra = data['location_id'] == 74
        data = data.loc[~is_andorra].reset_index(drop=True)
        manipulation_metadata['andorra'] = 'dropped all hospitalizations'
        
        ## false point in January (from deaths in imputation)
        is_ohio = data['location_id'] == 558
        is_pre_march = data['date'] < pd.Timestamp('2020-02-18')
        data = data.loc[~(is_ohio & is_pre_march)].reset_index(drop=True)
        manipulation_metadata['ohio'] = 'dropped death before Feb 18'
        
        ## late, starts in Feb 2021 (also probably too low)
        is_guinea_bissau = data['location_id'] == 209
        data = data.loc[~is_guinea_bissau].reset_index(drop=True)
        manipulation_metadata['guinea_bissau'] = 'dropped all hospitalizations'
        
        ## late, starts in June 2021 (also too low)
        is_zimbabwe = data['location_id'] == 198
        data = data.loc[~is_zimbabwe].reset_index(drop=True)
        manipulation_metadata['zimbabwe'] = 'dropped all hospitalizations'
        
        ## too low
        is_malawi = data['location_id'] == 182
        data = data.loc[~is_malawi].reset_index(drop=True)
        manipulation_metadata['malawi'] = 'dropped all hospitalizations'

    elif input_measure == 'deaths':
        ## false point in January
        is_ohio = data['location_id'] == 558
        is_pre_march = data['date'] < pd.Timestamp('2020-03-01')
        data = data.loc[~(is_ohio & is_pre_march)].reset_index(drop=True)
        manipulation_metadata['ohio'] = 'dropped death before March 1'
    
    else:
        raise ValueError(f'Input measure {input_measure} does not have a protocol for exclusions.')
    
    return data, manipulation_metadata


def draw_check(n_draws: int, n_draws_in_data: int,):
    if n_draws > n_draws_in_data:
        raise ValueError(f'User specified {n_draws} draws; only {n_draws_in_data} draws available in data.')
    elif n_draws < n_draws_in_data:
        logger.warning(f'User specified {n_draws} draws; {n_draws_in_data} draws available in data. '
                       f'Crudely taking first {n_draws} draws from rates.')


def load_ifr(rates_root: Path, n_draws: int,) -> pd.DataFrame:
    data_path = rates_root / 'ifr_draws.parquet'
    data = pd.read_parquet(data_path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.rename(columns={'ifr': 'ratio',
                                'ifr_fe': 'ratio_fe'})
    
    n_draws_in_data = data['draw'].max() + 1    
    draw_check(n_draws, n_draws_in_data,)
    data = (data
            .set_index('draw')
            .loc[list(range(n_draws))]
            .reset_index())
    
    data = (data
            .set_index(['location_id', 'draw', 'date'])
            .sort_index()
            .loc[:, ['ratio', 'ratio_fe']])
    
    return data


def load_ifr_rr(rates_root: Path, n_draws: int,) -> pd.DataFrame:
    data_path = rates_root / 'ifr_rr_draws.parquet'
    data = pd.read_parquet(data_path)
    data['date'] = pd.to_datetime(data['date'])
    
    n_draws_in_data = data['draw'].max() + 1    
    draw_check(n_draws, n_draws_in_data,)
    data = (data
            .set_index('draw')
            .loc[list(range(n_draws))]
            .reset_index())
    
    data = (data
            .set_index(['location_id', 'draw', 'date'])
            .sort_index()
            .loc[:, ['ifr_lr_rr', 'ifr_hr_rr']])
    
    return data


def load_ifr_data(rates_root: Path) -> pd.DataFrame:
    data_path = rates_root / 'ifr_model_data.parquet'
    data = pd.read_parquet(data_path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.rename(columns={'ifr_mean': 'ratio_mean',
                                'ifr_std': 'ratio_std',})
    data = (data
            .set_index(['location_id', 'date'])
            .sort_index()
            .loc[:, ['ratio_mean', 'ratio_std', 'is_outlier']])
    
    return data


def load_vaccine_data(rates_root: Path) -> pd.DataFrame:
    data_path = rates_root / 'vaccine_coverage.parquet'
    data = pd.read_parquet(data_path)
    data['date'] = pd.to_datetime(data['date'])
    data = (data
            .loc[:, ['location_id', 'date', 'cumulative_all_effective',]]
            .set_index(['location_id', 'date'])
            .sort_index())
    
    return data
    

def load_ihr(rates_root: Path, n_draws: int,) -> pd.DataFrame:
    data_path = rates_root / 'ihr_draws.parquet'
    data = pd.read_parquet(data_path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.rename(columns={'ihr': 'ratio',
                                'ihr_fe': 'ratio_fe'})
    
    n_draws_in_data = data['draw'].max() + 1    
    draw_check(n_draws, n_draws_in_data,)
    data = (data
            .set_index('draw')
            .loc[list(range(n_draws))]
            .reset_index())
    
    data = (data
            .set_index(['location_id', 'draw', 'date'])
            .sort_index()
            .loc[:, ['ratio', 'ratio_fe']])
    
    return data


def load_ihr_data(rates_root: Path) -> pd.DataFrame:
    data_path = rates_root / 'ihr_model_data.parquet'
    data = pd.read_parquet(data_path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.rename(columns={'ihr_mean': 'ratio_mean',
                                'ihr_std': 'ratio_std',})
    data = (data
            .set_index(['location_id', 'date'])
            .sort_index()
            .loc[:, ['ratio_mean', 'ratio_std', 'is_outlier']])

    
    return data


def load_idr(rates_root: Path, n_draws: int, limits: Tuple[float, float],) -> pd.DataFrame:
    data_path = rates_root / 'idr_draws.parquet'
    data = pd.read_parquet(data_path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.rename(columns={'idr': 'ratio',
                                'idr_fe': 'ratio_fe'})
    
    n_draws_in_data = data['draw'].max() + 1    
    draw_check(n_draws, n_draws_in_data,)
    data = (data
            .set_index('draw')
            .loc[list(range(n_draws))]
            .reset_index())
    
    data = (data
            .set_index(['location_id', 'draw', 'date'])
            .sort_index()
            .loc[:, ['ratio', 'ratio_fe']])
    data = data.clip(*limits)
    
    return data


def load_idr_data(rates_root: Path) -> pd.DataFrame:
    data_path = rates_root / 'idr_model_data.parquet'
    data = pd.read_parquet(data_path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.rename(columns={'idr_mean': 'ratio_mean',
                                'idr_std': 'ratio_std',})
    data = (data
            .set_index(['location_id', 'date'])
            .sort_index()
            .loc[:, ['ratio_mean', 'ratio_std', 'is_outlier']])

    
    return data


def load_sero_data(rates_root: Path) -> pd.DataFrame:
    data_path = rates_root / 'sero_data.csv'
    data = pd.read_csv(data_path)
    data = (data
            .loc[:, ['location_id', 'infection_date',
                     'seroprevalence', 'seroprevalence_no_vacc',
                     'sero_sample_mean', 'sero_sample_std',
                     'is_outlier']])
    data = data.rename(columns={'infection_date':'date'})
    data['date'] = pd.to_datetime(data['date'])
    data = (data
            .set_index(['location_id', 'date'])
            .sort_index())
    
    return data


def load_cross_variant_immunity(rates_root: Path, n_draws: int,) -> List:
    data_path = rates_root / 'cross_variant_immunity.pkl'
    with data_path.open('rb') as file:
        data = pickle.load(file)
    
    n_draws_in_data = len(data)
    draw_check(n_draws, n_draws_in_data,)
    data = data[:n_draws]
    
    return data


def load_variant_risk_ratio(rates_root: Path, n_draws: int,) -> List:
    data_path = rates_root / 'variant_risk_ratio.pkl'
    with data_path.open('rb') as file:
        data = pickle.load(file)
    
    n_draws_in_data = len(data)
    draw_check(n_draws, n_draws_in_data,)
    data = data[:n_draws]
    
    return data


def load_escape_variant_prevalence(rates_root: Path) -> pd.DataFrame:
    data_path = rates_root / 'variants.parquet'
    data = pd.read_parquet(data_path)
    data['date'] = pd.to_datetime(data['date'])
    data = (data
            .set_index(['location_id', 'date'])
            .sort_index()
            .loc[:, ['escape_variant_prevalence']])
    
    return data


def load_reinfection_inflation_factor(rates_root: Path) -> pd.DataFrame:
    data_path = rates_root / 'reinfection_inflation_factor.parquet'
    data = pd.read_parquet(data_path)
    data['date'] = pd.to_datetime(data['date'])
    data = (data
            .set_index(['location_id', 'date'])
            .sort_index()
            .loc[:, ['inflation_factor']])
    
    return data


def load_test_data(rates_root: Path) -> pd.DataFrame:
    data_path = rates_root / 'testing.parquet'
    data = pd.read_parquet(data_path)
    data['date'] = pd.to_datetime(data['date'])
    data = (data
            .set_index(['location_id', 'date'])
            .sort_index())
    
    return data


def load_em_scalars(rates_root: Path, n_draws: int,) -> pd.DataFrame:
    data_path = rates_root / 'excess_mortality.parquet'
    data = pd.read_parquet(data_path)
    data = data.reset_index()
    
    n_draws_in_data = data['draw'].max() + 1    
    draw_check(n_draws, n_draws_in_data,)
    data = (data
            .set_index('draw')
            .loc[list(range(n_draws))]
            .reset_index())
    
    data = (data
            .set_index(['location_id', 'draw',])
            .sort_index())

    return data


def load_durations(rates_root: Path, n_draws: int,) -> List[Dict[str, int]]:
    data_path = rates_root / 'durations.pkl'
    with data_path.open('rb') as file:
        data = pickle.load(file)
    
    n_draws_in_data = len(data)
    draw_check(n_draws, n_draws_in_data,)
    data = data[:n_draws]
        
    return data


def load_model_inputs(model_inputs_root:Path, hierarchy: pd.DataFrame, input_measure: str,) -> Tuple[pd.Series, pd.Series, Dict]:
    if input_measure == 'deaths':
        data_path = model_inputs_root / 'full_data_unscaled.csv'
    else:
        data_path = model_inputs_root / 'use_at_your_own_risk' / 'full_data_extra_hospital.csv'
    data = pd.read_csv(data_path)
    data = data.rename(columns={'Confirmed': 'cumulative_cases',
                                'Hospitalizations': 'cumulative_hospitalizations',
                                'Deaths': 'cumulative_deaths',})
    data['date'] = pd.to_datetime(data['Date'])
    keep_cols = ['location_id', 'date', f'cumulative_{input_measure}']
    data = data.loc[:, keep_cols].dropna()
    data['location_id'] = data['location_id'].astype(int)
    
    data = (data.groupby('location_id', as_index=False)
            .apply(lambda x: fill_dates(x, [f'cumulative_{input_measure}']))
            .reset_index(drop=True))

    data, manipulation_metadata = evil_doings(data, hierarchy, input_measure)
    
    data[f'daily_{input_measure}'] = (data
                                      .groupby('location_id')[f'cumulative_{input_measure}']
                                      .apply(lambda x: x.diff())
                                      .fillna(data[f'cumulative_{input_measure}']))
    data = data.dropna()
    data = (data
            .set_index(['location_id', 'date'])
            .sort_index())
    
    cumulative_data = data[f'cumulative_{input_measure}']
    daily_data = data[f'daily_{input_measure}']

    return cumulative_data, daily_data, manipulation_metadata


def fill_dates(data: pd.DataFrame, interp_vars: List[str]) -> pd.DataFrame:
    data = data.set_index('date').sort_index()
    data = data.asfreq('D').reset_index()
    data[interp_vars] = data[interp_vars].interpolate(axis=0, limit_area='inside')
    data['location_id'] = data['location_id'].fillna(method='pad')
    data['location_id'] = data['location_id'].astype(int)

    return data[['location_id', 'date'] + interp_vars]


def trim_leading_zeros(cumul_data: List[pd.Series],
                       daily_data: List[pd.Series],) -> Tuple[pd.Series]:
    cumul = pd.concat(cumul_data, axis=1)
    cumul = cumul.fillna(0)
    cumul = cumul.sum(axis=1)
    cumul = cumul.loc[cumul > 0]
    start_dates = cumul.reset_index().groupby('location_id')['date'].min()
    start_dates -= pd.Timedelta(days=14)
    start_dates = start_dates.rename('start_date').reset_index()
    
    def _trim_leading_zeros(data: pd.Series, start_dates: pd.DataFrame) -> pd.Series:
        data_name = data.name
        data = data.reset_index().merge(start_dates, how='left')
        data = data.loc[~(data['date'] < data['start_date'])]  # do it this way so we keep NaTs
        del data['start_date']
        data = data.set_index(['location_id', 'date']).loc[:, data_name]
        
        return data
    
    trimmed_data = (_trim_leading_zeros(data, start_dates) for data in cumul_data + daily_data)
    
    return trimmed_data


def load_hierarchy(model_inputs_root:Path) -> pd.DataFrame:
    data_path = model_inputs_root / 'locations' / 'modeling_hierarchy.csv'
    data = pd.read_csv(data_path)
    data = data.sort_values('sort_order').reset_index(drop=True)
#     logger.warning('Using ZAF subnats...')
#     gbd_path = model_inputs_root / 'locations' / 'gbd_analysis_hierarchy.csv'
#     covid_path = model_inputs_root / 'locations' / 'modeling_hierarchy.csv'

#     # get ZAF only from GBD for now
#     covid = pd.read_csv(covid_path)
#     covid_is_zaf = covid['path_to_top_parent'].apply(lambda x: '196' in x.split(','))
#     if not covid_is_zaf.sum() == 1:
#         raise ValueError('Already have ZAF subnats in Covid hierarchy.')
#     sort_order = covid.loc[covid_is_zaf, 'sort_order'].item()
#     covid = covid.loc[~covid_is_zaf]

#     gbd = pd.read_csv(gbd_path)
#     gbd_is_zaf = gbd['path_to_top_parent'].apply(lambda x: '196' in x.split(','))
#     gbd = gbd.loc[gbd_is_zaf].reset_index(drop=True)
#     gbd['sort_order'] = sort_order + gbd.index

#     covid.loc[covid['sort_order'] > sort_order, 'sort_order'] += len(gbd) - 1

#     data = pd.concat([covid, gbd]).sort_values('sort_order').reset_index(drop=True)

    return data


def load_population(model_inputs_root: Path) -> pd.DataFrame:
    data_path = model_inputs_root / 'output_measures' / 'population' / 'all_populations.csv'
    data = pd.read_csv(data_path)
    is_2019 = data['year_id'] == 2019
    is_bothsex = data['sex_id'] == 3
    is_alllage = data['age_group_id'] == 22
    data = (data
            .loc[is_2019 & is_bothsex & is_alllage]
            .set_index('location_id')
            .loc[:, ['population']])

    return data


def write_infections_draws(data: pd.DataFrame,
                           infections_draws_dir: Path,):
    draw_col = np.array([c for c in data.columns if c.startswith('draw_')]).item()
    draw = int(draw_col.split('_')[-1])
    data = data.rename(columns={draw_col:'infections_draw'})
    data['draw'] = draw
    
    out_path = infections_draws_dir / f'{draw_col}.csv'
    data.reset_index().to_csv(out_path, index=False)
    # data = data.reset_index()
    # data['date'] = data['date'].astype(str)
    # out_path = infections_draws_dir / f'{draw_col}.parquet'
    # data.to_parquet(out_path, engine='fastparquet', compression='gzip')
    
    return out_path


def write_ratio_draws(data_list: List[pd.Series],
                      estimated_ratio: str,
                      ratio_draws_dir: Path,
                      variant_risk_ratio: List[float],
                      durations: List[int],):
    if estimated_ratio == 'ifr':
        if len(data_list) != 3:
            raise ValueError('IFR, but not 3 elements in data list.')
        data = data_list[0]
        data_lr = data_list[1]
        data_hr = data_list[2]
    else:
        if len(data_list) != 1:
            raise ValueError('Not IFR, but multiple elements in data list.')
        data = data_list[0]
    draw_col = data.name
    draw = int(draw_col.split('_')[-1])
    data = data.rename(f'{estimated_ratio}_draw')
    data = data.to_frame()
    if estimated_ratio == 'ifr':
        data['ifr_lr_draw'] = data['ifr_draw'] * data_lr
        data['ifr_hr_draw'] = data['ifr_draw'] * data_hr
    data['draw'] = draw
    data['duration'] = durations[draw]
    if estimated_ratio in ['ifr', 'ihr']:
        data['variant_risk_ratio'] = variant_risk_ratio[draw]

    out_path = ratio_draws_dir / f'{draw_col}.csv'
    data.reset_index().to_csv(out_path, index=False)
    # data = data.reset_index()
    # data['date'] = data['date'].astype(str)
    # out_path = ratio_draws_dir / f'{draw_col}.parquet'
    # data.to_parquet(out_path, engine='fastparquet', compression='gzip')
    
    return out_path


# def store_df(data: pd.DataFrame, path: Path, filename: str, fmt: str,):
#     if fmt == 'csv':
#         full_path = path / f'{filename}.csv'
#         data.to_csv(full_path)
#     elif fmt == 'parquet':
#         full_path = path / f'{filename}.parquet'
#         data.to_parquet(full_path, engine='fastparquet', compression='gzip')
#     else:
#         raise ValueError(f'Invalid file format specified: {fmt}')


# def load_df(path: Path, filename: str, fmt: str,):
#     if fmt == 'csv':
#         full_path = path / f'{filename}.csv'
#         data = pd.read_csv(full_path)
#     elif fmt == 'parquet':
#         full_path = path / f'{filename}.parquet'
#         data = pd.read_parquet(full_path, engine='fastparquet')
#     else:
#         raise ValueError(f'Invalid file format specified: {fmt}')
