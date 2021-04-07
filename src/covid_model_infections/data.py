from typing import List, Tuple, Dict
from pathlib import Path

import pandas as pd
import numpy as np


def evil_doings(data: pd.DataFrame, hierarchy: pd.DataFrame, input_measure: str) -> Tuple[pd.DataFrame, Dict]:
    manipulation_metadata = {}
    if input_measure == 'cases':
        is_peru = data['location_id'] == 123
        data = data.loc[~is_peru].reset_index(drop=True)
        manipulation_metadata['peru'] = 'dropped all cases'
        
        is_ecuador = data['location_id'] == 122
        data = data.loc[~is_ecuador].reset_index(drop=True)
        manipulation_metadata['ecuador'] = 'dropped all cases'
        
        is_kazakhstan = data['location_id'] == 36
        data = data.loc[~is_kazakhstan].reset_index(drop=True)
        manipulation_metadata['kazakhstan'] = 'dropped all cases'
        
        # india_location_ids = hierarchy.loc[hierarchy['path_to_top_parent'].apply(lambda x: '163' in x.split(',')),
        #                                    'location_id'].to_list()
        # india_location_ids = [i for i in india_location_ids if i not in [4842, 4862, 4863, 4864, 4869, 60896]]
        # is_india = data['location_id'].isin(india_location_ids)
        # data = data.loc[~is_india].reset_index(drop=True)
        # manipulation_metadata['india'] = 'dropped all cases'

    elif input_measure == 'hospitalizations':
        # is_oman = data['location_id'] == 150
        # data = data.loc[~is_oman].reset_index(drop=True)
        # manipulation_metadata['oman'] = 'dropped all hospitalizations'

        # is_greece = data['location_id'] == 82
        # data = data.loc[~is_greece].reset_index(drop=True)
        # manipulation_metadata['greece'] = 'dropped all hospitalizations'

        # is_norway = data['location_id'] == 90
        # data = data.loc[~is_norway].reset_index(drop=True)
        # manipulation_metadata['norway'] = 'dropped all hospitalizations'

        is_argentina = data['location_id'] == 97
        data = data.loc[~is_argentina].reset_index(drop=True)
        manipulation_metadata['argentina'] = 'dropped all hospitalizations'

        is_romania = data['location_id'] == 52
        data = data.loc[~is_romania].reset_index(drop=True)
        manipulation_metadata['romania'] = 'dropped all hospitalizations'

        is_slovenia = data['location_id'] == 55
        data = data.loc[~is_slovenia].reset_index(drop=True)
        manipulation_metadata['slovenia'] = 'dropped all hospitalizations'

        is_vietnam = data['location_id'] == 20
        data = data.loc[~is_vietnam].reset_index(drop=True)
        manipulation_metadata['vietnam'] = 'dropped all hospitalizations'

        is_murcia = data['location_id'] == 60366
        data = data.loc[~is_murcia].reset_index(drop=True)
        manipulation_metadata['murcia'] = 'dropped all hospitalizations'

        is_dc = data['location_id'] == 531
        data = data.loc[~is_dc].reset_index(drop=True)
        manipulation_metadata['washington_dc'] = 'dropped all hospitalizations'

        pakistan_location_ids = hierarchy.loc[hierarchy['path_to_top_parent'].apply(lambda x: '165' in x.split(',')),
                                              'location_id'].to_list()
        is_pakistan = data['location_id'].isin(pakistan_location_ids)
        data = data.loc[~is_pakistan].reset_index(drop=True)
        manipulation_metadata['pakistan'] = 'dropped all hospitalizations'

        # wa_location_ids = hierarchy.loc[hierarchy['path_to_top_parent'].apply(lambda x: '570' in x.split(',')),
        #                                           'location_id'].to_list()
        # is_wa = data['location_id'].isin(wa_location_ids)
        # data = data.loc[~is_wa].reset_index(drop=True)
        # manipulation_metadata['washington'] = 'dropped all hospitalizations'

        is_poland = data['location_id'] == 51
        data = data.loc[~is_poland].reset_index(drop=True)
        manipulation_metadata['poland'] = 'dropped all hospitalizations'

        is_philippines = data['location_id'] == 16
        data = data.loc[~is_philippines].reset_index(drop=True)
        manipulation_metadata['philippines'] = 'dropped all hospitalizations'

        is_portugal = data['location_id'] == 91
        data = data.loc[~is_portugal].reset_index(drop=True)
        manipulation_metadata['portugal'] = 'dropped all hospitalizations'

        # is_jordan = data['location_id'] == 144
        # data = data.loc[~is_jordan].reset_index(drop=True)
        # manipulation_metadata['jordan'] = 'dropped all hospitalizations'

        # is_bc = data['location_id'] == 43859
        # data = data.loc[~is_bc].reset_index(drop=True)
        # manipulation_metadata['british_columbia'] = 'dropped all hospitalizations'
    
    elif input_measure == 'deaths':
        uk_location_ids = hierarchy.loc[hierarchy['path_to_top_parent'].apply(lambda x: '95' in x.split(',')),
                                                  'location_id'].to_list()
        is_uk = data['location_id'].isin(uk_location_ids)
        is_pre_mar1 = data['date'] < pd.Timestamp('2020-03-01')
        data = data.loc[~(is_uk & is_pre_mar1)].reset_index(drop=True)
        manipulation_metadata['uk'] = 'dropped leading 0s in deaths from UK (pre March 1)'
        
        is_arkansas = data['location_id'] == 526
        is_pre_mar15 = data['date'] < pd.Timestamp('2020-03-15')
        data = data.loc[~(is_arkansas & is_pre_mar15)].reset_index(drop=True)
        manipulation_metadata['arkansas'] = 'dropped leading 0s in deaths from Arkansas (pre March 15)'
        
        is_ohio = data['location_id'] == 558
        is_pre_mar1 = data['date'] < pd.Timestamp('2020-03-01')
        data = data.loc[~(is_ohio & is_pre_mar1)].reset_index(drop=True)
        manipulation_metadata['ohio'] = 'dropped leading 0s in deaths from Ohio (pre March 1)'
    
    else:
        raise ValueError(f'Input measure {input_measure} does not have a protocol for exclusions.')
    
    return data, manipulation_metadata


def load_ifr(infection_fatality_root: Path) -> pd.DataFrame:
    data_path = infection_fatality_root / 'allage_ifr_by_loctime.csv'
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.rename(columns={'ifr':'ratio',
                                'ifr_no_random_effect':'ratio_fe'})
    data = (data
            .set_index(['location_id', 'date'])
            .sort_index()
            .loc[:, ['ratio', 'ratio_fe']])
    
    return data


def load_ifr_risk_adjustment(infection_fatality_root: Path) -> pd.Series:
    data_path = infection_fatality_root / 'terminal_ifr.csv'
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])
    data['lr_adj'] = data['ifr_lr'] / data['ifr']
    data['hr_adj'] = data['ifr_hr'] / data['ifr']
    data = (data
            .set_index(['location_id', 'date'])
            .loc[:, ['lr_adj', 'hr_adj']])
    
    return data


def load_ifr_data(infection_fatality_root: Path) -> pd.DataFrame:
    data_path = infection_fatality_root / 'ifr_model_data.csv'
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.loc[data['ifr'].notnull()]
    data = data.rename(columns={'ifr':'ratio'})
    data = (data
            .set_index(['location_id', 'date'])
            .sort_index()
            .loc[:, ['ratio', 'is_outlier']])
    
    return data
    

def load_ihr(infection_hospitalization_root: Path) -> pd.DataFrame:
    data_path = infection_hospitalization_root / 'allage_ihr_by_loctime.csv'
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.rename(columns={'ihr':'ratio',
                                'ihr_no_random_effect':'ratio_fe'})
    data = (data
            .set_index(['location_id', 'date'])
            .sort_index()
            .loc[:, ['ratio', 'ratio_fe']])
    
    return data


def load_ihr_data(infection_hospitalization_root: Path) -> pd.DataFrame:
    data_path = infection_hospitalization_root / 'ihr_model_data.csv'
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.loc[data['ihr'].notnull()]
    data = data.rename(columns={'ihr':'ratio'})
    data = (data
            .set_index(['location_id', 'date'])
            .sort_index()
            .loc[:, ['ratio', 'is_outlier']])
    
    return data


def load_idr(infection_detection_root: Path, limits: Tuple[float, float]) -> pd.DataFrame:
    data_path = infection_detection_root / 'pred_idr.csv'
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.rename(columns={'idr':'ratio',
                                'idr_fe':'ratio_fe'})
    data = (data
            .set_index(['location_id', 'date'])
            .sort_index()
            .loc[:, ['ratio', 'ratio_fe']])
    data = data.clip(*limits)
    
    return data


def load_idr_data(infection_detection_root: Path) -> pd.DataFrame:
    data_path = infection_detection_root / 'idr_plot_data.csv'
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.rename(columns={'idr':'ratio',})
    data = (data
            .set_index(['location_id', 'date'])
            .sort_index()
            .loc[:, ['ratio', 'is_outlier']])
    
    return data


def load_sero_data(infection_detection_root: Path) -> pd.DataFrame:
    data_path = infection_detection_root / 'sero_data.csv'
    data = pd.read_csv(data_path)
    data = (data
            .loc[:, ['location_id', 'infection_date',
                     'seroprev_mean', 'seroprev_mean_no_vacc', 'seroprev_mean_no_vacc_waning',
                     'geo_accordance', 'manual_outlier']])
    data.loc[data['geo_accordance'] == 0, 'manual_outlier'] == 1
    data = data.rename(columns={'infection_date':'date'})
    data['date'] = pd.to_datetime(data['date'])
    data = (data
            .set_index(['location_id', 'date'])
            .sort_index()
            .loc[:, ['seroprev_mean', 'seroprev_mean_no_vacc', 'seroprev_mean_no_vacc_waning',
                     'manual_outlier']])
    
    return data


def load_reinfection_data(infection_fatality_root: Path) -> pd.Series:
    data_path = infection_fatality_root / 'reinfection_data.csv'
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])
    
    data = (data
            .set_index(['location_id', 'date'])
            .loc[:, 'inflation_factor'])
    
    return data


def load_testing_data(infection_detection_root: Path):
    data_path = infection_detection_root / 'test_data.csv'
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])
    data = (data
            .set_index(['location_id', 'date'])
            .sort_index())
    
    return data


def load_em_scalars(infection_fatality_root: Path) -> pd.DataFrame:
        data_path = infection_fatality_root / 'excess_mortalilty.csv'
        data = pd.read_csv(data_path)
        
        return data


def load_model_inputs(model_inputs_root:Path, hierarchy: pd.DataFrame, input_measure: str,
                      excess_mortality: bool = True,) -> Tuple[pd.Series, pd.Series, Dict]:
    data_path = model_inputs_root / 'use_at_your_own_risk' / 'full_data_extra_hospital.csv'
    data = pd.read_csv(data_path)
    data = data.rename(columns={'Confirmed':'cumulative_cases',
                                'Hospitalizations':'cumulative_hospitalizations',})
    if input_measure == 'deaths' and excess_mortality:
        data = data.rename(columns={'Deaths':'cumulative_deaths',})
    else:
        data = data.rename(columns={'UNSCALED Deaths':'cumulative_deaths',})
    data['date'] = pd.to_datetime(data['Date'])
    keep_cols = ['location_id', 'date', f'cumulative_{input_measure}']
    data = data.loc[:, keep_cols].dropna()
    data['location_id'] = data['location_id'].astype(int)
    
    data = (data.groupby('location_id', as_index=False)
            .apply(lambda x: fill_dates(x, [f'cumulative_{input_measure}']))
            .reset_index(drop=True))

    data[f'daily_{input_measure}'] = (data
                                      .groupby('location_id')[f'cumulative_{input_measure}']
                                      .apply(lambda x: x.diff())
                                      .fillna(data[f'cumulative_{input_measure}']))
    data = data.dropna()
    data, manipulation_metadata = evil_doings(data, hierarchy, input_measure)
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


def load_hierarchy(model_inputs_root:Path) -> pd.DataFrame:
    data_path = model_inputs_root / 'locations' / 'modeling_hierarchy.csv'
    data = pd.read_csv(data_path)
    data = data.sort_values('sort_order').reset_index(drop=True)
    
    return data


def load_population(model_inputs_root: Path) -> pd.Series:
    data_path = model_inputs_root / 'output_measures' / 'population' / 'all_populations.csv'
    data = pd.read_csv(data_path)
    is_2019 = data['year_id'] == 2019
    is_bothsex = data['sex_id'] == 3
    is_alllage = data['age_group_id'] == 22
    data = (data
            .loc[is_2019 & is_bothsex & is_alllage]
            .set_index('location_id')
            .loc[:, 'population'])

    return data


def write_infections_draws(data: pd.DataFrame,
                           infections_draws_dir: Path,):
    draw_col = np.array([c for c in data.columns if c.startswith('draw_')]).item()
    draw = int(draw_col.split('_')[-1])
    data = data.rename(columns={draw_col:'infections_draw'})
    data['draw'] = draw
    
    out_path = infections_draws_dir / f'{draw_col}.csv'
    data.reset_index().to_csv(out_path, index=False)
    
    return out_path


def write_ratio_draws(data: pd.DataFrame,
                      estimated_ratio: str,
                      ratio_draws_dir: Path,
                      duration: int,):
    draw_col = np.array([c for c in data.columns if c.startswith('draw_')]).item()
    draw = int(draw_col.split('_')[-1])
    data = data.rename(columns={draw_col:f'{estimated_ratio}_draw'})
    if estimated_ratio == 'ifr':
        data['ifr_lr_draw'] = data['ifr_draw'] * data['lr_adj']
        data['ifr_hr_draw'] = data['ifr_draw'] * data['hr_adj']
        del data['lr_adj']
        del data['hr_adj']
    data['draw'] = draw
    data['duration'] = duration

    out_path = ratio_draws_dir / f'{draw_col}.csv'
    data.reset_index().to_csv(out_path, index=False)
    
    return out_path
