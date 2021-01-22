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
    elif input_measure == 'hospitalizations':
        is_greece = data['location_id'] == 82
        data = data.loc[~is_greece].reset_index(drop=True)
        manipulation_metadata['greece'] = 'dropped all hospitalizations'
        
        is_norway = data['location_id'] == 90
        data = data.loc[~is_norway].reset_index(drop=True)
        manipulation_metadata['norway'] = 'dropped all hospitalizations'
        
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
    elif input_measure == 'deaths':
        is_india_hierarchy = hierarchy['path_to_top_parent'].apply(lambda x: '163' in x.split(','))
        india_location_ids = hierarchy.loc[is_india_hierarchy, 'location_id'].to_list()
        is_india_data = data['location_id'].isin(india_location_ids)
        data.loc[is_india_data, f'cumulative_{input_measure}'] *= 1.98
        manipulation_metadata['india'] = 'scale deaths up by a factor of 1.98'
        
        is_russia = data['location_id'] == 62
        data.loc[is_russia, f'cumulative_{input_measure}'] *= 3.38
        manipulation_metadata['russia'] = 'scale deaths up by a factor of 3.38'
    else:
        raise ValueError(f'Input measure {input_measure} does not have a protocol for exclusions.')
    
    return data, manipulation_metadata


def load_ifr(infection_fatality_root: Path) -> pd.DataFrame:
    data_path = infection_fatality_root / '20210118_v57_allage_ifr_by_loctime_v19_predbyranef_covidlocs.csv'
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['datevar'])
    data = data.rename(columns={'allage_ifr':'ratio',
                                'allage_ifr_nore':'ratio_fe'})
    data = (data
            .set_index(['location_id', 'date'])
            .sort_index()
            .loc[:, ['ratio', 'ratio_fe']])
    
    return data


def load_ifr_risk_adjustment(infection_fatality_root: Path) -> pd.Series:
    data_path = infection_fatality_root / '20210118_v57_allage_ifr_by_loctime_v19_predbyranef_covidlocs_agegtlt65.csv'
    data = pd.read_csv(data_path)
    data['lr_adj'] = data['ifr_lr'] / data['ifr']
    data['hr_adj'] = data['ifr_hr'] / data['ifr']
    data = (data
            .set_index('location_id')
            .loc[:, ['lr_adj', 'hr_adj']])
    
    return data


def load_ifr_data(infection_fatality_root: Path) -> pd.DataFrame:
    data_path = infection_fatality_root / 'dev_output_dirs' / '57_rsoren' / 'df_prepped_ifr_v19.csv'
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.loc[data['ifr'].notnull()]
    data = data.rename(columns={'ifr':'ratio',
                                'is_outlier_ifr':'is_outlier'})
    data = (data
            .set_index(['location_id', 'date'])
            .sort_index()
            .loc[:, ['ratio', 'is_outlier']])
    
    return data
    

def load_ihr(infection_hospitalization_root: Path) -> pd.DataFrame:
    data_path = infection_hospitalization_root / '20210118_v57_allage_ihr_by_loctime_v19.csv'
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['datevar'])
    data = data.rename(columns={'allage_hir':'ratio',
                                'allage_hir_nore':'ratio_fe'})
    data = (data
            .set_index(['location_id', 'date'])
            .sort_index()
            .loc[:, ['ratio', 'ratio_fe']])
    
    return data


def load_ihr_data(infection_hospitalization_root: Path) -> pd.DataFrame:
    data_path = infection_hospitalization_root / 'dev_output_dirs' / '57_rsoren' / 'df_prepped_ihr_v19.csv'
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])
    data = data.loc[data['ihr'].notnull()]
    data = data.rename(columns={'ihr':'ratio',
                                'is_outlier_ihr':'is_outlier'})
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
            .loc[:, ['location_id', 'infection_date', 'seroprev_mean', 'geo_accordance', 'manual_outlier']])
    data = data.rename(columns={'infection_date':'date'})
    data['date'] = pd.to_datetime(data['date'])
    data = (data
            .set_index(['location_id', 'date'])
            .sort_index()
            .loc[:, ['seroprev_mean', 'geo_accordance', 'manual_outlier']])
    
    return data


def load_testing_data(infection_detection_root: Path):
    data_path = infection_detection_root / 'test_data.csv'
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])
    data = (data
            .set_index(['location_id', 'date'])
            .sort_index())
    
    return data


def load_model_inputs(model_inputs_root:Path, hierarchy: pd.DataFrame, input_measure: str) -> Tuple[pd.Series, pd.Series, Dict]:
    #data_path = model_inputs_root / 'output_measures' / input_measure / 'cumulative.csv'
    data_path = model_inputs_root / 'use_at_your_own_risk' / 'full_data_extra_hospital.csv'
    data = pd.read_csv(data_path)
    data = data.rename(columns={'Deaths':'cumulative_deaths',
                                'Confirmed':'cumulative_cases',
                                'Hospitalizations':'cumulative_hospitalizations',})
    data['date'] = pd.to_datetime(data['Date'])
    #is_all_ages = data['age_group_id'] == 22
    #is_both_sexes = data['sex_id'] == 3
    #data = data.loc[is_all_ages & is_both_sexes]
    #data = data.rename(columns={'value': f'cumulative_{input_measure}'})
    keep_cols = ['location_id', 'date', f'cumulative_{input_measure}']
    data = data.loc[:, keep_cols].dropna()
    data['location_id'] = data['location_id'].astype(int)
    
    data, manipulation_metadata = evil_doings(data, hierarchy, input_measure)
    
    data = (data.groupby('location_id', as_index=False)
            .apply(lambda x: fill_dates(x, [f'cumulative_{input_measure}']))
            .reset_index(drop=True))
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
                           infections_draws_dir: Path,
                           inf_to_death: int,):
    draw_col = np.array([c for c in data.columns if c.startswith('draw_')]).item()
    draw = int(draw_col.split('_')[-1])
    data = data.rename(columns={draw_col:'infections_draw'})
    data['draw'] = draw
    #data['observed_infections'] = data['infections_mean'].notnull().astype(int)
    #data['observed_deaths'] = data['deaths'].notnull().astype(int)
    data['duration'] = inf_to_death
    
    out_path = infections_draws_dir / f'{draw_col}.csv'
    data.reset_index().to_csv(out_path, index=False)
    
    return out_path


def write_ratio_draws(data: pd.DataFrame,
                      ratio_draws_dir: Path,):
    draw_col = np.array([c for c in data.columns if c.startswith('draw_')]).item()
    draw = int(draw_col.split('_')[-1])
    data = data.rename(columns={draw_col:'ifr_draw'})
    data['ifr_lr_draw'] = data['ifr_draw'] * data['lr_adj']
    data['ifr_hr_draw'] = data['ifr_draw'] * data['hr_adj']
    data['draw'] = draw
    del data['lr_adj']
    del data['hr_adj']

    out_path = ratio_draws_dir / f'{draw_col}.csv'
    data.reset_index().to_csv(out_path, index=False)
    
    return out_path
