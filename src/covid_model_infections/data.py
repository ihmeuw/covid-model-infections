from typing import List, Tuple
from pathlib import Path

import pandas as pd


def load_ifr(infection_fatality_root: Path) -> pd.Series:
    data_path = infection_fatality_root / '20210110_allage_ifr_by_loctime_v3_predbyranef.csv'
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['datevar'])
    data = data.rename(columns={'allage_ifr':'ifr'})
    data = (data
            .set_index(['location_id', 'date'])
            .sort_index()
            .loc[:, 'ifr'])
    
    return data


def load_ihr(infection_hospitalization_root: Path) -> pd.Series:
    data_path = infection_hospitalization_root / '20210110_v57_allage_hir_by_loctime_v5.csv'
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['datevar'])
    data = data.rename(columns={'allage_hir':'ihr'})
    data = (data
            .set_index(['location_id', 'date'])
            .sort_index()
            .loc[:, 'ihr'])
    
    return data


def load_idr(infection_detection_root: Path, limits: Tuple[float, float] = (0.03, 0.7)) -> pd.Series:
    data_path = infection_detection_root / 'pred_idr.csv'
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])
    data = (data
            .set_index(['location_id', 'date'])
            .sort_index()
            .loc[:, 'idr'])
    data = data.clip(*limits)
    
    return data


def load_sero_data(infection_detection_root: Path):
    data_path = infection_detection_root / 'sero_data.csv'
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])
    
    return data


def load_testing_data(infection_detection_root: Path):
    data_path = infection_detection_root / 'test_data.csv'
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])
    data = (data
            .set_index(['location_id', 'date'])
            .sort_index())
    
    return data


def load_model_inputs(model_inputs_root:Path, input_measure: str) -> Tuple[pd.Series, pd.Series]:
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

    return cumulative_data, daily_data


def fill_dates(data: pd.DataFrame, interp_vars: List[str]) -> pd.DataFrame:
    data = data.set_index('date').sort_index()
    data = data.asfreq('D').reset_index()
    data[interp_vars] = data[interp_vars].interpolate(axis=0)
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


def write_seir_inputs(data: pd.DataFrame,
                      out_dir: Path,):
    draw_col = np.array([c for c in data.columns if c.startswith('draw_')]).item()
    draw = int(draw_col.split('_')[-1])
    
    out_path = out_root / f'draw_{draw}.csv'
    data.rename(columns={draw_col:'infections_draw',}).reset_index().to_csv(out_path, index=False)
    
    return out_path
