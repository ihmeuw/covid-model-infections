import sys
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import numpy as np

from covid_model_infections.model import plotter


def aggregate_md_data_dict(md_data: Dict, hierarchy: pd.DataFrame, measures: List[str]):
    parent_ids = hierarchy.loc[hierarchy['most_detailed'] != 1, 'location_id'].to_list()
    
    agg_data = {parent_id: create_parent_dict(md_data, parent_id, hierarchy, measures) for parent_id in parent_ids}
    
    return agg_data


def get_child_ids(parent_id: int, hierarchy: pd.DataFrame) -> List:
    is_child = (hierarchy['path_to_top_parent'].apply(lambda x: str(parent_id) in x.split(',')))
    is_most_detailed = hierarchy['most_detailed'] == 1
    child_ids = hierarchy.loc[is_child & is_most_detailed, 'location_id'].to_list()
    
    return child_ids

    
def create_parent_dict(md_data: Dict, parent_id: int, hierarchy: pd.DataFrame, measures: str) -> Dict:
    child_ids = get_child_ids(parent_id, hierarchy)
    children_data = [md_data.get(child_id, None) for child_id in child_ids]
    children_data = [cd for cd in children_data  if cd is not None]
    
    parent_data = sum_data_from_child_dicts(children_data, measures)
    
    return parent_data

    
def sum_data_from_child_dicts(children_data: Dict, measures: List[str]):
    child_dict = {}
    for measure in measures:
        measure_dict = {}
        metrics = [list(child_data.get(measure, {}).keys()) for child_data in children_data]
        if not all([bool(m) for m in metrics]):
            metrics = []
        else:
            metrics = metrics[0]
        for metric in metrics:
            metric_data = []
            for child_data in children_data:
                if metric == 'ratio':
                    ratio_data = (child_data[measure]['daily'] / child_data[measure][metric]['ratio']).rename('ratio')
                    ratio_fe_data = (child_data[measure]['daily'] / child_data[measure][metric]['ratio_fe']).rename('ratio_fe')
                    metric_data.append(pd.concat([ratio_data, ratio_fe_data], axis=1).dropna())
                else:
                    metric_data.append(child_data[measure][metric])
            if isinstance(metric_data[0], int):
                metric_data = metric_data[0]
            else:
                metric_data = pd.concat(metric_data)
                if metric_data.index.names != ['date']:
                    raise ValueError('Cannot aggregate multi-index.')
                metric_data_count = metric_data.groupby(level=0).count()
                keep_idx = metric_data_count[metric_data_count == len(children_data)].index
                metric_data = metric_data.groupby(level=0).sum()
                metric_data = metric_data.loc[keep_idx]
            measure_dict.update({metric: metric_data})
        
        if 'ratio' in metrics:
            if measure_dict['daily'].empty:
                 measure_dict['ratio'] = (measure_dict['ratio'] * np.nan).dropna()
            else:
                ratio_data = (measure_dict['daily'] / measure_dict['ratio']['ratio']).rename('ratio')
                ratio_fe_data = (measure_dict['daily'] / measure_dict['ratio']['ratio_fe']).rename('ratio_fe')
                measure_dict['ratio'] = pd.concat([ratio_data, ratio_fe_data], axis=1).dropna()
            
        if metrics:
            child_dict.update({measure: measure_dict})
    
    return child_dict
    
    
def aggregate_md_draws(md_draws: pd.DataFrame, hierarchy: pd.DataFrame) -> pd.DataFrame:
    parent_ids = hierarchy.loc[hierarchy['most_detailed'] != 1, 'location_id'].to_list()
    
    agg_draws = []
    for parent_id in tqdm(parent_ids, total=len(parent_ids), file=sys.stdout):
        agg_draws.append(create_parent_draws(md_draws, parent_id, hierarchy))
    agg_draws = pd.concat(agg_draws)
    
    return agg_draws
    

def create_parent_draws(md_draws: pd.DataFrame, parent_id: int, hierarchy: pd.DataFrame):
    child_ids = get_child_ids(parent_id, hierarchy)
    child_ids = [i for i in child_ids if i in md_draws.reset_index()['location_id'].to_list()]
    
    parent_draws = md_draws.loc[child_ids]
    if parent_draws.index.names != ['location_id', 'date']:
        raise ValueError("Multi-index differs from expected (['location_id', 'date']).")
    parent_draws_count = parent_draws.groupby(level=1).count().iloc[:,0]
    keep_idx = parent_draws_count[parent_draws_count == len(child_ids)].index
    parent_draws = parent_draws.groupby(level=1).sum()
    parent_draws = parent_draws.loc[keep_idx]
    parent_draws['location_id'] = parent_id
    parent_draws = (parent_draws
                    .reset_index()
                    .set_index(['location_id', 'date'])
                    .sort_index())
    
    return parent_draws

def plot_aggregate(location_id: int,
                   model_data: Dict, outputs: Dict, infections_draws: pd.DataFrame,
                   hierarchy: pd.DataFrame,
                   pop_data: pd.Series,
                   sero_data: pd.DataFrame,
                   ifr_model_data: pd.DataFrame,
                   ihr_model_data: pd.DataFrame,
                   idr_model_data: pd.DataFrame,
                   plot_dir: Path):
    sero_data = sero_data.reset_index()
    sero_data = (sero_data
                 .loc[sero_data['location_id'] == location_id]
                 .drop('location_id', axis=1)
                 .set_index('date'))
    
    population = pop_data[location_id]
    
    location_name = hierarchy.loc[hierarchy['location_id'] == location_id, 'location_name'].item()
    
    infections_mean = infections_draws.mean(axis=1).rename('infections')
    
    ratio_model_inputs = {}
    for measure, ratio_model_data in [('deaths', ifr_model_data),
                                      ('hospitalizations', ihr_model_data),
                                      ('cases', idr_model_data)]:
        ratio_model_data = idr_model_data.reset_index()
        ratio_model_data = (ratio_model_data
                            .loc[ratio_model_data['location_id'] == location_id]
                            .drop('location_id', axis=1)
                            .set_index('date'))
        ratio_model_inputs.update({measure:ratio_model_data})

    plotter.plotter(
        plot_dir, location_id, location_name,
        model_data, sero_data, ratio_model_inputs,
        outputs, infections_mean, infections_draws, population
    )
    