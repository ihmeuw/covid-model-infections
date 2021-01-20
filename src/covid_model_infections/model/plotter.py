from typing import Dict, Tuple
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

MEASURE_COLORS = {
    'deaths':{'light':'indianred', 'dark':'darkred'},
    'cases':{'light':'mediumseagreen', 'dark':'darkgreen'},
    'hospitalizations':{'light':'dodgerblue', 'dark':'navy'}
}


def get_dates(input_data: Dict, output_data: Dict) -> Tuple[pd.Timestamp, pd.Timestamp]:
    input_dates = [v['cumul'].reset_index()['date'] for k, v in input_data.items()]
    output_dates = [v['infections_cumul'].reset_index()['date'] for k, v in output_data.items()]
    dates = pd.concat(input_dates + output_dates)
    start_date = dates.min() - pd.Timedelta(days=7)
    end_date = dates.max() + pd.Timedelta(days=7)
    
    return start_date, end_date


def plotter(plot_dir, location_id, location_name,
            input_data,
            test_data, sero_data, ratio_model_inputs,
            output_data, smooth_infections, output_draws, population,
            measures=['cases', 'hospitalizations', 'deaths']):
    start_date, end_date = get_dates(input_data, output_data)
    
    n_cols = 3
    n_rows = 12
    widths = [2, 1, 2]
    heights = [1] * n_rows
    
    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = fig.add_gridspec(n_rows, n_cols, width_ratios=widths, height_ratios=heights)
    
    # line1 = plt.Line2D((0.41, 0.41),(0., 0.975), color='darkgrey', linewidth=2)
    # line2 = plt.Line2D((0.65, 0.65),(0., 0.975), color='darkgrey', linewidth=2)
    # fig.add_artist(line1)
    # fig.add_artist(line2)
    
    for i, measure in enumerate(measures):
        daily_ax = fig.add_subplot(gs[i*4:i*4+4, 0])
        cumul_ax = fig.add_subplot(gs[i*4:i*4+2, 1])
        if measure in list(input_data.keys()):
            # if i == 0:
            #     daily_title = 'Daily'
            #     cumul_title = 'Cumulative (in thousands)'
            # else:
            #     daily_title = None
            #     cumul_title = None
            daily_title = None
            cumul_title = None
            data_plot(daily_ax, daily_title, measure.capitalize(),
                      input_data[measure]['daily'][1:], output_data[measure]['daily'][1:],
                      MEASURE_COLORS[measure]['light'], MEASURE_COLORS[measure]['dark'],
                      start_date, end_date, measure==measures[-1])

            data_plot(cumul_ax, cumul_title, measure.capitalize(),
                      input_data[measure]['cumul'], output_data[measure]['cumul'],
                      MEASURE_COLORS[measure]['light'], MEASURE_COLORS[measure]['dark'],
                      start_date, end_date)
        else:
            daily_ax.axis('off')
            cumul_ax.axis('off')
    
    ratio_names = {'deaths':'IFR', 'hospitalizations':'IHR', 'cases':'IDR'}
    for i, measure in enumerate(measures):
        ratio_ax = fig.add_subplot(gs[i*4+2:i*4+4, 1])
        if measure in list(input_data.keys()):
            adj_ratio = smooth_infections.copy()
            adj_ratio.index += pd.Timedelta(days=input_data[measure]['lag'])
            adj_ratio = output_data[measure]['daily'] / adj_ratio
            adj_ratio = adj_ratio.dropna()
            if ratio_names[measure] == 'IFR':
                adj_ratio = adj_ratio.clip(0, 0.1)
            elif ratio_names[measure] == 'IHR':
                adj_ratio = adj_ratio.clip(0, 0.2)
            elif ratio_names[measure] == 'IDR':
                adj_ratio = adj_ratio.clip(0, 1.)
            else:
                raise ValueError('Unexpected ratio present in plotting.')
            ratio_plot(ratio_ax, ratio_names[measure],
                       pd.concat([input_data[measure]['ratio'], input_data[measure]['daily']], axis=1)['ratio'].dropna(),
                       pd.concat([input_data[measure]['ratio'], input_data[measure]['daily']], axis=1)['ratio_fe'].dropna(),
                       adj_ratio,
                       ratio_model_inputs[measure],
                       MEASURE_COLORS[measure]['light'],
                       MEASURE_COLORS[measure]['dark'],
                       start_date, end_date, measure==measures[-1])
        else:
            ratio_ax.axis('off')
    
    model_measures = [m for m in measures if m in list(output_data.keys())]
    whitespace_top = fig.add_subplot(gs[0:1, 2])
    whitespace_top.axis('off')
    dailymodel_ax = fig.add_subplot(gs[1:5, 2])
    infection_daily_data = {mm: output_data[mm]['infections_daily'][1:] for mm in model_measures}
    model_plot(dailymodel_ax, 'Daily infections', infection_daily_data, None,
               smooth_infections,
               output_draws, start_date, end_date, False)
    whitespace_mid = fig.add_subplot(gs[5:7, 2])
    whitespace_mid.axis('off')
    cumulmodel_ax = fig.add_subplot(gs[7:11, 2])
    infection_cumul_data = {mm: (output_data[mm]['infections_cumul'] / population) * 100 for mm in model_measures}
    model_plot(cumulmodel_ax, 'Cumulative infections (%)', infection_cumul_data, sero_data,
               (smooth_infections.cumsum() / population) * 100,
               (output_draws.cumsum() / population) * 100, start_date, end_date, True)
    whitespace_bottom = fig.add_subplot(gs[11:12, 2])
    whitespace_bottom.axis('off')
    
    fig.suptitle(f'{location_name} ({location_id})', fontsize=20)
    if plot_dir is not None:
        fig.savefig(plot_dir / f'{location_id}.pdf', bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    


def data_plot(ax, title, ylabel, raw_data, smooth_data, clight, cdark, start_date, end_date, include_xticks=False):
    ax.scatter(raw_data.index, raw_data,
               c=clight, edgecolors=cdark, alpha=0.4)
    ax.plot(raw_data, color=clight, alpha=0.2)
    ax.plot(smooth_data, color=cdark, alpha=1.)

    if title:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlim(start_date, end_date)
    if include_xticks:
        ax.tick_params('x', labelrotation=60)
    else:
        ax.set_xticklabels([])
    
    #ax.spines['left'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    #ax.spines['top'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)


def ratio_plot(ax, ylabel, ratio_data, ratio_data_fe, adj_ratio, ratio_input_data, clight, cdark, start_date, end_date, include_xticks=True):
    ax.plot(ratio_data, color=cdark, alpha=0.8)
    ax.plot(ratio_data_fe, linestyle='--', color=cdark, alpha=0.8)
    ax.plot(adj_ratio, color=clight, alpha=0.8)
    
    ax.scatter(ratio_input_data.loc[ratio_input_data['is_outlier'] == 0].index,
               ratio_input_data.loc[ratio_input_data['is_outlier'] == 0, 'ratio'],
               color=cdark, alpha=0.8, marker='o', facecolors='none')
    ax.scatter(ratio_input_data.loc[ratio_input_data['is_outlier'] == 1].index,
               ratio_input_data.loc[ratio_input_data['is_outlier'] == 1, 'ratio'],
               color=cdark, alpha=0.8, marker='x')
    
    ax.set_ylabel(ylabel)
    ax.set_xlim(start_date, end_date)
    
        
    if include_xticks:
        ax.tick_params('x', labelrotation=60)
    else:
        ax.set_xticklabels([])
    
    #ax.spines['left'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    #ax.spines['top'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)


def model_plot(ax, title, measure_data, sero_data, smooth_infections, output_draws, start_date, end_date, include_xticks=False):
    if sero_data is not None:
        ax.scatter(sero_data.loc[(sero_data['manual_outlier'] == 0) & (sero_data['geo_accordance'] == 1)].index,
                   sero_data.loc[(sero_data['manual_outlier'] == 0) & (sero_data['geo_accordance'] == 1), 'seroprev_mean'] * 100, s=100,
                   c='mediumorchid', edgecolors='darkmagenta', alpha=0.6)
        ax.scatter(sero_data.loc[(sero_data['manual_outlier'] == 0) & (sero_data['geo_accordance'] == 0)].index,
                   sero_data.loc[(sero_data['manual_outlier'] == 0) & (sero_data['geo_accordance'] == 0,) 'seroprev_mean'] * 100, s=100,
                   c='orange', edgecolors='darkorange', alpha=0.6, marker='^')
        ax.scatter(sero_data.loc[(sero_data['manual_outlier'] == 1) & (sero_data['geo_accordance'] == 1)].index,
                   sero_data.loc[(sero_data['manual_outlier'] == 1) & (sero_data['geo_accordance'] == 1), 'seroprev_mean'] * 100, s=100,
                   c='darkmagenta', edgecolors='darkmagenta', alpha=0.6, marker='x')
        ax.scatter(sero_data.loc[(sero_data['manual_outlier'] == 1) & (sero_data['geo_accordance'] == 0)].index,
                   sero_data.loc[(sero_data['manual_outlier'] == 1) & (sero_data['geo_accordance'] == 0,) 'seroprev_mean'] * 100, s=100,
                   c='darkorange', edgecolors='darkorange', alpha=0.6, marker='x')

    ax.plot(output_draws.mean(axis=1), color='black', alpha=0.8)
    ax.plot(smooth_infections, color='black', linestyle=':', alpha=0.6)
    ax.fill_between(output_draws.index,
                    np.percentile(output_draws, 2.5, axis=1),
                    np.percentile(output_draws, 97.5, axis=1),
                    color='black', alpha=0.2)
    for m, md in measure_data.items():
        ax.plot(md, color=MEASURE_COLORS[m]['dark'], linestyle='--', alpha=0.6)
    # if title:
    #     ax.set_title(title)
    ax.set_ylabel(title)
    ax.set_xlim(start_date, end_date)
    if include_xticks:
        ax.tick_params('x', labelrotation=60)
    else:
        ax.set_xticklabels([])
    
    #ax.spines['left'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    #ax.spines['top'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
