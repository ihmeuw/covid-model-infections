import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

MEASURE_COLORS = {
    'deaths':{'light':'firebrick', 'dark':'maroon'},
    'cases':{'light':'mediumseagreen', 'dark':'darkgreen'},
    'hospitalizations':{'light':'dodgerblue', 'dark':'navy'}
}


def plotter(plot_dir, location_id, location_name,
            input_data,
            test_data, sero_data,
            output_data, output_draws, population,
            measures=['cases', 'hospitalizations', 'deaths']):
    sns.set_style('whitegrid')
    
    n_cols = 4
    n_rows = 6
    widths = [1, 1, 1, 2]
    heights = [1] * n_rows
    
    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = fig.add_gridspec(n_rows, n_cols, width_ratios=widths, height_ratios=heights)
    
    line1 = plt.Line2D((0.4125, 0.4125),(0.03, 0.97), color='darkgrey', linewidth=2)
    line2 = plt.Line2D((0.6525, 0.6525),(0.03, 0.97), color='darkgrey', linewidth=2)
    fig.add_artist(line1)
    fig.add_artist(line2)
    
    for i, measure in enumerate(measures):
        daily_ax = fig.add_subplot(gs[i*2:i*2+2, 0])
        cumul_ax = fig.add_subplot(gs[i*2:i*2+2, 1])
        if measure in list(input_data.keys()):
            if i == 0:
                daily_title = 'Daily'
                cumul_title = 'Cumulative (in thousands)'
            else:
                daily_title = None
                cumul_title = None
            data_plot(daily_ax, daily_title, measure.capitalize(),
                      input_data[measure]['daily'], output_data[measure]['daily'],
                      MEASURE_COLORS[measure]['light'], MEASURE_COLORS[measure]['dark'])

            data_plot(cumul_ax, cumul_title, measure.capitalize(),
                      input_data[measure]['cumul'] / 1000, output_data[measure]['cumul'] / 1000,
                      MEASURE_COLORS[measure]['light'], MEASURE_COLORS[measure]['dark'])
        else:
            daily_ax.axis('off')
            cumul_ax.axis('off')
    
    for i, measure in enumerate(measures):
        ratio_ax = fig.add_subplot(gs[i*2:i*2+2, 2])
        if measure in list(input_data.keys()):
            if measure == 'cases':
                alt_data = test_data.copy() * 1e5
                alt_data = alt_data[input_data[measure]['daily'].index]
                alt_measure = 'Tests per 100K'
            else:
                alt_data = None
                alt_measure = None
            ratio_plot(ratio_ax, input_data[measure]['ratio'].name.upper(), alt_measure,
                       input_data[measure]['ratio'][input_data[measure]['daily'].index],
                       alt_data, MEASURE_COLORS[measure]['dark'])
        else:
            ratio_ax.axis('off')
    
    model_measures = [m for m in measures if m in list(output_data.keys())]
    dailymodel_ax = fig.add_subplot(gs[0:3, 3])
    infection_daily_data = {mm: output_data[mm]['infections_daily'] for mm in model_measures}
    model_plot(dailymodel_ax, 'Daily', infection_daily_data, None, output_draws)
    cumulmodel_ax = fig.add_subplot(gs[3:6, 3])
    infection_cumul_data = {mm: (output_data[mm]['infections_cumul'] / population) * 100 for mm in model_measures}
    model_plot(cumulmodel_ax, 'Cumulative infections (%)', infection_cumul_data, sero_data,
               (output_draws.cumsum() / population) * 100)
    
    fig.suptitle(f'{location_name} ({location_id})', fontsize=20)
    fig.savefig(plot_dir / f'{location_id}.pdf', bbox_inches='tight')
    plt.close(fig)


def data_plot(ax, title, ylabel, raw_data, smooth_data, clight, cdark):
    ax.scatter(raw_data.index, raw_data,
               c=clight, edgecolors=cdark, alpha=0.4)
    ax.plot(raw_data, color=cdark, alpha=0.2)
    ax.plot(smooth_data, color=cdark, alpha=1.)

    if title:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params('x', labelrotation=60)
    
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


def ratio_plot(ax, ylabel, alt_ylabel, ratio_data, alt_data, cdark):
    ax.plot(ratio_data, color=cdark, alpha=0.8)
    ax.set_ylabel(ylabel)
    ax.tick_params('x', labelrotation=60)
    
    if alt_data is not None:
        ax_alt = ax.twinx()
        ax_alt.plot(alt_data, color=cdark, linestyle='--', alpha=0.8)
        ax_alt.set_ylabel(alt_ylabel, verticalalignment='bottom', rotation=270) 
    
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


def model_plot(ax, title, measure_data, sero_data, output_draws):
    if sero_data is not None:
        sero_date = sero_data.index
        sero_mean = sero_data['seroprev_mean']
        sero_var = (sero_mean * (1 - sero_mean)) / sero_data['sample_size']
        sero_size = (1 / sero_var) / 1e4
        sero_size = sero_size.clip(25, 250)
        ax.scatter(sero_date, sero_mean * 100, s=sero_size,
                   c='mediumorchid', edgecolors='darkmagenta', alpha=0.6)
    ax.plot(output_draws.mean(axis=1), color='black', alpha=0.8)
    ax.fill_between(output_draws.index,
                    np.percentile(output_draws, 2.5, axis=1),
                    np.percentile(output_draws, 97.5, axis=1),
                    color='black', alpha=0.2)
    for m, md in measure_data.items():
        ax.plot(md, color=MEASURE_COLORS[m]['light'], linestyle='--', alpha=0.6)
    if title:
        ax.set_title(title)
    ax.set_ylabel('Infections')
    ax.tick_params('x', labelrotation=60)
    
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
