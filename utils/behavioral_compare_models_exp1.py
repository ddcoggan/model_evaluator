'''
This scripts makes plots and runs statistical analyses comparing model
performance in the behavioral benchmarks.
'''

import os
import os.path as op
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import time
from itertools import product as itp
import pingouin as pg
from statsmodels.stats.anova import AnovaRM
from copy import deepcopy

from .model_contrasts import (model_contrasts, HUMAN_COLOR,
                              HUMAN_MARKER, HUMAN_CONFIG)
from utils.benchmarks.occlusion_behavioral_exp1 import load_trials, reshape_metrics
from . import now, MODEL_BASE
from .plot_utils import make_legend, custom_defaults
plt.rcParams.update(custom_defaults)
from .math_functions import sigmoid
from .make_legends import make_legends
from .model_contrasts import model_properties

sys.path.append(f'..')
from p022_occlusion.in_vivo.behavioral.exp1.analysis import CFG as EXP1

METRICS = ['accuracy', 'true_class_prob', 'entropy']
OBJ_VARIABLES = ['object_animacy', 'object_class']
OBJ_ANIMACIES = ['animate', 'inanimate']
OBJ_CLASSES = EXP1.object_classes
OBJ_CLS_IDCS = EXP1.class_idxs
OBJ_CLS_DIRS = EXP1.synsets
OCC_VARIABLES = ['visibility', 'occluder_class', 'occluder_color']
OCC_COLORS = ['black', 'white']
OCC_CLASSES = EXP1.occluder_classes
UNOCC_COLOR = (.8, .8, .8)
VISIBILITIES = EXP1.visibilities
RES_DIR = 'benchmarking/occlusion_behavioral/exp1'


def bar_plot(data_config, df, plot_config, plot_file, samples=None,
             ceiling=None, multicycle=False, xticklabel=False):

    invert = 'invert' in plot_config and plot_config['invert']
    if multicycle:
        if 'humans' in data_config:
            del data_config['humans']
        hum_val, bottom = df[df.group == 'humans']['value'].mean(), 0
        if invert:
            hum_val, bottom = 1, hum_val
    group_counts = get_group_counts(data_config)
    groupby = ['cycle']
    if samples:
        groupby.extend(samples['columns'])
    figsize = (1 + len(group_counts)/5 + sum(group_counts)/4,
               3 + int(xticklabel) * 2)
    fig, axes = plt.subplots(
        ncols=len(group_counts), sharey='row', figsize=figsize,
        gridspec_kw={'width_ratios': group_counts})
    min_y, max_y = np.inf, -np.inf
    for g, (group, models) in enumerate(data_config.items()):
        ax = axes[g] if len(group_counts) > 1 else axes
        xticks, xticklabels = [], []
        max_x = 0
        for m, (model, info) in enumerate(models.items()):
            xpos = info['xpos'] if 'xpos' in info else m
            xticks.append(xpos)
            max_x = max(max_x, xpos)
            xticklabels.append(model.replace('\n', ' '))
            df_model = (df[(df.group == group) & (df.model == model)]
                        .groupby(groupby, observed=False, dropna=False)
                        .agg({'value': 'mean'}).reset_index())
            if multicycle:  # plot without samples
                ax.axhline(y=hum_val, color=HUMAN_COLOR, linestyle='solid')
                cycles = df_model.cycle.unique()
                xposs, width = split_bar(len(cycles), xpos)
                for x, cycle in zip(xposs, cycles):
                    yval = df_model[df_model.cycle == cycle]['value'].mean()
                    bottom = yval if invert else 0
                    yval = 1 if invert else yval
                    ax.bar(x, yval, bottom=bottom, color=info['color'],
                           edgecolor=info['edgecolor'], width=width, zorder=2)
                    min_y, max_y = min(min_y, yval), max(max_y, yval)
            else:  # plot with samples
                yvals = df_model.sort_values(by=samples['columns'])['value'
                    ].to_list()
                if 'robustness' in plot_file:
                    yval = remove_unoccluded(df_model)['value'].mean()
                else:
                    yval = np.nanmean(yvals) if any(np.isfinite(yvals)) else np.nan
                bottom = yval if invert else 0
                yval = 1 if invert else yval
                ax.bar(xpos, yval, bottom=bottom, color=info['color'],
                       edgecolor=info['edgecolor'], linewidth=1, zorder=2)
                sns.stripplot(
                    x=xpos, y=yvals, zorder=3, clip_on=False,
                    native_scale=True, dodge=True, ax=ax,
                    color=samples['color'], size=samples['size'],
                    alpha=samples['alpha'], linewidth=samples['linewidth'],
                    edgecolor=samples['markeredgecolor'])
                if 'errorbar' in samples and samples['errorbar']:
                    ax.errorbar(xpos, np.nanmean(yvals), stats.sem(yvals),
                                color='k', capsize=4, zorder=4)
                min_y, max_y = min(min_y, min(yvals)), max(max_y, max(yvals))

        # x-axis formatting
        if xticklabel:
            ax.set_xticks(xticks, labels=xticklabels, rotation=90,
                          ha='center', va='top')
        else:
            ax.set_xlabel(group, size=10, rotation=0, labelpad=10)
            ax.set_xticks([])
        xlims = (-.5, max_x + .5)
        ax.set_xlim(xlims)

        # y-axis formatting
        if ceiling is not None:
            ax.fill_between(xlims, ceiling[0], ceiling[1], color='tab:gray',
                            lw=0, zorder=1)
        if 'chance' in plot_config:
            ax.axhline(y=plot_config['chance'], color='k', linestyle='dotted')
        ax.tick_params(axis='y', which='both', left=False)
        if g == 0:
            ax.set_ylabel(plot_config['ylabel'])
            ax.set_yticks(plot_config['yticks'])
        else:
            ax.spines['left'].set_visible(False)
        if 'ylims' in plot_config:
            ylims = plot_config['ylims']
        else:
            # round up/down to nearest 0.2 for min/max
            ylims = [np.floor(min_y * 5) / 5, np.ceil(max_y * 5) / 5]
        if min(ylims) < 0:
            if len(group_counts) > 1:
                for a in axes:
                    a.spines['bottom'].set_visible(False)
            else:
                ax.spines['bottom'].set_visible(False)
        ax.set_ylim(ylims)
        ax.grid(axis='y', linestyle='solid', alpha=.25, zorder=0, clip_on=False)

    plt.tight_layout(pad=1)
    fig.savefig(plot_file)
    fig.savefig(plot_file.replace('.svg', '.pdf'))
    plt.close()


def bar_plot_mirror(data_config, df, plot_config, plot_file, ceiling=None):

    group_counts = get_group_counts(data_config)
    figsize = (7, 2 + (sum(group_counts) / 4))
    fig, axes = plt.subplots(
        ncols=2, nrows=len(group_counts), sharey='row', sharex='col',
        figsize=figsize, gridspec_kw={'height_ratios': group_counts})
    max_params = model_properties.num_params.max()

    # left column: number of parameters
    for c, metric in enumerate(['number of parameters', plot_config['ylabel']]):
        for g, (group, models) in enumerate(data_config.items()):
            ax = axes[g, c] if len(group_counts) > 1 else axes[c]
            if c == 0 and group == 'humans':
                ax.axis('off')
                continue
            for m, (model, info) in enumerate(models.items()):

                ypos = info['xpos'] if 'xpos' in info else m

                # left column: number of parameters
                if c == 0:
                    if model == 'humans':
                        xval = 0
                    else:
                        xval = model_properties.num_params[
                            model_properties.architecture == model].item()
                    ax.barh(ypos, xval, color='tab:red', zorder=2)

                # right column: behavioral benchmark
                elif c == 1:
                    xval = df[(df.group == group) & (df.model == model)][
                        'value'].mean()
                    ax.barh(ypos, xval, color=info['color'], zorder=2)

                    # axis labels
                    label_xpos = plot_config['ylims'][0] if 'ylims' in \
                        plot_config else 0
                    ax.text(label_xpos - .3, ypos, model, ha='center',
                            va='center', fontsize=10)

            # y-axis formatting
            ax.set_yticks([])
            ax.tick_params(axis='y', which='both', left=False)
            ax.grid(axis='x', linestyle='solid', zorder=0, clip_on=False)
            if g < len(group_counts) - 1:
                ax.spines['bottom'].set_visible(False)
            ylims = [-.5, group_counts[g] - .5]
            if g == len(group_counts) - 1:
                ylims[0] -= .1
            ax.set_ylim(ylims)

            # x-axis formatting
            if c == 0:
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(True)
                if g == len(group_counts) - 1:
                    ax.set_xlabel('number of parameters', size=10)
                    ax.set_xlim((1e9, 1e6))
                    ax.set_xscale('log')
            if c == 1:
                ax.spines['right'].set_visible(False)
                if g == len(group_counts) - 1:
                    ax.set_xlabel(plot_config['ylabel'], size=10)
                    ax.set_xticks(plot_config['yticks'])
                else:
                    ax.tick_params(axis='x', which='major', bottom=False)
                if 'chance' in plot_config:
                    ax.axvline(x=plot_config['chance'], color='k',
                               linestyle='dotted', zorder=3)
                if 'ylims' in plot_config:
                    ax.set_xlim(plot_config['ylims'])
                    if min(plot_config['ylims']) < 0:
                        ax.spines['left'].set_visible(False)
                        ax.axvline(x=0, color='k', linestyle='solid')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.6)
    fig.savefig(plot_file)
    fig.savefig(plot_file.replace('.svg', '.pdf'))
    plt.close()


def scatterplot(data_config, human_likeness, robustness, hum_config,
                rob_config, plot_file, noise_ceiling=None, multicycle=False):

    """ Creates scatterplot with different cycles outlined in different
    colors. Secondary model colors are not used, but such distinctions can
    be implemented through marker shape by specifying in the model contrast
    file. Markers default to different shapes for each group, chosen from
    the list below. """

    fig, ax = plt.subplots(figsize=(4,4))
    legend_labels, legend_markers, legend_colors, legend_edgecolors = ([], [],
                                                                       [], [])
    for g, (group, models) in enumerate(data_config.items()):
        for m, (model, info) in enumerate(models.items()):
            # plot human data only if a noise ceiling is available
            if noise_ceiling:
                y = noise_ceiling
                metric = 'accuracy' if 'exp1' in plot_file else 'visibility'
                x = robustness[(robustness.model == 'humans') &
                               (robustness.metric == metric)]['value'].mean()
                ax.scatter(x, y, color=HUMAN_COLOR, marker=HUMAN_MARKER, s=32,
                           zorder=3)
                legend_labels.append('Humans')
                legend_markers.append(HUMAN_MARKER)
                legend_colors.append(HUMAN_COLOR)
            # plot model data
            hum_model = human_likeness[
                (human_likeness.group == group) &
                (human_likeness.model == model)]
            rob_model = robustness[
                (robustness.group == group) &
                (robustness.model == model)]
            if multicycle:
                cycles = hum_model.cycle.unique()
                edge_cols = [matplotlib.cm.inferno.colors[int(i)] for i in
                     np.linspace(0, 255, len(cycles))]

                for cycle, edge_col in zip(cycles, edge_cols):
                    x = rob_model[rob_model.cycle == cycle]['value'].mean()
                    y = hum_model[hum_model.cycle == cycle]['value'].mean()
                    ax.scatter(x, y, color=info['color'], edgecolor=edge_col,
                               s=32, marker=info['marker'], linewidth=1,
                               zorder=2)
                    legend_labels.append(model)
                    legend_markers.append(info['marker'])
                    legend_colors.append(info['color'])
                    legend_edgecolors.append(edge_col)
            else:
                x = rob_model['value'].mean()
                y = hum_model['value'].mean()
                ax.scatter(x, y, color=info['color'], s=32,
                           marker=info['marker'], zorder=1)

    ax.set_xlabel(rob_config['title'], size=10, rotation=0)
    if 'invert' in rob_config and rob_config['invert']:
        ax.invert_xaxis()
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlim(rob_config['ylims'])
    ax.axvline(x=rob_config['chance'], color='k', linestyle='dotted')
    if 'chance' in hum_config:
        ax.axhline(y=hum_config['chance'], color='k', linestyle='dotted')
    ax.grid(axis='both', linestyle='solid', alpha=.25, zorder=0,
            clip_on=False)
    ax.set_ylabel(hum_config['ylabel'], size=10)
    if 'invert' in hum_config and hum_config['invert']:
        ax.invert_yaxis()
    ax.set_yticks(hum_config['yticks'])
    if 'ylims' in hum_config:
       ax.set_ylim(hum_config['ylims'])
    plt.tight_layout()
    fig.savefig(plot_file)
    fig.savefig(plot_file.replace('.svg', '.pdf'))
    plt.close()

    if multicycle:
        make_legend(
            outpath=f'{op.dirname(plot_file)}/legend_multicycle.svg',
            labels=legend_labels,
            markers=legend_markers,
            colors=legend_colors,
            markeredgecolors=legend_edgecolors,
            linestyles='None')


def curve_plot(data_config, df, curves, plot_config, plot_file, exp='exp1'):

    fig, ax = plt.subplots(figsize=(4, 4))
    curve_x = np.linspace(0, 1, 1000)
    xvals = VISIBILITIES + [1]

    for g, (group, models) in enumerate(data_config.items()):
        for m, (model, info) in enumerate(models.items()):

            # point properties
            yvals = (df
                [(df.group == group) & (df.model == model)]
                .groupby('visibility')
                .agg({'value': 'mean'})
                .sort_values(by='visibility')['value'].to_list())
            zorder = 2 if model == 'humans' else 1
            ax.scatter(xvals, yvals, s=32, zorder=zorder, clip_on=False,
                       color=info['color'], marker=info['marker'],
                       edgecolor=info['markeredgecolor'],
                       linestyle=info['linestyle'])

            # fitted curves
            curves.subject = curves.subject.replace(np.nan, 'group')
            popt = curves[
                (curves.group == group) &
                (curves.model == model) &
                (curves.subject == 'group') &
                (curves.occluder_class == 'all') &
                (curves.occluder_color == 'all')][
                ['L', 'x0', 'k', 'b']]
            assert len(popt) == 1
            curve_y = sigmoid(curve_x, *popt.values[0])
            ax.plot(curve_x, curve_y, color=info['linecolor'],
                    linestyle=info['linestyle'])

    # format plot
    ax.grid(axis='both', linestyle='solid', alpha=.25, zorder=0, clip_on=False)
    ax.set_xticks(xvals)
    ax.set_xlim((0, 1))
    ax.set_yticks(plot_config['yticks'])
    ax.set_ylim(plot_config['ylims'])
    ax.tick_params(axis='both', which='major', labelsize=7)
    if 'chance' in plot_config:
        ax.axhline(y=plot_config['chance'], color='k', ls='dotted')
    ax.set_xlabel('visibility')
    ax.set_ylabel(plot_config['ylabel'])
    plt.tight_layout()
    fig.savefig(plot_file)
    fig.savefig(plot_file.replace('.svg', '.pdf'))
    plt.close()


def line_plot(data_config, df, plot_config, samples, plot_file, ceiling=None):

    fig, ax = plt.subplots(figsize=(3, 3))

    for g, (group, models) in enumerate(data_config.items()):
        for m, (model, info) in enumerate(models.items()):

            df_model = df[(df.group == group) & (df.model == model)]
            yvals = (df_model
                .groupby('cycle')
                .agg({'value': 'mean'})
                .sort_values(by='cycle')['value'].to_list())
            if model == 'humans':
                ax.axhline(y=yvals[0], color=info['color'],
                           linestyle=info['linestyle'],
                           zorder=1)
            else:
                cycles = sorted([int(i) for i in df_model.cycle.unique()])
                ax.scatter(cycles, yvals, color=info['color'], s=32, zorder=2,
                           clip_on=False, marker=info['marker'],
                           edgecolor=info['markeredgecolor'])
                if 'errorbar' in samples and samples['errorbar']:
                    yerrs = (df_model
                        .groupby(['cycle', 'subject'])
                        .agg({'value': 'mean'})
                        .groupby('cycle')
                        .agg({'value': 'sem'})
                        .sort_values(by='cycle')['value'].to_list())
                    ax.errorbar(cycles, yvals, yerrs, color=info['linecolor'],
                                capsize=2, zorder=3)
                else:
                    ax.plot(cycles, yvals, color=info['linecolor'],
                            linestyle=info['linestyle'], zorder=1)


    # format plot
    ax.grid(axis='both', linestyle='solid', alpha=.25, zorder=0, clip_on=False)
    max_cycles = df.cycle.max()
    xlims = (-.5, max_cycles + .5)
    ax.set_xticks(np.arange(max_cycles + 1))
    ax.set_xlim(xlims)
    ax.set_yticks(plot_config['yticks'])
    ax.set_ylim(plot_config['ylims'])
    ax.tick_params(axis='both', which='major', labelsize=7)
    if ceiling is not None:
        ax.fill_between(xlims, ceiling[0], ceiling[1], color=HUMAN_COLOR,
                        lw=0, zorder=1)
    if 'chance' in plot_config:
        ax.axhline(y=plot_config['chance'], color='k', ls='dotted')
    ax.set_xlabel('cycle')
    ax.set_ylabel(plot_config['ylabel'])
    plt.tight_layout()
    fig.savefig(plot_file)
    fig.savefig(plot_file.replace('.svg', '.pdf'))
    plt.close()


def compare_models(overwrite=False):

    def _robustness_plots(df, curves, model_config, plot_configs, results_dir,
                          overwrite):

        if all(df.model == 'humans'):
            return

        metric = df.name
        for dataset in ['all', 'artificial']:
            out_dir = (f'{results_dir}/occlusion_robustness_dataset-{dataset}'
                       f'/{metric}')
            os.makedirs(out_dir, exist_ok=True)

            if dataset == 'all':
                cond_samples = {
                    'columns': ['occluder_class', 'occluder_color'],
                    'color': EXP1.plot_colors + [(1.,1.,1.)],
                    'markeredgecolor': EXP1.plot_colors + [(0.,0.,0.)],
                    'linewidth': [0] * len(EXP1.plot_colors) + [1],
                    'size': 5,
                    'alpha': 1,
                }
            else:
                df = df[df.occluder_class != 'naturalUntexturedCropped2']
                cond_samples = {
                    'columns': ['occluder_class', 'occluder_color'],
                    'color': EXP1.plot_colors[:-2] + [(1.,1.,1.)],
                    'markeredgecolor': EXP1.plot_colors[:-2] + [(0.,0.,0.)],
                    'linewidth': [0] * (len(EXP1.plot_colors) - 2) + [1],
                    'size': 5,
                    'alpha': 1,
                }

            # all cycles
            if df.cycle.max() > 0:

                # bar plot
                plot_file = f'{out_dir}/all_cycles.svg'
                if not op.isfile(plot_file) or overwrite:
                    data_config = {}
                    if 'humans' in df.model.unique():
                        data_config['humans'] = HUMAN_CONFIG
                    data_config.update(model_config)
                    bar_plot(
                        data_config=data_config,
                        df=df,
                        samples=cond_samples,
                        plot_config=plot_configs[metric],
                        plot_file=plot_file,
                        multicycle=True)

                # line plot
                plot_file = f'{out_dir}/all_cycles_lineplot.svg'
                if not op.isfile(plot_file) or overwrite:
                    data_config = {}
                    if 'humans' in df.model.unique():
                        data_config['humans'] = HUMAN_CONFIG
                    data_config.update(model_config)
                    line_plot(
                        data_config=data_config,
                        df=df,
                        plot_config=plot_configs[metric],
                        samples={'errorbar': True},
                        plot_file=plot_file)

            # final cycle only
            plot_file = f'{out_dir}/last_cycle.svg'
            anova_path = f'{out_dir}/anova.csv'
            ph_path = anova_path.replace('anova', 'posthocs')
            if not all([op.isfile(i) for i in [plot_file, anova_path, ph_path]]
                       ) or overwrite:
                df_plot = (df
                     .groupby(['group', 'model', 'layer'])
                     .apply(lambda d: d[d.cycle == d.cycle.max()])
                     .reset_index(drop=True))
                df_plot = add_unoccluded(df_plot)
                data_config = {}
                if 'humans' in df.model.unique():
                    data_config['humans'] = HUMAN_CONFIG
                data_config.update(model_config)
                bar_plot(
                    data_config=data_config,
                    df=df_plot,
                    samples=cond_samples,
                    plot_config=plot_configs[metric],
                    plot_file=plot_file)

                #factors = [i for i in df_plot.columns if i.startswith(
                #    'factor_')]
                #if factors:
                #    df_stats = (df_plot
                #
                #        .groupby(['model', 'subject'] + factors)
                #        .mean('value')
                #       .reset_index())
                df_plot = df_plot[df_plot.visibility < 1]
                inferential_stats(df_plot, out_dir)

            # pooled across model groups for each condition
            plot_file = f'{out_dir}/last_cycle_pooled.svg'
            if not op.isfile(plot_file) or overwrite:

                df_plot = (df
                    .groupby(['subject', 'model', 'layer'])
                    .apply(lambda d: d[d.cycle == d.cycle.max()])
                    .reset_index(drop=True))
                xlabel_padding = max([len(i.split('\n')) for i in
                                      model_config]) - 1
                xlabel = 'pooled models' + '\n' * xlabel_padding
                df_plot.group = [xlabel if i != 'humans' else 'humans'
                                 for i in df_plot.group]
                df_plot.cycle = -1
                df_plot = add_unoccluded(df_plot)
                df_plot = (df_plot
                    .groupby([i for i in df_plot.columns if i != 'value'],
                             dropna=False, observed=False)
                    .agg({'value': 'mean'})
                    .reset_index())
                df_plot = df_plot[~df_plot['value'].isnull()]
                data_config = {}
                if 'humans' in df.model.unique():
                    data_config['humans'] = HUMAN_CONFIG
                data_config[xlabel] = list(model_config.values())[0]
                bar_plot(
                    data_config=data_config,
                    df=df_plot,
                    samples=cond_samples,
                    plot_config=plot_configs[metric],
                    plot_file=plot_file,
                    xticklabel=True)


            # performance curves (last cycle only)
            plot_file = f'{out_dir}/{metric}_curves.svg'
            if not op.isfile(plot_file) or overwrite:
                df_plot = (df
                           .groupby(['group', 'model', 'layer'])
                           .apply(lambda d: d[d.cycle == d.cycle.max()])
                           .reset_index(drop=True))
                df_plot = add_unoccluded(df_plot)
                curves_plot = (curves
                    [curves.metric == metric]
                    .groupby(['group', 'model', 'layer'])
                    .apply(lambda d: d[d.cycle == d.cycle.max()])
                    .reset_index(drop=True))
                data_config = {}
                if 'humans' in df.model.unique():
                    data_config['humans'] = HUMAN_CONFIG
                data_config.update(model_config)
                curve_plot(
                    data_config=data_config,
                    df=df_plot,
                    curves=curves_plot,
                    plot_config=plot_configs[metric],
                    plot_file=plot_file,
                    exp='exp1')

            # bar plot with model labels, sorted by robustness
            plot_file = f'{out_dir}/last_cycle_sorted.svg'
            if model_contrast == 'public_models' and (
                    not op.isfile(plot_file) or overwrite):
                df_plot = (df[df.visibility < 1]
                    .groupby(['group', 'model', 'layer'])
                    .apply(lambda d: d[d.cycle == d.cycle.max()])
                    .reset_index(drop=True)
                    .groupby(['group', 'model', 'layer'])
                    .agg({'value': 'mean'})
                    .reset_index())
                df_plot_order = (df_plot[df_plot.group == 'public_models']
                    .sort_values(by='value', ascending=True)
                    .reset_index(drop=True))
                model_config_ordered = deepcopy(model_config)
                for m, model in enumerate(df_plot_order.model):
                    config = model_config_ordered['public_models'][model]
                    config['xpos'] = m
                    config['color'] = (0.75,0.75,0.75)
                data_config = {}
                if 'humans' in df.model.unique():
                    data_config['humans'] = HUMAN_CONFIG
                data_config.update(model_config_ordered)
                bar_plot_mirror(
                    data_config=data_config,
                    df=df_plot,
                    plot_config=plot_configs[metric],
                    plot_file=plot_file)


    def _human_likeness_plots(df, robustness, model_config, hum_configs,
                              rob_config, noise_ceiling, results_dir,
                              overwrite):

        dataset, level, metric_model, metric_sim, within, between = df.name

        nc_df = noise_ceiling[
            (noise_ceiling.dataset == dataset) &
            (noise_ceiling.level == level) &
            (noise_ceiling.metric_sim == metric_sim) &
            (noise_ceiling.within == within)]

        if len(nc_df):
            assert len(nc_df) == 30, 'more than one noise ceiling found'
            nc = nc_df[['lwr', 'upr']].mean().values
        else:
            assert level == 'trial-wise', (
                'There should be a noise-ceiling for this metric')
            nc = None

        # human likeness
        out_dir = (f'{results_dir}/human_likeness_dataset-{dataset}'
                   f'/{metric_model}/{level}_{metric_sim}_within_{within}_'
                   f'between_{between}')
        os.makedirs(out_dir, exist_ok=True)

        # all cycles
        if df.cycle.max() > 0:

            # bar plot
            plot_file = f'{out_dir}/all_cycles.svg'
            if not op.isfile(plot_file) or overwrite:
                bar_plot(
                    data_config=model_config,
                    df=df,
                    plot_config=hum_configs[level][metric_sim],
                    plot_file=plot_file,
                    ceiling=nc,
                    multicycle=True)

            # line plot
            plot_file = f'{out_dir}/all_cycles_lineplot.svg'
            if not op.isfile(plot_file) or overwrite:
                line_plot(
                    data_config=model_config,
                    df=df,
                    plot_config=hum_configs[level][metric_sim],
                    samples={'errorbar': True},
                    plot_file=plot_file,
                    ceiling=nc)

        # final cycle
        plot_file = f'{out_dir}/last_cycle.svg'
        anova_path = f'{out_dir}/anova.csv'
        ph_path = anova_path.replace('anova', 'posthocs')
        all_paths = [plot_file, anova_path, ph_path]
        if not all([op.isfile(i) for i in all_paths]) or overwrite:
            df_plot = (df
                .groupby(['subject', 'model', 'layer'])
                .apply(lambda d: d[d.cycle == d.cycle.max()])
                .reset_index(drop=True))
            plot_config = hum_configs[level][metric_sim]
            samples = {'columns': ['subject'],
                       'color': HUMAN_COLOR,
                       'markeredgecolor': 'None',
                       'linewidth': 0,
                       'size': 3,
                       'alpha': .5,
                       'errorbar': True,
                       }
            bar_plot(
                data_config=model_config,
                df=df_plot,
                samples=samples,
                plot_config=plot_config,
                plot_file=plot_file,
                ceiling=nc)

            inferential_stats(df_plot, out_dir, nc)

        # pooled across model groups for each condition
        plot_file = f'{out_dir}/last_cycle_pooled.svg'
        if not op.isfile(plot_file) or overwrite:
            df_plot = (df
                .groupby(['subject', 'group', 'model'])
                .apply(lambda d: d[d.cycle == d.cycle.max()])
                .reset_index(drop=True)
                .groupby(['subject', 'group'])
                .agg({'value': 'mean'})
                .reset_index())
            #xlabel_padding = max([len(i.split('\n')) for i in
            #                      model_config]) - 1
            #xlabel = 'pooled models' + '\n' * xlabel_padding
            #df_plot['model'] = [xlabel if i != 'humans' else 'humans' for i
            #                 in df_plot.group]
            df_plot = df_plot.rename(columns={'group': 'model'})
            df_plot['group'] = 'pooled models'
            df_plot['cycle'] = -1
            df_plot = (df_plot
                .groupby([i for i in df_plot.columns if i != 'value'])
                .agg({'value': 'mean'})
                .reset_index())
            config_pooled = {
                k: list(v.values())[0] for k, v in model_config.items()}
            data_config = {'pooled models': config_pooled}
            plot_config = hum_configs[level][metric_sim]
            samples = {'columns': ['subject'],
                       'color': HUMAN_COLOR,
                       'markeredgecolor': 'None',
                       'linewidth': 0,
                       'size': 3,
                       'alpha': .5,
                       'errorbar': True,
                       }
            bar_plot(
                data_config=data_config,
                df=df_plot,
                samples=samples,
                plot_config=plot_config,
                plot_file=plot_file,
                ceiling=nc,
                xticklabel=False)

            inferential_stats(df_plot, out_dir, nc, metric_sim)


        # human likeness v occlusion robustness
        out_dir = (f'{results_dir}/human_likeness_v_occlusion_robustness_'
                   f'dataset-{dataset}/{metric_model}/{level}_{metric_sim}_'
                   f'within_{within}_between_{between}')
        os.makedirs(out_dir, exist_ok=True)

        # remove unoccluded and select model metric from robustness
        robustness = remove_unoccluded(robustness)
        if dataset != 'all':
            robustness = robustness[
                robustness.occluder_class != 'naturalUntexturedCropped2']
        robustness = robustness[
            (robustness.model == 'humans') |
            (robustness.metric == metric_model)]

        nc = nc[0] if nc is not None else None

        # all cycles
        plot_file = f'{out_dir}/all_cycles.svg'
        if df.cycle.max() > 0 and not op.isfile(plot_file) or overwrite:
            scatterplot(
                data_config=model_config,
                human_likeness=df,
                robustness=robustness,
                hum_config=hum_configs[level][metric_sim],
                rob_config=rob_config,
                plot_file=plot_file,
                noise_ceiling=nc,
                multicycle=True)

        # final cycle
        plot_file = f'{out_dir}/last_cycle.svg'
        if not op.isfile(plot_file) or overwrite:
            df_plot = (
                df
                .groupby(['subject', 'model', 'layer'])
                .apply(lambda d: d[d.cycle == d.cycle.max()])
                .reset_index(drop=True))
            scatterplot(
                data_config=model_config,
                human_likeness=df_plot,
                robustness=robustness,
                hum_config=hum_configs[level][metric_sim],
                rob_config=rob_config,
                plot_file=plot_file,
                noise_ceiling=nc)

    for model_contrast, model_config in model_contrasts.items():

        results_dir = (f'../p022_occlusion/data/in_silico/analysis/'
                       f'{model_contrast}/behavior/exp1')
        robustness, curves, likeness, noise_ceiling = collate_data(
            model_config)

        print(f'{now()} | Comparing models (exp1) | {model_contrast}')

        # robustness plots
        rob_configs = {
            'accuracy': {
                'title': 'classification accuracy',
                'ylabel': 'classification accuracy',
                'yticks': np.arange(0, 2, .2),
                'ylims': (0, 1),
                'chance': 1 / 8},
            'true_class_prob': {
                'title': 'probability estimate for true class',
                'ylabel': 'probability',
                'yticks': np.arange(0, 2, .2),
                'ylims': (0, 1),
                'chance': 1 / 8},
            'entropy': {
                'title': 'uncertainty across 8 classes',
                'ylabel': 'shannon entropy',
                'yticks': np.arange(0, 4, 1),
                'ylims': (0, 3)},
            'reconstruction_loss': {
                'title': 'reconstruction error',
                'ylabel': 'loss',
                'yticks': np.arange(0, 4, 1),
                'ylims': (0, 3)}
        }
        (robustness
            .groupby('metric')
            .apply(_robustness_plots, curves, model_config, rob_configs,
                   results_dir, overwrite))

        # human likeness plots
        hum_configs = {
            'condition-wise': {
                'cond_pearson_r': {
                    'title': 'accuracy similarity to humans\n(condition-wise)',
                    'ylabel': r"Pearson's $\it{r}$",  #r"correlation ($\it{r}$)"
                    'yticks': np.arange(-2, 2, .5),
                    'ylims': (-1, 1),
                    'chance': 0},
                'curve_pearson_r': {
                    'title': 'curve similarity to humans\n(condition-wise)',
                    'ylabel': r"Pearson's $\it{r}$",  #r"correlation ($\it{r}$)"
                    'yticks': np.arange(-2, 2, .2),
                    'ylims': (.5, 1),
                    'chance': 0},
                'curve_norm_pearson_r': {
                    'title': 'normed curve similarity to humans\n(condition-wise)',
                    'ylabel': r"Pearson's $\it{r}$",
                    # r"correlation ($\it{r}$)"
                    'yticks': np.arange(-2, 2, .5),
                    'ylims': (-.5, .5),
                    'chance': 0},
                'accuracy_distance': {
                    'title': 'accuracy distance to humans\n(condition-wise)',
                    'ylabel': 'normalized distance',
                    'yticks': np.arange(-2, 2, .5),
                    'ylims': (1, 0),
                    'invert': True}},
            'trial-wise': {
                'c_inacc': {
                    'title': 'inaccurate consistency with humans\n(trial-wise)',
                    'ylabel': 'inaccurate consistency',
                    'yticks': np.arange(0, 1, .1),
                    'ylims': (0, .5),
                    'chance': 1 / 7},
                'c_obs': {
                    'title': 'observed consistency with humans\n(trial-wise)',
                    'ylabel': 'observed consistency',
                    'yticks': np.arange(0, 2, .5),
                    'ylims': (0, 1)},
                'c_err': {
                    'title': 'error consistency with humans\n(trial-wise)',
                    'ylabel': "Cohen's $\\kappa$",
                    'yticks': np.arange(0, 1, .2),
                    'ylims': (-.1, .6)},
                'c_err_weighted': {
                    'title': 'weighted error consistency with humans\n'
                             '(trial-wise)',
                    'ylabel': "Cohen's $\\kappa$",
                    'yticks': np.arange(0, 1, .2),
                    'ylims': (-.1, .6)}}}
        (likeness
            .groupby(['dataset', 'level', 'metric_model', 'metric_sim',
                   'within', 'between'])
            .apply(_human_likeness_plots, robustness, model_config, hum_configs,
                   rob_configs['accuracy'], noise_ceiling, results_dir,
                   overwrite))

    # save occluder legend separately
    outpath = f'{op.dirname(results_dir)}/occluder_types_legend.svg'
    if not op.isfile(outpath) or overwrite:
        make_legend(
            outpath=outpath,
            labels=[f'{o} {t}' for o, t in itp(
                EXP1.occluder_labels, OCC_COLORS)],
            markers='o',
            colors=EXP1.plot_colors,
            markeredgecolors=None,
            linestyles='None')

    # save occluder legend with unoccluded bar separately
    outpath = f'{op.dirname(results_dir)}/occluder_types_legend_w-unocc.svg'
    if not op.isfile(outpath) or overwrite:
        make_legend(
            outpath=outpath,
            labels=['unoccluded'] + [f'{o} {t}' for o, t in itp(
                EXP1.occluder_labels, OCC_COLORS)],
            markers='o',
            colors=[(1.,1.,1.)].extend(EXP1.plot_colors),
            markeredgecolors=['k'] + [None] * 18,
            linestyles=['solid'] + ['None'] * 18)


def inferential_stats(df, out_dir, nc=None, metric=None):

    df = df[df.model != 'humans']
    if metric == 'pearson_r':
        df['value'] = np.arctanh(df['value'])
    #factors = [i for i in df.columns if i.startswith('factor_')]
    #if all(df.value.isnull()) or len(factors) == 0:
    #    return

    # get all factor combinations
    #factor_sets = [['factor_all']] + [[i] for i in factors]
    #if len(factors) > 1:
    #    for num_factors in range(2, len(factors) + 1):
    #        factor_sets.extend(list(itc(factors, num_factors)))
    #df['factor_all'] = 'all'

    # try to perform a anova across entire dataset. If not balanced, try one
    # anova per group
    try:
        anova = AnovaRM(df, depvar='value', subject='subject',
            within=['model', 'group'], aggregate_func='mean').fit().anova_table
        post_hocs = pg.pairwise_tests(dv='value', within=['model', 'group'],
            subject='subject', data=df, padjust='bonf')
        # pingouin version (> 2 repeated measures not supported)
        #anova = pd.concat([anova, df.groupby(groupbys).apply(
        #    lambda d: pg.rm_anova(dv='value', within=withins,
        #        subject='subject', data=d, detailed=True))])

    except:
        try:
            anova = df.groupby('group').apply(
                lambda d: AnovaRM(d, depvar='value',
                    subject='subject', within=['model'],
                    aggregate_func='mean').fit().anova_table
            ).reset_index()
            post_hocs = df.groupby('group').apply(
                lambda d: pg.pairwise_tests(dv='value', within=['model'],
                    subject='subject', data=d, padjust='bonf'))
        except:
            return

    # summary statistics
    summary = (df
               .drop(columns='subject')[['model', 'value']]
               .groupby('model').agg(['mean', 'sem'], numeric_only=True)
               .reset_index())
    summary.columns = ['model', 'mean', 'sem']
    if nc is not None:
        summary = pd.concat([summary, pd.DataFrame({
            'model': ['nc_lwr', 'nc_upr'],
            'mean': nc,
            'sem': [np.nan, np.nan]})])

    # save results
    pooled_str = '_pooled' if 'pooled models' in df.group.unique() else ''
    anova.to_csv(f'{out_dir}/anova{pooled_str}.csv')
    post_hocs.to_csv(f'{out_dir}/posthocs{pooled_str}.csv', index=False)
    summary.to_csv(f'{out_dir}/summary{pooled_str}.csv')


def collate_data(model_contrast):

    # human data
    human_dir = f'../p022_occlusion/data/in_vivo/behavioral/exp1'
    groupbys = ['subject', 'occluder_class', 'occluder_color', 'visibility']
    robustness = load_trials(drop_human=False)
    robustness = reshape_metrics(robustness, 'long')
    robustness = (robustness
        .groupby(groupbys, dropna=False, observed=False)
        .agg({'value': 'mean'}).reset_index())
    robustness['model'] = 'humans'
    robustness['group'] = 'humans'
    robustness['layer'] = 'humans'
    robustness['cycle'] = -1
    robustness['metric'] = 'accuracy'
    curves = pd.read_parquet(f'{human_dir}/analysis/robustness_curves.parquet')
    curves = curves[curves.metric == 'accuracy']
    curves['group'] = 'humans'
    curves['model'] = 'humans'
    curves['layer'] = 'humans'
    curves['cycle'] = -1
    noise_ceiling = pd.read_csv(f'{human_dir}/analysis/noise_ceiling.csv')

    # model data
    likeness = pd.DataFrame()
    for group, models in model_contrast.items():
        for m, (label, info) in enumerate(models.items()):

            data_dir = op.join(MODEL_BASE, info['path'], RES_DIR)
            layer = info['readout_layer']

            # robustness
            df = pd.read_parquet(f'{data_dir}/trials.parquet')
            df = reshape_metrics(df, 'long')
            df = (df[df.layer == layer]
                .groupby(groupbys + ['cycle', 'metric'],
                         observed=False, dropna=False)
                .agg({'value': 'mean'}))
            df = df[~df['value'].isnull()].reset_index()
            assert len(df), 'no data for this model'
            df['model'] = label
            df['layer'] = layer
            df['group'] = group
            robustness = pd.concat([robustness, df])

            # visibility curve functions
            df = pd.read_parquet(f'{data_dir}/robustness_curves.parquet')
            df = df[df.layer == layer]
            df['model'] = label
            df['layer'] = layer
            df['group'] = group
            assert len(df), 'no data for this model'
            curves = pd.concat([curves, df])

            # human likeness
            df = pd.read_parquet(f'{data_dir}/human_likeness.parquet')
            df = df[~df.cycle.isnull()]
            df.to_parquet(f'{data_dir}/human_likeness.parquet')
            df = df[df.layer == layer]
            df['model'] = label
            df['layer'] = layer
            df['group'] = group
            assert len(df), 'no data for this model'
            #assert len(df) == 1140 * len(df.cycle.unique())
            likeness = pd.concat([likeness, df])

    return robustness, curves, likeness, noise_ceiling


def split_bar(num_cycles, xbase):
    bar_gap = .2
    width = (1 - bar_gap) / num_cycles
    left_bar = xbase - .5 + bar_gap / 2 + width / 2
    right_bar = xbase + .5 - bar_gap / 2 - width / 2
    xposs = np.linspace(left_bar, right_bar, num_cycles)
    return xposs, width


def get_group_counts(data_config):
    group_counts = []
    for group, models in data_config.items():
        group_counts.append(int(np.round(max(
            len(models), max([models[key]['xpos'] for key in models]) + 1 or 0
        ))))
    return group_counts


def add_unoccluded(df):
    df = df.copy()
    df.occluder_class = pd.Categorical(df.occluder_class,
        categories=EXP1.occluder_classes + ['unoccluded'], ordered=True)
    df.occluder_class.fillna('unoccluded')
    df.occluder_color = pd.Categorical(df.occluder_color,
        categories=OCC_COLORS + ['unoccluded'], ordered=True)
    df.occluder_color.fillna('unoccluded')
    return df


def remove_unoccluded(df):
    df = df.copy()
    df = df[(df.occluder_class != 'unoccluded') & (~df.occluder_color.isna())]
    return df




if __name__ == "__main__":

    from seconds_to_text import seconds_to_text

    start = time.time()
    compare_models(overwrite=False)
    finish = time.time()
    print(f'analysis took {seconds_to_text(finish - start)} to complete')
