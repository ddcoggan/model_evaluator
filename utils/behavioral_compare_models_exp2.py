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
from p022_occlusion.in_vivo.behavioral.exp2.analysis import CFG as EXP2

METRICS = ['accuracy', 'true_class_prob', 'entropy']
OBJ_VARIABLES = ['object_animacy', 'object_class']
OBJ_ANIMACIES = ['animate', 'inanimate']
OBJ_CLASSES = EXP1.object_classes
OBJ_CLS_IDCS = EXP1.class_idxs
OBJ_CLS_DIRS = EXP1.synsets
OCC_VARIABLES = ['visibility', 'occluder_class', 'occluder_color']
OCC_COLORS = ['black', 'white']
OCC_CLASSES = EXP2.occluder_classes
UNOCC_COLOR = (.8, .8, .8)
NUM_FRAMES = 17
VISIBILITIES = np.linspace(0, 1, NUM_FRAMES)
FRAMES = np.linspace(1, 360, NUM_FRAMES).astype(int)
RES_DIR = 'benchmarking/occlusion_behavioral/exp2'


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
        for m, (model, info) in enumerate(models.items()):
            xpos = info['xpos'] if 'xpos' in info else m
            xticks.append(xpos)
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
        #xlims = (-.5, group_counts[g] - .5)
        #ax.set_xlim(xlims)

        # y-axis formatting
        if ceiling is not None:
            ax.fill_between(ax.get_xlim(), ceiling[0], ceiling[1],
                            color='tab:gray', lw=0, zorder=1)
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
                metric = 'visibility'
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


def curve_plot(data_config, df, curves, plot_config, plot_file):

    fig, ax = plt.subplots(figsize=(4, 4))
    curve_x = np.linspace(0, 1, 1000)
    xvals = VISIBILITIES[exp] + [1]

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

    def _robustness_plots(df, model_config, level, plot_configs, results_dir,
                              overwrite):

        metric = df.name
        out_dir = f'{results_dir}/occlusion_robustness/{metric}/{level}'
        os.makedirs(out_dir, exist_ok=True)

        # plot all cycles (barplot)
        plot_file = f'{out_dir}/all_cycles.svg'
        if trials_m.cycle.max() > 0 and \
                (not op.isfile(plot_file) or overwrite):

            groupbys = ['group', 'model', 'layer', 'cycle']
            data_human = (trials_h[trials_h.accuracy == 1]
                          [groupbys + ['visibility']]
                          .groupby(groupbys)
                          .visibility.mean()
                          .reset_index())
            data_human.rename(columns={'visibility': 'value'}, inplace=True)

            if level == 'condition-wise':
                data_model = (df[groupbys + ['visibility', 'value']]
                    .groupby(groupbys)
                    .apply(lambda d: d[d['value'] > .5].visibility.min())
                    .reset_index()
                    .rename(columns={0: 'value'}))

            else:  # elif level == 'trial-wise':
                data_model = (df[(df.metric == metric) & (df.accuracy == 1)]
                    .groupby(groupbys)
                    .agg({'visibility': 'mean'})
                    .reset_index()
                    .rename(columns={'visibility': 'value'}))
            df_plot = pd.concat([data_human, data_model])

            data_config = {}
            if 'humans' in df.model.unique():
                data_config['humans'] = HUMAN_CONFIG
            data_config.update(model_config)
            bar_plot(
                data_config=data_config,
                df=df_plot,
                plot_config=plot_configs[metric],
                plot_file=plot_file,
                multicycle=True)

        # final cycle
        plot_file = f'{out_dir}/last_cycle.svg'
        anova_path = f'{out_dir}/anova.csv'
        ph_path = anova_path.replace('anova', 'posthocs')
        if not all([op.isfile(i) for i in [plot_file, anova_path, ph_path]]
                   ) or overwrite:
            groupbys = ['group', 'model', 'layer', 'occluder_class',
                        'occluder_color']
            data_human = (trials_h[trials_h.accuracy == 1]
                          .rename(columns={'visibility': 'value'}))

            if level == 'condition-wise':
                data_model = (df
                    [groupbys + ['cycle', 'visibility', 'value']]
                    .groupby(groupbys, observed=False, dropna=False)
                    .apply(lambda d: d[d.cycle == d.cycle.max()])
                    .reset_index(drop=True)
                    .groupby(groupbys, observed=False, dropna=False)
                    .apply(lambda d: d[d['value'] > .5].visibility.min())
                    .reset_index()
                    .rename(columns={0: 'value'})
                    .copy())
                data_model.value = data_model.value.fillna(1)

            else:  # elif level == 'trial-wise':
                data_model = (df
                    [(df.metric == metric) & (df.accuracy == 1)]
                    .groupby(groupbys, observed=False, dropna=False)
                    .apply(lambda d: d[d.cycle == d.cycle.max()])
                    .reset_index(drop=True)
                    .groupby(groupbys, observed=False, dropna=False)
                    .agg({'visibility': 'mean'})
                    .reset_index()
                    .dropna()
                    .rename(columns={'visibility': 'value'})
                    .copy())
            df_plot = pd.concat([data_human, data_model])

            data_config = model_config.copy()
            if 'humans' in df_plot.model.unique():
                data_config['humans'] = HUMAN_CONFIG
            samples = {'columns': ['occluder_class', 'occluder_color'],
                       'color': EXP2.plot_colors, 'size': 5, 'alpha': 1,
                       'linewidth': 0, 'markeredgecolor': 'None'}

            bar_plot(
                data_config=data_config,
                df=df_plot,
                samples=samples,
                plot_config=plot_configs[metric],
                plot_file=plot_file)

            # no inferential stats on robustness, only human-likeness

            """
            # performance curves
            if analysis == 'condition-wise':
                outpath = f'{out_dir}/{metric}_curves.svg'
                if not op.isfile(outpath) or overwrite:

                    fig, ax = plt.subplots(figsize=(4, 4))
                    curve_x = np.linspace(0, 1, 1000)
                    xvals = VISIBILITIES

                    # models
                    curves_m_last_cycle = (curves_m.groupby('model')
                    .apply(
                        lambda d: d[d.cycle == d.cycle.max()]))
                    for m, (label, info) in enumerate(config.items()):
                        color = info['color']

                        # plot model curve function
                        popt = curves_m_last_cycle[
                            (curves_m_last_cycle.metric == metric) &
                            (curves_m_last_cycle.model == label) &
                            (curves_m_last_cycle.occluder_class == 'all') &
                            (curves_m_last_cycle.occluder_color == 'all') &
                            (curves_m_last_cycle.stimulus_set == 'all')][
                            ['L', 'x0', 'k', 'b']].values
                        assert len(
                            popt) == 1, 'wrong number of rows selected'
                        curve_y = sigmoid(curve_x, *popt[0])
                        ax.plot(curve_x, curve_y, color=info['color'],
                                clip_on=False)

                        # plot model performance
                        yvals = (trials_m_last_cycle[
                                     (trials_m_last_cycle.model == label) &
                                     (trials_m_last_cycle.metric == metric)]
                                 .groupby('visibility')
                                 .agg('mean', numeric_only=True)
                                 .sort_values(by='visibility')
                                 .value.to_list())
                        ax.scatter(xvals, yvals, color=info['color'], s=2)

                    # format plot
                    # ax.set_title(EXP1.occluder_labels[o], size=7)
                    ax.set_xticks((0, 1))
                    ax.set_xlim((-.02, 1.02))
                    ax.set_yticks(yticks)
                    ax.set_ylim(ylims)
                    ax.tick_params(axis='both', which='major', labelsize=7)
                    ax.axhline(y=chance, color='k', linestyle='dotted')
                    ax.set_xlabel('visibility')
                    ax.set_ylabel(metric)
                    plt.tight_layout()
                    fig.savefig(outpath)
                    plt.close()
                    """

    def _human_likeness_plots(df, model_config, plot_configs, noise_ceiling,
                             results_dir, overwrite):

        level, metric = df.name
        out_dir = f'{results_dir}/human_likeness/{metric}/{level}'
        os.makedirs(out_dir, exist_ok=True)

        nc = noise_ceiling[noise_ceiling.level == level][
            ['lwr', 'upr']].values[0]

        # plot all cycles (barplot)
        plot_file = f'{out_dir}/all_cycles.svg'
        if df.cycle.max() > 0 and metric in plot_configs[level] and (
                not op.isfile(plot_file) or overwrite):
            bar_plot(
                data_config=model_config,
                df=df,
                plot_config=plot_configs[level][metric],
                plot_file=plot_file,
                ceiling=nc,
                multicycle=True)

        # final cycle
        plot_file = f'{out_dir}/last_cycle.svg'
        anova_path = f'{out_dir}/anova.csv'
        ph_path = anova_path.replace('anova', 'posthocs')
        all_paths = [plot_file, anova_path, ph_path]
        if not all([op.isfile(i) for i in all_paths]) or overwrite:
            df_plot = (df
               .groupby(['subject', 'group', 'model', 'layer'], observed=False)
               .apply(lambda d: d[d.cycle == d.cycle.max()])
               .reset_index(drop=True))
            plot_config = plot_configs[level][metric]
            samples = {'columns': ['subject'], 'color': HUMAN_COLOR, 'size': 3,
                       'alpha': .5, 'linewidth': 0, 'markeredgecolor': 'None'}
            bar_plot(
                data_config=model_config,
                df=df_plot,
                samples=samples,
                plot_config=plot_config,
                plot_file=plot_file,
                ceiling=nc)

            factors = [i for i in df_plot.columns if i.startswith(
                'factor_')]
            if factors:
                df_stats = (df_plot
                            [df_plot.visibility < 1]
                            .groupby(['model', 'subject'] + factors)
                            .mean('value')
                            .reset_index())
                inferential_stats(df_stats, out_dir, nc)

    for model_contrast, model_config in model_contrasts.items():

        results_dir = f'../p022_occlusion/data/in_silico/analysis/{model_contrast}/behavior/exp2'
        trials_h, trials_m, trials_m_rt, likeness, noise_ceiling = (
            collate_data(model_config))

        print(f'{now()} | Comparing models (exp2) | {model_contrast}')

        # occlusion robustness measures
        plot_configs = {
            'condition-wise': {
                'accuracy': {
                    'title': 'visibility threshold for\n'
                             '50% classification accuracy',
                    'ylabel': 'visibility',
                    'yticks': np.arange(0, 2, .2),
                    'ylims': (1, 0),
                    'invert': True},
                'entropy': {
                    'title': 'visibility threshold for\nentropy < .5',
                    'ylabel': 'visibility',
                    'yticks': np.arange(0, 2, .2),
                    'ylims': (1, 0),
                    'invert': True},
                'true_class_prob': {
                    'title': 'visibility threshold for\n'
                             '50% confidence in true class',
                    'ylabel': 'visibility',
                    'yticks': np.arange(0, 2, .2),
                    'ylims': (1, 0),
                    'invert': True},
                'reconstruction_loss': {
                    'title': 'visibility threshold for\nrecon loss < .5',
                    'ylabel': 'visibility',
                    'yticks': np.arange(0, 2, .2),
                    'ylims': (1, 0),
                    'invert': True}},
            'trial-wise': {
                'accuracy': {
                    'title': 'visibility at first accurate response',
                    'ylabel': 'visibility',
                    'yticks': np.arange(0, 2, .2),
                    'ylims': (1, 0),
                    'invert': True},
                'entropy': {
                    'title': 'visibility at first entropy < .5',
                    'ylabel': 'visibility',
                    'yticks': np.arange(0, 2, .2),
                    'ylims': (1, 0),
                    'invert': True},
                'true_class_prob': {
                    'title': 'visibility at first confidence > 50%',
                    'ylabel': 'visibility',
                    'yticks': np.arange(0, 2, .2),
                    'ylims': (1, 0),
                    'invert': True},
                'reconstruction_loss': {
                    'title': 'visibility threshold for\nrecon loss < .5',
                    'ylabel': 'visibility',
                    'yticks': np.arange(0, 2, .2),
                    'ylims': (1, 0),
                    'invert': True}}
        }

        trials_m.groupby(['metric']).apply(
            _robustness_plots, model_config, 'condition-wise',
            plot_configs['condition-wise'], results_dir, overwrite)
        trials_m_rt.groupby(['metric']).apply(
            _robustness_plots, model_config, 'trial-wise',
            plot_configs['trial-wise'], results_dir, overwrite)


        # human likeness measures
        plot_configs = {
            'condition-wise': {
                'accuracy': {
                    'title': 'model accuracy v. human accuracy\n'
                             '(condition-wise)',
                    'ylabel': r"Pearson's $\it{r}$",#"correlation ($\it{r}$)",
                    'yticks': np.arange(-1, 2, .5),
                    'ylims': (-.5, 1)},
                'entropy': {
                   'title': 'model entropy v. human accuracy\n(condition-wise)',
                    'ylabel': r"Pearson's $\it{r}$",
                    'yticks': np.arange(-1, 2, .2),
                   'ylims': (-1, 1)},
                'true_class_prob': {
                   'title': 'model true class prob v. human accuracy\n'
                            '(condition-wise)',
                    'ylabel': r"Pearson's $\it{r}$",
                    'yticks': np.arange(-1, 2, .2),
                   'ylims': (0, 1)},
                'reconstruction_loss': {
                    'title': 'model reconstruction error v. human accuracy\n'
                             '(condition-wise)',
                    'ylabel': r"Pearson's $\it{r}$",#"correlation ($\it{r}$)",
                    'yticks': np.arange(-1, 2, .2),
                    'ylims': (-1, 1)}},
            'trial-wise': {
                'accuracy': {
                    'title': 'RT similarity to humans\n'
                             '(trial-wise, accuracy-based)',
                    'ylabel': r"Pearson's $\it{r}$",#"correlation ($\it{r}$)",
                    'yticks': np.arange(-1, 2, .5),
                    'ylims': (-.5, 1)},
                'entropy': {
                   'title': 'RT similarity to humans\n'
                            '(trial-wise, entropy-based)',
                    'ylabel': r"Pearson's $\it{r}$",
                    'yticks': np.arange(-1, 2, .2),
                   'ylims': (0, 1)},
                'true_class_prob': {
                   'title': 'RT similarity to humans\n'
                            '(trial-wise, true class probability-based)',
                    'ylabel': r"Pearson's $\it{r}$",
                    'yticks': np.arange(-1, 2, .2),
                   'ylims': (0, 1)},
                'reconstruction_loss': {
                    'title': 'RT similarity to humans\n'
                             '(trial-wise, reconstruction-loss-based)',
                    'ylabel': r"Pearson's $\it{r}$",
                    'yticks': np.arange(-1, 2, .2),
                    'ylims': (0, 1)},
            }}

        (likeness
            .groupby(['level', 'metric'], observed=False, dropna=False)
            .apply(_human_likeness_plots, model_config, plot_configs,
                   noise_ceiling, results_dir, overwrite))

        # panel legend with one label per dataset
        if model_contrast == 'CCN_2024_poster':
            legend_file = f'{results_dir}/legend_panel.svg'
            if 'group' in likeness.columns and (
                    not op.isfile(legend_file) or overwrite):
                leg_colors = []
                leg_labels = []
                for k, v in model_config.items():
                    if v['color'] not in leg_colors:
                        leg_colors.append(v['color'])
                        try:
                            leg_labels.append(k.split(', ')[-2])
                        except:
                            leg_labels.append(k)
                make_legend(
                    outpath=legend_file,
                    labels=leg_labels,
                    markers='s',
                    colors=leg_colors,
                    markeredgecolors=None,
                    linestyles='None')

    # save occluder legend separately
    outpath = f'{op.dirname(results_dir)}/occluder_types_legend.svg'
    if not op.isfile(outpath) or overwrite:
        make_legend(
            outpath=outpath,
            labels=[f'{o} {t}' for o, t in itp(EXP2.occluder_labels,
                                               OCC_COLORS)],
            markers='o',
            colors=EXP2.plot_colors,
            markeredgecolors=None,
            linestyles='None')


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
    groupbys = ['occluder_class', 'occluder_color']
    trials_h = load_trials(drop_human=False)
    trials_h = (trials_h
        .groupby(groupbys + ['subject'], observed=False)
        .agg('mean', numeric_only=True)
        .reset_index())
    trials_h['model'] = 'humans'
    trials_h['group'] = 'humans'
    trials_h['layer'] = 'humans'
    trials_h['cycle'] = 0
    noise_ceiling = pd.read_csv('../p022_occlusion/data/in_vivo/behavioral/exp2/analysis/'
                                'performance/noise_ceiling.csv')

    # model data
    trials_m = pd.DataFrame()
    trials_m_rt = pd.DataFrame()
    curves_m = pd.DataFrame()
    likeness = pd.DataFrame()
    for group, models in model_contrast.items():
        for m, (label, info) in enumerate(models.items()):

            data_dir = op.join(MODEL_BASE, info['path'], RES_DIR)
            layer = info['readout_layer']

            # trials
            df = pd.read_parquet(f'{data_dir}/trials.parquet')
            df = df[df.layer == layer]
            df = (df
                  .groupby(groupbys + ['visibility', 'cycle', 'metric'],
                           observed=False)
                  .agg('mean', numeric_only=True)
                  .reset_index())
            df['model'] = label
            df['layer'] = layer
            df['group'] = group
            df = reshape_metrics(df, 'long')
            trials_m = pd.concat([trials_m, df])

            # estimated RTs
            df = pd.read_parquet(f'{data_dir}/trials_RT.parquet')
            df = df[df.layer == layer]
            df['model'] = label
            df['layer'] = layer
            df['group'] = group
            trials_m_rt = pd.concat([trials_m_rt, df])

            # visibility curve functions
            df = pd.read_parquet(f'{data_dir}/robustness_curves.parquet')
            df = df[df.layer == layer]
            df = (df
                  .groupby(groupbys + ['cycle'])
                  .agg('mean', numeric_only=True)
                  .reset_index())
            df['model'] = label
            df['layer'] = layer
            df['group'] = group
            curves_m = pd.concat([curves_m, df])

            # human-likeness
            df = pd.read_parquet(f'{data_dir}/human_likeness.parquet')
            df = df[df.layer == layer]
            df['model'] = label
            df['layer'] = layer
            df['group'] = group
            likeness = pd.concat([likeness, df])

    return trials_h, trials_m, trials_m_rt, likeness, noise_ceiling


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



if __name__ == "__main__":

    from seconds_to_text import seconds_to_text

    start = time.time()
    compare_models(overwrite=False)
    finish = time.time()
    print(f'analysis took {seconds_to_text(finish - start)} to complete')
