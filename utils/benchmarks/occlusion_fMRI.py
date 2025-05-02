import os
import os.path as op
import sys
import glob
import time
import pickle as pkl
import numpy as np
from types import SimpleNamespace
import shutil
import itertools
import torch
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colormaps
from scipy.stats import kendalltau, sem
from PIL import Image
from frrsa import frrsa
import torchvision.transforms.v2 as transforms
from torch import float32
from itertools import product as itp

from get_activations import get_activations
from image_processing import tile
from seconds_to_text import seconds_to_text
from plot_utils import export_legend, custom_defaults
plt.rcParams.update(custom_defaults)

from utils import insert_cycle, now
from utils.model_contrasts import region_to_layer
from utils.get_trained_model import (
    get_trained_model, get_model, load_params)
from in_vivo.fMRI.utils import CFG as fMRI
from in_vivo.fMRI.utils import RSA_dataset, RSA, clustered_barplot, line_plot

FMRI_DIR = '../data/in_vivo/fMRI'
H_REGIONS = ['V1','V2','hV4','ventral_stream_sub_ret']
REGIONS = ['V1', 'V2', 'V4', 'IT']

# image directory
image_dir = '../data/in_vivo/fMRI/exp1/stimuli/images'
images = []
for obj, occ in itp(fMRI.exemplars, fMRI.occluder_labels):
    image = f'{image_dir}/{obj}_{occ}.png'
    assert op.isfile(image)
    images.append(image)


def get_transform(architecture):

    imsize = 256 if architecture == 'pix2pix' else 224
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(float32, scale=True),
        transforms.Resize(imsize),
        transforms.CenterCrop(imsize),
        transforms.Grayscale(num_output_channels=3),
        transforms.Normalize(mean=[0.445, 0.445, 0.445],
                             std=[0.269, 0.269, 0.269]),
    ])
    return transform


def get_human_data(analysis_dir_fmri):

    """ loads RSMs dict, returns 3D array of RSMs """

    # load RSMs (switching utils module to allow pickle to load)
    from in_vivo.fMRI import utils
    sys.modules['utils'] = utils  # allows pickle to load module
    RSAs = pkl.load(open(f'{analysis_dir_fmri}/RSA.pkl', 'rb'))
    del utils

    # determine which experiment and subjects
    exp = 'exp1' if 'exp1' in analysis_dir_fmri else 'exp2'
    subjects = fMRI.subjects_final[exp]

    # get RSMs and noise ceiling for this layer
    RSMs, nc = {}, {}
    for region, h_region in zip(REGIONS, H_REGIONS):
        RSMs[region] = np.empty((len(subjects), fMRI.n_img, fMRI.n_img))
        for s, subject in enumerate(subjects):
            RSMs[region][s, :, :] = RSAs[h_region][subject].RSM
        nc[region] = RSAs[h_region]['group'].noise_ceiling

    return RSMs, nc


def get_model_responses(model_dir, m=0, total_models=0, overwrite=False):

    model_info = model_dir.split('models/')[-1].split('/')
    architecture, identifier = model_info[:2]
    transfer_dir = model_info[2] if len(model_info) == 3 else 'X'

    # output directory
    out_dir = f'{model_dir}/fMRI'
    if op.isdir(out_dir) and overwrite:
        shutil.rmtree(out_dir)
    os.makedirs(f'{out_dir}', exist_ok=True)

    # load any previous model responses
    responses_path = f'{out_dir}/responses.pkl'
    if not op.isfile(responses_path) or overwrite:

        print(f'{now()} Measuring responses '
              f'model({m + 1}/{total_models} '
              f'{architecture}/{identifier}/{transfer_dir})')

        layers = sorted(set(region_to_layer[architecture].values()))
        model = get_trained_model(model_dir, architecture, True, layers)
        transform = get_transform(architecture)
        responses = {}

        # get model responses
        activations = get_activations(
            model, architecture, images, batch_size=24,
            layers=layers, transform=transform, shuffle=False)
        activations = insert_cycle(activations)
        for layer_cyc in activations:
            responses[layer_cyc] = RSA_dataset(
                responses=activations[layer_cyc].reshape((fMRI.n_img, -1)))
            tsne_path = f'{out_dir}/TSNE/{layer_cyc}.svg'
            responses[layer_cyc].plot_TSNE(tsne_path)
        pkl.dump(responses, open(responses_path, 'wb'))

        return True

    return False


def get_prednet_responses(model_dir, overwrite=False):

    model_info = model_dir.split('models/')[-1].split('/')
    architecture, identifier = model_info[:2]
    transfer_dir = model_info[2] if len(model_info) == 3 else 'X'

    # output directory
    out_dir = f'{model_dir}/fMRI'
    if op.isdir(out_dir) and overwrite:
        shutil.rmtree(out_dir)
    os.makedirs(f'{out_dir}', exist_ok=True)

    # load any previous model responses
    responses_path = f'{out_dir}/responses.pkl'
    if not op.isfile(responses_path) or overwrite:

        print(f'{now()} Measuring responses {architecture}/{identifier}/{transfer_dir})')
        transform = get_transform(architecture)
        responses = {}
        for unit_type, layer in itertools.product(
                ['A', 'Ahat', 'R', 'E'], np.arange(4)):

            kwargs = {
                'stack_sizes': (3, 48, 96, 192),
                'R_stack_sizes': (3, 48, 96, 192),
                'A_filter_sizes': (3, 3, 3),
                'Ahat_filter_sizes': (3, 3, 3, 3),
                'R_filter_sizes': (3, 3, 3, 3),
                'output_mode': f'{unit_type}{layer}',
                'data_format': 'channels_first',
                'return_sequences': True}
            model = get_model(architecture, kwargs)
            params_path = sorted(glob.glob(f"{model_dir}/params/*.pt*"))[-1]
            model = load_params(params_path, model, 'model')

            # get model responses
            inputs = torch.stack([transform(Image.open(i)) for i in images])
            num_cycles = 8
            inputs_time = torch.stack([inputs] * num_cycles, dim=1)
            outputs = model(inputs_time)
            activations = insert_cycle(outputs)
            for layer_cyc in activations:
                responses[layer_cyc] = RSA_dataset(
                    responses=activations[layer_cyc].reshape((fMRI.n_img, -1)))
                tsne_path = f'{out_dir}/TSNE/{layer_cyc}.svg'
                responses[layer_cyc].plot_TSNE(tsne_path)
        pkl.dump(responses, open(responses_path, 'wb'))

        return True

    return False


def compare_layers_cycles(RSAs, architecture, analysis_dir):


    # get useful cycle, layer information
    layers = sorted(set(region_to_layer[architecture].values()))
    cycles = {layer: set() for layer in layers}
    max_cycles = 1
    for key, RSA in RSAs.items():
        if 'cyc' in key:
            layer, cycle = key.split('_')
        else:
            layer, cycle = key, 'cyc-1'
        cycles[layer].add(cycle)
        max_cycles = max(max_cycles, int(cycle[-2:]))
    cycle_colors = colormaps['viridis'].colors[::(256 // max_cycles - 1)]

    # condition-wise similarities
    df = pd.DataFrame()
    for key, RSA in RSAs.items():
        if 'cyc' in key:
            layer, cycle = key.split('_')
        else:
            layer, cycle = key, 'cyc-1'
        if cycle == sorted(cycles[layer])[-1]:
            temp = RSA.RSM_table
            temp['layer'] = layer
            temp['cycle'] = cycle
            df = pd.concat([df, temp])
    df['layer'] = df['layer'].astype('category').cat.reorder_categories(layers)
    df = df.drop(columns=['exemplar_a', 'exemplar_b', 'occluder_a','occluder_b']
        ).groupby(['analysis', 'level', 'layer']).agg(
        'mean', numeric_only=True).dropna().reset_index()
    df.columns = ['analysis', 'level', 'layer', 'value']
    df['error'] = np.nan
    for analysis, params in fMRI.occlusion_robustness_analyses.items():
        plot_df = df[df.analysis == analysis]
        ylabel = fMRI.similarities[op.basename(analysis_dir)]
        out_path = f'{analysis_dir}/{analysis}/cond-wise_sims.svg'
        clustered_barplot(
            plot_df, out_path, params=params, ylabel=ylabel, x_var='layer')
        plot_df.to_csv(f'{analysis_dir}/{analysis}/cond-wise_sims.csv')


    # occlusion robustness indices (object completion and occlusion invariance)
    df = pd.DataFrame()
    for key, RSA in RSAs.items():
        if 'cyc' in key:
            layer, cycle = key.split('_')
        else:
            layer, cycle = key, 'cyc-1'
        temp = RSA.occlusion_robustness
        temp['layer'] = layer
        temp['cycle'] = cycle
        df = pd.concat([df, temp])
    df['layer'] = df['layer'].astype('category').cat.reorder_categories(layers)
    for analysis, params in fMRI.occlusion_robustness_analyses.items():
        for subtype in df.subtype.unique():

            # layer*cycle plot
            outpath = (f'{analysis_dir}/{analysis}/{params["index_label"]}_'
                       f'{subtype}_cyclewise.svg')
            os.makedirs(op.dirname(outpath), exist_ok=True)

            fig, ax = plt.subplots(figsize=(2, 2))
            x_pos = np.arange(len(layers))

            for l, layer in enumerate(layers):
                num_cycles = len(cycles[layer])
                indices = []
                for cycle in cycles[layer]:
                    indices.append(df.value[
                                       (df.analysis == analysis) &
                                       (df.subtype == subtype) &
                                       (df.layer == layer) &
                                       (df.cycle == cycle)].item())

                # add lines between cycles
                x_pos_adj = [x_pos[l]]
                if num_cycles > 1:
                    x_pos_adj += np.linspace(-.35, .35, num_cycles)
                ax.plot(x_pos_adj, indices, color='k')

                # add points
                for c in range(num_cycles):
                    ax.plot(x_pos_adj[c],
                            indices[c],
                            color=cycle_colors[c],
                            marker='o',
                            markerfacecolor='white',
                            lw=0,
                            markersize=3)

            ceiling_x = np.arange(-.5, len(layers) + .5)
            ax.fill_between(ceiling_x, 1.0, 2, color='black', alpha=.2, lw=0)
            ax.set_yticks((0, .5, 1), labels=['0', '.5', '1'])
            ax.set_ylabel(params['index_label'])
            ax.set_ylim((-.05, 1.2))
            ax.set_xticks(ticks=x_pos, labels=layers)
            ax.set_xlim((-.5, len(layers) - .5))
            plt.tight_layout()
            plt.savefig(outpath, dpi=300)
            plt.close()

            # legend
            f = lambda m, c: \
                plt.plot([], [], marker=m, color='white', markeredgecolor=c,
                         linestyle='None')[0]
            handles = [f('o', color) for color in cycle_colors]
            labels = [str(c) for c in range(max_cycles)]
            legend = plt.legend(handles, labels, loc=3)
            export_legend(legend, filename=
                f'{analysis_dir}/{analysis}/cycles_legend.svg')
            plt.close()

            # plot for last cycle only
            plot_df = pd.DataFrame()
            for layer in layers:
                temp = df[
                    (df.analysis == analysis) &
                    (df.subtype == subtype) &
                    (df.layer == layer) &
                    (df.cycle == sorted(cycles[layer])[-1])
                    ].reset_index(drop=True).drop(columns=[
                    'analysis', 'subtype', 'cycle'])
                plot_df = pd.concat([plot_df, temp])
            outpath = (f'{analysis_dir}/{analysis}/'
                       f'{params["index_label"]}_{subtype}.svg')
            line_plot(plot_df, outpath, title=analysis.replace('_', ' '),
                            ylabel=params["index_label"], x_var='layer',
                            cond_var='model', col_var='color', ceiling=[1],
                            figsize=(3.5, 4))

        df[(df.analysis == analysis)].to_csv(
            f'{analysis_dir}/{analysis}/{params["index_label"]}.csv')


    # regression models
    df = pd.DataFrame()
    for key, RSA in RSAs.items():
        if 'cyc' in key:
            layer, cycle = key.split('_')
        else:
            layer, cycle = key, 'cyc-1'
        if cycle == sorted(cycles[layer])[-1]:
            temp = RSAs[key].model_fits
            temp['layer'] = layer
            df = pd.concat([df, temp])
    df['layer'] = df['layer'].astype('category').cat.reorder_categories(layers)
    df = df.rename(columns={'model': 'level', 'beta': 'value', 'mse': 'error'})
    clustered_barplot(df,
                      f'{analysis_dir}/regression/model_fits.svg',
                      fMRI.RSM_models,
                      fMRI.RSM_models['ylabel'],
                      x_var='layer')
    df.to_csv(f'{analysis_dir}/regression/regression.csv')


    # Compare representational geometry with human visual cortex
    responses = pkl.load(open(f'{op.dirname(op.dirname(analysis_dir))}/'
                              f'responses.pkl', 'rb'))
    model_df = pd.DataFrame()
    similarity = op.basename(analysis_dir)
    norm = analysis_dir.split('/')[-2].split('_')[0][5:]
    norm_method = analysis_dir.split('/')[-2].split('_')[-1]

    for (exp, task), RSA_type, cond_set, level in \
            itertools.product(
                fMRI.exps_tasks,
                ['cRSA'], ['identity', 'exemplar'],
                ['ind', 'group']):

        # matched_norm = matched_norms[norm] if exp == 'exp2' else norm
        analysis_dir_fmri = f'{FMRI_DIR}/{exp}/derivatives/RSA/{task}_' \
                            f'space-standard/norm-{norm}_{norm_method}/' \
                            f'{similarity}'

        RSMs_fMRI, noise_ceiling = get_human_data(analysis_dir_fmri)
        nc = {'upper': [noise_ceiling[reg]['upper'] for reg in REGIONS for _
                        in range(2)],
              'lower': [noise_ceiling[reg]['lower'] for reg in REGIONS for _
                        in range(2)]}

        df = pd.DataFrame()


        for region in REGIONS:

            layer = region_to_layer[architecture][region]
            keys = [key for key in RSAs if key.startswith(layer)]
            for key in keys:
                if 'cyc' in key:
                    layer, cycle = key.split('_')
                else:
                    layer, cycle = key, 'cyc-1'
                RSA = RSAs[key]
                model_RSM = RSA.RSM
                model_responses = responses[key].responses

                # get subjectwise RSMs at 3D array
                human_RSMs = RSMs_fMRI[region]

                # get group mean if necessary
                human_RSMs = human_RSMs if level == 'ind' else \
                    np.mean(human_RSMs, axis=0, keepdims=True)

                n_RSMs = human_RSMs.shape[0]

                if RSA_type == 'frRSA':
                    targ_RSMs = np.moveaxis(human_RSMs, 0, -1)
                    targ_sim = {'pearson': 'pearson_sim',
                                'spearman': 'spearman_sim'}[similarity]
                    scores, _, _, _ = frrsa(
                        targ_RSMs,
                        model_responses.T,
                        preprocess=True,
                        nonnegative=False,
                        measures=['dot', targ_sim],
                        cv=[5, 10],
                        hyperparams=np.linspace(.05, 1, 20),
                        score_type='pearson',
                        wanted=['predicted_matrix'],
                        parallel='10',
                        random_state=None,
                    )

                    # pred_RSMs = np.moveaxis(pred_RSMs, -1, 0)
                    rs = scores.score

                else:

                    rs = np.empty((n_RSMs))

                    # 'image' excludes diagonal, 'exemplar' excludes same exemplar
                    RSM_mask = np.array(1 - fMRI.RSM_models['matrices'][
                        cond_set].flatten(), dtype=bool)

                    pred_RSMs = np.stack([model_RSM] * n_RSMs, axis=0)
                    for s in range(n_RSMs):
                        targ_RSM_flat = human_RSMs[s].flatten()[RSM_mask]
                        pred_RSM_flat = pred_RSMs[s].flatten()[RSM_mask]
                        rs[s] = np.corrcoef(targ_RSM_flat, pred_RSM_flat)[1, 0]
                        # rs[s] = kendalltau(targ_RSM_flat, pred_RSM_flat).correlation

                # print(f'{RSA_type} score: {np.mean(rs):.3f}({sem(rs):.3f})')
                df = pd.concat([
                    df,
                    pd.DataFrame({
                        'region': [region],
                        'layer': [layer],
                        'cycle': [cycle],
                        'task': [task],
                        'RSA_type': [RSA_type],
                        'level': level,
                        'cond_set': [cond_set],
                        'mean': [np.mean(rs)],
                        'sem': [sem(rs) if level == 'ind' else np.nan],
                    }),
                ]).reset_index(drop=True)

        # add to analyses
        model_df = pd.concat([model_df, df]).reset_index(drop=True)

        # layer*cycle plot
        outpath = f'{analysis_dir}/human_likeness/{exp}_{task}_' \
                  f'{RSA_type}_{cond_set}_{level}_cyclewise.svg'
        os.makedirs(op.dirname(outpath), exist_ok=True)

        fig, ax = plt.subplots(figsize=(2, 2))

        x_pos = np.arange(len(REGIONS))

        for l, (region, layer) in enumerate(region_to_layer[architecture].items()):
            num_cycles = len(cycles[layer])

            means, sems = [], []
            for cycle in cycles[layer]:
                df_c = df[
                    (df.task == task) &
                    (df.RSA_type == RSA_type) &
                    (df.level == level) &
                    (df.region == region) &
                    (df.layer == layer) &
                    (df.cycle == cycle)]
                means.append(df_c['mean'].item())
                sems.append(df_c['sem'].item())

            # add lines between cycles
            x_pos_adj = [x_pos[l]]
            if num_cycles > 1:
                x_pos_adj += np.linspace(-.35, .35, num_cycles)
            ax.plot(x_pos_adj, means, color='k')

            # add points with error bar if available
            for c in range(num_cycles):
                if np.isfinite(sems[c]):
                    ax.errorbar(x_pos_adj[c],
                                means[c],
                                yerr=sems[c],
                                color=cycle_colors[c],
                                capsize=2)
                ax.plot(x_pos_adj[c],
                        means[c],
                        color=cycle_colors[c],
                        marker='o',
                        markerfacecolor='white',
                        lw=0,
                        markersize=3)

        cl_x = np.repeat(np.arange(len(REGIONS) + 1), 2)[1:-1] - .5
        ax.fill_between(cl_x, nc['lower'], nc['upper'], color='black',
                        alpha=.2, lw=0)
        ax.set_yticks(np.arange(0, 1, .2))
        ax.set_ylabel(f'{RSA_type} correlation (r)')
        ax.set_ylim((-.05, .6))
        ax.set_xticks(ticks=x_pos, labels=REGIONS)
        ax.set_xlim((-.5, len(layers) - .5))
        plt.tight_layout()
        plt.savefig(outpath, dpi=300)
        plt.close()

        # legend
        f = lambda m, c: \
            plt.plot([], [], marker=m, color='white', markeredgecolor=c,
                     linestyle='None')[0]
        handles = [f('o', color) for color in cycle_colors]
        labels = [str(c) for c in range(max_cycles)]
        legend = plt.legend(handles, labels, loc=3)
        export_legend(legend, filename=
            f'{analysis_dir}/human_likeness/cycles_legend.svg')
        plt.close()


        # layerwise plot
        outpath = f'{analysis_dir}/human_likeness/{exp}_{task}_' \
                  f'{RSA_type}_{cond_set}_{level}.svg'
        df_last_cycle = pd.DataFrame()
        for region, layer in region_to_layer[architecture].items():
            cycle = sorted(cycles[layer])[-1]
            temp = df[(df.region == region) &
                      (df.layer == layer) &
                      (df.cycle == cycle)].copy(deep=True)
            df_last_cycle = pd.concat([df_last_cycle, temp])
        line_plot(df_last_cycle, outpath, ylabel=f'{RSA_type} correlation (r)',
                  x_var='region', y_var='mean', error_var='sem', ceiling=nc,
                  ylims=(0,1), x_tick_labels=REGIONS)
        #df, outpath, ylabel = None, x_var = 'region', cond_var = 'cond',
        #col_var = 'color', error_var = 'error', ceiling = None, floor = None,
        #ylims = (0, 1.25), hline = None, yticks = np.arange(-1, 1.1, .5),
        #figsize = (3.5, 2)

    # save
    model_df.to_csv(f'{analysis_dir}/human_likeness/human_likeness.csv')


def RSA_fMRI(model_dir, m=0, total_models=0, overwrite=False):

    model_info = model_dir.split('models/')[-1].split('/')
    architecture, identifier = model_info[:2]
    transfer_dir = model_info[2] if len(model_info) == 3 else 'X'

    similarities = {
        # 'cosine': 'cosine similarity',
        'pearson': 'pearson correlation'}
        #'spearman': "Kendall's correlation"}

    for norm, norm_method, (similarity, similarity_label) in \
        itertools.product(fMRI.norms, fMRI.norm_methods, similarities.items()):

        analysis_dir = (f'{model_dir}/fMRI/norm-{norm}_{norm_method}/'
                        f'{similarity}')
        os.makedirs(analysis_dir, exist_ok=True)

        RSA_path = f"{analysis_dir}/RSA.pkl"

        if not op.isfile(RSA_path) or overwrite:

            print(f'{now()} Performing RSA | model({m + 1}/{total_models}) '
                  f'{architecture}/{identifier}/{transfer_dir})')

            responses = pkl.load(open(f'{model_dir}/fMRI/responses.pkl', 'rb'))
            RSAs = {}

            for key, responses_layer in responses.items():

                RSAs[key] = {}
                if 'cyc' in key:
                    layer, cycle = key.split('_')
                else:
                    layer, cycle = key, 'cyc-1'
                n_units = responses_layer.responses.shape[1]

                print_string = (f'| {cycle} | {layer} | norm-{norm} '
                                f'| {norm_method} | {similarity} '
                                f'| {n_units} units')

                # calculate RSM
                # print(f'Calculating RSM {print_string}')
                RSAs[key] = responses_layer.calculate_RSM(
                    norm, norm_method, similarity)

                # plot RSM
                # print(f'Plotting RSM {print_string}')
                RSAs[key].plot_RSM(
                    vmin=None, vmax=None,
                    fancy=False,
                    title=f'layer: {layer}',
                    labels=fMRI.cond_labels['exp1'],
                    outpath=f'{analysis_dir}/RSMs/{cycle}_{layer}.svg',
                    measure=similarity)

                # MDS
                # print(f'Plotting MDS {print_string}')
                outpath = f'{analysis_dir}/MDS/{layer}.svg'
                RSAs[key].plot_MDS(
                    title=f'cycle: {cycle}, layer: {layer}',
                    outpath=outpath)

                # perform contrasts and make plots
                RSAs[key].RSM_to_table()
                RSAs[key].calculate_occlusion_robustness()
                RSAs[key].fit_models()

            # save RSA data
            pkl.dump(RSAs, open(RSA_path, 'wb'))

            # perform analyses across layers and make plots
            compare_layers_cycles(RSAs, architecture, analysis_dir)



def generate_reconstructions(model_dir, overwrite=False):

    model_info = model_dir.split('models/')[-1].split('/')
    architecture, identifier = model_info[:2]
    transfer_dir = model_info[2] if len(model_info) == 3 else 'X'
    num_cycles = 8

    # output directory
    out_dir = f'{model_dir}/fMRI/image_reconstructions'
    os.makedirs(f'{out_dir}', exist_ok=True)

    tiled_recons = f'{out_dir}/tiled.png'
    if not op.isfile(tiled_recons) or overwrite:

        print(f'{now()} | Making reconstructions | '
              f'{architecture}/{identifier}/{transfer_dir}')

        model = get_trained_model(model_dir, architecture)
        transform = get_transform(architecture)
        inputs = torch.stack([transform(Image.open(x)) for x in images])
        inputs_time = torch.stack([inputs]*num_cycles, dim=1)

        # put on device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        inputs_time.to(device)

        outputs = model(inputs_time)

        for t, i in itertools.product(np.arange(num_cycles), np.arange(24)):
            image = outputs[t][i]
            image -= image.min()
            image /= image.max()
            image = transforms.ToPILImage()(image)
            exemplar = fMRI.exemplars[i // 3]
            occluder = fMRI.occluders[i % 3]
            image.save(f'{out_dir}/{exemplar}_{occluder}_cyc{t+1:02}.png')

        for occ in fMRI.occluder_labels:
            paths = list()
            for exem in fMRI.exemplars:
                input_path = [i for i in images if f'{exem}_{occ}' in i][0]
                paths.append(input_path)
                paths += sorted(glob.glob(f'{out_dir}/{exem}_{occ}_cyc*.png'))
            tile(paths, tiled_recons.replace('.png', f'_{occ}.png'),
                 num_rows=8, num_cols=num_cycles + 1, base_gap=4)
        image_paths = sorted(glob.glob(f'{out_dir}/tiled_*.png'))
        tile(image_paths, tiled_recons, num_rows=1, num_cols=3, base_gap=32)


