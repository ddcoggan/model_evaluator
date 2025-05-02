'''
This script tests the accuracy of CNNs on classifying the exact images presented in the human behavioral experiment.
'''

import os
import os.path as op
import glob
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy import special
import pickle as pkl
import pandas as pd
from scipy.optimize import curve_fit
from itertools import product as itp
from tqdm import tqdm
import gc

sys.path.append(op.expanduser('~/david/projects/p022_occlusion'))
from in_vivo.behavioral.exp1.analysis import CFG as EXP1
from in_vivo.behavioral.exp2.analysis import CFG as EXP2
from in_vivo.behavioral.exp2.analysis import condwise_robustness_plot
from torchvision.utils import save_image

from utils import insert_cycle, now, MODEL_BASE
from utils.model_contrasts import model_contrasts
from utils.get_trained_model import get_trained_model
from utils.image_processing import tile
from utils.get_activations import get_activations
from utils.math_functions import sigmoid
from utils.plot_utils import custom_defaults
from utils.benchmarks.occlusion_behavioral_exp1 import get_transform
plt.rcParams.update(custom_defaults)

np.random.seed(42)

METRICS = ['accuracy', 'true_class_prob', 'entropy']  # always available
METRICS_ALL = METRICS + ['reconstruction_loss']  # sometimes available
OBJ_VARIABLES = ['object_animacy', 'object_class']
OBJ_ANIMACIES = ['animate', 'inanimate']
OBJ_CLASSES = EXP1.object_classes
OBJ_CLS_IDCS = EXP1.class_idxs
OBJ_CLS_DIRS = EXP1.synsets
OCC_VARIABLES = ['visibility', 'occluder_class', 'occluder_color']
OCC_COLORS = ['black', 'white']
OCC_CLASSES = EXP2.occluder_classes
NUM_FRAMES = 17
VISIBILITIES = np.linspace(0, 1, NUM_FRAMES)
FRAMES = np.linspace(1, 360, NUM_FRAMES).astype(int)
RES_DIR = 'benchmarking/occlusion_behavioral'


def load_trials(drop_human=False, model_dir=None):

    if model_dir is None:
        trials = pd.read_parquet(
            f'../p022_occlusion/data/in_vivo/behavioral'
            f'/exp2/analysis/trials.parquet')
    else:
        trials = pd.read_parquet(op.join(
            MODEL_BASE, model_dir, RES_DIR, 'exp2/trials.parquet'))
        drop_human = False  # human data not present

    # drop human data
    if drop_human:  # human data
        trials = (trials
            [trials.subject.isin(['sub-01', 'sub-02'])]
            .drop(columns=[
                'trial', 'RT', 'prediction', 'accuracy', 'visibility',
                'subject'] + [c for c in trials.columns if c.startswith('fix')])
            .sort_values(by='stimulus_id').reset_index(drop=True))
        num_trials = len(trials)
        trials = pd.concat([trials] * len(FRAMES)).reset_index(drop=True)
        trials['visibility'] = [
            x for x in VISIBILITIES for _ in range(num_trials)]

    # enforce ordering of categorical variables
    trials.object_animacy = pd.Categorical(
        trials.object_animacy, OBJ_ANIMACIES, ordered=True)
    trials.object_class = pd.Categorical(
        trials.object_class, OBJ_CLASSES, ordered=True)
    trials.occluder_class = pd.Categorical(
        trials.occluder_class, OCC_CLASSES, ordered=True)
    trials.occluder_color = pd.Categorical(
        trials.occluder_color, OCC_COLORS, ordered=True)

    return trials


def list_images():
    image_dir = f'../p022_occlusion/data/in_vivo/behavioral/exp2/stimuli/final'
    images = np.ravel([sorted(glob.glob(
        f'{image_dir}/set-?/frames/*/{f:03}.png')) for f in FRAMES]).tolist()
    return images


def list_unoccluded_images():
    images_occ = list_images()
    images = [op.join(op.dirname(i), '360.png') for i in images_occ]
    return images


def get_responses(model_dir, architecture, layers, batch_size, m=0,
                  total_models=0, num_procs=1, overwrite=False):

    results_dir = op.join(MODEL_BASE, model_dir, RES_DIR, 'exp2')
    os.makedirs(results_dir, exist_ok=True)

    out_path = f'{results_dir}/trials.parquet'
    if not op.isfile(out_path):
        prev_trials = pd.DataFrame()
    else:
        prev_trials = pd.read_parquet(out_path)
        if overwrite:
            prev_trials = prev_trials[~prev_trials.layer.isin(layers)]
        layers = [i for i in layers if i not in prev_trials.layer.unique()]
        if not len(layers):
            return False

    non_output_layers = [l for l in layers if l != 'output']
    if len(non_output_layers):
        print(f'{now()} | Loading PCA and SVC objects...')
        transfer_dir = op.join(MODEL_BASE, model_dir, RES_DIR,
                               'transfer_learning')
        svcs = {}
        for layer in non_output_layers:
            svc_path = op.join(transfer_dir, f'{layer}.pkl')
            with open(svc_path, 'rb') as f:
                svcs[layer] = pkl.load(f)

    print(f'{now()} | Measuring responses for exp2 stimuli, '
          f'model: {m + 1}/{total_models} at {model_dir}, '
          f'layers: {layers}')

    # get responses to test images
    images = list_images()
    sampler = np.random.permutation(len(images))
    trials = load_trials(drop_human=True)
    new_trials = pd.DataFrame()

    # process in batches inside a function so we can free memory with gc
    def _process_batch(model, architecture, inputs, trials_batch):

        activations = get_activations(
            model, architecture, inputs, num_workers=num_procs,
            shuffle=False, batch_size=batch_size, layers=layers,
            transform=get_transform(architecture, model_dir))
        activations = insert_cycle(activations)
        trials_batch_lcs = pd.DataFrame()  # collates layers and cycles

        for layer, cycles in activations.items():
            for cycle, activs in cycles.items():

                # get a copy of the trial data and add relevant info
                trials_batch_lc = trials_batch.copy()
                trials_batch_lc['layer'] = layer
                trials_batch_lc['cycle'] = int(cycle[3:])

                # get pca and svc objects, if necessary
                kwargs = {}
                if layer != 'output':
                    kwargs['pca_object'] = svcs[layer][cycle]['pca']
                    kwargs['svc_object'] = svcs[layer][cycle]['svc']

                # get predictions
                print(f'{now()} | Generating predictions for {layer} {cycle}')
                trials_batch_lc = get_predictions(trials=trials_batch_lc,
                    activations=activs, readout_layer=layer, **kwargs)
                trials_batch_lcs = pd.concat(
                    [trials_batch_lcs, trials_batch_lc])

        return trials_batch_lcs

    # loop through batches
    superbatch_size = 2000  # reduces memory demand
    num_batches = np.ceil(len(images) / superbatch_size).astype(int)
    for b in range(num_batches):
        print(f'{now()} | Batch {b + 1}/{num_batches}')
        model = get_trained_model(model_dir, architecture, True, layers)
        first = b * superbatch_size
        last = min(first + superbatch_size, len(images))
        batch_ids = sampler[first:last]
        trials_batch = pd.concat([trials[trials.index == i
                                  ] for i in batch_ids])
        inputs = [images[i] for i in batch_ids]
        trials_batch = _process_batch(model, architecture, inputs,
                                      trials_batch)
        gc.collect()
        new_trials = pd.concat([new_trials, trials_batch])

    # reorder based on layer and cycle
    new_trials = new_trials \
        .sort_values(by=['layer', 'cycle', 'stimulus_id']) \
        .reset_index(drop=True)

    # print out accuracy for each layer and cycle
    new_trials.groupby(['layer', 'cycle']).apply(
        lambda df: print(
            f'{now()} | {df.name} accuracy: {df.accuracy.mean():.4}'))

    # reformat to combine all metric columns into long format
    new_trials = reshape_metrics(new_trials, 'long')

    # if generative model, evaluate reconstructions
    if 'pix2pix' in model_dir:
        trials = evaluate_reconstructions(model_dir, num_procs)
        trials = reshape_metrics(trials, 'long')
        new_trials = pd.concat([new_trials, trials])

    # save trials
    all_trials = pd.concat([prev_trials, new_trials]).reset_index(drop=True)
    all_trials.to_parquet(out_path, index=False)

    return True


def get_predictions(trials, activations, readout_layer, pca_object=None,
                    svc_object=None):

    # check we have the right number of activations
    assert activations.shape[0] == len(trials), \
        'different number of images and activations'

    # predictions based on output layer or svc object
    if readout_layer == 'output':
        probs = special.softmax(activations[:, OBJ_CLS_IDCS], axis=1)
        classes_ordered = OBJ_CLASSES
    else:
        pca_weights = pca_object.transform(
            activations.reshape((len(trials), -1)))[:, :1000]
        probs = svc_object.predict_proba(pca_weights)
        classes_ordered = list(svc_object.classes_)
    assert (probs.sum(1).round(2) == 1).all(), 'probabilities do not sum to 1'

    # add predictions and other measures to trials
    trials['prediction'] = [classes_ordered[c] for c in probs.argmax(axis=1)]
    trials['accuracy'] = pd.Series(
        trials.prediction == trials.object_class, dtype=int)
    trials['true_class_prob'] = [probs[i, classes_ordered.index(c)]
                                 for i, c in enumerate(trials.object_class)]
    trials['entropy'] = stats.entropy(probs, axis=1)

    return trials


def evaluate_reconstructions(model_dir, num_procs):

    import torch
    from torch.utils.data import DataLoader, Dataset
    from PIL import Image

    print(f'{now()} | Evaluating reconstructions for exp2 stimuli')

    #os.makedirs(op.dirname(out_path), exist_ok=True)
    images_occ = list_images()
    images_unocc = list_unoccluded_images()
    sampler = np.random.permutation(len(images_occ))
    reload_every = 4000  # overcomes memory leak with some recurrent models
    num_blocks = np.ceil(len(images_occ) / reload_every).astype(int)
    batch_size = 64

    class ReconstructionImages(Dataset):
        def __init__(self,
                     mode: str = 'eval',
                     direction: str = 'B2A',
                     files_unocc: list = [],
                     files_occ: list = []):
            self.mode = mode
            self.direction = direction
            self.transform = get_transform('pix2pix')
            self.files_unocc = files_unocc
            self.files_occ = files_occ

        def __len__(self, ):
            return len(self.files_unocc)

        def __getitem__(self, idx):

            imgA = Image.open(self.files_unocc[idx]).convert('RGB')
            imgA = self.transform(imgA)
            imgB = Image.open(self.files_occ[idx]).convert('RGB')
            imgB = self.transform(imgB)
            if self.direction == 'A2B':
                return imgA, imgB
            else:
                return imgB, imgA

    loss = torch.nn.BCEWithLogitsLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    losses = []
    for b in range(num_blocks):

        print(f'{now()} | Block {b + 1}/{num_blocks}')

        model = get_trained_model(model_dir).to(device) # reload model
        first = b * reload_every
        last = min(first + reload_every, len(images_occ))
        inputs_occ = [images_occ[i] for i in sampler[first:last]]
        inputs_unocc = [images_unocc[i] for i in sampler[first:last]]
        dataset = ReconstructionImages(mode='eval',
                                       files_unocc=inputs_unocc,
                                       files_occ=inputs_occ)
        dataloader = DataLoader(dataset, batch_size=batch_size//2,
                                shuffle=False,
                                num_workers=num_procs)
        with tqdm(dataloader, unit=f"batch({batch_size//2})") as tepoch:
            for batch, (occ, unocc) in enumerate(tepoch):
                occ = occ.to(device)
                unocc = unocc.to(device)
                pred = model(occ)
                losses.extend([loss(p, u).detach().cpu().numpy().item()
                               for p, u in zip(pred, unocc)])
                if b + batch == 0:
                    outdir = op.join(MODEL_BASE, model_dir, RES_DIR,
                                     'exp2/reconstructions')
                    os.makedirs(outdir, exist_ok=True)
                    outpaths = []
                    for inputs in [occ, unocc, pred]:
                        inputs -= inputs.min()
                        inputs /= inputs.max()
                    for i, (label, inputs) in itp(
                            range(32), zip(['input', 'target', 'prediction'],
                                           [occ, unocc, pred])):
                        outpath = f'{outdir}/{i:04}_{label}.png'
                        save_image(inputs[i, :, :, :], outpath)
                        outpaths.append(outpath)
                    tile(outpaths, f'{outdir}/tiled.png', base_gap=4,
                         colgap=16, colgapfreq=3, num_cols=12)
                    for out_path in outpaths:
                        os.remove(out_path)

    trials = load_trials(drop_human=True)
    trials['reconstruction_loss'] = [losses[i] for i in np.argsort(sampler)]
    trials['layer'] = 'output'
    trials['cycle'] = -1
    trials = trials.reset_index(drop=True)

    print(f'{now()} | Recon loss: {trials.reconstruction_loss.mean():.4f}')

    return trials


def measure_robustness(trials):
    agg_dict = dict(accuracy='mean', true_class_prob='mean', entropy='mean')
    agg_vars = ['stimulus_set']
    agg_vars.extend(OCC_VARIABLES + OBJ_VARIABLES)
    robustness = (
        trials.groupby(agg_vars, dropna=False).agg(agg_dict).reset_index())
    robustness['layer'] = trials.layer.values[0]
    robustness['cycle'] = trials.cycle.values[0]

    return robustness


def estimate_model_RTs(trials_long):

    def _estimate_RT(df, metric):
        sustain_period = 3
        if metric == 'accuracy':
            performance = list(df.value.to_numpy() == 1)
        elif metric == 'true_class_prob':
            performance = list(df.value.to_numpy() > .5)
        else:  # if metric == 'entropy' or 'reconstruction_loss':
            performance = list(df.value.to_numpy() < .5)
        frame_index = len(performance) - 1  # default value if no criteria met
        for i in range(len(performance) - 2):
            if performance[i:i + sustain_period] == [True] * sustain_period:
                frame_index = i
                break
        else:
            # for cases where performance hits criteria within 2 highest vis
            if max(performance[-2:]):
                frame_index = (performance[-2:].index(True) +
                               (len(performance) - 2))
        vis = frame_index / (len(performance) - 1)

        return vis

    metric = trials_long.name[-1]
    trials = (trials_long[trials_long.visibility == 1])
    trials['visibility'] = (
        trials_long.groupby('stimulus_id').apply(_estimate_RT, metric)).values

    return trials


def fit_visibility_curves(trials):

    def _fit_curve(xvals, yvals, thr=.5):
        try:
            init_params = [max(yvals), np.median(xvals), 1, 0]
            popt, pcov = curve_fit(
                sigmoid, xvals, yvals, init_params, maxfev=int(10e5))
            curve = sigmoid(np.linspace(0, 1, 1000), *popt)
            threshold = sum(curve < thr) / 1000
        except:
            UserWarning('Curve fitting failed, returning NaNs')
            popt, threshold = [np.nan] * 4, np.nan
        return popt, threshold

    curves = pd.DataFrame()
    metric = trials.name[-1]

    # separate curve per stimulus_set, occluder_class, occluder_color
    for stimulus_set, occluder_class, occluder_color in (
            itp(['a', 'b'], OCC_CLASSES, OCC_COLORS)):
        yvals = (trials[
                     (trials['stimulus_set'] == stimulus_set) &
                     (trials['occluder_class'] == occluder_class) &
                     (trials['occluder_color'] == occluder_color)]
                 .groupby('visibility')
                 .mean(numeric_only=True).value.to_list())
        yval_mean = np.mean(yvals)

        # fit curve function
        popt, threshold = _fit_curve(VISIBILITIES, yvals)
        curves = pd.concat(
            [curves, pd.DataFrame({
                'stimulus_set': [stimulus_set],
                'subject': ['group'],
                'occluder_class': [occluder_class],
                'occluder_color': [occluder_color],
                'L': [popt[0]],
                'x0': [popt[1]],
                'k': [popt[2]],
                'b': [popt[3]],
                'threshold_50': [threshold],
                'mean': [yval_mean],
            })]).reset_index(drop=True)

    # single function across entire dataset
    yvals = (trials.groupby('visibility').mean(
        numeric_only=True).value.to_list())
    yval_mean = np.mean(yvals)
    popt, threshold = _fit_curve(VISIBILITIES, yvals)
    curves = pd.concat(
        [curves, pd.DataFrame({
            'stimulus_set': ['all'],
            'subject': ['group'],
            'occluder_class': ['all'],
            'occluder_color': ['all'],
            'L': [popt[0]],
            'x0': [popt[1]],
            'k': [popt[2]],
            'b': [popt[3]],
            'threshold_50': [threshold],
            'mean': [yval_mean],
        })]).reset_index(drop=True)

    return curves


def measure_human_likeness(trials_model, trials_model_rt, trials_human):
    def _trial_corrs(trials_h, trials_m):
        mutual_cols = list(np.intersect1d(trials_h.columns, trials_m.columns))
        [mutual_cols.remove(x) for x in ['accuracy', 'visibility']]
        trials = trials_h.merge(trials_m, on=mutual_cols, suffixes=('_h', '_m'))
        trials_acc = trials[(trials.accuracy_h == 1) & (trials.accuracy_m == 1)]
        r = np.corrcoef(trials_acc.visibility_h, trials_acc.visibility_m)[0, 1]
        return r

    def _cond_corrs(trials_h, vis_m):
        vis_h = (trials_h
                 .groupby(['occluder_class', 'occluder_color'], observed=True)
                 .visibility.mean())
        r = np.corrcoef(vis_h, vis_m)[0, 1]
        return r

    stimulus_set, layer, cycle, metric = trials_model.name

    # trial-wise
    trials_model_rt = trials_model_rt[
        (trials_model_rt.stimulus_set == stimulus_set) &
        (trials_model_rt.layer == layer) &
        (trials_model_rt.cycle == cycle) &
        (trials_model_rt.metric == metric)]
    trials_h = trials_human[trials_human.stimulus_set == stimulus_set]
    trial_wise = pd.DataFrame(trials_h.groupby('subject')
                              .apply(_trial_corrs, trials_model_rt))
    trial_wise.rename(columns={0: 'value'}, inplace=True)
    trial_wise['metric_human'] = 'visibility'
    trial_wise['metric_model'] = 'visibility'
    trial_wise['metric_sim'] = 'correlation'
    trial_wise['level'] = 'trial-wise'
    trial_wise['within'] = 'subject'
    trial_wise['between'] = 'trial'
    trial_wise.reset_index(inplace=True)

    # condition-wise
    vis_m = (trials_model
             .groupby(['occluder_class', 'occluder_color', 'visibility'],
                      observed=True)
             .mean(numeric_only=True).reset_index()
             .groupby(['occluder_class', 'occluder_color'], observed=False)
             .apply(lambda d: d[d.value > .5].visibility.min()))
    vis_m.columns = ['visibility']
    cond_wise = pd.DataFrame(dict(
        value=trials_h.groupby('subject').apply(_cond_corrs, vis_m)))
    cond_wise['metric_human'] = 'visibility'
    cond_wise['metric_model'] = 'visibility'
    cond_wise['metric_sim'] = 'correlation'
    cond_wise['level'] = 'condition-wise'
    cond_wise['within'] = 'subject'
    cond_wise['between'] = 'occluder_class_x_occluder_color'
    cond_wise.reset_index(inplace=True)

    human_likeness = pd.concat([cond_wise, trial_wise]).reset_index(drop=True)

    return human_likeness


def reshape_metrics(df, shape):
    """ Reshape dataframe based on different model performance metrics to help
    with grouping, plotting, etc. """

    if shape == 'long' and 'metric' not in df.columns:
        metrics = [i for i in df.columns if i in METRICS_ALL]
        df = df.melt(id_vars=[c for c in df.columns if c not in metrics],
            value_vars=metrics, var_name='metric')
    elif shape == 'wide' and 'metric' in df.columns:
        df = df.pivot(index=[c for c in df.columns if c not in [
            'metric', 'value']], columns='metric', values='value').reset_index()
    return df


def existing_results(path, layers, overwrite):

    if op.isfile(path):
        df = pd.read_parquet(path, columns=['layer'])
        if overwrite:
            return df[~df.layer.isin(layers)]
        elif all(layer in df.layer.unique() for layer in layers):
            return True
        return df
    return pd.DataFrame()


def analyse_performance(model_dir, m=0, total_models=0,
                        layers=['output'], overwrite=False, remake_plots=False):

    results_dir = op.join(MODEL_BASE, model_dir, RES_DIR, 'exp2')
    mod_str = f'model {m + 1}/{total_models} at {model_dir}'
    recompare_models = False
    groupby = ['layer', 'cycle', 'metric']

    # fit performance curves
    curves_path = f'{results_dir}/robustness_curves.parquet'
    existing_curves = existing_results(curves_path, layers, overwrite)
    if existing_curves is not True:
        print(f'{now()} | Analysing performance curves (exp2) | {mod_str}')
        trials_model = reshape_metrics(
            load_trials(model_dir=model_dir), shape='long')
        curves = (trials_model
                  .groupby(groupby)
                  .apply(fit_visibility_curves)
                  .reset_index(groupby))
        curves = pd.concat([existing_curves, curves]).reset_index(drop=True)
        curves.to_parquet(curves_path, index=False)
        recompare_models, remake_plots = True, True

    # generate response times for each model
    trials_model_rt_path = f'{results_dir}/trials_RT.parquet'
    existing_trials = existing_results(trials_model_rt_path, layers,
                                       overwrite)
    if existing_trials is not True:
        print(f'{now()} | Estimating model RTs (exp2) | {mod_str}')
        trials_model = reshape_metrics(
            load_trials(model_dir=model_dir), shape='long')
        trials_model_rt = (trials_model
                           .groupby(['stimulus_set'] + groupby)
                           .apply(estimate_model_RTs).reset_index(drop=True))
        trials_model_rt = trials_model_rt.merge(
            trials_model[(trials_model.visibility == 1) &
                         (trials_model.metric == 'accuracy')][
                groupby[:-1] + ['stimulus_id', 'value']].rename(columns={
                'value': 'accuracy'}),
            on=groupby[:-1] + ['stimulus_id'])  # include model acc at vis = 1
        trials_model_rt = pd.concat(
            [existing_trials, trials_model_rt]).reset_index(drop=True)
        trials_model_rt.to_parquet(trials_model_rt_path, index=False)
        recompare_models, remake_plots = True, True

    # measure human likeness
    likeness_path = f'{results_dir}/human_likeness.parquet'
    existing_likeness = existing_results(likeness_path, layers, overwrite)
    if existing_likeness is not True:
        print(f'{now()} | Analysing human likeness (exp2) | {mod_str}')
        trials_human = load_trials(drop_human=False)
        trials_model = reshape_metrics(load_trials(model_dir=model_dir),
                                       shape='long')
        trials_model_rt = pd.read_parquet(trials_model_rt_path)
        likeness = (trials_model
                    .groupby(['stimulus_set'] + groupby)
                    .apply(measure_human_likeness, trials_model_rt,
                           trials_human)
                    .reset_index(level=['stimulus_set'] + groupby))

        likeness = pd.concat([existing_likeness, likeness]).reset_index(drop=True)
        likeness.to_parquet(likeness_path, index=False)
        recompare_models, remake_plots = True, True

    # plot model performance
    if remake_plots or not op.isdir(op.join(results_dir, 'plots')):
        plot_performance(model_dir)

    return recompare_models


def plot_performance(model_dir):

    def _make_curve_plots(robustness, curves, info):

        layer, cycle, metric = info

        plot_cfg = dict(
            accuracy=dict(
                ylabel='classification accuracy',
                ylims=(0, 1),
                yticks=(0, 1),
                chance=1 / 8),
            true_class_prob=dict(
                ylabel='true class probability',
                ylims=(0, 1),
                yticks=(0, 1),
                chance=1 / 8),
            entropy=dict(
                ylabel='uncertainty',
                ylims=(0, 2.5),
                yticks=np.arange(0, 2.5),
                chance=None),
            reconstruction_loss=dict(
                ylabel='reconstruction error',
                ylims=(0, 2.5),
                yticks=np.arange(0, 2.5),
                chance=None),
            )[metric]

        outpath = f'{plot_dir}/cyc{cycle:02}_{layer}_{metric}.svg'

        # accuracy for each occluder * visibility
        fig, axes = plt.subplots(
            2, 3, figsize=(3.5, 2.6), sharex=True, sharey=True)

        for o, occluder_class in enumerate(OCC_CLASSES):

            ax = axes.flatten()[o]
            for (c, color), (stimulus_set, edge_color) in itp(enumerate(
                    OCC_COLORS), zip(['a', 'b'], ['k', 'tab:grey'])):
                # plot curve function underneath
                face_color = EXP2.plot_colors[o * 2 + c]
                popt = (curves[
                            (curves.metric == metric) &
                            (curves.stimulus_set == stimulus_set) &
                            (curves.occluder_class == occluder_class) &
                            (curves.occluder_color == color)]
                        [['L', 'x0', 'k', 'b']].values)
                assert len(popt) == 1, 'more than one curve fit'
                curve_x = np.linspace(0, 1, 1000)
                curve_y = sigmoid(curve_x, *popt[0])
                #ax.plot(curve_x, curve_y, color=face_color,
                #        clip_on=False, zorder=1)

                # plot accuracies on top
                yvals = robustness[
                    #(robustness.stimulus_set == stimulus_set) &
                    (robustness.occluder_class == occluder_class) &
                    (robustness.occluder_color == color) &
                    (robustness.metric == metric)].groupby(
                    'visibility').value.mean().values
                ax.scatter(VISIBILITIES, yvals, clip_on=False,
                           facecolor=face_color, zorder=2)
                #edgecolor=edge_color, )

            # format plot
            #if o == 0:
            #    ax.set_title('biological', size=7)
            #else:
            #    ax.set_title(EXP2.occluder_labels[o], size=7)
            ax.set_xticks((0, 1))
            ax.set_xlim((0, 1))
            ax.set_yticks(plot_cfg['yticks'])
            ax.set_ylim(plot_cfg['ylims'])
            ax.tick_params(axis='both', which='major',
                           labelsize=7)
            if plot_cfg['chance']:
                ax.axhline(y=plot_cfg['chance'], color='k',
                           linestyle='dotted')
            if o == 4:
                ax.set_xlabel('visibility', size=10)

        fig.text(0, 0.58, plot_cfg['ylabel'], va='center',
                 rotation='vertical')
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close()


    def _make_scatterplots(robustness_m, robustness_h, info):

        layer, cycle, metric = info
        outpath = (
            f'{plot_dir}/cyc{cycle:02}_{layer}_{metric}_human-likeness.svg')
        colors = EXP2.plot_colors
        metric_human = 'visibility'
        rob_h = robustness_h.rename(columns={metric_human: f'human_value'})
        rob_m = robustness_m.rename(columns={'value': f'model_value'})
        rob_h = (rob_h[
                     rob_h.occluder_class.isin(OCC_CLASSES)]
                 .groupby(['occluder_class', 'occluder_color'], observed=True)
                 .agg({'human_value': 'mean'}))
        rob_m = (rob_m[
                     rob_m.occluder_class.isin(OCC_CLASSES)]
                 .groupby(['occluder_class', 'occluder_color'], observed=True)
                 .agg({'model_value': 'mean'}))
        plot_data = pd.merge(rob_h, rob_m,
                             on=['occluder_class', 'occluder_color'])
        xvals, yvals = plot_data[['human_value', 'model_value']].values.T
        plt.scatter(xvals, yvals, color=colors)
        plt.xlabel(f'human {metric_human}')
        plt.ylabel(f'model {metric}')
        x = [min(xvals), max(xvals)]
        y = np.poly1d(np.polyfit(xvals, yvals, 1))(x)
        plt.plot(x, y, color='k')
        r = np.corrcoef(xvals, yvals)[0, 1]
        plt.title(f'condition-wise accuracy scatterplot\nr = {r:.2f}')
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close()


    results_dir = op.join(MODEL_BASE, model_dir, RES_DIR, 'exp2')
    plot_dir = f'{results_dir}/plots'
    os.makedirs(plot_dir, exist_ok=True)
    trials_h = load_trials()
    trials_m = load_trials(model_dir=model_dir)
    curves = pd.read_parquet(f'{results_dir}/robustness_curves.parquet')

    # calculate mean performance for each condition

    # model
    groupby = ['layer', 'cycle', 'occluder_class', 'occluder_color',
               'metric', 'visibility', 'stimulus_set']
    robustness_m = (reshape_metrics(trials_m, 'long').groupby(groupby)
                    .agg({'value': 'mean'}).dropna().reset_index())

    # human
    groupby = ['occluder_class', 'occluder_color', 'stimulus_set']
    robustness_h = (trials_h.groupby(groupby, observed=False)
                    .agg('mean', numeric_only=True).reset_index())

    # condition-wise scatter_plots, model vs human
    for rob_m in robustness_m.groupby(['layer', 'cycle', 'metric']):
        _make_scatterplots(rob_m[1], robustness_h, rob_m[0])

    # performance curves, model only
    trials_m = reshape_metrics(trials_m, 'long')
    groupby = ['layer', 'cycle', 'metric']
    for rob, cur in zip(trials_m.groupby(groupby), curves.groupby(groupby)):
        _make_curve_plots(rob[1], cur[1], rob[0])

    # condition-wise performance (occluder class * occluder color)
    trials_m_rt = pd.read_parquet(f'{results_dir}/trials_RT.parquet')
    groupby = ['layer', 'cycle', 'metric']
    for rob in robustness_m.groupby(groupby):

        layer, cycle, metric = rob[0]

        # trial-wise mean performance across conditions
        outpath = (f'{plot_dir}/cyc{cycle:02}_{layer}_'
                   f'{metric}_trial-wise_mean.svg')
        trials = trials_m_rt[
            (trials_m_rt.cycle == cycle) &
            (trials_m_rt.layer == layer) &
            (trials_m_rt.metric == metric)]
        condwise_robustness_plot(
            df=trials,
            outpath=outpath,
            metric_column=f'value',
            class_column='occluder_class',
            color_column='occluder_color',
            sample_column=None,
            invert_y=metric in ['entropy', 'reconstruction_loss'],
            title=f'mean {metric}')

        # trial-wise visibility threshold
        outpath = (f'{plot_dir}/cyc{cycle:02}_{layer}_'
                   f'{metric}_trial-wise_thresh.svg')
        trials = trials_m_rt[
            (trials_m_rt.cycle == cycle) &
            (trials_m_rt.layer == layer) &
            (trials_m_rt.metric == metric)]
        condwise_robustness_plot(
            df=trials,
            outpath=outpath,
            metric_column=f'visibility',
            class_column='occluder_class',
            color_column='occluder_color',
            sample_column=None,
            acc_only=True,
            title='object visibility at RT')

        # condition-wise mean performance across visibilities
        outpath = (f'{plot_dir}/cyc{cycle:02}_{layer}_'
                   f'{metric}_cond-wise_mean.svg')
        vis_m = (rob[1]
             .groupby(['occluder_class', 'occluder_color'])
             .mean('value')[['value']])
        condwise_robustness_plot(
            df=vis_m,
            outpath=outpath,
            metric_column='value',
            class_column='occluder_class',
            color_column='occluder_color',
            sample_column=None,
            title=f'mean {metric}',
            invert_y=metric in ['entropy', 'reconstruction_loss'],
            ylabel=metric)

        # condition-wise visibility threshold
        outpath = (f'{plot_dir}/cyc{cycle:02}_{layer}_'
                   f'{metric}_cond-wise_thresh.svg')
        vis_m = (rob[1].groupby(
            ['occluder_class', 'occluder_color', 'visibility'])
                 .mean(numeric_only=True).reset_index()
                 .groupby(['occluder_class', 'occluder_color'])
                 .apply(lambda d: d[d.value > .5].visibility.min()))
        vis_m = pd.DataFrame(dict(visibility=vis_m))
        condwise_robustness_plot(
            df=vis_m,
            outpath=outpath,
            metric_column='visibility',
            class_column='occluder_class',
            color_column='occluder_color',
            sample_column=None,
            title='mean accuracy',
            ylabel='visibility',
            legend_path=f'{plot_dir}_legend.svg')




