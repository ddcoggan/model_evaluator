'''
This script tests the accuracy of CNNs on classifying the exact images
presented in the human behavioral experiment 1.
'''

import os
import os.path as op
import glob
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import colormaps
from scipy import stats
from scipy import special
import pickle as pkl
import pandas as pd
from scipy.optimize import curve_fit
import math
import time
from types import SimpleNamespace
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from itertools import product as itp
from datetime import datetime
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
import gc
import torchvision
import torchvision.transforms.v2 as transforms
from torchvision.utils import save_image
from torch import float32
from utils import insert_cycle, now, MODEL_BASE
from utils.model_contrasts import model_contrasts
from utils.get_trained_model import get_trained_model
from utils.image_processing import tile
from utils.get_activations import get_activations
from utils.math_functions import sigmoid
from utils.plot_utils import custom_defaults
plt.rcParams.update(custom_defaults)
sys.path.append(op.expanduser('~/david/projects/p022_occlusion'))
from in_vivo.behavioral.exp1.analysis import CFG as EXP1
from in_vivo.behavioral.exp1.analysis import (
    condwise_robustness_plot_array)

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
OCC_CLASSES = EXP1.occluder_classes
VISIBILITIES = EXP1.visibilities
RES_DIR = 'benchmarking/occlusion_behavioral'

def get_transform(architecture, model_dir):

    if hasattr(torchvision.models, architecture) and 'pretrained' in model_dir:
        model_attr = str([i for i in torchvision.models.__dict__ if
                             i.lower() == f'{architecture}_weights'][0])
        weights_attr = model_dir.split('pretrained_')[-1]
        transform = getattr(getattr(torchvision.models, model_attr),
                            weights_attr).transforms()
        return transform

    imsize = 256 if architecture == 'pix2pix' else 224
    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(float32, scale=True),
        transforms.Resize(imsize),
        transforms.CenterCrop(imsize),  # in case resize is off by a pixel
        transforms.Grayscale(num_output_channels=3),  # should be redundant
        transforms.Normalize(mean=[0.445, 0.445, 0.445],
                             std=[0.269, 0.269, 0.269]),
    ])
    return transform


def make_svc_dataset(overwrite=False):
    """ Creates a dataframe of images to train a linear support vector
    machine classifier (SVC) . The images are from the same classes as the
    behavioral experiments but are not the same images. """

    dataset_dir = op.expanduser(f'~/Datasets/ILSVRC2012')
    svc_images_path = 'utils/SVC_images.csv'

    if not op.isfile(svc_images_path) or overwrite:

        # get set of all images used in behavioral exps
        behavioral_set = set()
        for exp in ['exp1', 'exp2']:
            trials = load_trials(exp)
            behavioral_set.update([op.basename(x).split('.')[0]
                                   for x in trials.object_path.values])

        # create independent training set
        classes, images = [], []
        ims_per_class_svc = 256
        for class_dir, class_label in zip(OBJ_CLS_DIRS, OBJ_CLASSES):
            im_counter = 0
            image_paths = sorted(
                glob.glob(f'{dataset_dir}/train/{class_dir}/*'))
            while im_counter < ims_per_class_svc:
                image_path = image_paths.pop(0)
                if op.basename(image_path) not in behavioral_set:
                    classes.append(class_label)
                    images.append(image_path)
                    im_counter += 1
        svc_images = pd.DataFrame({'object_class': classes, 'filepath': images})
        svc_images.to_csv(svc_images_path, index=False)


def make_pca_dataset(overwrite=False):
    """ Creates a dataframe containing images with which to measure
    the principle components of layer activations. Images selected are from
    the 992 imagenet_classes not used in the behavioral experiments. """

    dataset_dir = op.expanduser(f'~/Datasets/ILSVRC2012')

    pca_images_path = 'utils/PCA_images.csv'
    if not op.isfile(pca_images_path) or overwrite:

        imagenet_classes = [op.basename(path) for path in sorted(glob.glob(
            f'{dataset_dir}/val/*'))]
        for d in OBJ_CLS_DIRS:
            imagenet_classes.remove(d)
        classes, images = [], []
        ims_per_class = 2
        for imagenet_class in imagenet_classes:
            images += sorted(glob.glob(
                f'{dataset_dir}/val/{imagenet_class}/*'))[:ims_per_class]
            classes += [imagenet_class] * ims_per_class
        pca_images = pd.DataFrame({'class': classes, 'filepath': images})
        pca_images.to_csv(pca_images_path, index=False)


def train_svc(model_dir, architecture, batch_size, layer, m='?',
              total_models='?', num_procs=8, overwrite=False):

    if layer == 'output':
        return False

    transfer_dir = op.join(MODEL_BASE, model_dir, RES_DIR,
                           'transfer_learning')
    os.makedirs(transfer_dir, exist_ok=True)

    # check if old format of svc file exists, where all layers are combined
    svc_path = op.join(op.dirname(transfer_dir), 'SVC.pkl')
    if op.isfile(svc_path):
        with open(svc_path, 'rb') as f:
            svcs = pkl.load(f)
        layers = set([k.split('_')[0] for k in svcs.keys()])
        for layer in layers:
            new_svcs = {k: v for k, v in svcs.items() if k.startswith(layer)}
            with open(op.join(transfer_dir, f'{layer}.pkl'), 'wb') as f:
                pkl.dump(new_svcs, f)
        os.remove(svc_path)
    svc_contents_path = op.join(op.basename(transfer_dir), 'SVC_contents.csv')
    if op.isfile(svc_contents_path):
        os.remove(svc_contents_path)

    # do PCA and SVC inside function to free memory
    def pca_svc(layer, out_path):

        print(f'{now()} | Performing transfer learning for model '
              f'{m + 1}/{total_models}, {model_dir}, layer: {layer}')

        transform = get_transform(architecture, model_dir)
        svcs = {}

        # use PCA for dimensionality reduction
        print(f'{now()} | Running PCA...')
        pca_images = pd.read_csv('utils/PCA_images.csv').filepath.values
        model = get_trained_model(model_dir, architecture, True, layer)
        activations = get_activations(
            model, architecture, pca_images, layers=layer,
            num_workers=num_procs, batch_size=batch_size,
            transform=transform, shuffle=True)
        activations = insert_cycle(activations)
        for cyc, activ in activations[layer].items():
            svcs[cyc] = {'pca': PCA().fit(activ.reshape([len(pca_images), -1]))}

        # train a support vector machine on responses to the training set
        print(f'{now()} | Running SVC...')
        svc_dataset = pd.read_csv('utils/SVC_images.csv')
        svc_images = svc_dataset['filepath'].values
        sampler = np.random.permutation(len(svc_images))
        svc_classes = [svc_dataset.object_class.values[s] for s in sampler]
        model = get_trained_model(model_dir, architecture, True, layer)
        activations = get_activations(
            model, architecture, svc_images, num_workers=num_procs,
            batch_size=batch_size, layers=layer, transform=transform,
            sampler=sampler)
        activations = insert_cycle(activations)
        for cyc, activ in activations[layer].items():
            pca_weights = svcs[cyc]['pca'].transform(
                activ.reshape((len(svc_images), -1)))[:, :1000]
            clf = OneVsRestClassifier(BaggingClassifier(
                SVC(kernel='linear', probability=True),
                max_samples=1 / num_procs, n_estimators=num_procs))
            clf.fit(pca_weights, svc_classes)
            train_acc = np.mean(clf.predict(pca_weights) == svc_classes)
            svcs[cyc]['svc'] = clf
            print(f'{now()} | Training accuracy ({cyc}): {train_acc:.4f}')

        with open(out_path, 'wb') as f:
            pkl.dump(svcs, f)

    transfer_path = op.join(transfer_dir, f'{layer}.pkl')
    if overwrite or not op.isfile(transfer_path):
        pca_svc(layer, transfer_path)
        gc.collect()
        return True
    return False


def load_trials(drop_human=False, model_dir=None):
    if model_dir is None:
        trials = pd.read_parquet(
            f'../p022_occlusion/data/in_vivo/behavioral'
            f'/exp1/analysis/trials.parquet')
    else:
        trials = pd.read_parquet(op.join(
            MODEL_BASE, model_dir, RES_DIR, 'exp1', 'trials.parquet'))
        drop_human = False  # human data not present

    # drop human data
    if drop_human:  # human data
        trials.drop(columns=['prediction', 'accuracy', 'RT'], inplace=True)

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
    images = sorted(glob.glob(
        f'../p022_occlusion/data/in_vivo/behavioral/exp1/images/final/*.png'))
    return images


def list_unoccluded_images():
    exp_dir = f'../p022_occlusion/data/in_vivo/behavioral/exp1'
    images = sorted(glob.glob(f'{exp_dir}/images/final_unoccluded/*.png'))
    return images


def get_responses(model_dir, architecture, layers, batch_size, m=0,
                  total_models=0, num_procs=1, overwrite=False):

    results_dir = op.join(MODEL_BASE, model_dir, RES_DIR, 'exp1')
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

    print(f'{now()} | Measuring responses for exp1 stimuli, '
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
        trials_batch = pd.concat([trials[trials.stimulus_id == f'{i:05}'
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
        trials = evaluate_reconstructions(model_dir, 'exp1', num_procs)
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

    print(f'{now()} | Evaluating reconstructions for exp1 stimuli')

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
                    outdir = op.join(MODEL_BASE, model_dir, RES_DIR, 'exp1',
                                     'reconstructions')
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
    agg_vars = ['subject']
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

    vis = VISIBILITIES + [1]
    if metric != 'entropy':
        vis = [0] + vis

    # separate function for each occluder_class * occluder_color
    for occluder_class, occluder_color in itp(OCC_CLASSES, OCC_COLORS):

        yvals = (trials[
                     (trials['occluder_class'] == occluder_class) &
                     (trials['occluder_color'] == occluder_color)]
                 .groupby('visibility').mean(
            numeric_only=True).value.to_list())
        yval_mean = np.mean(yvals)

        unocc = trials[trials['visibility'] == 1].value.mean()
        yvals += [unocc]
        if metric != 'entropy':
            yvals = [1 / 8] + yvals

        # fit curve function
        popt, threshold = _fit_curve(vis, yvals)
        curves = pd.concat(
            [curves, pd.DataFrame({
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
    for dataset in ['all', 'artificial']:
        if dataset == 'artificial':
            trials = trials[trials.occluder_class !=
                            'naturalUntexturedCropped2']
        yvals = (trials.groupby('visibility').mean(
            numeric_only=True).value.to_list())
        yval_mean = np.mean(yvals)
        if metric != 'entropy':
            yvals = [1 / 8] + yvals
        popt, threshold = _fit_curve(vis, yvals)
        curves = pd.concat(
            [curves, pd.DataFrame({
                'subject': ['group'],
                'occluder_class': [dataset],
                'occluder_color': ['all'],
                'L': [popt[0]],
                'x0': [popt[1]],
                'k': [popt[2]],
                'b': [popt[3]],
                'threshold_50': [threshold],
                'mean': [yval_mean],
            })]).reset_index(drop=True)

    return curves


def measure_human_likeness(trials_model, trials_human):

    def _c_obs(trials):
        return np.mean(trials.human_performance == trials.model_performance)

    def _c_err(trials, weighted=False):
        hum = trials.human_performance.mean()
        mod = trials.model_performance.mean()
        ceil = 1 - np.abs(hum - mod)  # ceil = 1 only if a == b
        chance_acc = hum * mod
        chance_inacc = (1 - hum) * (1 - mod)
        chance = chance_acc + chance_inacc
        c_obs = _c_obs(trials)
        c_err = (c_obs - chance) / (ceil - chance) if ceil > chance else np.nan
        if weighted:
            weight = ceil * len(trials) if ceil > chance else np.nan
            return c_err, weight
        return c_err

    def _c_inacc(trials):
        both_inacc = len(trials[(trials.human_performance == 0) &
                                (trials.model_performance == 0)])
        if both_inacc:
            c_inacc = (len(trials[
                              (trials.human_performance == 0) &
                              (trials.human_prediction ==
                               trials.model_prediction)]) / both_inacc)
            return c_inacc
        else:
            return np.nan

    def _measure_consistencies(trials, metrics, groupby=[]):

        df = pd.DataFrame()
        if 'c_obs' in metrics:
            df['c_obs'] = [_c_obs(trials)]
        if 'c_err' in metrics:
            if groupby:
                df['c_err'] = trials.groupby(groupby).apply(_c_err)
            else:
                df['c_err'] = [_c_err(trials)]
        if 'c_err_weighted' in metrics:
            temp = trials.groupby(groupby).apply(_c_err, weighted=True)
            c_errs = [i[0] for i in temp]
            weights = [i[1] for i in temp]
            weights_sqrt = np.sqrt(weights)
            weights_normed = weights_sqrt / np.nanmean(weights_sqrt)
            c_errs_weighted = c_errs * weights_normed
            df['c_err_weighted'] = c_errs_weighted
        if 'c_inacc' in metrics:
            df['c_inacc'] = [_c_inacc(trials)]
        df = df.reset_index().melt(
            id_vars=groupby, value_vars=metrics, var_name='metric_sim')
        if groupby:  # average across conditions
            df = df.groupby('metric_sim').agg({'value': 'mean'}).reset_index()
        df['level'] = 'trial-wise'
        df['within'] = '_x_'.join(['subject'] + groupby)
        df['between'] = 'trial'
        return df

    def _accuracy_distance(trials):
        acc_hum = trials.human_performance.mean()
        acc_mod = trials.model_performance.mean()
        max_dist = max(acc_hum, 1 - acc_hum)
        acc_dist = np.abs(acc_hum - acc_mod) / max_dist
        return acc_dist

    def _curve_correlation(trials, unocc_subject, unocc_model, norm_curve=None):
        trials_agg = trials.groupby('visibility').mean(numeric_only=True)
        curve_subject = trials_agg.human_performance.to_list() + [unocc_subject]
        curve_model = trials_agg.model_performance.to_list() + [unocc_model]
        if norm_curve is not None:
            curve_subject = np.array(curve_subject) - norm_curve
            curve_model = np.array(curve_model) - norm_curve
        corr = np.corrcoef(curve_subject, curve_model)[0, 1]
        return corr

    human_likeness = pd.DataFrame()

    # combine model and human data
    trials_h = (trials_human
        .copy(deep=True)
        .rename(columns={'accuracy': 'human_performance',
                         'prediction': 'human_prediction'}))
    trials_m = (trials_model
        .copy(deep=True)
        .rename(columns={'value': 'model_performance',
                         'prediction': 'model_prediction'}))
    mutual_cols = list(np.intersect1d(trials_h.columns,
                                      trials_m.columns))
    trials = trials_h.merge(trials_m, on=mutual_cols,
                                suffixes=('_h', '_m'))

    # trial-wise metrics
    if trials_model.name[-1] != 'accuracy':
        return pd.DataFrame()

    # all trials, all metrics (except c_err_weighted to keep subject parity)
    df = trials.groupby('subject').apply(_measure_consistencies,
        metrics=['c_obs', 'c_err', 'c_inacc']).reset_index('subject')
    human_likeness = pd.concat([human_likeness, df])

    # measure c_err(_weighted) at successively finer condition levels
    groupby = []
    for var in ['visibility', 'occluder_class', 'occluder_color']:
        groupby.append(var)
        df = (trials
            .groupby('subject')
            .apply(_measure_consistencies, groupby=groupby,
                   metrics=['c_err', 'c_err_weighted'])
            .reset_index('subject'))
        human_likeness = pd.concat([human_likeness, df])

    # condition-wise metrics
    max_groupby = ['occluder_class', 'occluder_color', 'visibility']
    for subject in trials.subject.unique():

        # all and occluded-only trials, unocc performance for subject
        trials_subject = trials[trials.subject == subject]
        trials_subject_occ = (trials_subject[trials_subject.visibility < 1]
            .groupby(max_groupby)
            .agg('mean', numeric_only=True)
            .reset_index())
        unocc_subject = (trials_subject[trials_subject.visibility == 1]
            .human_performance.mean())

        # all and occluded-only trials, unocc performance for rem grp and model
        trials_rem_grp = trials[trials.subject != subject]
        trials_rem_grp_occ = (trials_rem_grp[trials_rem_grp.visibility < 1]
            .groupby(max_groupby)
            .agg('mean', numeric_only=True)
            .reset_index())
        unocc_rem_grp = (trials_rem_grp[trials_rem_grp.visibility == 1]
            .human_performance.mean())
        unocc_model = (trials_rem_grp[trials_rem_grp.visibility == 1]
            .model_performance.mean())

        # mean visibility-accuracy performance curve for remaining groupby
        mean_curve_rem_grp = (trials_rem_grp_occ
            .groupby('visibility')
            .agg('mean', numeric_only=True)
            .sort_values(by='visibility')
            .human_performance.to_list()
            + [unocc_rem_grp])

        # metrics at different levels of condition resolution
        groupby = []
        for var in max_groupby:
            groupby.append(var)
            between = [i for i in max_groupby if i not in groupby]

            # accuracy distance
            subject_trials = trials_subject[groupby + ['human_performance']]
            model_trials = trials_rem_grp[groupby + ['model_performance']]
            both = subject_trials.merge(model_trials, on=groupby)
            value = (both
                  .groupby(groupby)
                  .apply(_accuracy_distance)
                  .agg('mean', numeric_only=True))
            human_likeness = pd.concat([human_likeness, pd.DataFrame(dict(
                subject=[subject],
                level=['condition-wise'],
                within=['_x_'.join(['subject'] + groupby)],
                between=['_x_'.join(between)],
                metric_sim=['accuracy_distance'],
                value=[value]))])

            # condition-wise accuracy correlation (all 91 conditions
            # including unoccluded)
            if 'visibility' in groupby:
                value = np.corrcoef(both.human_performance,
                                    both.model_performance)[0, 1]
                human_likeness = pd.concat([human_likeness, pd.DataFrame(dict(
                    subject=[subject],
                    level=['condition-wise'],
                    within=['_x_'.join(['subject'] + groupby)],
                    between=['_x_'.join(between)],
                    metric_sim=['cond_pearson_r_unocc'],
                    value=[value]))])

            # all other metrics require separating occluded and unoccluded
            subject_trials = trials_subject_occ[groupby + ['human_performance']]
            model_trials = trials_rem_grp_occ[groupby + ['model_performance']]
            both = (subject_trials
                .merge(model_trials, on=groupby)
                .dropna()
                .groupby(groupby)
                .agg('mean', numeric_only=True))

            # condition-wise accuracy correlation (occluded conditions only)
            value = np.corrcoef(both.human_performance,
                               both.model_performance)[0,1]
            human_likeness = pd.concat([human_likeness, pd.DataFrame(dict(
                subject=[subject],
                level=['condition-wise'],
                within=['_x_'.join(['subject'] + groupby)],
                between=['_x_'.join(between)],
                metric_sim=['cond_pearson_r'],
                value=[value]))])

            # 2 metrics that require all visibilities
            if 'visibility' not in groupby:

                # ensure visibility column is retained
                subject_trials = trials_subject_occ[
                    groupby + ['visibility', 'human_performance']]
                model_trials = trials_rem_grp_occ[
                    groupby + ['visibility', 'model_performance']]
                both = (subject_trials
                    .merge(model_trials,  on=groupby + ['visibility'])
                    .dropna())

                # correlation across visibility-accuracy curve
                value = (both
                    .groupby(groupby)
                    .apply(_curve_correlation, unocc_subject, unocc_model)
                    .agg('mean', numeric_only=True))
                human_likeness = pd.concat([human_likeness, pd.DataFrame(dict(
                    subject=[subject],
                    level=['condition-wise'],
                    within=['_x_'.join(['subject'] + groupby)],
                    between=['_x_'.join(between)],
                    metric_sim=['curve_pearson_r'],
                    value=[value]))])

                # correlation across normed curve (subtract mean curve x rem grp)
                value = (both
                    .groupby(groupby)
                    .apply(_curve_correlation, unocc_subject,
                           unocc_model, mean_curve_rem_grp)
                    .agg('mean', numeric_only=True))
                human_likeness = pd.concat([human_likeness, pd.DataFrame(dict(
                    subject=[subject],
                    level=['condition-wise'],
                    within=['_x_'.join(['subject'] + groupby)],
                    between=['_x_'.join(between)],
                    metric_sim=['curve_norm_pearson_r'],
                    value=[value]))])

    human_likeness = (human_likeness
        .sort_values(by=['level', 'metric_sim', 'subject'])
        .reset_index(drop=True))
    human_likeness['metric_human'] = 'accuracy'
    human_likeness['metric_model'] = trials_model.name[-1]

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
                        layers=('output'), overwrite=False, remake_plots=False):

    results_dir = op.join(MODEL_BASE, model_dir, RES_DIR, 'exp1')
    mod_str = f'model {m + 1}/{total_models} at {model_dir}'
    recompare_models = False
    groupby = ['layer', 'cycle', 'metric']

    # fit performance curves
    curves_path = f'{results_dir}/robustness_curves.parquet'
    existing_curves = existing_results(curves_path, layers, overwrite)
    if existing_curves is not True:
        print(f'{now()} | Analysing performance curves (exp1) | {mod_str}')
        trials_model = reshape_metrics(
            load_trials(model_dir=model_dir), shape='long')
        curves = (trials_model
                  .groupby(groupby)
                  .apply(fit_visibility_curves)
                  .reset_index(groupby))
        curves = pd.concat([existing_curves, curves]).reset_index(drop=True)
        curves.to_parquet(curves_path, index=False)
        recompare_models, remake_plots = True, True

    # measure human likeness
    likeness_path = f'{results_dir}/human_likeness.parquet'
    existing_likeness = existing_results(likeness_path, layers, overwrite)
    if existing_likeness is not True:
        print(f'{now()} | Analysing human likeness (exp1) | {mod_str}')
        trials_human = load_trials(drop_human=False)
        trials_model = reshape_metrics(load_trials(model_dir=model_dir),
                                       shape='long')

        # full dataset
        likeness = (trials_model
            .groupby(groupby)
            .apply(measure_human_likeness, trials_human)
            .reset_index(level=groupby[:-1]))
        likeness['dataset'] = 'all'

        # without the natural occluder type
        trials_model_artificial = trials_model[
            trials_model.occluder_class != 'naturalUntexturedCropped2']
        trials_human_artificial = trials_human[
            trials_human.occluder_class != 'naturalUntexturedCropped2']
        likeness_a = (trials_model_artificial
          .groupby(groupby)
          .apply(measure_human_likeness, trials_human_artificial)
          .reset_index(level=groupby[:-1]))
        likeness_a['dataset'] = 'artificial'
        likeness = pd.concat([likeness, likeness_a]).reset_index(drop=True)

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
        robustness.rename(columns={'value': metric}, inplace=True)
        condwise_robustness_plot_array(
            df=robustness,
            df_curves=curves,
            metric=metric,
            outpath=outpath,
            ylabel=plot_cfg['ylabel'],
            yticks=plot_cfg['yticks'],
            ylims=plot_cfg['ylims'],
            chance=plot_cfg['chance'],
            legend_path=outpath.replace(metric, 'legend'))


    def _make_scatterplots(robustness_m, robustness_h, info):

        layer, cycle, metric = info
        outpath = (
            f'{plot_dir}/cyc{cycle:02}_{layer}_{metric}_human-likeness.svg')
        colors = EXP1.plot_colors
        metric_human = 'accuracy'
        rob_h = robustness_h.rename(columns={metric_human: f'human_value'})
        rob_m = robustness_m.rename(columns={'value': f'model_value'})
        rob_h = (rob_h[
                     rob_h.occluder_class.isin(OCC_CLASSES)]
                 .groupby(['occluder_class', 'occluder_color'])
                 .agg({'human_value': 'mean'}))
        rob_m = (rob_m[
                     rob_m.occluder_class.isin(OCC_CLASSES)]
                 .groupby(['occluder_class', 'occluder_color'])
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


    results_dir = op.join(MODEL_BASE, model_dir, RES_DIR, 'exp1')
    plot_dir = f'{results_dir}/plots'
    os.makedirs(plot_dir, exist_ok=True)
    trials_h = load_trials()
    trials_m = load_trials(model_dir=model_dir)
    curves = pd.read_parquet(f'{results_dir}/robustness_curves.parquet')

    # calculate mean performance for each condition

    # model
    groupby = ['layer', 'cycle', 'occluder_class', 'occluder_color',
               'metric', 'visibility']
    robustness_m = (reshape_metrics(trials_m, 'long').groupby(groupby)
                    .agg({'value': 'mean'}).dropna().reset_index())

    # human
    groupby = ['occluder_class', 'occluder_color', 'visibility']
    robustness_h = (trials_h.groupby(groupby)
                    .agg('mean', numeric_only=True).reset_index())

    # condition-wise scatter_plots, model vs human
    for rob_m in robustness_m.groupby(['layer', 'cycle', 'metric']):
        _make_scatterplots(rob_m[1], robustness_h, rob_m[0])

    # performance curves, model only
    trials_m.occluder_class = (
        trials_m.occluder_class.cat.add_categories('unoccluded'))
    trials_m.occluder_color = (
        trials_m.occluder_color.cat.add_categories('unoccluded'))
    trials_m.occluder_class[trials_m.visibility == 1] = 'unoccluded'
    trials_m.occluder_color[trials_m.visibility == 1] = 'unoccluded'
    trials_m.occluder_class = pd.Categorical(
        trials_m.occluder_class, OCC_CLASSES + ['unoccluded'],
        ordered=True)
    trials_m = reshape_metrics(trials_m, 'long')
    groupby = ['layer', 'cycle', 'metric']
    for rob, cur in zip(trials_m.groupby(groupby), curves.groupby(groupby)):
        _make_curve_plots(rob[1], cur[1], rob[0])



