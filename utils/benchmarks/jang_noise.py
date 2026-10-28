'''
This script tests the accuracy of ANNs on fourier and gaussian noise
'''

import os
import os.path as op
import glob
import gc
import numpy as np
import pandas as pd
from types import SimpleNamespace
from itertools import product as itp
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from PIL import Image
import pickle as pkl
from scipy import special
from scipy import io
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning
import warnings
import h5py
import torch
import torchvision.transforms.v2 as transforms
import torchvision
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import float32
from utils import now, insert_cycle, MODEL_BASE
from utils.accuracy import accuracy
from utils.AverageMeter import AverageMeter
from utils.math_functions import sigmoid
from utils.get_activations import get_activations
from utils.get_trained_model import get_trained_model
from utils.Noise import Noise
from utils.CustomDataset import CustomDataset
from utils.get_transform import get_transform
from utils.load_benchmark_scores import load_benchmark_scores

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

np.random.seed(42)

DATASET_BASE = op.expanduser('~/Datasets/Jang-Noise/behavioral')
NOISE_TYPES = ['gaussian', 'fourier']
SSNRS = np.arange(0.05, 1.01, 0.05)

#HUMAN_RSM = np.load(f'{DATASET_BASE}/fmri/human_RSMs.npy')
#ROIS = ['V1', 'V2', 'V3', 'V4', 'LOC', 'FFA', 'PPA', 'EVC', 'HVC']
RES_DIR = 'benchmarking/noise_behavioral'
RAW_TRIALS = pd.read_parquet(f'{DATASET_BASE}/human/trials_fig2.parquet')
OBJ_CLASSES = sorted(list(set([i.split('/')[1] for i in RAW_TRIALS.image_path.unique()])))
IMAGENET_DIR = '/home/david/Datasets/ILSVRC2012/val'
IMAGENET_LABELS = pd.read_csv(f'{op.dirname(IMAGENET_DIR)}/labels.csv')
SYNSETS = [IMAGENET_LABELS[IMAGENET_LABELS.label == i].directory.item() for i in OBJ_CLASSES]
OBJ_CLS_IDCS = [IMAGENET_LABELS[IMAGENET_LABELS.label == i].index.item() for i in OBJ_CLASSES]
CAT_TO_SYN = lambda i: SYNSETS[OBJ_CLASSES.index(i)]
CAT_TO_IDX = lambda i: OBJ_CLS_IDCS[OBJ_CLASSES.index(i)]
METRICS = ['accuracy', 'true_class_prob']  # always available
METRICS_ALL = METRICS


def make_jang_noise_dataset(overwrite=False, num_procs=1):

    """ This dataset contains several benchmarks across behavioral and neural domains from the Jang and Tong 2021 PLOS
    Comp. Bio. paper. It is different to the ImageNet-Noise dataset in several ways, such as using a 16-AFC task and
    grayscale rather than color images.

    Behavioral benchmarks
    The underlying DNN inputs are the same for both benchmarks. They need to be evaluated on 800 object images applied
    with Fourier or Gaussian noise at 20 different SSNRs.

    1. In the first experiment, humans classified individual images at different levels of Fourier or Gaussian noise.
    In the corresponding benchmark, DNN accuracies are averaged within each noise type and SSNR, then two SSNR-accuracy
    curves are fit. These are compared with corresponding curves calculated for each subject based on the 50% accuracy
    threshold. This is a robustness comparison in that indiviudal thresholds are compared between humans and DNNs.

    2. In the second experiment, human participants observed a video of each unique object gradually emerging from a
    Gaussian noise pattern, then responded when they were confident they could identify the object. In the
    corresponding benchmark, DNN accuracies are fitted with an SSNR-accuracy curve for each unique object image, then
    90% accuracy thresholds are calculated. These trial-level SSNRs are then correlated with the SSNRs at which humans
    responded, which is a finer-grained measure of whether humans and DNNs found the same images to be more or less
    difficult to recognize.

    """
    """ noise_ceiling for fig 5 """
    if not op.isfile(f'{DATASET_BASE}/human/noise_ceiling_fig5.csv') or overwrite:
        trials = pd.read_parquet(f'{DATASET_BASE}/human/trials_fig5.parquet')
        nc = pd.DataFrame()
        for subject in trials.subject_id.unique():

            # get accurate trials for this noise type
            df_noise = trials[trials.accuracy == 1]

            # exclude trials where too few subjects responded correctly
            df_noise = (df_noise
                #.groupby('trial_id', dropna=False)
                #.filter(lambda g: len(g) >= 9)
                .sort_values('trial_id'))

            # subject data
            subject_trials = df_noise[df_noise.subject_id == subject]

            # lower bound: individual vs remaining group
            rem_group_trials = (df_noise
                [df_noise.subject_id != subject]
                .groupby('trial_id', observed=True)
                .agg({'ssnr': 'mean'})
                .reset_index())
            ind_v_rem_group = subject_trials.merge(rem_group_trials, on='trial_id', suffixes=('_ind', '_group'))
            lwr = np.corrcoef(ind_v_rem_group.ssnr_ind, ind_v_rem_group.ssnr_group)[0, 1]

            # upper bound: individual vs entire group
            group_trials = (df_noise
                .groupby('trial_id', observed=True)
                .agg({'ssnr': 'mean'})
                .reset_index())
            ind_v_group = subject_trials.merge(group_trials, on='trial_id', suffixes=('_ind', '_group'))
            upr = np.corrcoef(ind_v_group.ssnr_ind, ind_v_group.ssnr_group)[0, 1]

            # add to dataframe and save out
            nc = pd.concat([nc, pd.DataFrame({
                'subject_id': [subject],
                'lower': [lwr],
                'upper': [upr]
            })])
        nc.to_csv(f'{DATASET_BASE}/human/noise_ceiling_fig5.csv', index=False)


    transform = transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(float32, scale=True),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),
    ])

    # get clear images from imagenet, apply cropping / greyscale, save out
    image_paths = sorted(list(RAW_TRIALS.image_path.unique()))

    clear_dir = f'{DATASET_BASE}/objects'
    for image_path in image_paths:
        image_path = image_path.replace('JPEG', 'jpg')
        category = image_path.split('/')[1]
        out_subdir = op.join(clear_dir, category)
        basename = op.basename(image_path).split('.')[0]
        out_path = f'{out_subdir}/{basename}.png'
        if not op.isfile(out_path) or overwrite:
            print(image_path)
            os.makedirs(out_subdir, exist_ok=True)
            object_path = op.join(IMAGENET_DIR, CAT_TO_SYN(category), op.basename(image_path))
            object_img = transform(Image.open(object_path))
            save_image(object_img, out_path)
    image_paths = sorted(glob.glob(f'{clear_dir}/*/*.png'))
    assert len(image_paths) == 800

    # get Fourier amplitude spectrum of clear images
    magnitude_path = f'{clear_dir}/magnitude.pt'
    if op.isfile(magnitude_path) and not overwrite:
        magnitude = torch.load(magnitude_path)
    else:
        dataset = CustomDataset(image_paths, transform=transform)
        loader = DataLoader(dataset, batch_size=50, shuffle=False,
                            num_workers=num_procs)
        magnitudes = []
        with tqdm(loader) as tepoch:
            for images in tepoch:
                magnitudes_batch = []
                for image in images:
                    magnitudes_batch.append(torch.fft.fft2(image.mean(0)))
                magnitude_batch = torch.stack(magnitudes_batch).abs().mean(0)
                magnitudes.append(magnitude_batch)
        magnitude = torch.stack(magnitudes).mean(0)
        torch.save(magnitude, magnitude_path)


    def _apply_noise(image, noise, ssnr):
        image, noise = image.clone(), noise.clone()
        signal = (image - 0.5) * ssnr + 0.5
        noise = noise * (1 - ssnr)
        image = torch.clamp(signal + noise, 0, 1)
        return image


    def _make_dataset(noise_type, image_path):

        print(noise_type, image_path)

        # image properties
        category = image_path.split('/')[-2]
        basename = op.basename(image_path).split('.')[0]

        # get clear image
        object_img = transform(Image.open(image_path))

        # make noise image
        if noise_type == 'gaussian':
            noise = torch.normal(0, 1 / 6, object_img.shape[1:])
        elif noise_type == 'fourier':
            image_fft = np.fft.fft2(object_img[0])
            image_fft_phase = np.angle(image_fft)
            # np.random.shuffle(image_fft_phase) # wrong! only does 1st dim
            np.random.shuffle(image_fft_phase.flat)  # correct way to do it
            image_fft_shuffled = np.multiply(
                magnitude, np.exp(1j * image_fft_phase))
            noise = abs(np.fft.ifft2(image_fft_shuffled))
            noise = (noise - np.min(noise)) / (
                    np.max(noise) - np.min(noise))
            noise = torch.tensor(noise) - 0.5
        noise = torch.stack([noise] * 3, 0)

        # apply noise at each ssnr and save out
        out_subdir = op.join(DATASET_BASE, noise_type, category)
        os.makedirs(out_subdir, exist_ok=True)
        for ssnr in SSNRS[::-1]:
            noisy_img = _apply_noise(object_img, noise, ssnr)
            out_path = f'{out_subdir}/{basename}_SSNR-{ssnr:.2f}.png'
            if not op.isfile(out_path) or overwrite:
                save_image(noisy_img, out_path)

    Parallel(n_jobs=num_procs)(
        delayed(_make_dataset)(n, s) for n, s in itp(NOISE_TYPES, image_paths)
    )

    # make empty trials df
    if not op.isfile(f'{DATASET_BASE}/trials.parquet') or overwrite:
        trials = pd.DataFrame()
        image_paths = list_images()
        object_ids = list(set([op.basename(i).split('_SSNR')[0] for i in image_paths]))
        for i, image_path in enumerate(image_paths):
            noise_type, obj_class, img_basename = image_path.split('/')[-3:]
            object_id = img_basename.split('_SSNR')[0]
            trial_id = object_ids.index(object_id)
            ssnr = np.float16(img_basename[-8:-4])
            trials = pd.concat([trials, pd.DataFrame({
                'trial_id': [trial_id],
                'image_id': [i],
                'image_path': [image_path.split('behavioral/')[-1]],
                'noise_type': [noise_type],
                'object_class': [obj_class],
                'object_synset': [CAT_TO_SYN(obj_class)],
                'object_idx': [CAT_TO_IDX(obj_class)],
                'ssnr': [f'{ssnr:.2f}'],
            })])

        # enforce ordering of categorical variables
        trials.object_class = pd.Categorical(
            trials.object_class, OBJ_CLASSES, ordered=True)
        trials.noise_type = pd.Categorical(
            trials.noise_type, NOISE_TYPES, ordered=True)
        
        trials.to_parquet(f'{DATASET_BASE}/trials.parquet')


    # order human trials by trial_id
    human_ordered_path = f'{DATASET_BASE}/human/trials_fig2.parquet'
    if not op.isfile(human_ordered_path) or overwrite:
        trials_ordered = pd.read_parquet(f'{DATASET_BASE}/trials.parquet')
        trials_ordered = (trials_ordered
            [trials_ordered.ssnr == "1.00"]
            .sort_values(['noise_type', 'trial_id'], ascending=[True, True])
            .drop(columns=['image_id', 'ssnr'])
        )
        human_fourier = pd.read_csv(f'{DATASET_BASE}/human/human_fourier.csv')
        human_fourier['noise_type'] = 'fourier'
        human_gaussian = pd.read_csv(f'{DATASET_BASE}/human/human_gaussian.csv')
        human_gaussian['noise_type'] = 'gaussian'
        human_trials = pd.concat([human_fourier, human_gaussian])
        human_ordered = pd.DataFrame()
        for _, row in trials_ordered.iterrows():
            print(row.noise_type, row.trial_id)
            for subject in human_trials.subject_id.unique():
                image_id = op.basename(row.image_path).split('_SSNR')[0]
                human_row = human_trials[
                    (human_trials.subject_id == subject) &
                    (human_trials.noise_type == row.noise_type) &
                    (human_trials.image_path.str.contains(image_id))
                ]
                assert len(human_row) == 1, f'Expected 1 row, got {len(human_row)}'
                new_row = pd.DataFrame({
                    'subject_id': subject,
                    'trial_id': row.trial_id,
                    'image_path': row.image_path,
                    'noise_type': row.noise_type,
                    'object_class': row.object_class,
                    'object_synset': row.object_synset,
                    'object_idx': row.object_idx,
                    'ssnr': human_row.SSNR,
                    'accuracy': human_row.correct,
                })
                human_ordered = pd.concat([human_ordered, new_row])
        human_ordered.to_parquet(human_ordered_path)


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


def load_trials(model_dir=None, overwrite=False):

    trials_path = op.join(MODEL_BASE, model_dir, RES_DIR, 'trials.parquet')
    if not op.isfile(trials_path) or overwrite:
        trials = pd.read_parquet(f'{DATASET_BASE}/trials.parquet')
    else:
        trials = pd.read_parquet(trials_path)

    return trials


def list_images():
    image_data = pd.read_parquet(f'{DATASET_BASE}/trials.parquet')
    images = [f'{DATASET_BASE}/{i}' for i in image_data.image_path]
    return images


@torch.no_grad()
def get_responses(model_dir, architecture, layers, batch_size, m=0,
                  total_models=0, num_procs=1, overwrite=False):

    results_dir = op.join(MODEL_BASE, model_dir, RES_DIR)
    os.makedirs(results_dir, exist_ok=True)

    out_path = f'{results_dir}/trials.parquet'
    if not op.isfile(out_path):
        prev_trials = pd.DataFrame()
    else:
        prev_trials = pd.read_parquet(out_path)
        if 'weights' in prev_trials.columns:
            prev_trials = prev_trials.drop(columns=['weights'])
            prev_trials.to_parquet(out_path)
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

    print(f'{now()} | Measuring responses for noise stimuli, '
          f'model: {m + 1}/{total_models} at {model_dir}, '
          f'layers: {layers}')

    # get responses to test images
    images = list_images()
    sampler = np.random.permutation(len(images))
    trials = load_trials(model_dir)
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
        trials_batch = pd.concat([trials[trials.image_id == i] for i in batch_ids])
        inputs = [images[i] for i in batch_ids]
        trials_batch = _process_batch(model, architecture, inputs, trials_batch)
        gc.collect()
        new_trials = pd.concat([new_trials, trials_batch])

    # reorder based on layer and cycle
    new_trials = new_trials \
        .sort_values(by=['layer', 'cycle', 'image_id']) \
        .reset_index(drop=True)

    # print out accuracy for each layer and cycle
    new_trials.groupby(['layer', 'cycle'], observed=False).apply(
        lambda df: print(
            f'{now()} | {df.name} accuracy: {df.accuracy.mean():.4}'))

    # reformat to combine all metric columns into long format
    new_trials = reshape_metrics(new_trials, 'long')

    # save trials
    all_trials = pd.concat([prev_trials, new_trials]).reset_index(drop=True)
    all_trials.to_parquet(out_path, index=False)

    return True


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
    assert (abs(probs.sum(1) - 1) <= .01).all(), 'probabilities do not sum to 1'

    # add predictions and other measures to trials
    trials['prediction'] = [classes_ordered[c] for c in probs.argmax(axis=1)]
    trials['accuracy'] = pd.Series(
        trials.prediction == trials.object_class, dtype=int)
    trials['true_class_prob'] = [probs[i, classes_ordered.index(c)]
                                 for i, c in enumerate(trials.object_class)]

    return trials


def fit_visibility_curves(trials):

    trials = (trials.copy()
              .groupby('ssnr').agg('mean', numeric_only=True)
              .sort_values('ssnr', ascending=True))
    assert len(trials) == len(SSNRS), 'missing / additional ssnr values'

    # Don't fit curves if performance is too high for noisiest image
    # Note: accuracy is binary for individual trials, so threshold is tailored to
    # true class probability and average accuracies
    if trials.value[0] > .5:
        popt, threshold_50, threshold_90 = [np.nan] * 4, np.nan, np.nan
    else:
        try:
            init_params = [max(trials.value), np.median(SSNRS), 1, 0]
            popt, pcov = curve_fit(
                sigmoid, SSNRS, trials.value, init_params, maxfev=int(10e5))
            curve = sigmoid(np.linspace(0, 1, 1000), *popt)
            threshold_50 = sum(curve < .5) / 1000
            threshold_90 = sum(curve < .9) / 1000
        except:
            UserWarning('Curve fitting failed, returning NaNs')
            popt, threshold_50, threshold_90 = [np.nan] * 4, np.nan, np.nan

    df = pd.DataFrame({
            'L': [popt[0]],
            'x0': [popt[1]],
            'k': [popt[2]],
            'b': [popt[3]],
            'thr-50': [threshold_50],
            'thr-90': [threshold_90],
            'mean': [trials.value.mean()]
        })
    return df


def analyse_performance(model_dir, m=0, total_models=0, layers=('output'),
                        overwrite=False, remake_plots=False):

    results_dir = op.join(MODEL_BASE, model_dir, RES_DIR)
    mod_str = f'model {m + 1}/{total_models} at {model_dir}'
    groupby = ['layer', 'cycle', 'metric', 'noise_type', 'object_class', 'trial_id']

    """" fit performance curves """
    # trial
    curves_path = f'{results_dir}/robustness_curves.parquet'
    if not op.isfile(curves_path) or overwrite:
        print(f'{now()} | Analysing performance curves (trial) | {mod_str}')
        trials_model = reshape_metrics(
            load_trials(model_dir=model_dir), shape='long')
        curves = (trials_model
                  .groupby(groupby, observed=False)
                  .apply(fit_visibility_curves)
                  .sort_values(groupby)
                  .reset_index(groupby))
        curves = curves.sort_values(groupby).reset_index(drop=True)
        curves.to_parquet(curves_path, index=False)

    # category
    groupby_class = [i for i in groupby if i != 'trial_id']
    curves_path_class = f'{results_dir}/robustness_curves_object_class.parquet'
    if not op.isfile(curves_path_class) or overwrite:
        print(f'{now()} | Analysing performance curves (object_class) | {mod_str}')
        trials_model = reshape_metrics(
            load_trials(model_dir=model_dir), shape='long')
        curves_class = (trials_model
                  .groupby(groupby_class, observed=False)
                  .apply(fit_visibility_curves)
                  .reset_index(groupby_class))
        curves_class.to_parquet(curves_path_class, index=False)
    else:
        curves_class = pd.read_parquet(curves_path_class)

    # noise type level
    curves_path_noise = f'{results_dir}/robustness_curves_noise_type.parquet'
    if not op.isfile(curves_path_noise) or overwrite:
        print(f'{now()} | Analysing performance curves (noise_type) | {mod_str}')
        trials_model = reshape_metrics(
            load_trials(model_dir=model_dir), shape='long')
        curves_noise = (trials_model
                        .groupby(['layer', 'cycle', 'metric', 'noise_type'], observed=False)
                        .apply(fit_visibility_curves)
                        .reset_index(['layer', 'cycle', 'metric', 'noise_type']))
        curves_noise.to_parquet(curves_path_noise, index=False)
    else:
        curves_noise = pd.read_parquet(curves_path_noise)

    # plot curves
    outpaths = [f'{results_dir}/plots/{noise_type}_{metric}.png' for \
                    noise_type, metric in itp(NOISE_TYPES, METRICS)]
    if not all([op.isfile(i) for i in outpaths]) or overwrite:
        trials = reshape_metrics(
            load_trials(model_dir=model_dir), shape='long')
        curves = pd.read_parquet(curves_path)
        curves_class = pd.read_parquet(curves_path_class)

    for noise_type, metric in itp(NOISE_TYPES, METRICS):
        outpath = f'{results_dir}/plots/{noise_type}_{metric}.png'
        if op.isfile(outpath) and not overwrite:
            continue

        os.makedirs(f'{results_dir}/plots', exist_ok=True)
        fig, axes = plt.subplots(
            4, 4, figsize=(8, 8), sharex=True, sharey=True)

        for o, object_class in enumerate(OBJ_CLASSES):
            ax = axes.flatten()[o]
            trials_class = trials[trials.object_class == object_class]
            curve_x = np.linspace(0, 1, 1000)

            """ image-level performance """
            for trial_id in trials_class.trial_id.unique():
                popt = (curves[
                    (curves.noise_type == noise_type) &
                    (curves.metric == metric) &
                    (curves.trial_id == trial_id)]
                    [['L', 'x0', 'k', 'b']]).values[0]
                if np.isfinite(popt).all():

                    # plot curve function
                    curve_y = sigmoid(curve_x, *popt)
                    ax.plot(curve_x, curve_y, color='tab:gray', zorder=1)

                    # plot accuracies
                    yvals = trials[
                        (trials.trial_id == trial_id) &
                        (trials.noise_type == noise_type) &
                        (trials.metric == metric)].sort_values('ssnr', ascending=True).value.values
                    ax.scatter(SSNRS, yvals, color='tab:gray', clip_on=False, zorder=2)

            """ category-level performance"""
            popt = (curves_class[
                (curves_class.noise_type == noise_type) &
                (curves_class.metric == metric) &
                (curves_class.object_class == object_class)]
                [['L', 'x0', 'k', 'b']]).values[0]
            if np.isfinite(popt).all():

                # plot curve function
                curve_y = sigmoid(curve_x, *popt)
                ax.plot(curve_x, curve_y, color='tab:green', zorder=3, clip_on=False)

                # plot accuracies
                yvals = (trials_class
                         .groupby('ssnr', observed=False)
                         .agg('mean', numeric_only=True)
                         .sort_values('ssnr', ascending=True)).value.values
                ax.scatter(SSNRS, yvals, color='tab:green', zorder=4, clip_on=False)

            # format plot
            ax.set_title(object_class.split(',')[0], size=7)
            ax.set_xticks((0, 1))
            ax.set_xlim((0, 1))
            ax.set_yticks((0, 1))
            ax.set_ylim((0, 1))
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.axhline(y=1/16, color='k', linestyle='dotted')
            if o > 11:
                ax.set_xlabel('SSNR', size=10)
            if o % 4 == 0:
                ax.set_ylabel(metric.replace('_', ' ').capitalize(), size=10)
        plt.suptitle(f'{noise_type.capitalize()} noise, {metric.replace("_", " ").capitalize()}', size=10)
        plt.tight_layout()
        plt.savefig(outpath)
        plt.savefig(outpath.replace('.png', '.pdf'))
        plt.close()

        """ noise type level performance """
        fig, ax = plt.subplots(figsize=(4, 4))
        curve_x = np.linspace(0, 1, 1000)
        popt = curves_noise[
            (curves_noise.noise_type == noise_type)][['L', 'x0', 'k', 'b']].values[0]
        if np.isfinite(popt).all():
            # plot curve function
            curve_y = sigmoid(curve_x, *popt)
            ax.plot(curve_x, curve_y, color='tab:red', zorder=1)

        # plot accuracies
        yvals = (trials[(trials.noise_type == noise_type) &
                        (trials.metric == metric)]
                 .groupby('ssnr', observed=False)
                 .agg('mean', numeric_only=True)
                 .sort_values('ssnr', ascending=True)).value.values
        ax.scatter(SSNRS, yvals, color='tab:red', clip_on=False, zorder=2)

        # format plot
        ax.set_title(f'{noise_type.capitalize()} noise, {metric.replace("_", " ").capitalize()}', size=7)
        ax.set_xticks((0, 1))
        ax.set_xlim((0, 1))
        ax.set_yticks((0, 1))
        ax.set_ylim((0, 1))
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.axhline(y=1 / 16, color='k', linestyle='dotted')
        ax.set_xlabel('SSNR', size=10)
        ax.set_ylabel(metric.replace('_', ' ').capitalize(), size=10)
        plt.tight_layout()
        plt.savefig(outpath)
        plt.savefig(outpath.replace('.png', '.pdf'))
        plt.close()


    """ DNN-human alignment """
    trials = load_trials(model_dir)
    model_curves = pd.read_parquet(curves_path)
    human_ssnr = pd.read_parquet(f'{DATASET_BASE}/human/trials_fig5.parquet')

    def _correlate_thresholds(df):
        model_metric = df.name[2]

        # merge human and model data into single df to ensure trial alignment
        human_model = df.merge(human_ssnr, on='trial_id')
        human_model = human_model[human_model.accuracy == 1].dropna()

        # remove trials for which the model is accurate / high true class probability at the lowest SSNR (0.05)
        selected_trials = trials[
            (trials.metric == model_metric) &
            (trials.ssnr == '0.05') &
            (trials.value < 0.25)].trial_id.values
        human_model = human_model[human_model.trial_id.isin(selected_trials)]

        prefix = {'accuracy': 'acc', 'true_class_prob': 'tcp'}[model_metric]
        corrs_50 = (human_model
                 .groupby('subject_id', observed=True)
                 .apply(lambda x: np.corrcoef(x[f'thr-50'], x.ssnr)[0, 1])
                 .reset_index(name=f'{prefix}_r_thr-50'))
        corrs_90 = (human_model
                 .groupby('subject_id', observed=True)
                 .apply(lambda x: np.corrcoef(x[f'thr-90'], x.ssnr)[0, 1])
                 .reset_index(name=f'{prefix}_r_thr-90'))
        corrs = corrs_50.merge(corrs_90, on='subject_id')

        return corrs

    scores_path = f'{MODEL_BASE}/{model_dir}/benchmarking/benchmark_scores.csv'
    if op.isfile(scores_path):
        scores = pd.read_csv(scores_path)
    else:
        scores = pd.DataFrame({'benchmark': []})
    if not len(scores[scores.benchmark == 'Jang-Noise_Behavioral_Alignment']) or overwrite:
        scores = scores[scores.benchmark != 'Jang-Noise_Behavioral_Alignment']
        df = (model_curves[model_curves.noise_type == 'gaussian']
            .groupby(['layer', 'cycle', 'metric'], observed=False)
            .apply(_correlate_thresholds)
            .reset_index()
            .drop(columns=['metric', 'level_3'])
            .melt(
                id_vars=['layer', 'cycle', 'subject_id'],
                value_vars=['acc_r_thr-50', 'acc_r_thr-90', 'tcp_r_thr-50', 'tcp_r_thr-90'],
                var_name='metric')
            .dropna()
            .rename(columns={'value': 'score'}))
        df['benchmark'] = 'Jang-Noise_Behavioral_Fig5'
        df['path'] = 'Jang-Noise/behavioral'
        df['subject_id'] = [f'subject_{i:02}' for i in df.subject_id]
        df.rename(columns={'subject_id': 'level_1'}, inplace=True)
        scores = pd.concat([scores, df])
        scores.to_csv(scores_path, index=False)

