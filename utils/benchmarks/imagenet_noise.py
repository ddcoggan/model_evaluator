'''
This script tests the accuracy of ANNs on fourier and gaussian noise
'''

import os
import os.path as op
import glob
import numpy as np
import pandas as pd
from types import SimpleNamespace
from itertools import product as itp
from tqdm import tqdm
from joblib import Parallel, delayed
import torch
import torchvision.transforms.v2 as transforms
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import float32
from utils import now, insert_cycle
from utils.accuracy import accuracy
from utils.AverageMeter import AverageMeter
from utils.get_trained_model import get_trained_model
from utils.Noise import Noise
from utils.CustomDataset import CustomDataset
from utils.get_transform import get_transform
from utils.load_benchmark_scores import load_benchmark_scores

np.random.seed(42)

BENCHMARK = 'ImageNet-Noise'
DATASET_BASE = op.expanduser('~/Datasets/ImageNet-Noise')
NOISE_TYPES = ['gaussian', 'fourier']
SSNRS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DATASETS = list(itp(NOISE_TYPES, SSNRS))

def make_imagenet_noise_dataset(overwrite=False, num_procs=1):

    imagenet_dir = '/home/tonglab/Datasets/ILSVRC2012/val'
    out_dir = '/home/tonglab/Datasets/ImageNet-Noise'
    os.makedirs(op.join(out_dir, 'fourier'), exist_ok=True)

    # Measure the average Fourier spectrum of imagenet val images
    mean_magnitude_path = op.join(out_dir, 'fourier/mean_magnitude.pt')
    if not op.isfile(mean_magnitude_path) or overwrite:
        """
        Measure the average Fourier spectrum of a batch of images.
        """
        print(f'{now()} | Measuring mean magnitude of ImageNet images')
        transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(float32, scale=True),
            transforms.Resize(224),
            transforms.CenterCrop(224)])
        images = sorted(glob.glob(f'{imagenet_dir}/*/*'))
        dataset = CustomDataset(images, transform=transform)
        loader = DataLoader(dataset, batch_size=50, shuffle=False,
                            num_workers=num_procs)
        magnitudes = []
        with tqdm(loader) as tepoch:
            for images in tepoch:
                magnitudes_batch = []
                for image in images:
                    magnitudes_batch.append(
                        torch.stack([torch.fft.fft2(channel) for channel in image]))
                magnitude_batch = torch.stack(magnitudes_batch).abs().mean(0)
                magnitudes.append(magnitude_batch)
        magnitude = torch.stack(magnitudes).mean(0)
        torch.save(magnitude, mean_magnitude_path)

    def _make_dataset(noise_type, ssnr):

        ssnr_str = str(ssnr)[:3]
        noise_args = SimpleNamespace(
            type=noise_type, ssnr=ssnr, probability=1,
            mean_magnitude_path=mean_magnitude_path)
        transform = transforms.Compose([
            transforms.ToImage(),
            transforms.ToDtype(float32, scale=True),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            Noise(noise_args)])

        for synset in sorted(glob.glob(f'{imagenet_dir}/*')):
            dataset = CustomDataset(synset, transform=transform)
            loader = DataLoader(dataset, batch_size=50, shuffle=False,
                num_workers=num_procs, multiprocessing_context='fork')
            synset_name = op.basename(synset)
            out_dir_synset = op.join(
                out_dir, noise_type, ssnr_str, synset_name)
            os.makedirs(out_dir_synset, exist_ok=True)
            in_images = sorted(glob.glob(f'{synset}/*'))
            image_names = [op.basename(i) for i in in_images]
            out_images = [op.join(out_dir_synset, i) for i in image_names]
            if not all([op.isfile(i) for i in out_images]) or overwrite:
                print(f'{now()} | {noise_type}, SSNR: {ssnr_str}, {synset_name}')
                for images in loader:
                    for i, image in enumerate(images):
                        save_image(image, out_images[i])

    Parallel(n_jobs=num_procs)(
        delayed(_make_dataset)(n, s) for n, s in itp(NOISE_TYPES, SSNRS)
    )


@torch.no_grad()
def score_model(model_dir, architecture, batch_size, m=0, total_models=0,
                num_procs=1, overwrite=False):

    results, out_path = load_benchmark_scores(
        model_dir, BENCHMARK, overwrite)

    if results.empty:
        subsets_to_run = DATASETS
    else:
        subsets_to_run = [i for i in DATASETS if not len(results[
            (results.level_1 == i[0]) &
            (results.level_2 == str(i[1])[:3])])]

    if not len(subsets_to_run):
        return False

    print(f'{now()} | Measuring performance for ImageNet-Noise, '
          f'model: {m + 1}/{total_models} at {model_dir}')

    model = get_trained_model(model_dir, architecture, True, ['output'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    transform = get_transform(architecture, model_dir)

    for noise_type, ssnr in subsets_to_run:
        ssnr_str = str(ssnr)[:3]
        dataset = ImageFolder(op.join(DATASET_BASE, noise_type, ssnr_str),
                              transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_procs)

        # loop through batches
        with tqdm(loader, unit=f"batch({batch_size})") as tepoch:

            for batch, (inputs, targets) in enumerate(tepoch):

                tepoch.set_description(
                    f'{now()} | {noise_type}, SSNR: {ssnr_str}')

                # put inputs on device
                inputs = inputs.to(device)
                targets = targets.to(device)

                # pass through model with automatic mixed precision for speed
                with torch.autocast(device_type=device.type,
                                    dtype=torch.float16):
                    outputs = model(inputs)
                outputs = insert_cycle(outputs, batch_size=inputs.shape[0])

                # calculate accuracy
                if batch == 0:
                    performance = {k: AverageMeter() for k in outputs}
                for cycle, output in outputs.items():
                    acc = accuracy(output, targets, (1,)
                                   )[0].detach().cpu().item()
                    performance[cycle].update(acc)

                # print last and mean accuracy of final cycle
                tepoch.set_postfix_str(
                    f'acc1: {acc:.4f}({performance[cycle].avg_epoch:.4f})')

        # save results
        for cycle, perf in performance.items():
            new_results = pd.DataFrame({
                    'benchmark': [BENCHMARK],
                    'level_1': [noise_type],
                    'level_2': [ssnr_str],
                    'path': [f'{noise_type}/{ssnr_str}'],
                    'cycle': [int(cycle[3:])],
                    'metric': ['accuracy'],
                    'score': [perf.avg_epoch]
                })
            results = pd.concat([results, new_results]).reset_index(drop=True)
            print(f'{cycle} accuracy: {perf.avg_epoch:.4f}')

        results.to_csv(out_path, index=False)

    return True

