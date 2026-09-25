'''
This script tests the accuracy of ANNs on fourier and gaussian noise
'''

import os.path as op
import sys
import numpy as np
from scipy import special
import pandas as pd
from tqdm import tqdm
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from utils import now, insert_cycle
from utils.AverageMeter import AverageMeter
from utils.get_trained_model import get_trained_model
from utils.load_benchmark_scores import load_benchmark_scores
from utils.get_transform import get_transform
sys.path.append(op.expanduser('~/data/repos'))
from texture_vs_shape_bias.code import probabilities_to_decision

np.random.seed(42)

BENCHMARK = 'texture_vs_shape_bias'
DATASET_BASE = op.expanduser(
    '~/Datasets/texture_vs_shape_bias/style-transfer-preprocessed-512')

class ImageFolderWithPaths(ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder"""
    # Override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # This is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # The image file path
        path = self.imgs[index][0]
        # Make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path, index))
        return tuple_with_path


@torch.no_grad()
def score_model(model_dir, architecture, batch_size,  m=0, total_models=0,
                num_procs=1, overwrite=False):

    results, out_path = load_benchmark_scores(
        model_dir, BENCHMARK, overwrite)

    if not results.empty and len(results[results.benchmark == BENCHMARK]):
        return False

    print(f'{now()} | Measuring Texture vs Shape Bias, '
          f'model: {m + 1}/{total_models} at {model_dir}')

    model = get_trained_model(model_dir, architecture, True, ['output'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    transform = get_transform(architecture, model_dir)
    mapping = probabilities_to_decision.ImageNetProbabilitiesTo16ClassesMapping()
    dataset = ImageFolderWithPaths(DATASET_BASE, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=True, num_workers=num_procs)

    # loop through batches
    with tqdm(loader, unit=f"batch({batch_size})") as tepoch:

        for batch, (inputs, targets, paths, indices) in enumerate(tepoch):

            tepoch.set_description(f'{now()} | ')

            image_index = []
            for path, target in zip(paths, targets):
                image_index.append(
                    loader.dataset.samples.index((path, target)))

            # put inputs on device
            inputs = inputs.to(device)

            # pass through model with automatic mixed precision for speed
            with torch.autocast(device_type=device.type,
                                dtype=torch.float16):
                outputs = model(inputs)
            outputs = insert_cycle(outputs, batch_size=inputs.shape[0])

            # calculate score
            if batch == 0:
                performance = {k: AverageMeter() for k in outputs}
            for cycle, output in outputs.items():
                output_softmax = special.softmax(
                    output.detach().cpu().numpy(), axis=1)
                predictions = [mapping.probabilities_to_decision(i) for i in
                              output_softmax]
                path_info = [op.basename(p) for p in paths]
                target_shapes = [p.split('-')[0][:-1] for p in path_info]
                target_textures = [p.split('-')[1][:-5] for p in path_info]
                counts = np.zeros(4, dtype=int)
                for prediction, shape, texture in zip(
                        predictions, target_shapes, target_textures):
                    if prediction != shape and prediction != texture:
                        counts[0] += 1
                    elif prediction != shape and prediction == texture:
                        counts[1] += 1
                    elif prediction == shape and prediction != texture:
                        counts[2] += 1
                    elif prediction == shape and prediction == texture:
                        counts[3] += 1
                shape_bias = counts[2] / (counts[1] + counts[2])
                performance[cycle].update(shape_bias)

            # print last and mean performance of final cycle
            tepoch.set_postfix_str(
                f'shape bias: {shape_bias:.4f}'
                f'({performance[cycle].avg_epoch:.4f})')

    # save results
    for cycle, perf in performance.items():
        new_results = pd.DataFrame({
                'benchmark': ['texture_vs_shape_bias'],
                'path': [DATASET_BASE],
                'cycle': [int(cycle[3:])],
                'level_1': [None],
                'level_2': [None],
                'metric': ['shape_bias'],
                'score': [perf.avg_epoch]
            })
        results = pd.concat([results, new_results]).reset_index(drop=True)
        print(f'{cycle} shape bias: {perf.avg_epoch:.4f}')

    results.to_csv(out_path, index=False)

    return True


        


