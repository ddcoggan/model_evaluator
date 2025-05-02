'''
This script tests the accuracy of ANNs on public benchmarks
'''

import os.path as op
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from utils import now, insert_cycle
from utils.accuracy import accuracy
from utils.get_transform import get_transform
from utils.AverageMeter import AverageMeter
from utils.get_trained_model import get_trained_model
from utils.load_benchmark_scores import load_benchmark_scores

np.random.seed(42)

BENCHMARK = 'ImageNet-C'
DATASET_BASE = op.expanduser('~/Datasets/ImageNet-C')
DATASETS = [i[len(DATASET_BASE) + 1:] for i in sorted(glob.glob(
            f'{DATASET_BASE}/*/*/*'))]

@torch.no_grad()
def score_model(model_dir, architecture, batch_size, m=0, total_models=0,
                num_procs=1, overwrite=False):

    results, out_path = load_benchmark_scores(
        model_dir, BENCHMARK, overwrite)

    if results.empty:
        subsets_to_run = DATASETS
    else:
        subsets_to_run = [i for i in DATASETS if i not in results.path.unique()]
    if not len(subsets_to_run):
        return False

    print(f'{now()} | Measuring performance for {BENCHMARK}, '
          f'model: {m + 1}/{total_models} at {model_dir}')

    model = get_trained_model(model_dir, architecture, True, ['output'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    transform = get_transform(architecture, model_dir)

    for subset in subsets_to_run:

        print(f'{now()} | model: {m + 1}/{total_models}, subset: {subset}')

        dataset = ImageFolder(op.join(DATASET_BASE, subset),
                              transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=num_procs)

        # loop through batches
        with tqdm(loader, unit=f"batch({batch_size})") as tepoch:

            for batch, (inputs, targets) in enumerate(tepoch):

                tepoch.set_description(f'{now()} | {BENCHMARK}')

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
            level_1, level_2, level_3 = subset.split(f'{DATASET_BASE}/')[
                -1].split('/')
            new_results = pd.DataFrame({
                'benchmark': [BENCHMARK],
                'path': [subset],
                'cycle': [int(cycle[3:])],
                'level_1': [level_1],
                'level_2': [level_2],
                'level_3': [level_3],
                'metric': ['accuracy'],
                'score': [perf.avg_epoch]
            })
            results = pd.concat([results, new_results]).reset_index(drop=True)
            print(f'{subset}, {cycle} accuracy: {perf.avg_epoch:.4}')

        results.to_csv(out_path, index=False)

    return True

