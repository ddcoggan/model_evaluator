'''
This script tests the accuracy of ANNs on public benchmarks
'''

import os.path as op
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import now, insert_cycle
from utils.accuracy import accuracy
from utils.AverageMeter import AverageMeter
from utils.get_trained_model import get_trained_model
from utils.get_transform import get_transform
from utils.load_benchmark_scores import load_benchmark_scores

np.random.seed(42)
torch.manual_seed(42)

BENCHMARK = 'ImageNet-CutMix'
DATASET_BASE = op.expanduser('~/Datasets/ILSVRC2012/val')

def cutmix_50(input: torch.tensor, targets: torch.tensor) -> torch.tensor:

    # get image size
    n, c, h, w = input.shape

    # get random location
    fg_w = fg_h = int(np.sqrt((h * w) / 2))
    x1 = np.random.randint(0, w - fg_w)
    x2 = x1 + fg_w
    y1 = np.random.randint(0, h - fg_h)
    y2 = y1 + fg_h

    # cutmix images
    cutmix = input.clone()
    for i in range(n):
        j = (i + 1) % n
        cutmix[i, :, y1:y2, x1:x2] = input[j, :, y1:y2, x1:x2]

    # foreground targets
    targets_fg = targets.clone()
    targets_fg = torch.cat((targets_fg[1:], targets_fg[:1]))

    return cutmix, targets_fg


@torch.no_grad()
def score_model(model_dir, architecture, m=0, total_models=0, batch_size=64,
                 num_procs=1, overwrite=False):

    results, out_path = load_benchmark_scores(
        model_dir, BENCHMARK, overwrite)

    if not results.empty and len(results[results.benchmark == BENCHMARK]):
        return False

    print(f'{now()} | Measuring performance for {BENCHMARK}, '
          f'model: {m + 1}/{total_models} at {model_dir}')

    model = get_trained_model(model_dir, architecture, True, ['output'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    transform = get_transform(architecture, model_dir)
    dataset = ImageFolder(DATASET_BASE, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_procs)

    # loop through batches
    with (tqdm(loader, unit=f'batch({batch_size})') as tepoch):

        for batch, (inputs_bg, targets_bg) in enumerate(tepoch):

            tepoch.set_description(f'{now()} | {BENCHMARK}')

            # apply cutmix
            inputs, targets_fg = cutmix_50(inputs_bg, targets_bg)

            """
            # save some images
            if batch == 0:
                for inputs_to_save, input_type in zip(
                        [inputs, inputs_bg], ['mixed', 'background']):
                    outdir = f'/home/tonglab/Desktop/cutmix_50/{input_type}'
                    os.makedirs(outdir, exist_ok=True)
                    images = inputs_to_save.permute(0, 2, 3, 1)
                    images -= images.min()
                    images /= images.max()
                    images *= 255
                    images = np.array(images, dtype=np.uint8)
                    for i in range(images.shape[0]):
                        Image.fromarray(images[i].squeeze()).save(
                            op.join(outdir, f'{i:02}.png'))
            """

            # put inputs on device
            inputs = inputs.to(device)
            targets_fg = targets_fg.to(device)
            targets_bg = targets_bg.to(device)

            # pass through model with automatic mixed precision for speed
            with torch.autocast(device_type=device.type,
                                dtype=torch.float16):
                outputs = model(inputs)
            outputs = insert_cycle(outputs, batch_size=inputs.shape[0])

            # calculate performance
            if batch == 0:
                performance = {
                    'fg_acc': {k: AverageMeter() for k in outputs},
                    'bg_acc': {k: AverageMeter() for k in outputs},
                    'bg_bias': {k: AverageMeter() for k in outputs},
                }
            for cycle, output in outputs.items():

                # accuracy
                acc_fg = accuracy(output, targets_fg, (1,)
                               )[0].detach().cpu().item()
                performance['fg_acc'][cycle].update(acc_fg)
                acc_bg = accuracy(output, targets_bg, (1,)
                                  )[0].detach().cpu().item()
                performance['bg_acc'][cycle].update(acc_bg)

                # background bias
                output_norm = F.softmax(output, 1)
                fg_prob = torch.tensor([output_norm[i, j].detach().cpu().item()
                                        for i, j in enumerate(targets_fg)])
                bg_prob = torch.tensor([output_norm[i, j].detach().cpu().item()
                                        for i, j in enumerate(targets_bg)])
                bg_bias = (bg_prob / (bg_prob + fg_prob))
                bg_bias[torch.isnan(bg_bias)] = 0.5
                bg_bias_norm = bg_bias.mean().item()
                performance['bg_bias'][cycle].update(bg_bias_norm)

            # print last and mean accuracy of final cycle
            fg_acc_avg = performance['fg_acc'][cycle].avg_epoch
            bg_acc_avg = performance['bg_acc'][cycle].avg_epoch
            bg_bias_avg = performance['bg_bias'][cycle].avg_epoch

            tepoch.set_postfix_str(
                f'acc fg: {acc_fg:.4f}({fg_acc_avg:.4f}) | '
                f'acc bg: {acc_bg:.4f}({bg_acc_avg:.4f}) | '
                f'bg bias: {bg_bias_norm:.4f}({bg_bias_avg:.4f})')

        # save results
        for ground, gr, metric, mt in zip(
                ['foreground', 'background', 'background'],
                ['fg', 'bg', 'bg'],
                ['accuracy', 'accuracy', 'bias'],
                ['acc', 'acc', 'bias']):
            for cycle, perf in performance[f'{gr}_{mt}'].items():
                results = pd.concat([results, pd.DataFrame({
                        'benchmark': [BENCHMARK],
                        'path': [DATASET_BASE],
                        'level_1': [ground],
                        'cycle': [int(cycle[3:])],
                        'metric': [metric],
                        'score': [perf.avg_epoch]
                    })]).reset_index(drop=True)
    print(f'{BENCHMARK}, {cycle} | '
          f'acc_fg: {performance["fg_acc"][cycle].avg_epoch:.4} '
          f'acc_bg: {performance["bg_acc"][cycle].avg_epoch:.4} '
          f'bg_bias: {performance["bg_bias"][cycle].avg_epoch:.4}')
    results.to_csv(out_path, index=False)

    return True

