'''
This script tests the accuracy of ANNs on COCO occluded vehicles from
Kortylewski et al. (2020) https://arxiv.org/pdf/2003.04490
'''

import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.nn.functional as F
from utils import now, insert_cycle
from utils.AverageMeter import AverageMeter
from utils.get_trained_model import get_trained_model
from utils.plot_utils import custom_defaults
from utils.get_transform import get_transform
from utils.load_benchmark_scores import load_benchmark_scores
plt.rcParams.update(custom_defaults)

np.random.seed(42)

BENCHMARK = 'PASCAL3D+_Occluded_Objects'
BENCHMARK_BASE = f'/home/david/Datasets/PASCAL3D+_occ'
CLASSES = {
    'aeroplane': [404, 895],
    'bicycle': [671, 444],
    'boat': [554, 625, 814],
    'bottle': [440, 737, 898, 907],
    'bus': [654, 779, 874],
    'car': [436, 468, 609, 656, 734, 751, 817],
    'chair': [559, 765, 423],
    'diningtable': [532],
    'motorbike': [670],
    'sofa': [831],
}

class PASCAL3DPlus_Occluded_Objects_Dataset(Dataset):

    def __init__(self, transform=None):
        self.main_dir = BENCHMARK_BASE
        self.image_paths = []
        self.targets = []
        for cls, trg in CLASSES.items():
            paths = sorted(glob.glob(f'{self.main_dir}/{cls}*/*.JPEG'))
            self.image_paths.extend(paths)
            self.targets.extend([trg] * len(paths))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image, self.targets[idx]


def accuracy(output, target):

    """Computes the precision@k for the specified values of k"""
    _, pred = output.topk(1, 1, True, True)
    res = []
    pred = pred.detach().cpu()
    for trg, prd in zip(target, pred):
        res.append(torch.tensor(int(prd in trg)).float())
    return torch.tensor(res)


def accuracy_afc(output, target):
    """Computes 6-AFC accuracy (max value)"""
    res = []
    for trg, prd in zip(target, output.detach().cpu()):
        targ = [i for i, (key, value) in enumerate(CLASSES.items()) if trg == value][0]
        preds = torch.tensor([torch.tensor([prd[v] for v in value]).max() for key, value in CLASSES.items()])
        pred = torch.argmax(preds)
        res.append(torch.tensor(pred == targ).float())
    return torch.tensor(res)


def accuracy_afc_sum(output, target):
    """Computes 6-AFC accuracy (sum of values)"""
    res = []
    for trg, prd in zip(target, output.detach().cpu()):
        targ = [i for i, (key, value) in enumerate(CLASSES.items()) if trg == value][0]
        preds = torch.tensor([torch.tensor([prd[v] for v in value]).sum() for key, value in CLASSES.items()])
        pred = torch.argmax(preds)
        res.append(torch.tensor(pred == targ).float())
    return torch.tensor(res)


@torch.no_grad()
def score_model(model_dir, architecture, batch_size, m=0, total_models=0,
                num_procs=1, overwrite=False):

    results, out_path = load_benchmark_scores(
        model_dir, BENCHMARK, overwrite)

    if not results.empty and len(results[results.benchmark == BENCHMARK]) == 4:
        return False

    print(f'{now()} | Measuring performance for {BENCHMARK}, '
          f'model: {m + 1}/{total_models} at {model_dir}')

    model = get_trained_model(model_dir, architecture, True, ['output'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    transform = get_transform(architecture, model_dir)

    dataset = PASCAL3DPlus_Occluded_Objects_Dataset(transform=transform)
    def collate_fn(data):
        img = torch.stack([i[0] for i in data])
        trg = [i[1] for i in data]
        return [img, trg]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_procs, collate_fn=collate_fn)

    # loop through batches
    with tqdm(loader, unit=f"batch({batch_size})") as tepoch:

        for batch, (inputs, targets) in enumerate(tepoch):

            tepoch.set_description(f'{now()} | ')

            # put inputs on device
            inputs = inputs.to(device)

            # pass through model
            with torch.autocast(device_type=device.type,
                                dtype=torch.float16):
                outputs = model(inputs)
            outputs = insert_cycle(outputs, batch_size=inputs.shape[0])

            # calculate accuracy
            if batch == 0:
                performance = {metric: {k: AverageMeter() for k in outputs}
                               for metric in ['accuracy', 'probability', 'accuracy_afc', 'accuracy_afc_sum']}
            for cycle, output in outputs.items():
                acc = accuracy(output, targets)  # .detach().cpu().item()
                for i in acc:
                    performance['accuracy'][cycle].update(i.item())

                acc_afc = accuracy_afc(output, targets)  # .detach().cpu().item()
                for i in acc_afc:
                    performance['accuracy_afc'][cycle].update(i.item())

                acc_afc_sum = accuracy_afc_sum(output, targets)  # .detach().cpu().item()
                for i in acc_afc_sum:
                    performance['accuracy_afc_sum'][cycle].update(i.item())

                output_norm = F.softmax(output, 1)
                prob = torch.tensor([output_norm[i, j].detach().cpu().mean()
                                     for i, j in enumerate(targets)])
                for i in prob:
                    performance['probability'][cycle].update(i.item())

            # print last and mean accuracy of final cycle
            tepoch.set_postfix_str(
                f'acc: {acc.mean():.4f}('
                f'{performance["accuracy"][cycle].avg_epoch:.4f}) | '
                f'prob: {prob.mean():.4f}('
                f'{performance["probability"][cycle].avg_epoch:.4f}) | '
                f'acc_afc: {acc_afc.mean():.4f}('
                f'{performance["accuracy_afc"][cycle].avg_epoch:.4f}) | '
                f'acc_afc_sum: {acc_afc_sum.mean():.4f}('
                f'{performance["accuracy_afc_sum"][cycle].avg_epoch:.4f})')

    # save results
    for metric, cycles in performance.items():
        for cycle, perf in cycles.items():
            new_results = pd.DataFrame({
                    'benchmark': [BENCHMARK],
                    'level_1': [''],
                    'level_2': [''],
                    'path': [BENCHMARK_BASE],
                    'cycle': [int(cycle[3:])],
                    'metric': [metric],
                    'score': [perf.avg_epoch]
                })
        results = pd.concat([results, new_results]).reset_index(drop=True)
        print(f'{cycle} {metric}: {perf.avg_epoch:.4f}')

    results.to_csv(out_path, index=False)

    return True
