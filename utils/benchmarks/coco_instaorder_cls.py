'''
This script tests the accuracy of ANNs on fourier and gaussian noise
'''

import os
import os.path as op
import glob
import shutil
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
import json
from pycocotools.coco import COCO
from PIL import Image
import skimage.io as io
from skimage.draw import polygon2mask
import cv2
import os
import string
import networkx as nx
import copy
import pylab

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # converts to order in nvidia-smi (not in cuda)
#os.environ["CUDA_VISIBLE_DEVICES"] = "1" # which device(s) to use

import torch
import torchvision
import torchvision.transforms.v2 as transforms
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch import float32
from torch.utils.data import Dataset

from utils import now, MODEL_BASE
from utils.AverageMeter import AverageMeter
from utils.calculate_batch_size import calculate_batch_size
from utils.get_trained_model import get_trained_model
from utils.image_processing import tile
from utils.math_functions import sigmoid
from utils.plot_utils import custom_defaults
from utils.Noise import Noise
from utils.CustomDataset import CustomDataset
from utils.get_transform import get_transform
from utils.load_benchmark_scores import load_benchmark_scores
plt.rcParams.update(custom_defaults)

np.random.seed(42)

BENCHMARK = 'COCO_InstaOrder_cls'
BENCHMARK_BASE = f'/home/tonglab/Datasets/{BENCHMARK}'
DATASET_BASE = op.expanduser('~/david/datasets/images/COCO')
DATA_TYPE = 'val2014'
METADATA_PATH = op.join(BENCHMARK_BASE, f'metadata_{DATA_TYPE}.csv')
IMAGENET_METADATA_PATH = op.expanduser(
    '~/david/datasets/images/ILSVRC2012/imagenet_class_index.json')
IMAGE_DIR = op.join(BENCHMARK_BASE, DATA_TYPE)
IMAGE_DIR_ANNOT = IMAGE_DIR + '_annot'

def make_coco_instaorder_cls_dataset(num_procs=1):

    """
    Create a dataset of 224x224 images from the COCO dataset, using the
    InstaOrder annotations to identify occluded objects. For each image in
    the InstaOrder dataset, the object occluded my the most other objects is
    selected, cropped to a square bounding box, resized to 224x224, and saved
    in the InstaOrder benchmark directory. The metadata for each image is
    saved in a csv file.
    """

    if op.isdir(IMAGE_DIR):
        shutil.rmtree(IMAGE_DIR)
    if op.isdir(IMAGE_DIR_ANNOT):
        shutil.rmtree(IMAGE_DIR_ANNOT)
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR_ANNOT, exist_ok=True)

    # coco images and annotations
    coco_json = op.join(DATASET_BASE, f'annotations/instances_{DATA_TYPE}.json')
    coco = COCO(coco_json)

    # instaorder annotations
    with open(op.join(DATASET_BASE, f'annotations/InstaOrder_{DATA_TYPE}.json'),
              'r+') as f:
        instaorder_data = json.load(f)['annotations']

    # imagenet synsets
    with open(IMAGENET_METADATA_PATH, 'r+') as f:
        imagenet_data = json.load(f)
    imagenet_df = pd.DataFrame(dict(
        target = list(imagenet_data.keys()),
        synset = [v[0] for v in imagenet_data.values()],
        class_name = [v[1] for v in imagenet_data.values()]))

    # mapping from instaorder to imagenet classes, reversing the mapping from
    # https://github.com/howardyclo/ImageNet2COCO, with further manual
    # additions during the dataset creation process
    DNN_utils_dir = op.expanduser('~/david/master_scripts/DNN/utils')
    coco_to_imagenet_path = op.join(DNN_utils_dir, 'coco_to_imagenet.json')
    imagenet_to_coco_path = op.join(DNN_utils_dir, 'imagenet2coco.txt')
    with open(imagenet_to_coco_path, 'r+') as f:
        imagenet_to_coco = f.readlines()
    coco_to_imagenet = {}
    for line in imagenet_to_coco:
        imagenet_synset, coco_classes, _ = line.split('\t')
        for coco_class in coco_classes.split(', '):
            if coco_class not in coco_to_imagenet:
                coco_to_imagenet[coco_class] = {imagenet_synset}
            else:
                coco_to_imagenet[coco_class].add(imagenet_synset)
    coco_to_imagenet = {k: list(v) for k, v in coco_to_imagenet.items()}
    with open(coco_to_imagenet_path, 'w+') as f:
        json.dump(coco_to_imagenet, f)


    # wrap function to parallelize image processing
    def _make_image(i, ann):

        # get image annotations
        img_coco = coco.loadImgs(int(ann['image_id']))[0]
        ann_ids = [int(i) for i in ann['instance_ids']]
        ann_coco = coco.loadAnns(ann_ids)
        num_inst = len(ann_ids)

        # create occlusion matrix and find most occluded object
        occ_matrix = np.zeros((num_inst, num_inst)).astype(np.uint8)
        for occ_str in ann['occlusion']:
            idx1, idx2 = occ_str['order'].split(' & ')[0].split('<')
            idx1, idx2 = int(idx1), int(idx2)
            if '&' in occ_str['order']:  # bidirection
                occ_matrix[idx1, idx2] = 1
                occ_matrix[idx2, idx1] = 1
            else:
                occ_matrix[idx1, idx2] = 1
        target_idx = np.argmax(occ_matrix.sum(0))

        # get imagenet class(es) and target number
        coco_class = coco.loadCats(
            ann_coco[target_idx]['category_id'])[0]['name']

        # look for imagenet class if item not present in coco_to_imagenet
        if coco_class in coco_to_imagenet:
            imagenet_synset = coco_to_imagenet[coco_class]
        if coco_class not in coco_to_imagenet:
            # match based on last word in class name, e.g. 'horse cart'
            # should be matched with 'cart' and not 'horse'
            imagenet_matches = [i for i in imagenet_df.class_name if
                                i.lower().split('_')[-1] == coco_class.lower()]
            if len(imagenet_matches):
                imagenet_synset = imagenet_df[
                    imagenet_df.class_name.isin(imagenet_matches)
                ].synset.to_list()
                coco_to_imagenet[coco_class] = imagenet_synset
                with open(coco_to_imagenet_path, 'w+') as f:
                    json.dump(coco_to_imagenet, f)
            else:
                print(f'{coco_class} not found in imagenet')
                return

        # get target number
        imagenet_target = [int(i) for i in imagenet_df[imagenet_df.synset.isin(
            imagenet_synset)].target]
        imagenet_class = imagenet_df[imagenet_df.synset.isin(
            imagenet_synset)].class_name.to_list()

        # get square bounding box around target object
        target_bbox = ann_coco[target_idx]['bbox']
        largest_dim = np.argmax(target_bbox[2:])
        length = int(target_bbox[largest_dim+2])
        center = (target_bbox[0] + target_bbox[2] / 2,
                  target_bbox[1] + target_bbox[3] / 2)
        left, top = int(center[0] - length / 2), int(center[1] - length / 2)
        right, bottom = left + length, top + length
        if left < 0:
            left, right, = 0, length
        elif right > img_coco['width']:
            left, right = img_coco['width'] - length, img_coco['width']
        if top < 0:
            top, bottom = 0, length
        elif bottom > img_coco['height']:
            top, bottom = img_coco['height'] - length, img_coco['height']

        # make test image
        img = io.imread(f"{DATASET_BASE}/{DATA_TYPE}/{img_coco['file_name']}")
        img_cropped = img[top:bottom, left:right]
        img_resized = cv2.resize(img_cropped, (224, 224))
        image_filename = f'{i:06}.png'
        io.imsave(op.join(IMAGE_DIR, image_filename), img_resized)

        # make test images with annotations
        img_w_inst = draw_instance_mask(img, [ann_coco[target_idx]],
                                        [coco_class])
        image_filename = f'{i:06}.png'
        io.imsave(op.join(IMAGE_DIR_ANNOT, image_filename), img_w_inst)

        return [image_filename, imagenet_synset, imagenet_target,
                imagenet_class, coco_class]

    # code for running in parallel, job only takes a few mins in serial
    #metadata = Parallel(n_jobs=num_procs)(
    #    delayed(_make_image)(i, ann) for i, ann in enumerate(instaorder_data))
    metadata = []
    for i, ann in enumerate(instaorder_data):
        print(f'{now()} | {i}/{len(instaorder_data)}')
        metadata.append(_make_image(i, ann))

    # save out dataframe of image filenames, synsets, and targets
    metadata = [i for i in metadata if i is not None]
    df = pd.DataFrame({'filename': [i[0] for i in metadata],
                       'synset': [i[1] for i in metadata],
                       'target': [i[2] for i in metadata],
                       'imagenet_class': [i[3] for i in metadata],
                       'coco_class': [i[4] for i in metadata]})
    df.to_csv(op.join(BENCHMARK_BASE, f'metadata_{DATA_TYPE}.csv'), index=False)

    """
    # Code for visualizing segmented images and depth/occlusion graphs
    
    plt.cla()
    image_h, image_w = img_coco['height'], img_coco['width']

    # read instance annotations
    img = io.imread(f"{DATASET_BASE}/{DATA_TYPE}/{img_coco['file_name']}")


    # 1. draw instances masks on the image
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.axis("off")
    img_w_inst = draw_instance_mask(img, ann_coco)
    plt.imshow(img_w_inst)

    # 2. draw occlusion graph
    plt.subplot(132)
    draw_occ_graph(ann['occlusion'], num_inst)

    # 3. draw depth graph
    plt.subplot(133)
    draw_depth_graph(ann['depth'], num_inst)

    plt.savefig('/home/tonglab/Desktop/instaorder.png')
    plt.show()
    """


class COCO_InstaOrder_Dataset(Dataset):

    def __init__(self, transform=None):
        self.main_dir = IMAGE_DIR
        metadata = pd.read_csv(METADATA_PATH)
        self.image_paths = [op.join(self.main_dir, f) for f in metadata.filename]
        self.targets = [torch.tensor([
            int(j) for j in i[1:-1].split(', ')]) for i in metadata.target]
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


@torch.no_grad()
def score_model(model_dir, architecture, batch_size, m=0, total_models=0,
                num_procs=1, overwrite=False):

    results, out_path = load_benchmark_scores(
        model_dir, BENCHMARK, overwrite)

    if len(results) and BENCHMARK in results.benchmark.unique():
        return False

    print(f'{now()} | Measuring performance for {BENCHMARK}, '
          f'model: {m + 1}/{total_models} at {model_dir}')

    model = get_trained_model(model_dir, architecture, True, ['output'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    transform = get_transform(architecture, model_dir)

    dataset = COCO_InstaOrder_Dataset(transform=transform)
    def collate_fn(data):
        img = torch.stack([i[0] for i in data])
        trg = [i[1] for i in data]
        return [img, trg]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
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

            # for recurrent models
            if len(outputs) > inputs.shape[0]:
                cycles = len(outputs) // inputs.shape[0]
                outputs = {f'cyc{c:02}': outputs[c::cycles] for c in
                           range(cycles)}
            else:
                outputs = {'cyc-1': outputs}

            # calculate accuracy
            if batch == 0:
                performance = {k: AverageMeter() for k in outputs}
            for cycle, output in outputs.items():
                acc = accuracy(output, targets)#.detach().cpu().item()
                for i in acc:
                    performance[cycle].update(i)

            # print last and mean accuracy of final cycle
            tepoch.set_postfix_str(
                f'accuracy: {acc.mean():.4f}'
                f'({performance[cycle].avg_epoch:.4f})')

    # save results
    for cycle, perf in performance.items():
        new_results = pd.DataFrame({
                'benchmark': [BENCHMARK],
                'level_1': [DATA_TYPE],
                'level_2': [''],
                'path': [DATA_TYPE],
                'cycle': [int(cycle[3:])],
                'metric': ['accuracy'],
                'score': [perf.avg_epoch.item()]
            })
        results = pd.concat([results, new_results]).reset_index(drop=True)
        print(f'{cycle} accuracy: {perf.avg_epoch:.4f}')

    results.to_csv(out_path, index=False)

    return True

COLORS = [
        [252, 15, 15],  # Red
        [252, 165, 15],  # orange
        [252, 232, 15],  # yellow
        [14, 227, 39],  # light green
        [10, 138, 37],  # dark green
        [9, 219, 216],  # light blue
        [9, 111, 219],  # dark blue
        [185, 90, 232],  # light purple
        [201, 40, 175],  # purple
        [245, 49, 166]  # pink
    ]

def get_mask_from_coco_ann(seg, image_shape):
    poly_xy = np.array(seg).reshape((int(len(seg) / 2), 2))
    poly_yx = np.zeros_like(poly_xy)
    poly_yx[:, 0], poly_yx[:, 1] = poly_xy[:, 1], poly_xy[:, 0]
    mask = polygon2mask(image_shape, poly_yx)
    return mask, poly_xy

def get_mid_top_loc(poly, image_h, image_w):
    top_idx = poly[:, 1].argmin()
    mid_top = (np.clip(int(poly[top_idx][0]), 15, image_w - 15),
               np.clip(int(poly[top_idx][1]), 15, image_h - 15))
    return mid_top

def fill_poly_with_list(I, polyfilled, poly_xy, c, alpha=0.6):
    polyfilled = cv2.fillPoly(polyfilled, pts=[
        poly_xy.reshape(-1, 1, 2).astype(np.int32)], color=c)
    I = cv2.addWeighted(I, alpha, polyfilled, 1 - alpha, 0)
    return I

def draw_polyline_with_list(I, poly_xy, c, isClosed=True, thickness=2):
    I = cv2.polylines(I, [poly_xy.reshape(-1, 1, 2).astype(np.int32)],
                      isClosed, c, thickness)
    return I

def draw_graph_with_color(matrix, color_matrix, color='red', pos=None):
    edges = np.where(matrix == 1)

    from_idx = edges[0].tolist()
    to_idx = edges[1].tolist()

    from_node = [string.ascii_uppercase[i] for i in from_idx]
    to_node = [string.ascii_uppercase[i] for i in to_idx]

    G = nx.DiGraph()
    for i in range(matrix.shape[0]):
        G.add_node(string.ascii_uppercase[i])

    pos = nx.circular_layout(G)
    G.add_edges_from(list(zip(from_node, to_node)))
    colors_tf = [color_matrix[pair[0], pair[1]] for pair in
                 list(zip(from_idx, to_idx))]
    edge_color = [color if color_tf else 'black' for color_tf in
                  colors_tf]
    node_size = [600] * len(G)

    nx.draw_networkx_nodes(G, pos, node_size=node_size)
    nx.draw_networkx_labels(G, pos, font_color='w', font_size=15)
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=20,
                           width=2, edge_color=edge_color)

    return pos

def write_text(img, text, location, c):
    cv2.rectangle(img, (location[0] - 5, location[1] - 10),
                  (location[0] + 25, location[1] + 5), c, -1)
    cv2.putText(img, text, location, 0, 0.3, (0, 0, 0), thickness=1,
                lineType=cv2.LINE_AA)
    return img

def draw_occ_graph(occ_str_all, num_inst):
    occ_matrix = np.zeros((num_inst, num_inst)).astype(np.uint8)
    is_overlap_matrix = np.zeros((num_inst, num_inst)).astype(bool)
    for occ_str in occ_str_all:
        idx1, idx2 = occ_str['order'].split(' & ')[0].split('<')
        idx1, idx2 = int(idx1), int(idx2)
        if '&' in occ_str['order']: #bidirection
            occ_matrix[idx1, idx2] = 1
            occ_matrix[idx2, idx1] = 1
        else:
            occ_matrix[idx1, idx2] = 1
    draw_graph_with_color(occ_matrix, is_overlap_matrix, color='green')

def draw_depth_graph(depth_str_all, num_inst):
    depth_matrix = np.zeros((num_inst, num_inst)).astype(np.uint8)
    is_overlap_matrix = np.zeros((num_inst, num_inst)).astype(bool)
    for depth_str in depth_str_all:
        if '=' in depth_str['order']:
            eq1_idx, eq2_idx = list(
                map(int, depth_str['order'].split('=')))

            depth_matrix[eq1_idx, eq2_idx] = 1
            depth_matrix[eq2_idx, eq1_idx] = 1
            is_overlap_matrix[eq1_idx, eq2_idx] = 1 if depth_str[
                                                           'overlap'] == True else 0
            is_overlap_matrix[eq2_idx, eq1_idx] = 1 if depth_str[
                                                           'overlap'] == True else 0

        elif '<' in depth_str['order']:
            near_idx, far_idx = list(
                map(int, depth_str['order'].split('<')))
            depth_matrix[near_idx, far_idx] = 1
            is_overlap_matrix[near_idx, far_idx] = 1 if depth_str[
                                                            'overlap'] == True else 0
    draw_graph_with_color(depth_matrix, is_overlap_matrix,
                          color='green')

def draw_instance_mask(img, anns, class_names, alpha=0.6):
    mid_tops = []
    for i, ann in enumerate(anns):
        poly_xy_instance = []
        for seg_ind, seg in enumerate(ann['segmentation']):
            mask, poly_xy = get_mask_from_coco_ann(seg,
                                                   image_shape=img.shape[
                                                               :2])
            poly_xy_instance.append(poly_xy)  # [(n1,2), (n2,2), (n3,2)]

        # text location
        mid_top = get_mid_top_loc(np.concatenate(poly_xy_instance),
                                  img.shape[0], img.shape[1])
        mid_tops.append(mid_top)

        # color every instance
        for poly_xy in poly_xy_instance:
            polyfilled = copy.deepcopy(img)
            img = fill_poly_with_list(img, polyfilled, poly_xy,
                                      COLORS[i],
                                      alpha=alpha)
            img = draw_polyline_with_list(img, poly_xy, COLORS[i])

    for c, class_name in enumerate(class_names):
        img = write_text(img, class_name, mid_tops[c], COLORS[c])
    return img