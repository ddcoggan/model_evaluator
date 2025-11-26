'''
This script tests the accuracy of ANNs on occlusion
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
#from pexpect.screen import screen
from scipy import stats
from scipy import special
import pickle as pkl
import pandas as pd
from itertools import product as itp
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
import gc
import json
from pycocotools.coco import COCO
import pycocotools.mask as mask_util
import matplotlib.colors as mplc
import matplotlib as mpl
import colorsys
from PIL import Image
import skimage.io as io
from skimage.draw import polygon2mask
import cv2
import os
import string
import networkx as nx
import copy
import pylab
from nltk.corpus import wordnet as wn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
sys.path.append('/mnt/HDD1_12TB/repos/ORCNN')
#from detectron2.data.datasets import load_coco_json
#from detectron2.data.datasets.coco import (
#    register_coco_instances, convert_to_coco_dict)
#from detectron2.utils.visualizer import Visualizer
from utils import now, insert_cycle
from utils.AverageMeter import AverageMeter
from utils.get_trained_model import get_trained_model
from utils.get_transform import get_transform
from utils.load_benchmark_scores import load_benchmark_scores

np.random.seed(42)

BENCHMARK = 'COCOA_cls'
BENCHMARK_BASE = f'/home/tonglab/Datasets/{BENCHMARK}'
DATASET_BASE = op.expanduser('~/david/datasets/images/COCO')
DATA_TYPE = 'val2014'
COCO_AMODAL_PATH = op.join(
    DATASET_BASE, f'annotations/COCO_amodal_{DATA_TYPE}.json')
ANNOTATIONS_PATH = op.join(BENCHMARK_BASE, f'cocoa_cls_{DATA_TYPE}.json')
IMAGENET_METADATA_PATH = op.expanduser(
    '~/david/datasets/images/ILSVRC2012/imagenet_class_index.json')
SAMPLE_IMAGE_DIR = op.join(BENCHMARK_BASE, 'sample_images')


class SynsetFinder(object):

    def __init__(self):

        # record responses to avoid repeating
        self.manual_mapping_path = op.join(BENCHMARK_BASE,
                                           'manual_mapping.json')
        if op.isfile(self.manual_mapping_path):
            with open(self.manual_mapping_path, 'r+') as f:
                manual_mapping = json.load(f)
        else:
            manual_mapping = {}
        self.excluded_words_path = op.join(BENCHMARK_BASE, 'excluded_words.json')
        if op.isfile(self.excluded_words_path):
            with open(self.excluded_words_path, 'r+') as f:
                excluded_words = json.load(f)
        else:
            excluded_words = []
        self.manual_mapping = manual_mapping
        self.excluded_words = excluded_words

        # get imagenet synsets and hypernyms thereof
        with open(IMAGENET_METADATA_PATH, 'r+') as f:
            imagenet_data = json.load(f)
        imagenet_df = pd.DataFrame(dict(
            target=list(imagenet_data.keys()),
            synset=[v[0] for v in imagenet_data.values()],
            class_name=[v[1] for v in imagenet_data.values()]))
        self.imagenet_df = imagenet_df
        imagenet_synsets = {}
        for id in imagenet_df.synset.values:
            synset = wn.synset_from_pos_and_offset('n', int(id[1:]))
            hypernym = synset.hypernyms()[0]
            hypernyms = []
            while hypernym.hypernyms():
                hypernyms.append(hypernym)
                hypernym = hypernym.hypernyms()[0]
            imagenet_synsets[synset] = hypernyms
        self.imagenet_synsets = imagenet_synsets


    def find_synsets(self, word, manual_mapping):

        # check manual mapping
        if word in manual_mapping:
            word = manual_mapping[word]

        # expand search with basic replacements and wordnet morphs
        word_forms = [word, word.strip(), word.replace(' ', '_'),
                      word.replace(' ', '-'),
                      word.replace(' ', '')]
        word_forms += [wn.morphy(word_form) for word_form in word_forms]
        word_forms = [i for i in word_forms if i is not None]

        # remove words that are not concrete nouns
        for word_form in word_forms:
            synsets = wn.synsets(word_form, pos='n')
            for synset in synsets:
                hypernyms = set([i for i in synset.closure(
                    lambda s:s.hypernyms())])
                if not any([h.name().split('.')[0] in [
                    'object', 'artifact', 'whole', 'sign'] for h in hypernyms]):
                    synsets.remove(synset)
            if len(synsets):
                break

        return word_form, synsets


    def word_to_synset(self, word, format='pos_offset_string'):

        if word in self.excluded_words:
            return None, None

        word_orig = word  # keep the original

        # find synset for object
        word, synset_results = self.find_synsets(word, self.manual_mapping)

        # search manually
        while not len(synset_results):
            word = input('options:\n'
                f'try another word for "{word_orig}"\n'
                'g: give up\n')
            if word == 'g':
                self.excluded_words.append(word_orig)
                with open(self.excluded_words_path, 'w+') as f:
                    json.dump(self.excluded_words, f)
                return None, None
            word, synset_results = self.find_synsets(word, self.manual_mapping)
        self.manual_mapping[word_orig] = word
        with open(self.manual_mapping_path, 'w+') as f:
            json.dump(self.manual_mapping, f)

        # if several synsets found, select based on those in imagenet
        if len(synset_results) > 1:

            # reduce to those contained in imagenet
            synsets_matched = [i for i in synset_results if
                               i in self.imagenet_synsets]
            # if none in imagenet, try hypernyms
            if not len(synsets_matched):
                synsets_matched = []
                for synset, hypernyms in itp(synset_results,
                                             self.imagenet_synsets.values()):
                    if synset in hypernyms and synset not in synsets_matched:
                        synsets_matched.append(synset)

            # if still multiple synsets found, pick most common. This is the
            # first one according to https://groups.google.com/g/nltk-users/c/YeWIzI-vrm4?pli=1
            if len(synsets_matched):
                synset_results = [synsets_matched[0]]
            else:
                synset_results = [synset_results[0]]

        # if found, add synset to annotation
        if len(synset_results):
            assert len(synset_results) == 1
            synset = synset_results[0]
            if format == 'pos_offset_string':
                synset = f'n{synset.offset():08}'
            return word, synset


    def find_imagenet_synsets_targets(self, synset):

        """
        Take the synsets found in the previous step and map them to imagenet
        """

        # look for exact matches in imagenet
        if synset in self.imagenet_df.synset:
            imagenet_target = [int(self.imagenet_df.target[
                                       self.imagenet_df.synset ==
                                                      synset].values[0])]
            return [synset], imagenet_target

        # look for imagenet synsets that are a hypernym of object synset
        synset_obj = wn.synset_from_pos_and_offset(
            'n', int(synset[1:]))
        hypernyms = set([i for i in synset_obj.closure(
            lambda s: s.hypernyms())])
        synsets_matched = [
            i for i in self.imagenet_synsets if i in hypernyms]
        assert len(synsets_matched) <= 1
        if len(synsets_matched):
            imagenet_synset = [f'n{synsets_matched[0].offset():08}']
            imagenet_target = [int(i) for i in self.imagenet_df.target[
                self.imagenet_df.synset == imagenet_synset[0]].values]
            return imagenet_synset, imagenet_target

        # look for imagenet synsets with hypernyms matching object name
        synsets_matched = [
            s for s, h in self.imagenet_synsets.items() if synset_obj in h]
        if len(synsets_matched):
            imagenet_synset = [f'n{s.offset():08}' for s in synsets_matched]
            imagenet_target = [int(self.imagenet_df.target[
                                   self.imagenet_df.synset == i].values[0]) for
                               i in imagenet_synset]
            return imagenet_synset, imagenet_target

        return None, None


def _read_COCOA(ann, h, w):

    amodal = mask_util.decode(mask_util.merge(
        mask_util.frPyObjects([ann['segmentation']], h, w)))
    bbox = _mask_to_bbox(amodal)

    if 'visible_mask' in ann:
        visible = mask_util.decode([ann['visible_mask']]).squeeze()
    else:
        visible = None

    if 'invisible_mask' in ann:
        invisible = mask_util.decode([ann['invisible_mask']]).squeeze()
    else:
        invisible = None

    return amodal, visible, invisible, bbox, ann['name']


def _mask_to_bbox(mask):
    mask = (mask == 1)
    if np.all(~mask):
        return [0, 0, 0, 0]
    assert len(mask.shape) == 2
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [cmin.item(), rmin.item(), cmax.item() + 1 - cmin.item(),
            rmax.item() + 1 - rmin.item()]  # xywh


def make_cocoa_cls_dataset():

    """
    Create a dataset of 224x224 images from the COCO dataset, using the
    COCOA annotations to identify occluded objects. For each image in the
    COCOA dataset, occluded objects are identified, square-cropped,
    and resized to 224x224. The metadata is saved in a csv file.
    """

    def _make_image_dataset():

        if op.isdir(SAMPLE_IMAGE_DIR):
            shutil.rmtree(SAMPLE_IMAGE_DIR)
        os.makedirs(SAMPLE_IMAGE_DIR, exist_ok=True)

        # get cocoa annotations
        coco_amodal = load_coco_json(COCO_AMODAL_PATH, op.join(
            DATASET_BASE, DATA_TYPE), extra_annotation_keys=['regions'])

        image_count = 0

        synset_finder = SynsetFinder()

        for i, cocoa_ann in enumerate(coco_amodal):

            print(f'{now()} | {i}/{len(coco_amodal)} image count: {image_count}')

            H, W = cocoa_ann['height'], cocoa_ann['width']
            instance = 0

            for ann in cocoa_ann['annotations'][0]['regions']:

                amd, vis, inv, bbox, object_class = _read_COCOA(ann, H, W)

                # skip objects based on several criteria:

                # objects that are not just 'stuff'
                if ann['isStuff']:
                    continue

                # objects without occlusion masks
                if inv is None or vis is None:
                    continue

                # visibility is too high or too low
                visibility = 1 - ann['occlude_rate']
                if visibility < .1 or visibility > .9:
                    continue

                #  too small
                if any(np.array(amd.shape) < 112):
                    continue

                # amodal mask not fully contained in the image
                if sum(amd[0, :] + amd[-1, :]) or sum(amd[:, 0] + amd[:, -1]):
                    continue

                image_count += 1

                # find square amodal bounding box around object
                length = max(bbox[2], bbox[3])
                length = min(length, W, H)  # handle objects larger than W or H
                center = [int(bbox[0] + bbox[2] / 2),
                          int(bbox[1] + bbox[3] / 2)]
                left, top = int(center[0] - length / 2), int(
                    center[1] - length / 2)
                right, bottom = left + length, top + length

                # shift if out of bounds
                if left < 0:
                    left, right, = 0, length
                elif right > W:
                    left = W - length
                if top < 0:
                    top, bottom = 0, length
                elif bottom > H:
                    top = H - length

                ann['square_amodal_bbox'] = [left, top, length, length]
                synset_name, synset = synset_finder.word_to_synset(ann['name'])
                if synset_name is not None:
                    ann['synset_name'] = synset_name
                    ann['synset'] = synset
                    imagenet_synset, imagenet_target = \
                        synset_finder.find_imagenet_synsets_targets(synset)
                    if imagenet_synset is not None:
                        ann['imagenet_synset'] = imagenet_synset
                        ann['imagenet_target'] = imagenet_target

                # make annotated images for a sample of objects
                if i < 32:
                    image = io.imread(cocoa_ann['file_name'])
                    outpath = op.join(SAMPLE_IMAGE_DIR,
                        op.basename(cocoa_ann['file_name']).replace('.jpg',
                        f'ins-{instance}_obj-{object_class}.jpg'))
                    v = Visualizer(image, cocoa_ann)
                    image = v.draw_binary_mask( binary_mask=vis, color='tab:green',
                        edge_color='tab:green', alpha=.5).get_image()
                    image = v.draw_binary_mask(binary_mask=inv, color='tab:red',
                        edge_color='tab:red', alpha=.5).get_image()
                    image = v.draw_binary_mask(binary_mask=amd, color='white',
                        edge_color='white', text=object_class, alpha=0).get_image()
                    io.imsave(outpath, image)
                    image_cropped = image[top:top+length, left:left+length]
                    io.imsave(outpath.replace('.jpg', '_cropped.jpg'),
                        image_cropped)

        # save annotations
        with open(ANNOTATIONS_PATH, 'w+') as f:
            json.dump(coco_amodal, f)


    def _home_window():
        rsp = 'x'
        first_attempt = True
        while rsp not in ['s', 'g', 'o', 'i', 'r']:
            if not first_attempt:
                print('invalid input, try again')
            rsp = input(f'\noptions:\n'
                        's: save progress\n'
                        'g: good\n'
                        'o: image does not match synset (try another word)\n'
                        'i: select subset of imagenet synsets\n'
                        'r: reject object if e.g.:\n\t'
                        'image or segmentation is poor quality\n\t'
                        'object synset is correct but '
                        'no imagenet synsets are acceptable\n\t'
                        'object is virtually unoccluded or totally occluded\n\t'
                        'object and occluder are the same category\n\t'
                        'object is not recognizable\n\n')
            first_attempt = False

        return rsp


    def _select_synsets_subset(ann, synset_finder):

        while True:
            slct = input('\noptions:\n'
                         '0 1 2: select specific synsets, '
                         'd 3 4 5: delete specific synsets, '
                         'a: accept all synsets, '
                         'o: try another word, '
                         'r: give up and reject object\n\n')
            if slct == 'r':
                included = False
                break
            if slct == 'a':
                included = True
                break
            if slct[0].isnumeric():  # keep some synsets
                idcs = [int(j) for j in slct.split(' ')]
                ann['imagenet_synset'] = [
                    ann['imagenet_synset'][j] for j in idcs]
                ann['imagenet_target'] = [
                    ann['imagenet_target'][j] for j in idcs]
                included = True
                break
            if slct.startswith('d'):  # remove synsets (can be quicker)
                idcs = [int(j) for j in slct[2:].split(' ')]
                ann['imagenet_synset'] = [k for j, k in enumerate(
                    ann['imagenet_synset']) if j not in idcs]
                ann['imagenet_target'] = [k for j, k in enumerate(
                    ann['imagenet_target']) if j not in idcs]
                included = True
                break
            if slct == 'o':
                included, ann = _new_word(ann, synset_finder)
                break
            print('invalid input, try again.')
        return included, ann

    def _new_word(ann, synset_finder):

        rsp = input(f'try another word for this object '
                     'or enter "r" to reject image')
        while True:

            if rsp == 'r':
                included = False
                break
            synset_results = wn.synsets(rsp, pos='n')
            if len(synset_results):
                print('\nOBJECT SYNSETS')
                for s, synset_obj in enumerate(synset_results):
                    print(f'{s}: {synset_obj.name()},'
                          f' {synset_obj.definition()}')
                rsp = input(f'\noptions:\n'
                      f'<number>: select the correct definition\n'
                      f'<word>: input another word to try\n'
                      f'r: reject image\n')
                if rsp.isnumeric():
                    synset = synset_results[int(rsp)]
                    ann['synset_name'] = synset.name().split('.')[0]
                    ann['synset'] = f'n{synset.offset():08}'
                    imagenet_synsets, imagenet_targets = \
                        synset_finder.find_imagenet_synsets_targets(
                            ann['synset'])
                    ann['imagenet_synset'] = imagenet_synsets
                    ann['imagenet_target'] = imagenet_targets
                    if imagenet_synsets is not None:
                        print('\nIMAGENET SYNSETS')
                        for s, imagenet_synset in enumerate(
                                imagenet_synsets):
                            imagenet_synset_obj = wn.synset_from_pos_and_offset(
                                'n', int(imagenet_synset[1:]))
                            print(f'{s}: {imagenet_synset_obj.name()},'
                                  f' {imagenet_synset_obj.definition()}')
                        included, ann = _select_synsets_subset(ann,
                                                               synset_finder)
                    else:
                        print('no imagenet synsets found, image rejected.')
                        included = False
                    break
                else:
                    print('invalid input, try again.')
            else:
                rsp = input('no synsets found, try another word or enter "r" '
                            'to reject image\n')

        return included, ann


    def _refine_image_dataset():
        """
        Go through images, ensure they are correctly labelled, and remove any
        low-quality images or objects that are occluded by other objects of
        the same class.
        """

        # make backup of annotations
        shutil.copy(ANNOTATIONS_PATH, ANNOTATIONS_PATH.replace('.json', '_backup.json'))

        # get cocoa annotations
        with open(ANNOTATIONS_PATH, 'r+') as f:
            cocoa_cls = json.load(f)

        synset_finder = SynsetFinder()
        total_images = 2320
        tested_count = 0
        included_count = 0

        for i, cocoa_ann in enumerate(cocoa_cls):


            H, W = cocoa_ann['height'], cocoa_ann['width']
            image_path = op.join(DATASET_BASE, cocoa_ann['file_name'])
            anns = [i for i in cocoa_ann['annotations'][0]['regions'] if
                    'included' not in i and 'imagenet_synset' in i]
            if not len(anns):
                tested_count += len(
                    [i for i in cocoa_ann['annotations'][0]['regions'] if
                    'imagenet_synset' in i])
                included_count += len(
                    [i for i in cocoa_ann['annotations'][0]['regions'] if
                    'included' in i and i['included']])

            for ann in anns:

                tested_count += 1

                amd, vis, inv, bbox, object_class = _read_COCOA(ann, H, W)
                bbox = ann['square_amodal_bbox']

                if bbox[2] < 112:
                    ann['included'] = False
                    continue

                # show image
                image = io.imread(image_path)
                v = Visualizer(image, cocoa_ann)
                image = v.draw_binary_mask(binary_mask=amd,
                                           color='white',
                                           edge_color='red',
                                           alpha=0).get_image()

                image = image[bbox[1]:bbox[1] + bbox[3],
                              bbox[0]:bbox[0] +  bbox[2]]
                plt.imshow(image)
                plt.show()

                # evaluate image
                included = None
                while included is None:

                    # print image metadata, object synset, and imagenet synsets
                    print('\n' * 25)
                    print(f'{now()} | {tested_count}/{total_images}, '
                          f' {included_count} included')
                    object_synset = wn.synset_from_pos_and_offset(
                        'n', int(ann['synset'][1:]))
                    print(f'\nOBJECT SYNSET\n{object_synset.name()},'
                          f' {object_synset.definition()}\n\nIMAGENET SYNSETS')
                    for s, imagenet_synset in enumerate(ann['imagenet_synset']):
                        imagenet_synset_obj = wn.synset_from_pos_and_offset(
                            'n', int(imagenet_synset[1:]))
                        print(f'{s}: {imagenet_synset_obj.name()},'
                              f' {imagenet_synset_obj.definition()}')

                    rsp = _home_window()

                    while rsp == 's':
                        print('saving progress...')
                        with open(ANNOTATIONS_PATH, 'w+') as f:
                            json.dump(cocoa_cls, f)
                        print('progress saved.')
                        rsp = _home_window()

                    if rsp == 'g':
                        included = True
                        continue

                    if rsp == 'i':
                        included, ann = _select_synsets_subset(ann, synset_finder)
                        continue

                    if rsp == 'o':
                        included, ann = _new_word(ann, synset_finder)
                        continue

                    if rsp == 'r':
                        included = False

                assert type(included) == bool
                ann['included'] = included
                if included:
                    included_count += 1

        # save annotations
        with open(ANNOTATIONS_PATH, 'w+') as f:
            json.dump(cocoa_cls, f)

    #_make_image_dataset()
    _refine_image_dataset()


class COCOA_cls_Dataset(Dataset):

    def __init__(self, transform=None):
        self.main_dir = DATASET_BASE
        with open(ANNOTATIONS_PATH, 'r+') as f:
            metadata = json.load(f)
        self.image_paths, self.bbox, self.targets = [], [], []
        for i, cocoa_ann in enumerate(metadata):
            for ann in cocoa_ann['annotations'][0]['regions']:
                if 'included' in ann and ann['included']:
                    self.image_paths.append(op.join(DATASET_BASE, cocoa_ann[
                        'file_name']))
                    self.targets.append(torch.tensor(ann['imagenet_target']))
                    self.bbox.append(ann['square_amodal_bbox'])


        #metadata = metadata.dropna()
        #self.image_paths = [op.join(IMAGE_DIR, f) for f in metadata.filename]
        #self.targets = [torch.tensor([int(j) for j in i[1:-1].split(', ')])
        #                        for i in metadata.imagenet_target]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        bbox = self.bbox[idx]
        image = image.crop(
            (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
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

    if len(results[results.benchmark == BENCHMARK]) == 2:
        return False

    print(f'{now()} | Measuring performance for {BENCHMARK}, '
          f'model: {m + 1}/{total_models} at {model_dir}')

    model = get_trained_model(model_dir, architecture, True, ['output'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    transform = get_transform(architecture, model_dir)

    dataset = COCOA_cls_Dataset(transform=transform)
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
                               for metric in ['accuracy', 'probability']}
            for cycle, output in outputs.items():
                acc = accuracy(output, targets)#.detach().cpu().item()
                for i in acc:
                    performance['accuracy'][cycle].update(i.item())
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
                f'{performance["probability"][cycle].avg_epoch:.4f})')

    # save results
    for metric, cycles in performance.items():
        for cycle, perf in cycles.items():
            new_results = pd.DataFrame({
                    'benchmark': [BENCHMARK],
                    'level_1': [DATA_TYPE],
                    'level_2': [''],
                    'path': [DATA_TYPE],
                    'cycle': [int(cycle[3:])],
                    'metric': [metric],
                    'score': [perf.avg_epoch]
                })
        results = pd.concat([results, new_results]).reset_index(drop=True)
        print(f'{cycle} {metric}: {perf.avg_epoch:.4f}')

    results.to_csv(out_path, index=False)

    return True

"""        
### CODE THAT MAY BE USEFUL IF THE CURRENT DATASET IS NOT GOOD ENOUGH ###
# if multiple synsets found
if len(synsets_matched) > 1:
    image = Image.open(op.join(IMAGE_DIR, row.filename))
    image.show()
    print(f'Which synset best matches {object_name}?\n')
    for j, synset_matched in enumerate(synsets_matched):
        print(f'{j}: {synset_matched.definition()}')
    idx = input()
    synsets_matched = [synsets_matched[int(idx)]]

# if match found, add to matched synsets
if len(synsets_matched):
    synset_matched = synsets_matched[0]
    synset_fmt = f'n{synset_matched.offset():08}'
    matched_synsets.append(synset_fmt)
    targets.append()
    continue

# look for imagenet synsets with hypernyms matching object name
for synset_result in synset_results:
    for id, hypernyms in imagenet_hypernyms.items():
        if synset_result in hypernyms:
            synsets_matched.append(id)
if len(synsets_matched):
    matched_synsets.append(synsets_matched)  # use all of these
    targets.append([int(i) for i in imagenet_df.target
                   [imagenet_df.synset.isin(synsets_matched)].values])
    continue

# else give up
matched_synsets.append('None')
targets.append(np.nan)

metadata['imagenet_synset'] = matched_synsets
metadata['target'] = targets

# check dataset
metadata = pd.read_csv(METADATA_PATH).dropna()
METADATA_PATH_IMAGENET = METADATA_PATH.replace('.csv', '_final.csv')
if op.isfile(METADATA_PATH_IMAGENET):
    metadata_final = pd.read_csv(METADATA_PATH_IMAGENET)
    start_point = metadata[metadata.filename ==
                           metadata_final.filename.values[-1]].index + 1
else:
    metadata_final = pd.DataFrame(columns=metadata.columns)
    start_point = 0

for i, row in metadata.iterrows():

    if i < start_point:
        continue

    # get data for new dataframe
    new_row = row.copy()

    # show image
    image = np.array(Image.open(op.join(IMAGE_DIR, row.filename)))
    plt.imshow(image)
    plt.show()

    # save metadata while image is checked
    metadata_final.to_csv(METADATA_PATH_IMAGENET, index=False)

    # get synset objects for this row
    synsets = row.imagenet_synset
    if synsets.startswith('['):
        synsets = synsets[2:-2].split("', '")
    else:
        synsets = [synsets]
    synsets = [imagenet_synsets[j] for j in synsets]

    # print image metadata and the name and definition of each synset
    print(f'{i+1}/{len(metadata)} | {row.object_name} ({row.filename})')
    for s, synset in enumerate(synsets):
        print(f'{s}: {synset.name()}, {synset.definition()}')

    idx = input('\n press "]" if good, '
                '"0 1 2" to keep specific synsets, '
                '"r 0 1 2" to remove specific synsets, '
                '"n bat" to rename object, '
                '"d" to reject object entirely\n\n')

    # get user input
    while True:

        if idx == 'd':  # give up on this object
            break
        if idx == ']':  # synset is good
            metadata_final = pd.concat([
                metadata_final, pd.DataFrame(new_row).T])
            break

        # if trying other synset names
        elif idx.startswith('n'):
            satisfied = False
            while not satisfied:
                base_word = idx[2:].strip()
                synset_results = wn.synsets(base_word, pos='n')
                while not len(synset_results):
                    base_word = input('no synsets found, try another word or '
                                   'enter "g" to give up')
                    if base_word == "g":
                        satisfied = True
                        break
                    else:
                        synset_results = wn.synsets(base_word, pos='n')
                print(f"\nSelect the correct definition of this object, "
                      f"or input 'n boxes' to try another rename")
                for i, synset_result in enumerate(synset_results):
                    print(f"{i}: {synset_result.definition()}")
                idx = input()
                if idx.startswith('n'):
                    continue
                synset_result = synset_results[int(idx)]
                satisfied = True

            # find in imagenet
            if synset_result in imagenet_synsets.values():
                synset_fmt = f'n{synset_result.offset():08}'
                new_row.object_name = base_word
                new_row.imagenet_synset = synset_fmt
                new_row.target = imagenet_df.target[
                    imagenet_df.synset == synset_fmt].values[0]
                metadata_final = pd.concat([
                    metadata_final, pd.DataFrame(new_row).T])
                break
            else:
                # look for imagenet synsets that are a hypernym of object
                hypernyms = set()
                synset = synset_result
                while synset.hypernyms():
                    synset = synset.hypernyms()[0]
                    hypernyms.add(synset)
                synsets_matched = [
                    i for i in imagenet_synsets.values() if i in hypernyms]

                # if multiple synsets found
                if len(synsets_matched) > 1:
                    image = Image.open(op.join(IMAGE_DIR, row.filename))
                    image.show()
                    print(f'Which synset best matches {base_word}?\n')
                    for j, synset_matched in enumerate(synsets_matched):
                        print(f'{j}: {synset_matched.definition()}')
                    idx = input()
                    synsets_matched = [synsets_matched[int(idx)]]

                # if match found, add to matched synsets
                if len(synsets_matched):
                    synset_matched = synsets_matched[0]
                    synset_fmt = f'n{synset_matched.offset():08}'
                    new_row.object_name = base_word
                    new_row.imagenet_synset = synset_fmt
                    new_row.target = imagenet_df.target[
                        imagenet_df.synset == synset_fmt].values[0]
                    metadata_final = pd.concat([
                        metadata_final, pd.DataFrame(new_row).T])
                    break

                # look for imagenet synsets with hypernyms matching object name
                synsets_matched_fmt = []
                for id, hypernyms in imagenet_synsets.items():
                    if synset_result in hypernyms:
                        synsets_matched_fmt.append(id)
                if len(synsets_matched):
                    new_row.object_name = base_word
                    new_row.imagenet_synset = synsets_matched_fmt  # use all
                    new_row.target = [int(i) for i in imagenet_df.target
                        [imagenet_df.synset.isin(synsets_matched)].values]
                    metadata_final = pd.concat([
                        metadata_final, pd.DataFrame(new_row).T])

                break

        else:
            print('incorrect input, try again')
            idx = input('\n press "]" if good, '
                        '"0 1 2" to keep specific synsets, '
                        '"r 0 1 2" to remove specific synsets, '
                        '"n bat" to rename object, '
                        '"d" to reject object entirely\n\n')
    plt.close()
metadata_final.to_csv(METADATA_PATH_IMAGENET, index=False)
"""