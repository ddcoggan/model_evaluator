"""
Created by David Coggan on 2023 02 14
contrasts for publicly available pretrained models
"""

import sys
import os.path as op
import os
import warnings
import torch
import torchvision.models
import pandas as pd
import glob
import numpy as np
import shutil
import gc
from .. import MODEL_BASE
from ..calculate_batch_size import calculate_batch_size


# make dict of torchvision architecture names, lower case: camel case
models_lower_case = [i for i in torchvision.models.list_models() if
    hasattr(torchvision.models, i)]  # some listed models are not available
[models_lower_case.remove(i) for i in [
    'regnet_x_1_6gf',  # duplicate regnet entries that are misspelled
    'regnet_x_3_2gf',
    'regnet_y_1_6gf',
    'regnet_y_3_2gf']]
model_names = {}
for lower_case in models_lower_case:
    model_names[lower_case] = [i.split('_Weights')[0] for i in
        torchvision.models.__dict__ if i.split('_Weights')[0] != lower_case and
        i.split('_Weights')[0].lower() == lower_case][0]

# process each architecture and weights set
models = {}
for a, (arch, arch_camel) in enumerate(models_tv.items()):

    # get list of available pretrained weights
    weight_names = [i for i in getattr(torchvision.models,
        f'{arch_camel}_Weights').__dict__ if 'IMAGENET' in i and
        'FEATURES' not in i]

    for weight_name in weight_names:

        # make model directory
        model_dir = f'{MODEL_BASE}/{arch}/pretrained_{weight_name}'
        params_dir = f'{model_dir}/params'
        os.makedirs(params_dir, exist_ok=True)

        # download weights
        metadata = getattr(getattr(torchvision.models,
                                f'{arch_camel}_Weights'), weight_name)
        weights_url = metadata.url
        weights_path = f'{params_dir}/{op.basename(weights_url)}'
        if not op.isfile(weights_path):
            # directly downloaded weights are incompatible for some models
            if arch.startswith('densenet') or arch.startswith('vit'):
                model = getattr(torchvision.models, arch)(weights=weight_name)
                torch.save(model.state_dict(), weights_path)
            else:  # standard method
                torch.hub.download_url_to_file(weights_url, weights_path)
            with open(f'{model_dir}/done', 'w') as _:  # create 'done' file
                pass

        color = COLS[model_counter % len(COLS)]
        edgecolor = EDGECOLORS[int(model_counter / len(COLS))]
        marker = MARKERS[int(model_counter / len(COLS))]
        model_label = arch_camel
        if len(weight_names) > 1:
            model_label += f'_{weight_name}'
        models[f'{arch}/pretrained_{weight_name}'] = {
            'architecture': arch,
            'readout_layer': 'output'}


# store model properties in a csv file
import pandas as pd

def select_final_embedding_layer(model):
    submodules = list(model.named_modules())

    # get last conv layer
    for i in submodules[::-1]:
        if isinstance(i[1], torch.nn.Conv2d):
            selected_module = i
            break

    # get activations after norm or LU layers
    for j in submodules[submodules.index(i)+1:]:
        module_name = str(type(j[1]))
        if any([k in module_name for k in ['LU', 'Norm']]):
            selected_module = j
        else:
            return selected_module[0]

properties_path = 'utils/model_contrasts/model_properties_24GB.csv'
if op.isfile(properties_path):
    model_properties = pd.read_csv(properties_path)
else:
    model_properties = pd.DataFrame()
new_models = False
for a, (arch, arch_camel) in enumerate(model_names.items()):
    if not len(model_properties) or arch not in \
            model_properties['architecture'].values:
        new_models = True

        # get model
        model = getattr(torchvision.models, arch)()

        # calculate batch size
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad() and torch.autocast(device_type=device.type,
                                dtype=torch.float16):
            batch_size = min(2048, calculate_batch_size(model, device))

        # batch size for efficientnet_b7 is hardcoded as 64 (128 fails with AMP)
        # https://github.com/pytorch/pytorch/issues/80020
        if arch == 'efficientnet_b7':
            batch_size = 64

        # number of trainable parameters
        num_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)

        # get final embedding layer
        layer = select_final_embedding_layer(model)

        # append to dataframe
        model_properties = pd.concat([model_properties, pd.DataFrame({
            'architecture': arch,
            'architecture_camel': arch_camel,
            'batch_size': batch_size,
            'num_params': num_params,
            'final_embedding_layer': layer}, index=[0])])

        del model
        gc.collect()
        torch.cuda.empty_cache()

if new_models:
    model_properties.to_csv(properties_path, index=False)
