# /usr/bin/python
# Created by David Coggan on 2024 01 04
import os.path as op
import pandas as pd
from .. import MODEL_BASE

# control which model sets are processed
model_sets = [
    #'public_models',
    #'brainscore_models',
    'occlusion_rare',
]

# create dict with each unique model instance
all_models = {}
for model_set in model_sets:

    # import the list of models for this set
    models = __import__(f'utils.model_contrasts.{model_set}', fromlist=['models']).models

    # loop through each model and add to the dict of all models
    for path, info in models.items():

        # set default properties
        if 'readout_layer' not in info:
            info['readout_layer'] = 'output'
        if 'image_size' not in info:
            info['image_size'] = 224
        if 'architecture' not in info:
            info['architecture'] = info['path'].split('/')[0]

        # assert that the model directory exists
        assert op.isdir(f'{MODEL_BASE}/{path}'), f'{path} not found'

        # skip models that have not finished optimizing
        if not op.isfile(f'{MODEL_BASE}/{info["path"]}/done'):
            print(f'{path} not finished optimizing, skipping evaluation.')
            continue

        # if item already exists, add any new readout layers and move on
        if path in all_models:
            all_models[path]['readout_layers'].add(info['readout_layer'])
            continue

        # add new item
        # get batch size from model properties file if available
        if info['architecture'] in model_properties.architecture.values:
            batch_size = model_properties.batch_size[
                model_properties.architecture == info['architecture']
            ].item()
        else:
            batch_size = 1024

        all_models[path] = {
            'architecture': info['architecture'],
            'readout_layers': {info['readout_layer']},
            'batch_size': batch_size,
            'image_size': info['image_size'],
        }





