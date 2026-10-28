"""
Created by David Coggan on 2026 10 28
Template for setting up model evaluation

This file should contain a dictionary called models, e.g.:

models = {
    'cornet_s_plus/pretrained': {
        'architecture': 'cornet_s_plus',
        'readout_layer': ['IT', 'output']
        'weights': '043'
    },
    'brainscore_models/convnext_large_mlp:clip_laion2b_augreg_ft_in1k_38': {
        'architecture': 'brainscore_models'
        'readout_layer': 'output'
    },
}

The top level keys are the paths to each model directory, starting after the
MODEL_BASE directory specified at 'utils/__init__.py'. This ensures that
each model has a unique identifier (its path), which helps avoid repeating
evaluations.

Each item in the dictionary is another dictionary containing evaluation
settings specific to this model. Not every item below needs to be specified,
as default settings will be applied for missing items at a later point.

    'architecture': the string used by 'utils/get_model.py' to load the
        architecture. Default is to use the base folder of the model directory.
    'architecture_args': a dictionary of any key-word arguments that need
        passing to the model when initializing it, e.g. {'pretrained': True}.
        Default is {}.
    'readout_layer': list or string corresponding to the nn.module(s) of the
        model from which responses are to be used for evaluation. Default is
        'output', which is always the output of the final model layer.
    'weights': string to determine which weights are used, when more
        than one set appears is available in <model_dir>/params/.
        'final': last epoch of training (when weights are named by train epoch)
        'best': looks for 'best_???.pt'
        '<actual_weights_filename>': not including path, must be in params dir.
"""

models = {
    'cornet_s_plus/pretrained': {},
}
