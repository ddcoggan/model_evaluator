"""
Created by David Coggan on 2026 09 24
"""
import os
import os.path as op
from .. import MODEL_BASE


model_ids = [
    "vit_large_patch14_clip_224:openai_ft_in1k",
    "convnext_large_mlp:clip_laion2b_augreg_ft_in1k_384",
]

models = {}
for model_id in model_ids:

    # add model to dict
    model_dir = op.join('brainscore_models', model_id)
    models[model_id] = {model_id: {'architecture': 'brainscore_models', 'path': model_dir}}

    # make model dir
    model_dir_abs = op.join(MODEL_BASE, model_dir)
    if not op.isdir(model_dir_abs):
        os.makedirs(model_dir_abs)

    # flag model as done
    with open(f'{model_dir_abs}/done', 'w') as _:
        pass
