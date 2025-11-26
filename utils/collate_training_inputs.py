'''
This scripts collates sample training inputs for quick inspection
'''
import os
import os.path as op
import glob
import sys
import matplotlib.pyplot as plt
from .model_contrasts import model_contrasts
from .behavioral_compare_models_exp1 import get_group_counts
from . import MODEL_BASE

def collate_training_inputs(overwrite=False):

    for (model_contrast, model_config) in model_contrasts.items():

        outdir = f'../p022_occlusion/data/in_silico/analysis/{model_contrast}'
        outpath = op.join(outdir, 'sample_training_inputs.svg')
        if not op.isfile(outpath) or overwrite:
            os.makedirs(outdir, exist_ok=True)
            num_models = sum(get_group_counts(model_config))
            figsize = (num_models*2, 3)
            fig, axes = plt.subplots(ncols=num_models, figsize=figsize)
            model_counter = 0
            for g, (group, models) in enumerate(model_config.items()):
                for m, (model, info) in enumerate(models.items()):
                    if 'pretrained' in info['path']:
                        continue
                    ax = axes[model_counter]
                    image_path = op.join(MODEL_BASE, info['path'],
                                         'sample_train_inputs', 'tiled.png')
                    if not op.isfile(image_path):
                        image_path = op.join(MODEL_BASE, info['path'],
                                             'sample_train_inputs.png')
                    assert op.isfile(image_path), 'training inputs not found'
                    ax.imshow(plt.imread(image_path))
                    ax.axis('off')
                    ax.set_title(f'{group}\n{model}')
                    model_counter += 1
            plt.tight_layout()
            plt.savefig(outpath)
            plt.savefig(outpath.replace('.svg', '.pdf'))
            plt.close()

