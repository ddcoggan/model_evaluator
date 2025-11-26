# /usr/bin/python
# Created by David Coggan on 2024 01 04
import os.path as op
import pandas as pd

# get all model contrasts and place in dict
from .exp1_mnscrpt_occ_type import models as exp1_mnscrpt_occ_type
from .exp1_mnscrpt_occ_type_vit import models as exp1_mnscrpt_occ_type_vit
from .exp1_mnscrpt_occ_type_pooled import models as exp1_mnscrpt_occ_type_pooled
from .exp1_mnscrpt_occ_type_vit_pooled import (
    models as exp1_mnscrpt_occ_type_vit_pooled)
from .exp1_mnscrpt_occ_strength import models as exp1_mnscrpt_occ_strength
from .exp1_mnscrpt_occ_strength_pooled import (
    models as exp1_mnscrpt_occ_strength_pooled)
from .exp1_mnscrpt_occ_strength_pooled_type_strn import (
    models as exp1_mnscrpt_occ_strength_pooled_type_strn)
from .exp1_mnscrpt_noise_blur import models as exp1_mnscrpt_noise_blur
from .exp1_mnscrpt_occ_task import models as exp1_mnscrpt_occ_task
from .exp1_mnscrpt_occ_task_pooled import models as exp1_mnscrpt_occ_task_pooled
from .exp2_mnscrpt_recurrence_v26 import models as exp2_mnscrpt_recurrence_v26
from .exp2_mnscrpt_recurrence_v27 import models as exp2_mnscrpt_recurrence_v27
#from utils.model_contrasts.exp1_mnscrpt_recurrence_last_cycle import (
#    models as exp1_mnscrpt_recurrence_last_cycle)
#from utils.model_contrasts.old.exp1_mnscrpt_recurrence_last_cycle import
# models as FLaBnet
from .public_models import models as public_models
from .pix2pix import models as pix2pix
from .vit import models as vit
from .recurrence import models as recurrence
#from utils.model_contrasts.VSS_2024_abstract import models as
# VSS_2024_abstract
#from utils.model_contrasts.CCN_2024_abstract import models as
# CCN_2024_abstract
from .. import MODEL_BASE

HUMAN_COLOR = 'tab:gray'
HUMAN_MARKER = 'P'
HUMAN_CONFIG = {'humans': {
    'color': HUMAN_COLOR,
    'edgecolor': HUMAN_COLOR,
    'marker': HUMAN_MARKER,
    'markerfillcolor': HUMAN_COLOR,
    'markeredgecolor': HUMAN_COLOR,
    'linestyle': 'solid',
    'linecolor': HUMAN_COLOR,
    'xpos': 0}}

# control which model sets are processed
model_contrasts = dict(

    public_models=public_models,

    #exp1_mnscrpt_diet=exp1_mnscrpt_diet,
    #exp1_mnscrpt_occ_type=exp1_mnscrpt_occ_type,
    #exp1_mnscrpt_occ_type_vit=exp1_mnscrpt_occ_type_vit,
    #exp1_mnscrpt_occ_type_pooled=exp1_mnscrpt_occ_type_pooled,
    #exp1_mnscrpt_occ_type_vit_pooled=exp1_mnscrpt_occ_type_vit_pooled,
    #exp1_mnscrpt_occ_strength=exp1_mnscrpt_occ_strength,
    #exp1_mnscrpt_occ_strength_pooled=exp1_mnscrpt_occ_strength_pooled,
    #exp1_mnscrpt_occ_strength_pooled_type_strn=\
    # exp1_mnscrpt_occ_strength_pooled_type_strn,
    #exp1_mnscrpt_occ_task=exp1_mnscrpt_occ_task,
    #exp1_mnscrpt_occ_task_pooled=exp1_mnscrpt_occ_task_pooled,
    #exp1_mnuscrpt_recurrence_all_cycles=exp1_mnscrpt_recurrence_all_cycles,
    #exp1_mnscrpt_recurrence_last_cycle=exp1_mnscrpt_recurrence_last_cycle,
    #exp1_mnscrpt_noise_blur=exp1_mnscrpt_noise_blur,

    #exp2_mnscrpt_recurrence_v26=exp2_mnscrpt_recurrence_v26,
    #exp2_mnscrpt_recurrence_v27=exp2_mnscrpt_recurrence_v27,

    #FLaBnet=FLaBnet,
    #FovealBlock=FovealBlock,
    #cornet_s_custom_recurrence=cornet_s_custom_recurrence,
    #learning_objective=learning_objective,
    #recurrence=recurrence,
    #pix2pix=pix2pix,
    #VSS_2024_abstract=VSS_2024_abstract,
    #VSS_2024_poster=VSS_2024_poster,
    #CCN_2024_abstract=CCN_2024_abstract,
    #CCN_2024_poster_exp1=CCN_2024_poster_exp1,
    #CCN_2024_poster_exp2=CCN_2024_poster_exp2,
    #cornet_rt=cornet_rt,
    #generative_models=generative_models,
    #cornet_s_custom_base=cornet_s_custom_base,
    #cornet_s_custom_large=cornet_s_custom_large,
    #cornet_s_unshared=cornet_s_unshared,
    #vit=vit,
    #VVRC_poster=VVRC_poster,
    #**main_effects,
)


# fill in missing properties with defaults and get batch size for evaluation
for contrast, groups in model_contrasts.items():
    for group, models in groups.items():
        for model, info in models.items():

            # independent of other properties
            if 'readout_layer' not in info:
                info['readout_layer'] = 'output'
            if 'image_size' not in info:
                info['image_size'] = 224
            if 'fillcolor' not in info:
                info['fillcolor'] = 'tab:blue'
            if 'edgecolor' not in info:
                info['edgecolor'] = 'None'
            if 'linestyle' not in info:
                info['linestyle'] = 'solid'
            if 'marker' not in info:
                info['marker'] = 'o'


            # dependent on other properties
            if 'markerfillcolor' not in info:
                info['markerfillcolor'] = info['fillcolor']
            if 'markeredgecolor' not in info:
                info['markeredgecolor'] = info['edgecolor']
            if 'linecolor' not in info:
                info['linecolor'] = info['fillcolor']
            if 'architecture' not in info:
                info['architecture'] = info['path'].split('/')[0]

# list all unique model directories and layers to analyze
all_models = {}
models_to_exclude = []
model_properties = pd.read_csv('utils/model_contrasts/model_properties.csv')
for contrast, groups in model_contrasts.items():
    for group, models in groups.items():
        for model, info in models.items():

            architecture = info['architecture']
            layer = info['readout_layer']
            path = info['path']  # dict keys are path to ensure uniqueness
            assert op.isdir(f'{MODEL_BASE}/{path}'), f'{path} not found'

            if not op.isfile(f'{MODEL_BASE}/{info["path"]}/done'):
                models_to_exclude.append([contrast, group, model])
            else:

                # if item already exists, just add any new readout layers
                if path in all_models:
                    all_models[path]['readout_layers'].add(layer)

                # else add new item
                else:
                    if architecture in model_properties.architecture.values:
                        batch_size = model_properties.batch_size[
                            model_properties.architecture == architecture
                        ].item()
                    else:
                        batch_size = 1024
                    all_models[path] = {'architecture': architecture,
                                        'readout_layers': {layer},
                                        'batch_size': batch_size,
                                        'image_size': info['image_size']}

# remove models that have not finished optimizing from model contrasts
for contrast, group, model in models_to_exclude:
    del model_contrasts[contrast][group][model]




