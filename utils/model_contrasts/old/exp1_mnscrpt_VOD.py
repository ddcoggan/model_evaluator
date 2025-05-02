# /usr/bin/python
"""
Created by David Coggan on 2023 02 14
"""

import matplotlib
import numpy as np
from .model_properties import region_to_layer

TAB20B = matplotlib.cm.tab20b.colors
TAB20C = matplotlib.cm.tab20c.colors
cols = [matplotlib.cm.viridis.colors[int(i)] for i in np.linspace(0, 255, 3)]
#cols_light = [list(c) for c in 1 - ((1 - np.array(cols)) * .8)]
#cols_dark = [list(c) for c in np.array(cols) * .8]

properties = {
    'all': {},
    'level1': {},
    'level2': {
        'no occlusion': {'color': 'w', 'edgecolor': 'k', 'xpos': 0},
        'artificial shape': {'color': cols[0], 'xpos': 1},
        'artificial shape test': {'color': cols[0], 'edgecolor': 'k',
                                  'xpos': 2},
        'natural shape': {'color': cols[1], 'xpos': 3},
        'natural shape and texture': {'color': cols[2], 'xpos': 4},
    }}

models = {
    'CORnet-S+\nweak occlusion\nclassification': {
        'no occlusion': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/xform-cont',
        },
        'artificial shape': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-art3-weak_xform-cont',
        },
        'artificial shape test': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-art-weak_xform-cont',
        },
        'natural shape': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-nat-untex-weak_xform-cont',
        },
        'natural shape and texture': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-nat-weak_xform-cont',
        },
    },
    'ResNet101\nweak occlusion\nclassification': {
        'no occlusion': {
            'architecture': 'resnet101',
            'path': 'resnet101/xform-cont',
        },
        'artificial shape': {
            'architecture': 'resnet101',
            'path': 'resnet101/occ-art3-weak_xform-cont',
        },
        'artificial shape test': {
            'architecture': 'resnet101',
            'path': 'resnet101/occ-art-weak_xform-cont',
        },
        'natural shape': {
            'architecture': 'resnet101',
            'path': 'resnet101/occ-nat-untex-weak_xform-cont',
        },
        'natural shape and texture': {
            'architecture': 'resnet101',
            'path': 'resnet101/occ-nat-weak_xform-cont',
        },
    },
    'ResNet101\nstrong occlusion\nclassification': {
        'no occlusion': {
            'architecture': 'resnet101',
            'path': 'resnet101/xform-cont',
        },
        'artificial shape': {
            'architecture': 'resnet101',
            'path': 'resnet101/occ-art3-strong_xform-cont',
        },
        'artificial shape test': {
            'architecture': 'resnet101',
            'path': 'resnet101/occ-art-strong_xform-cont',
        },
        'natural shape': {
            'architecture': 'resnet101',
            'path': 'resnet101/occ-nat-untex-strong_xform-cont',
        },
        'natural shape and texture': {
            'architecture': 'resnet101',
            'path': 'resnet101/occ-nat-strong_xform-cont',
        },
    },
    'EfficientNet-B1\nweak occlusion\nclassification': {
        'no occlusion': {
            'architecture': 'efficientnet_b1',
            'path': 'efficientnet_b1/xform-cont',
        },
        'artificial shape': {
            'architecture': 'efficientnet_b1',
            'path': 'efficientnet_b1/occ-art3-weak_xform-cont',
        },
        'artificial shape test': {
            'architecture': 'efficientnet_b1',
            'path': 'efficientnet_b1/occ-art-weak_xform-cont',
        },
        'natural shape': {
            'architecture': 'efficientnet_b1',
            'path': 'efficientnet_b1/occ-nat-untex-weak_xform-cont',
        },
        'natural shape and texture': {
            'architecture': 'efficientnet_b1',
            'path': 'efficientnet_b1/occ-nat-weak_xform-cont',
        },
    },
    'FLaBnet\nall outputs optimized': {
        'no occlusion': {
            'architecture': 'cognet_v25',
            'path': 'cognet/v25',
        },
        'artificial shape': {
            'architecture': 'cognet_v25',  # use v25 to output all cycles
            'path': 'cognet/v26_occ-art3_all-cycles',
        },
        'artificial shape test': {
            'architecture': 'cognet_v25',  # use v25 to output all cycles
            'path': 'cognet/v26_occ-art_all-cycles',
        },
        'natural shape': {
            'architecture': 'cognet_v25',  # use v25 to output all cycles
            'path': 'cognet/v26_occ-nat-untex_all-cycles',
        },
        'natural shape and texture': {
            'architecture': 'cognet_v25',
            'path': 'cognet/v25_natural_occluders',
        },
    },
    'FLaBnet\nfinal output optimized': {
        'no occlusion': {
            'architecture': 'cognet_v25',  # use v25 to output all cycles
            'path': 'cognet/v22',
        },
        'artificial shape': {
            'architecture': 'cognet_v25',  # use v25 to output all cycles
            'path': 'cognet/v26_occ-art3_last-cycle',
        },
        'artificial shape test': {
            'architecture': 'cognet_v25',  # use v25 to output all cycles
            'path': 'cognet/v26_occ-art_last-cycle',
        },
        'natural shape': {
            'architecture': 'cognet_v25',  # use v25 to output all cycles
            'path': 'cognet/v26_occ-nat-untex_last-cycle',
        },
        'natural shape and texture': {
            'architecture': 'cognet_v25',  # use v25 to output all cycles
            'path': 'cognet/v26_occ-nat_last-cycle',
        },
    },
}

for level1, level2s in models.items():
    for level2, config in level2s.items():
        models[level1][level2] = {**config, **properties['all'],
                                  **properties['level1'][level1],
                                  **properties['level2'][level2]}