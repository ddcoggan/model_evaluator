"""
Created by David Coggan on 2023 02 14
"""

import matplotlib
import numpy as np


TAB20B = matplotlib.cm.tab20b.colors
TAB20C = matplotlib.cm.tab20c.colors
#cols = [matplotlib.cm.viridis.colors[int(i)] for i in np.linspace(0, 255, 3)]
#cols_light = [list(c) for c in 1 - ((1 - np.array(cols)) * .8)]
#cols_dark = [list(c) for c in np.array(cols) * .8]

cols = dict(
    blue=(0.122, 0.467, 0.706),
    red=(0.839, 0.153, 0.157),
    brown=(0.549, 0.337, 0.294),
    green=(0.172, 0.627, 0.172),
)
cols_light = {}
cols_dark = {}
for color, rgb in cols.items():
    cols_light[color] = 1 - ((1 - np.array(rgb)) * .7)
    cols_dark[color] = np.array(rgb) * .7

properties = {
    'all': {},
    'level1': {
        'CORnet-S+\nclassification': {
            'linestyle': 'solid', 'readout_layer': 'output'},
        'CORnet-S+\nSimCLR': {
            'linestyle': 'solid',
            'readout_layer': 'IT.output'},
        'ResNet101\nclassification': {
            'linestyle': 'dashed', 'readout_layer': 'output'},
        'EfficientNet-B1\nclassification': {
            'linestyle': 'dotted', 'readout_layer': 'output'},
        'ViT-B/16\nclassification': {
            'linestyle': 'dashdot', 'readout_layer': 'output'},
        },
    'level2': {
        'no occlusion': {'color': 'w', 'edgecolor': 'k', 'xpos': 0},
        'natural': {'color': cols['green'], 'xpos': 1},
        'natural silhouette': {'color': cols['brown'], 'xpos': 2},
        'artificial 1': {'color': cols['blue'], 'xpos': 3},
        'artificial 2': {'color': cols['red'], 'xpos': 4},
    }}

models = {
    'CORnet-S+\nclassification': {
        'no occlusion': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/xform-cont-weak-resize',
        },
        'natural': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-nat_xform-cont-weak-resize',
        },
        'natural silhouette': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-nat-untex_xform-cont-weak-resize',
        },
        'artificial 1': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-art_xform-cont-weak-resize',
        },
        'artificial 2': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-art3_xform-cont-weak-resize',
        },

    },
    'CORnet-S+\nSimCLR': {
        #'CORnet-S+, unoccluded, SimCLR (strong resize)': {
        #    'architecture': 'cornet_s_plus',
        #    'path': 'cornet_s_plus/task-cont_strong-resize',
        #},
        'no occlusion': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/task-cont-weak-resize',
        },
        'natural': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-nat_task-cont-weak-resize',
        },
        'natural silhouette': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-nat-untex_task-cont-weak-resize',
        },
        'artificial 1': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-art_task-cont-weak-resize',
        },
        'artificial 2': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-art3_task-cont-weak-resize',
        },
    },
    'ResNet101\nclassification': {
        'no occlusion': {
            'architecture': 'resnet101',
            'path': 'resnet101/xform-cont-weak-resize',
        },
        'natural': {
            'architecture': 'resnet101',
            'path': 'resnet101/occ-nat_xform-cont-weak-resize',
        },
        'natural silhouette': {
            'architecture': 'resnet101',
            'path': 'resnet101/occ-nat-untex_xform-cont-weak-resize',
        },
        'artificial 1': {
            'architecture': 'resnet101',
            'path': 'resnet101/occ-art_xform-cont-weak-resize',
        },
        'artificial 2': {
            'architecture': 'resnet101',
            'path': 'resnet101/occ-art3_xform-cont-weak-resize',
        },
    },
    'EfficientNet-B1\nclassification': {
        'no occlusion': {
            'architecture': 'efficientnet_b1',
            'path': 'efficientnet_b1/xform-cont-weak-resize',
        },
        'natural': {
            'architecture': 'efficientnet_b1',
            'path': 'efficientnet_b1/occ-nat_xform-cont-weak-resize',
        },
        'natural silhouette': {
            'architecture': 'efficientnet_b1',
            'path': 'efficientnet_b1/occ-nat-untex_xform-cont-weak-resize',
        },
        'artificial 1': {
            'architecture': 'efficientnet_b1',
            'path': 'efficientnet_b1/occ-art_xform-cont-weak-resize',
        },
        'artificial 2': {
            'architecture': 'efficientnet_b1',
            'path': 'efficientnet_b1/occ-art3_xform-cont-weak-resize',
        },
    },
    'ViT-B/16\nclassification': {
        'no occlusion': {
            'architecture': 'vit_b_16',
            'path': 'vit_b_16/xform-cont-weak-resize',
        },
        'natural': {
            'architecture': 'vit_b_16',
            'path': 'vit_b_16/occ-nat_xform-cont-weak-resize',
        },
        'natural silhouette': {
            'architecture': 'vit_b_16',
            'path': 'vit_b_16/occ-nat-untex_xform-cont-weak-resize',
        },
        'artificial 1': {
            'architecture': 'vit_b_16',
            'path': 'vit_b_16/occ-art_xform-cont-weak-resize',
        },
        'artificial 2': {
            'architecture': 'vit_b_16',
            'path': 'vit_b_16/occ-art3_xform-cont-weak-resize',
        },
    },
}

for level1, level2s in models.items():
    for level2, config in level2s.items():
        models[level1][level2] = {**config, **properties['all'],
                                  **properties['level1'][level1],
                                  **properties['level2'][level2]}