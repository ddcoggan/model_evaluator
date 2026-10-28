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
        },
    'level2': {
        'no occlusion': {'color': 'w', 'edgecolor': 'k'},
        'artificial 2 weak': {'color': cols_light['red']},
        'artificial 2 moderate': {'color': cols['red']},
        'artificial 2 strong': {'color': cols_dark['red']},
        'artificial 1 weak': {'color': cols_light['blue']},
        'artificial 1 moderate': {'color': cols['blue']},
        'artificial 1 strong': {'color': cols_dark['blue']},
        'natural silhouette weak': {'color': cols_light['brown']},
        'natural silhouette moderate': {'color': cols['brown']},
        'natural silhouette strong': {'color': cols_dark['brown']},
        'natural weak': {'color': cols_light['green']},
        'natural moderate': {'color': cols['green']},
        'natural strong': {'color': cols_dark['green']},
    }}

models = {
    'CORnet-S+\nclassification': {
        'no occlusion': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/xform-cont-weak-resize',
            'xpos': 0,
        },
        'natural weak': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-nat-weak_xform-cont-weak-resize',
            'xpos': 1.5,#4.5,
        },
        'natural moderate': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-nat_xform-cont-weak-resize',
            'xpos': 6,
        },
        'natural silhouette weak': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-nat-untex-weak_xform-cont-weak-resize',
            'xpos': 2.5,#3.5,
        },
        'natural silhouette moderate': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-nat-untex_xform-cont-weak-resize',
            'xpos': 7,
        },
        'artificial 1 weak': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-art-weak_xform-cont-weak-resize',
            'xpos': 3.5,#2.5,
        },
        'artificial 1 moderate': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-art_xform-cont-weak-resize',
            'xpos': 8,
        },

        'artificial 2 weak': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-art3-weak_xform-cont-weak-resize',
            'xpos': 4.5,#1.5,
        },
        'artificial 2 moderate': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-art3_xform-cont-weak-resize',
            'xpos': 9,
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
            'xpos': 0,
        },
        'natural moderate': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-nat_task-cont-weak-resize',
            'xpos': 1.5,
        },
        'natural silhouette moderate': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-nat-untex_task-cont-weak-resize',
            'xpos': 2.5,
        },
        'artificial 1 moderate': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-art_task-cont-weak-resize',
            'xpos': 3.5,
        },
        'artificial 2 moderate': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-art3_task-cont-weak-resize',
            'xpos': 4.5,
        },
    },
    'ResNet101\nclassification': {
        'no occlusion': {
            'architecture': 'resnet101',
            'path': 'resnet101/xform-cont-weak-resize',
            'xpos': 0,
        },
        'natural weak': {
            'architecture': 'resnet101',
            'path': 'resnet101/occ-nat-weak_xform-cont-weak-resize',
            'xpos': 1.5,#4.5,
        },
        'natural moderate': {
            'architecture': 'resnet101',
            'path': 'resnet101/occ-nat_xform-cont-weak-resize',
            'xpos': 6,
        },
        'natural strong': {
            'architecture': 'resnet101',
            'path': 'resnet101/occ-nat-strong_xform-cont-weak-resize',
            'xpos': 10.5,
        },
        'natural silhouette weak': {
            'architecture': 'resnet101',
            'path': 'resnet101/occ-nat-untex-weak_xform-cont-weak-resize',
            'xpos': 2.5,#3.5,
        },
        'natural silhouette moderate': {
            'architecture': 'resnet101',
            'path': 'resnet101/occ-nat-untex_xform-cont-weak-resize',
            'xpos': 7,
        },
        'natural silhouette strong': {
            'architecture': 'resnet101',
            'path': 'resnet101/occ-nat-untex-strong_xform-cont-weak-resize',
            'xpos': 11.5,
        },
        'artificial 1 weak': {
            'architecture': 'resnet101',
            'path': 'resnet101/occ-art-weak_xform-cont-weak-resize',
            'xpos': 3.5,#2.5,
        },
        'artificial 1 moderate': {
            'architecture': 'resnet101',
            'path': 'resnet101/occ-art_xform-cont-weak-resize',
            'xpos': 8,
        },
        'artificial 1 strong': {
            'architecture': 'resnet101',
            'path': 'resnet101/occ-art-strong_xform-cont-weak-resize',
            'xpos': 12.5,
        },
        'artificial 2 weak': {
            'architecture': 'resnet101',
            'path': 'resnet101/occ-art3-weak_xform-cont-weak-resize',
            'xpos': 4.5,#1.5,
        },
        'artificial 2 moderate': {
            'architecture': 'resnet101',
            'path': 'resnet101/occ-art3_xform-cont-weak-resize',
            'xpos': 9,
        },
        'artificial 2 strong': {
            'architecture': 'resnet101',
            'path': 'resnet101/occ-art3-strong_xform-cont-weak-resize',
            'xpos': 13.5,
        },
    },
    'EfficientNet-B1\nclassification': {
        'no occlusion': {
            'architecture': 'efficientnet_b1',
            'path': 'efficientnet_b1/xform-cont-weak-resize',
            'xpos': 0,
        },
        'natural weak': {
            'architecture': 'efficientnet_b1',
            'path': 'efficientnet_b1/occ-nat-weak_xform-cont-weak-resize',
            'xpos': 1.5,
        },
        'natural moderate': {
            'architecture': 'efficientnet_b1',
            'path': 'efficientnet_b1/occ-nat_xform-cont-weak-resize',
            'xpos': 6,
        },
        'natural strong': {
            'architecture': 'efficientnet_b1',
            'path': 'efficientnet_b1/occ-nat-strong_xform-cont-weak-resize',
            'xpos': 10.5,
        },
        'natural silhouette weak': {
            'architecture': 'efficientnet_b1',
            'path': 'efficientnet_b1/occ-nat-untex-weak_xform-cont-weak-resize',
            'xpos': 2.5,
        },
        'natural silhouette moderate': {
            'architecture': 'efficientnet_b1',
            'path': 'efficientnet_b1/occ-nat-untex_xform-cont-weak-resize',
            'xpos': 7,
        },
        'natural silhouette strong': {
            'architecture': 'efficientnet_b1',
            'path': 'efficientnet_b1/occ-nat-untex-strong_xform-cont-weak-resize',
            'xpos': 11.5,
        },
        'artificial 1 weak': {
            'architecture': 'efficientnet_b1',
            'path': 'efficientnet_b1/occ-art-weak_xform-cont-weak-resize',
            'xpos': 3.5,
        },
        'artificial 1 moderate': {
            'architecture': 'efficientnet_b1',
            'path': 'efficientnet_b1/occ-art_xform-cont-weak-resize',
            'xpos': 8,
        },
        'artificial 1 strong': {
            'architecture': 'efficientnet_b1',
            'path': 'efficientnet_b1/occ-art-strong_xform-cont-weak-resize',
            'xpos': 12.5,
        },
        'artificial 2 weak': {
            'architecture': 'efficientnet_b1',
            'path': 'efficientnet_b1/occ-art3-weak_xform-cont-weak-resize',
            'xpos': 4.5,
        },
        'artificial 2 moderate': {
            'architecture': 'efficientnet_b1',
            'path': 'efficientnet_b1/occ-art3_xform-cont-weak-resize',
            'xpos': 9,
        },
        'artificial 2 strong': {
            'architecture': 'efficientnet_b1',
            'path': 'efficientnet_b1/occ-art3-strong_xform-cont-weak-resize',
            'xpos': 13.5,
        },
    },
}

for level1, level2s in models.items():
    for level2, config in level2s.items():
        models[level1][level2] = {**config, **properties['all'],
                                  **properties['level1'][level1],
                                  **properties['level2'][level2]}