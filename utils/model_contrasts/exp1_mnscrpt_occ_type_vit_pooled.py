"""
Created by David Coggan on 2023 02 14
"""

import matplotlib.cm as cm
import numpy as np

TAB20B = cm.tab20b.colors
TAB20C = cm.tab20c.colors
cols = [cm.viridis.colors[int(i)] for i in np.linspace(0, 255, 3)]

properties = {
    'all': {},
    'level1': {
        'no occlusion': {
            'color': 'w', 'edgecolor': 'k', 'linestyle': 'solid', 'xpos': 0},
        'artificial 2': {
            'color': 'tab:red', 'linestyle': 'solid', 'xpos': 4},
        'artificial 1': {
            'color': 'tab:blue', 'linestyle': 'solid', 'xpos': 3},
        'natural silhouette': {
            'color': 'tab:brown', 'linestyle': 'solid', 'xpos': 2},
        'natural': {
            'color': 'tab:green', 'linestyle': 'solid', 'xpos': 1},
    },
    'level2': {
        'CORnet-S+\nclassification': {
            'readout_layer': 'output'},
        'CORnet-S+\nSimCLR': {
            'readout_layer': 'IT.output'},
        'ResNet101\nclassification': {
            'readout_layer': 'output'},
        'EfficientNet-B1\nclassification': {
            'readout_layer': 'output'},
        'ViT-B/16\nclassification': {
            'readout_layer': 'output'},
    }}

models = {
    'no occlusion': {
        'CORnet-S+\nclassification': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/xform-cont-weak-resize',
        },
        'CORnet-S+\nSimCLR': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/task-cont-weak-resize',
        },
        'ResNet101\nclassification': {
            'architecture': 'resnet101',
            'path': 'resnet101/xform-cont-weak-resize',
        },
        'EfficientNet-B1\nclassification': {
            'architecture': 'efficientnet_b1',
            'path': 'efficientnet_b1/xform-cont-weak-resize',
        },
        'ViT-B/16\nclassification': {
            'architecture': 'vit_b_16',
            'path': 'vit_b_16/xform-cont-weak-resize',
        },
    },
    'artificial 2': {
        'CORnet-S+\nclassification': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-art3_xform-cont-weak-resize',
        },
        'CORnet-S+\nSimCLR': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-art3_task-cont-weak-resize',
        },
        'ResNet101\nclassification': {
            'architecture': 'resnet101',
            'path': 'resnet101/occ-art3_xform-cont-weak-resize',
        },
        'EfficientNet-B1\nclassification': {
            'architecture': 'efficientnet_b1',
            'path': 'efficientnet_b1/occ-art3_xform-cont-weak-resize',
        },
        'ViT-B/16\nclassification': {
            'architecture': 'vit_b_16',
            'path': 'vit_b_16/occ-art3_xform-cont-weak-resize',
        },
    },
    'artificial 1': {
        'CORnet-S+\nclassification': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-art_xform-cont-weak-resize',
        },
        'CORnet-S+\nSimCLR': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-art_task-cont-weak-resize',
        },
        'ResNet101\nclassification': {
            'architecture': 'resnet101',
            'path': 'resnet101/occ-art_xform-cont-weak-resize',
        },
        'EfficientNet-B1\nclassification': {
            'architecture': 'efficientnet_b1',
            'path': 'efficientnet_b1/occ-art_xform-cont-weak-resize',
        },
        'ViT-B/16\nclassification': {
            'architecture': 'vit_b_16',
            'path': 'vit_b_16/occ-art_xform-cont-weak-resize',
        },
    },
    'natural silhouette': {
        'CORnet-S+\nclassification': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-nat-untex_xform-cont-weak-resize',
        },
        'CORnet-S+\nSimCLR': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-nat-untex_task-cont-weak-resize',
        },
        'ResNet101\nclassification': {
            'architecture': 'resnet101',
            'path': 'resnet101/occ-nat-untex_xform-cont-weak-resize',
        },
        'EfficientNet-B1\nclassification': {
            'architecture': 'efficientnet_b1',
            'path': 'efficientnet_b1/occ-nat-untex_xform-cont-weak-resize',
        },
        'ViT-B/16\nclassification': {
            'architecture': 'vit_b_16',
            'path': 'vit_b_16/occ-nat-untex_xform-cont-weak-resize',
        },
    },
    'natural': {
        'CORnet-S+\nclassification': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-nat_xform-cont-weak-resize',
        },
        'CORnet-S+\nSimCLR': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-nat_task-cont-weak-resize',
        },
        'ResNet101\nclassification': {
            'architecture': 'resnet101',
            'path': 'resnet101/occ-nat_xform-cont-weak-resize',
        },
        'EfficientNet-B1\nclassification': {
            'architecture': 'efficientnet_b1',
            'path': 'efficientnet_b1/occ-nat_xform-cont-weak-resize',
        },
        'ViT-B/16\nclassification': {
            'architecture': 'vit_b_16',
            'path': 'vit_b_16/occ-nat_xform-cont-weak-resize',
        },
    },
}

for level1, level2s in models.items():
    for level2, config in level2s.items():
        models[level1][level2] = {
            **config,
            **properties['all'],
            **properties['level1'][level1],
            **properties['level2'][level2],
        }