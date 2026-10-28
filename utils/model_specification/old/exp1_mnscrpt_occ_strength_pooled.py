"""
Created by David Coggan on 2023 02 14
"""

import matplotlib
import numpy as np

TAB20C = matplotlib.cm.tab20c.colors

properties = {
    'all': {'readout_layer': 'output'},
    'level1': {
        'no occlusion': {'color': 'w', 'edgecolor': 'k', 'xpos': 0},
        'weak occlusion': {
            'color': TAB20C[6], 'xpos': 1},
        'moderate occlusion': {
            'color': TAB20C[5], 'xpos': 2},
        'strong occlusion': {
            'color': TAB20C[4], 'xpos': 3},
    },
    'level2': {
        'no occlusion\nResNet101': {
            'architecture': 'resnet101', 'linestyle': 'solid'},
        'no occlusion\nEfficientNet-B1': {
            'architecture': 'efficientnet_b1', 'linestyle': 'dashed'},
        'artificial 2\nResNet101': {
            'architecture': 'resnet101', 'linestyle': 'solid'},
        'artificial 2\nEfficientNet-B1': {
            'architecture': 'efficientnet_b1', 'linestyle': 'dashed'},
        'artificial 1\nResNet101': {
            'architecture': 'resnet101', 'linestyle': 'solid'},
        'artificial 1\nEfficientNet-B1': {
            'architecture': 'efficientnet_b1', 'linestyle': 'dashed'},
        'natural silhouette\nResNet101': {
            'architecture': 'resnet101', 'linestyle': 'solid'},
        'natural silhouette\nEfficientNet-B1': {
            'architecture': 'efficientnet_b1', 'linestyle': 'dashed'},
        'natural\nResNet101': {
            'architecture': 'resnet101', 'linestyle': 'solid'},
        'natural\nEfficientNet-B1': {
            'architecture': 'efficientnet_b1', 'linestyle': 'dashed'},
    }}

models = {
    'no occlusion': {
        'no occlusion\nResNet101': {
            'path': 'resnet101/xform-cont-weak-resize',
        },
        'no occlusion\nEfficientNet-B1': {
            'path': 'efficientnet_b1/xform-cont-weak-resize',
        },
    },
    'weak occlusion': {
        'artificial 2\nResNet101': {
            'path': 'resnet101/occ-art3-weak_xform-cont-weak-resize',
        },
        'artificial 2\nEfficientNet-B1': {
            'path': 'efficientnet_b1/occ-art3-weak_xform-cont-weak-resize',
        },
        'artificial 1\nResNet101': {
            'path': 'resnet101/occ-art-weak_xform-cont-weak-resize',
        },
        'artificial 1\nEfficientNet-B1': {
            'path': 'efficientnet_b1/occ-art-weak_xform-cont-weak-resize',
        },
        'natural silhouette\nResNet101': {
            'path': 'resnet101/occ-nat-untex-weak_xform-cont-weak-resize',
        },
        'natural silhouette\nEfficientNet-B1': {
            'path': 'efficientnet_b1/occ-nat-untex-weak_xform-cont-weak-resize',
        },
        'natural\nResNet101': {
            'path': 'resnet101/occ-nat-weak_xform-cont-weak-resize',
        },
        'natural\nEfficientNet-B1': {
            'path': 'efficientnet_b1/occ-nat-weak_xform-cont-weak-resize',
        },
    },
    'moderate occlusion': {
        'artificial 2\nResNet101': {
            'path': 'resnet101/occ-art3_xform-cont-weak-resize',
        },
        'artificial 2\nEfficientNet-B1': {
            'path': 'efficientnet_b1/occ-art3_xform-cont-weak-resize',
        },
        'artificial 1\nResNet101': {
            'path': 'resnet101/occ-art_xform-cont-weak-resize',
        },
        'artificial 1\nEfficientNet-B1': {
            'path': 'efficientnet_b1/occ-art_xform-cont-weak-resize',
        },
        'natural silhouette\nResNet101': {
            'path': 'resnet101/occ-nat-untex_xform-cont-weak-resize',
        },
        'natural silhouette\nEfficientNet-B1': {
            'path': 'efficientnet_b1/occ-nat-untex_xform-cont-weak-resize',
        },
        'natural\nResNet101': {
            'path': 'resnet101/occ-nat_xform-cont-weak-resize',
        },
        'natural\nEfficientNet-B1': {
            'path': 'efficientnet_b1/occ-nat_xform-cont-weak-resize',
        },
    },
    'strong occlusion': {
        'artificial 2\nResNet101': {
            'path': 'resnet101/occ-art3-strong_xform-cont-weak-resize',
        },
        'artificial 2\nEfficientNet-B1': {
            'path': 'efficientnet_b1/occ-art3-strong_xform-cont-weak-resize',
        },
        'artificial 1\nResNet101': {
            'path': 'resnet101/occ-art-strong_xform-cont-weak-resize',
        },
        'artificial 1\nEfficientNet-B1': {
            'path': 'efficientnet_b1/occ-art-strong_xform-cont-weak-resize',
        },
        'natural silhouette\nResNet101': {
            'path': 'resnet101/occ-nat-untex-strong_xform-cont-weak-resize',
        },
        'natural silhouette\nEfficientNet-B1': {
            'path': 'efficientnet_b1/occ-nat-untex-strong_xform-cont-weak-resize',
        },
        'natural\nResNet101': {
            'path': 'resnet101/occ-nat-strong_xform-cont-weak-resize',
        },
        'natural\nEfficientNet-B1': {
            'path': 'efficientnet_b1/occ-nat-strong_xform-cont-weak-resize',
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