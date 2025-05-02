"""
Created by David Coggan on 2023 02 14
"""

import matplotlib
import numpy as np

TAB20C = matplotlib.cm.tab20c.colors
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
    'all': {'readout_layer': 'output'},
    'level1': {
        'no occlusion': {'color': 'w', 'edgecolor': 'k', 'xpos': 0},
        'artificial 1\nweak occlusion': {
            'color': cols_light['red'], 'xpos': 1.5},
        'artificial 1\nmoderate occlusion': {
            'color': cols['red'], 'xpos': 2.5},
        'artificial 1\nstrong occlusion': {
            'color': cols_dark['red'], 'xpos': 3.5},
        'artificial 3\nweak occlusion': {
            'color': cols_light['blue'], 'xpos': 5},
        'artificial 3\nmoderate occlusion': {
            'color': cols['blue'], 'xpos': 6},
        'artificial 3\nstrong occlusion': {
            'color': cols_dark['blue'], 'xpos': 7},
        'natural silhouette\nweak occlusion': {
            'color': cols_light['brown'], 'xpos': 8.5},
        'natural silhouette\nmoderate occlusion': {
            'color': cols['brown'], 'xpos': 9.5},
        'natural silhouette\nstrong occlusion': {
            'color': cols_dark['brown'], 'xpos': 10.5},
        'natural\nweak occlusion': {
            'color': cols_light['green'], 'xpos': 12},
        'natural\nmoderate occlusion': {
            'color': cols['green'], 'xpos': 13},
        'natural\nstrong occlusion': {
            'color': cols_dark['green'], 'xpos': 14},
    },
    'level2': {
        'ResNet101': {'architecture': 'resnet101', 'linestyle': 'solid'},
        'EfficientNet-B1': {'architecture': 'efficientnet_b1',
                            'linestyle': 'dashed'},
    }}

models = {
    'no occlusion': {
        'ResNet101': {
            'path': 'resnet101/xform-cont-weak-resize',
        },
        'EfficientNet-B1': {
            'path': 'efficientnet_b1/xform-cont-weak-resize',
        },
    },
    'artificial 1\nweak occlusion': {
        'ResNet101': {
            'path': 'resnet101/occ-art3-weak_xform-cont-weak-resize',
        },
        'EfficientNet-B1': {
            'path': 'efficientnet_b1/occ-art3-weak_xform-cont-weak-resize',
        },
    },
    'artificial 1\nmoderate occlusion': {
        'ResNet101': {
            'path': 'resnet101/occ-art3_xform-cont-weak-resize',
        },
        'EfficientNet-B1': {
            'path': 'efficientnet_b1/occ-art3_xform-cont-weak-resize',
        },
    },
    'artificial 1\nstrong occlusion': {
        'ResNet101': {
            'path': 'resnet101/occ-art3-strong_xform-cont-weak-resize',
        },
        'EfficientNet-B1': {
            'path': 'efficientnet_b1/occ-art3-strong_xform-cont-weak-resize',
        },
    },
    'artificial 3\nweak occlusion': {
        'ResNet101': {
            'path': 'resnet101/occ-art-weak_xform-cont-weak-resize',
        },
        'EfficientNet-B1': {
            'path': 'efficientnet_b1/occ-art-weak_xform-cont-weak-resize',
        },
    },
    'artificial 3\nmoderate occlusion': {
        'ResNet101': {
            'path': 'resnet101/occ-art_xform-cont-weak-resize',
        },
        'EfficientNet-B1': {
            'path': 'efficientnet_b1/occ-art_xform-cont-weak-resize',
        },
    },
    'artificial 3\nstrong occlusion': {
        'ResNet101': {
            'path': 'resnet101/occ-art-strong_xform-cont-weak-resize',
        },
        'EfficientNet-B1': {
            'path': 'efficientnet_b1/occ-art-strong_xform-cont-weak-resize',
        },
    },
    'natural silhouette\nweak occlusion': {
        'ResNet101': {
            'path': 'resnet101/occ-nat-untex-weak_xform-cont-weak-resize',
        },
        'EfficientNet-B1': {
            'path': 'efficientnet_b1/occ-nat-untex-weak_xform-cont-weak-resize',
        },
    },
    'natural silhouette\nmoderate occlusion': {
        'ResNet101': {
            'path': 'resnet101/occ-nat-untex_xform-cont-weak-resize',
        },
        'EfficientNet-B1': {
            'path': 'efficientnet_b1/occ-nat-untex_xform-cont-weak-resize',
        },
    },
    'natural silhouette\nstrong occlusion': {
        'ResNet101': {
            'path': 'resnet101/occ-nat-untex-strong_xform-cont-weak-resize',
        },
        'EfficientNet-B1': {
            'path': 'efficientnet_b1/occ-nat-untex-strong_xform-cont-weak-resize',
        },
    },
    'natural\nweak occlusion': {
        'ResNet101': {
            'path': 'resnet101/occ-nat-weak_xform-cont-weak-resize',
        },
        'EfficientNet-B1': {
            'path': 'efficientnet_b1/occ-nat-weak_xform-cont-weak-resize',
        },
    },
    'natural\nmoderate occlusion': {
        'ResNet101': {
            'path': 'resnet101/occ-nat_xform-cont-weak-resize',
        },
        'EfficientNet-B1': {
            'path': 'efficientnet_b1/occ-nat_xform-cont-weak-resize',
        },
    },
    'natural\nstrong occlusion': {
        'ResNet101': {
            'path': 'resnet101/occ-nat-strong_xform-cont-weak-resize',
        },
        'EfficientNet-B1': {
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