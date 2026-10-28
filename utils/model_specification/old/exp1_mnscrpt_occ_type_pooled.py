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
            'color': 'tab:red', 'linestyle': 'solid', 'xpos': 4.5},
        'artificial 1': {
            'color': 'tab:blue', 'linestyle': 'solid', 'xpos': 3.5},
        'natural silhouette': {
            'color': 'tab:brown', 'linestyle': 'solid', 'xpos': 2.5},
        'natural': {
            'color': 'tab:green', 'linestyle': 'solid', 'xpos': 1.5},
    },
    'level2': {
        'CORnet-S+\nno occlusion\nclassification': {
            'readout_layer': 'output'},
        'CORnet-S+\nweak occlusion\nclassification': {
            'readout_layer': 'output'},
        'CORnet-S+\nmoderate occlusion\nclassification': {
            'readout_layer': 'output'},
        'CORnet-S+\nno occlusion\nSimCLR': {
            'readout_layer': 'IT.output'},
        'CORnet-S+\nmoderate occlusion\nSimCLR': {
            'readout_layer': 'IT.output'},
        'ResNet101\nno occlusion\nclassification': {
            'readout_layer': 'output'},
        'ResNet101\nweak occlusion\nclassification': {
            'readout_layer': 'output'},
        'ResNet101\nmoderate occlusion\nclassification': {
            'readout_layer': 'output'},
        'ResNet101\nstrong occlusion\nclassification': {
            'readout_layer': 'output'},
        'EfficientNet-B1\nno occlusion\nclassification': {
            'readout_layer': 'output'},
        'EfficientNet-B1\nweak occlusion\nclassification': {
            'readout_layer': 'output'},
        'EfficientNet-B1\nmoderate occlusion\nclassification': {
            'readout_layer': 'output'},
        'EfficientNet-B1\nstrong occlusion\nclassification': {
            'readout_layer': 'output'}
    }}

models = {
    'no occlusion': {
        'CORnet-S+\nno occlusion\nclassification': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/xform-cont-weak-resize',
        },
        'CORnet-S+\nno occlusion\nSimCLR': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/task-cont-weak-resize',
        },
        'ResNet101\nno occlusion\nclassification': {
            'architecture': 'resnet101',
            'path': 'resnet101/xform-cont-weak-resize',
        },
        'EfficientNet-B1\nno occlusion\nclassification': {
            'architecture': 'efficientnet_b1',
            'path': 'efficientnet_b1/xform-cont-weak-resize',
        },
    },
    'artificial 2': {
        'CORnet-S+\nweak occlusion\nclassification': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-art3-weak_xform-cont-weak-resize',
        },
        'CORnet-S+\nmoderate occlusion\nclassification': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-art3_xform-cont-weak-resize',
        },
        'CORnet-S+\nmoderate occlusion\nSimCLR': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-art3_task-cont-weak-resize',
        },
        'ResNet101\nweak occlusion\nclassification': {
            'architecture': 'resnet101',
            'path': 'resnet101/occ-art3-weak_xform-cont-weak-resize',
        },
        'ResNet101\nmoderate occlusion\nclassification': {
            'architecture': 'resnet101',
            'path': 'resnet101/occ-art3_xform-cont-weak-resize',
        },
        'ResNet101\nstrong occlusion\nclassification': {
            'architecture': 'resnet101',
            'path': 'resnet101/occ-art3-strong_xform-cont-weak-resize',
        },
        'EfficientNet-B1\nweak occlusion\nclassification': {
            'architecture': 'efficientnet_b1',
            'path': 'efficientnet_b1/occ-art3_xform-cont-weak-resize',
        },
        'EfficientNet-B1\nmoderate occlusion\nclassification': {
            'architecture': 'efficientnet_b1',
            'path': 'efficientnet_b1/occ-art_xform-cont-weak-resize',
        },
        'EfficientNet-B1\nstrong occlusion\nclassification': {
            'architecture': 'efficientnet_b1',
            'path': 'efficientnet_b1/occ-art3-strong_xform-cont-weak-resize',
        },
    },
    'artificial 1': {
        'CORnet-S+\nweak occlusion\nclassification': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-art-weak_xform-cont-weak-resize',
        },
        'CORnet-S+\nmoderate occlusion\nclassification': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-art_xform-cont-weak-resize',
        },
        'CORnet-S+\nmoderate occlusion\nSimCLR': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-art_task-cont-weak-resize',
        },
        'ResNet101\nweak occlusion\nclassification': {
            'architecture': 'resnet101',
            'path': 'resnet101/occ-art-weak_xform-cont-weak-resize',
        },
        'ResNet101\nmoderate occlusion\nclassification': {
            'architecture': 'resnet101',
            'path': 'resnet101/occ-art_xform-cont-weak-resize',
        },
        'ResNet101\nstrong occlusion\nclassification': {
            'architecture': 'resnet101',
            'path': 'resnet101/occ-art-strong_xform-cont-weak-resize',
        },
        'EfficientNet-B1\nweak occlusion\nclassification': {
            'architecture': 'efficientnet_b1',
            'path': 'efficientnet_b1/occ-art-weak_xform-cont-weak-resize',
        },
        'EfficientNet-B1\nmoderate occlusion\nclassification': {
            'architecture': 'efficientnet_b1',
            'path': 'efficientnet_b1/occ-art_xform-cont-weak-resize',
        },
        'EfficientNet-B1\nstrong occlusion\nclassification': {
            'architecture': 'efficientnet_b1',
            'path': 'efficientnet_b1/occ-art-strong_xform-cont-weak-resize',
        },
    },
    'natural silhouette': {
        'CORnet-S+\nweak occlusion\nclassification': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-nat-untex-weak_xform-cont-weak-resize',
        },
        'CORnet-S+\nmoderate occlusion\nclassification': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-nat-untex_xform-cont-weak-resize',
        },
        'CORnet-S+\nmoderate occlusion\nSimCLR': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-nat-untex_task-cont-weak-resize',
        },
        'ResNet101\nweak occlusion\nclassification': {
            'architecture': 'resnet101',
            'path': 'resnet101/occ-nat-untex-weak_xform-cont-weak-resize',
        },
        'ResNet101\nmoderate occlusion\nclassification': {
            'architecture': 'resnet101',
            'path': 'resnet101/occ-nat-untex_xform-cont-weak-resize',
        },
        'ResNet101\nstrong occlusion\nclassification': {
            'architecture': 'resnet101',
            'path': 'resnet101/occ-nat-untex-strong_xform-cont-weak-resize',
        },
        'EfficientNet-B1\nweak occlusion\nclassification': {
            'architecture': 'efficientnet_b1',
            'path': 'efficientnet_b1/occ-nat-untex-weak_xform-cont-weak-resize',
        },
        'EfficientNet-B1\nmoderate occlusion\nclassification': {
            'architecture': 'efficientnet_b1',
            'path': 'efficientnet_b1/occ-nat-untex_xform-cont-weak-resize',
        },
        'EfficientNet-B1\nstrong occlusion\nclassification': {
            'architecture': 'efficientnet_b1',
            'path': 'efficientnet_b1/occ-nat-untex-strong_xform-cont-weak-resize',
        },
    },
    'natural': {
        'CORnet-S+\nweak occlusion\nclassification': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-nat-weak_xform-cont-weak-resize',
        },
        'CORnet-S+\nmoderate occlusion\nclassification': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-nat_xform-cont-weak-resize',
        },
        'CORnet-S+\nmoderate occlusion\nSimCLR': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-nat_task-cont-weak-resize',
        },
        'ResNet101\nweak occlusion\nclassification': {
            'architecture': 'resnet101',
            'path': 'resnet101/occ-nat-weak_xform-cont-weak-resize',
        },
        'ResNet101\nmoderate occlusion\nclassification': {
            'architecture': 'resnet101',
            'path': 'resnet101/occ-nat_xform-cont-weak-resize',
        },
        'ResNet101\nstrong occlusion\nclassification': {
            'architecture': 'resnet101',
            'path': 'resnet101/occ-nat-strong_xform-cont-weak-resize',
        },
        'EfficientNet-B1\nweak occlusion\nclassification': {
            'architecture': 'efficientnet_b1',
            'path': 'efficientnet_b1/occ-nat-weak_xform-cont-weak-resize',
        },
        'EfficientNet-B1\nmoderate occlusion\nclassification': {
            'architecture': 'efficientnet_b1',
            'path': 'efficientnet_b1/occ-nat_xform-cont-weak-resize',
        },
        'EfficientNet-B1\nstrong occlusion\nclassification': {
            'architecture': 'efficientnet_b1',
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