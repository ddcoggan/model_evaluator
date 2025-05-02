# /usr/bin/python
"""
Created by David Coggan on 2023 02 14
"""

import matplotlib
import numpy as np
COLS = [matplotlib.cm.viridis.colors[int(i)] for i in np.linspace(0, 255, 3)]
TAB20B = matplotlib.cm.tab20b.colors
TAB20C = matplotlib.cm.tab20c.colors

models = {
    'ResNet101\nclassification': {
        'ResNet101, unoccluded, classification': {
            'path': 'resnet101/xform-cont',
            'color': 'k',
            'xpos': 0,
            'architecture': 'resnet101',
            'dataset': 'no occlusion',
            'task': 'classification'
        },
        'ResNet101, artificial shapes 2, classification': {
            'path': 'resnet101/occ-art2-strong_xform-cont',
            'color': COLS[0],
            'xpos': 1,
            'architecture': 'resnet101',
            'dataset': 'artificial shapes',
            'task': 'classification'
        },
        'ResNet101, natural shapes, classification': {
            'path': 'resnet101/occ-nat-untex-strong_xform-cont',
            'color': COLS[1],
            'xpos': 2,
            'architecture': 'resnet101',
            'dataset': 'natural shapes',
            'task': 'classification'
        },
        'ResNet101, natural shapes and textures, classification': {
            'path': 'resnet101/occ-nat-strong_xform-cont',
            'color': COLS[2],
            'xpos': 3,
            'architecture': 'resnet101',
            'dataset': 'natural shapes and textures',
            'task': 'classification'
        }
    },
    'CORnet-S+\nclassification': {
        'CORnet-S+, unoccluded, classification': {
            'path': 'cornet_s_plus/xform-cont',
            'color': 'k',
            'xpos': 0,
            'architecture': 'cornet_s_plus',
            'dataset': 'no occlusion',
            'task': 'classification'
        },
        'CORnet-S+, artificial shapes 2, classification': {
            'path': 'cornet_s_plus/occ-art2-weak_xform-cont',
            'color': COLS[0],
            'xpos': 1,
            'architecture': 'cornet_s_plus',
            'dataset': 'artificial shapes',
            'task': 'classification'
        },
        'CORnet-S+, natural shapes, classification': {
            'path': 'cornet_s_plus/occ-nat-untex-weak_xform-cont',
            'color': COLS[1],
            'xpos': 2,
            'architecture': 'cornet_s_plus',
            'dataset': 'natural shapes',
            'task': 'classification'
        },
        'CORnet-S+, natural shapes and textures, classification': {
            'path': 'cornet_s_plus/occ-nat-weak_xform-cont',
            'color': COLS[2],
            'xpos': 3,
            'architecture': 'cornet_s_plus',
            'dataset': 'natural shapes and textures',
            'task': 'classification'
        }
    },
    'CORnet-S+\nSimCLR': {
        'CORnet-S+, unoccluded, SimCLR': {
            'path': 'cornet_s_plus/task-cont/transfer_unocc',
            'color': 'k',
            'xpos': 0,
            'architecture': 'cornet_s_plus',
            'dataset': 'no occlusion',
            'task': 'contrastive learning'
        },
        'CORnet-S+, artificial shapes 2, SimCLR': {
            'path': 'cornet_s_plus/occ-art2-weak_task-cont/transfer_unocc',
            'color': COLS[0],
            'xpos': 1,
            'architecture': 'cornet_s_plus',
            'dataset': 'artificial shapes',
            'task': 'contrastive learning'
        },
        'CORnet-S+, natural shapes, SimCLR': {
            'path': 'cornet_s_plus/occ-nat-untex-weak_task-cont/transfer_unocc',
            'color': COLS[1],
            'xpos': 2,
            'architecture': 'cornet_s_plus',
            'dataset': 'natural shapes',
            'task': 'contrastive learning'
        },
        'CORnet-S+, natural shapes and textures, SimCLR': {
            'path': 'cornet_s_plus/occ-nat-weak_task-cont/transfer_unocc',
            'color': COLS[2],
            'xpos': 3,
            'architecture': 'cornet_s_plus',
            'dataset': 'natural shapes and textures',
            'task': 'contrastive learning'
        }
    }
}
