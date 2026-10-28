# /usr/bin/python
"""
Created by David Coggan on 2023 02 14
"""

import matplotlib
import numpy as np
TAB20B = matplotlib.cm.tab20b.colors
TAB20C = matplotlib.cm.tab20c.colors

properties = {
    'all': {'readout_layer': 'IT.output'},
    #'level1': {},
    'level1': {
        'supervised classification': {
            'color': 'tab:pink', 'xpos': 0},
        'self-supervised contrastive learning': {
            'color': 'tab:purple', 'xpos': 1},
        #'self-supervised contrastive learning (no occlusion, weak resize)': {
        #    'color': 'tab:purple', 'xpos': 1},
        #'self-supervised contrastive learning (no occlusion, strong resize)': {
        #    'color': 'tab:orange', 'xpos': 2},
        #'self-supervised contrastive learning (one view occluded,
        # weak resize)': {
        #    'color': 'tab:purple', 'xpos': 1},
        #'self-supervised contrastive learning (both views occluded,
        # weak resize)': {
        #    'color': 'tab:purple', 'xpos': 2},
    }}

models = {
    'supervised classification': {
        'no occlusion': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/xform-cont-weak-resize',
        },
        'artificial\nshape 1': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-art3_xform-cont-weak-resize',
        },
        'artificial\nshape 3': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-art_xform-cont-weak-resize',
        },
        'natural\nshape': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-nat-untex_xform-cont-weak-resize',
        },
        'natural\nshape and\ntexture': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-nat_xform-cont-weak-resize',
        },
    },
    'self-supervised contrastive learning': {
        'no occlusion': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/task-cont-weak-resize',
        },
        'artificial\nshape 1': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-art3_task-cont-weak-resize',
        },
        'artificial\nshape 3': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-art_task-cont-weak-resize',
        },
        'natural\nshape': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-nat-untex_task-cont-weak-resize',
        },
        'natural\nshape and\ntexture': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-nat_task-cont-weak-resize',
        },
    },
}

for level1, level2s in models.items():
    for level2, config in level2s.items():
        models[level1][level2] = {
            **config,
            **properties['all'],
            **properties['level1'][level1],
            #**properties['level2'][level2],
        }