# /usr/bin/python
"""
Created by David Coggan on 2023 02 14
contrasts for publically available pretrained models
"""

import matplotlib
TABCOLS = matplotlib.cm.tab10.colors

models = {
    'ViT_B_16': {
        'pretrained_IMAGENET1K_V1': {
            'path': 'vit_b_16/pretrained_IMAGENET1K_V1',
            'color': TABCOLS[1],
            'xpos': 0,
            #'image_size': 384,
        },
        'pretrained_IMAGENET1K_SWAG_E2E_V1': {
            'path': 'vit_b_16/pretrained_IMAGENET1K_SWAG_E2E_V1',
            'color': TABCOLS[4],
            'xpos': 1,
            'image_size': 384,
        },
        'pretrained_IMAGENET1K_SWAG_LINEAR_V1': {
            'path': 'vit_b_16/pretrained_IMAGENET1K_SWAG_LINEAR_V1',
            'color': TABCOLS[6],
            'xpos': 2,
        },
        'no occlusion': {
            'path': 'vit_b_16/xform-cont-weak-resize',
            'color': 'w',
            'linecolor': 'k',
            'xpos': 3.5,
            #'image_size': 384,
        },
        'natural': {
            'path': 'vit_b_16/occ-nat_xform-cont-weak-resize',
            'color': 'tab:green',
            'xpos': 4.5,
            #'image_size': 384,
        },
        'natural silhouette': {
            'path': 'vit_b_16/occ-nat-untex_xform-cont-weak-resize',
            'color': 'tab:brown',
            'xpos': 5.5,
            #'image_size': 384,
        },
        'artificial 1': {
            'path': 'vit_b_16/occ-art_xform-cont-weak-resize',
            'color': 'tab:blue',
            'xpos': 6.5,
            #'image_size': 384,
        },
        'artificial 2': {
            'path': 'vit_b_16/occ-art3_xform-cont-weak-resize',
            'color': 'tab:red',
            'xpos': 7.5,
            #'image_size': 384,
        },
    }
}
