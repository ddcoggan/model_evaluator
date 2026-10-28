# /usr/bin/python
"""
Created by David Coggan on 2023 02 14
contrasts for larger model developed for better contrastive learning
"""

import matplotlib
TAB20B = matplotlib.cm.tab20b.colors
TAB20C = matplotlib.cm.tab20c.colors
cols = matplotlib.cm.viridis.colors[::(256//3)]

models = {
    #'supervised, unoccluded, standard transform (pretrained)': {
    #    'path': 'cornet_s/pretrained',
    #    'color': TAB20C[0],
    #},
    'supervised\nCNNs': {
        'unoccluded, supervised': {
            'path': 'cornet_s_plus/xform-cont',
            'color': cols[0],#TAB20C[0],
            'xpos': 0,
        },
        'artificial shapes, supervised': {
            'path': 'cornet_s_plus/occ-art-weak_xform-cont',
            'color': cols[1],#TAB20C[4],
            'xpos': 1,
        },
        'natural shapes, supervised': {
            'path': 'cornet_s_plus/occ-nat-untex-weak_xform'
                    '-cont',
            'color': cols[2],#TAB20C[8],
            'xpos': 2,
        },
        'natural shapes and textures, supervised': {
            'path': 'cornet_s_plus/occ-nat-weak_xform-cont',
            'color': cols[3],  # TAB20B[4],
            'xpos': 3,
        },
    },
    'self-supervised\nCNNs': {
        'unoccluded, self-supervised': {
            'path': 'cornet_s_plus/task-cont/transfer_unocc',
            'color': cols[0],#TAB20C[0],
            'xpos': 0,
        },      
        'artificial shapes, self-supervised': {
            'path': 'cornet_s_plus/occ-art-weak_task-cont/transfer_unocc',
            'color': cols[1],#TAB20C[4],
            'xpos': 1,
        },
        'natural shapes, self-supervised': {
            'path': 'cornet_s_plus/occ-nat-untex-weak_task'
                    '-cont/transfer_unocc',
            'color': cols[2],#TAB20C[8],
            'xpos': 2,
        },   
        'natural shapes and textures, self-supervised': {
            'path': 'cornet_s_plus/occ-nat-weak_task-cont/transfer_unocc',
            'color': cols[3],#TAB20B[4],
            'xpos': 3,
        },
    },
}
