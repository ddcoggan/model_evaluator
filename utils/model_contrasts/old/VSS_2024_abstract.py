# /usr/bin/python
"""
Created by David Coggan on 2023 02 14
contrasts for larger model developed for better contrastive learning
"""

import matplotlib
TAB20B = matplotlib.cm.tab20b.colors
TAB20C = matplotlib.cm.tab20c.colors

models = {
    'supervised': {
        'supervised, unoccluded (contrastive xform)': {
            'path': 'cornet_s_plus/xform-cont',
            'color': TAB20C[1],
        },
        'supervised, natural occluders (contrastive xform)': {
            'path': 'cornet_s_plus/occ-nat_xform-cont',
            'color': TAB20C[9],
        },
        'supervised, natural untext. occluders (contrastive xform)': {
                'path': 'cornet_s_plus/occ-nat-untex_xform-cont',
                'color': TAB20B[5],
            },
        'supervised, behavioral occluders (contrastive xform)': {
                'path': 'cornet_s_plus/occ-beh_xform-cont',
                'color': TAB20C[5],
            },
    },
    'self-supervised': {
        'self-sup., unoccluded (contrastive xform)': {
            'path': 'cornet_s_plus/task-cont/transfer_unocc',
            'color': TAB20C[2],
        },
        'self-sup., natural occluders (contrastive xform)': {
            'path': 'cornet_s_plus/occ-nat_task-cont/'
                    'transfer_unocc',
            'color': TAB20C[10],
        },
        'self-sup., behavioral occluders (contrastive xform)': {
            'path': 'cornet_s_plus/occ-beh_task-cont/'
                    'transfer_unocc',
            'color': TAB20C[6],
        },
    },
}
