# /usr/bin/python
"""
Created by David Coggan on 2023 02 14
contrasts for larger model developed for better contrastive learning
"""

import matplotlib
TAB20B = matplotlib.cm.tab20b.colors
TAB20C = matplotlib.cm.tab20c.colors

models = {
    'CORnet_s_plus': {
        'supervised, unoccluded (contrastive xform)': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/xform-cont',
            'color': TAB20C[1],
        },
        'self-sup., unoccluded (contrastive xform)': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/task-cont/transfer_unocc',
            'color': TAB20C[2],
        },
        'supervised, natural occluders (standard xform)': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-nat',
            'color': TAB20C[8],
        },
        'supervised, natural occluders (contrastive xform)': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-nat_xform-cont',
            'color': TAB20C[9],
        },
        'self-sup., natural occluders (contrastive xform)': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-nat_task-cont/transfer_unocc',
            'color': TAB20C[10],
        },
        'supervised, natural untext. occluders (contrastive xform)': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-nat-untex_xform-cont',
            'color': TAB20B[5],
        },
        'supervised, behavioral occluders (standard xform)': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-beh',
            'color': TAB20C[4],
        },
        'supervised, behavioral occluders (contrastive xform)': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-beh_xform-cont',
            'color': TAB20C[5],
        },
        'self-sup., behavioral occluders (contrastive xform)': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-beh_task-cont/'
                    'transfer_unocc',
            'color': TAB20C[6],
        },
    },
}
