# /usr/bin/python
"""
Created by David Coggan on 2023 02 14
contrasts for base model with and without recurrence
"""

import matplotlib
TAB20C = matplotlib.cm.tab20c.colors

models = {
    'CORnet-S+\nunoccluded': {
        'supervised, unnoccluded, standard recurrence': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/xform-cont',
            'color': TAB20C[4],
            'xpos': 0
        },
        'supervised, unnoccluded, no recurrence': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_rec-0_xform-cont',
            'color': TAB20C[6],
            'xpos': 1
        },
        'self-sup., unnoccluded, standard recurrence': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/task-cont/transfer_unocc',
            'color': TAB20C[0],
            'xpos': 2
        },
        'self-sup., unnoccluded, no recurrence': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_rec-0_task-cont/transfer_unocc',
            'color': TAB20C[2],
            'xpos': 3
        }
    },
    'CORnet-S+\nartificial occ.': {
        'supervised, behavioral occluders, standard recurrence': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-beh_xform-cont',
            'color': TAB20C[4],
            'xpos': 0
        },
        'supervised, behavioral occluders, no recurrence': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_rec-0_occ-beh_xform-cont',
            'color': TAB20C[6],
            'xpos': 1
        },
        'self sup., behavioral occluders, standard recurrence': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-beh_task-cont/transfer_unocc',
            'xpos': 2,
            'color': TAB20C[0]
        },
        'self sup., behavioral occluders, no recurrence': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_rec-0_occ-beh_task-cont/transfer_unocc',
            'color': TAB20C[2],
            'xpos': 3
        }
    },
    'CORnet-S+\nnatural occ.': {
        'supervised, natural occluders, standard recurrence': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-nat-weak_xform-cont',
            'color': TAB20C[4],
            'xpos': 0
        },
        'supervised, natural occluders, no recurrence': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_rec-0_occ-nat_xform-cont',
            'color': TAB20C[6],
            'xpos': 1
        },
        'self sup., natural occluders, standard recurrence': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-nat-weak_task-cont/transfer_unocc',
            'color': TAB20C[0],
            'xpos': 2
        },
        'self sup., natural occluders, no recurrence': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_custom/hd-2_hw-3_V1f-128_rec-0_occ-nat_task-cont/transfer_unocc',
            'color': TAB20C[2],
            'xpos': 3
        }
    }
}