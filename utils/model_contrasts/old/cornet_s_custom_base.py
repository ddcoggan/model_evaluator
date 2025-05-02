# /usr/bin/python
"""
Created by David Coggan on 2023 02 14
contrasts for base model plus minor architectural changes
"""

import matplotlib
TAB20B = matplotlib.cm.tab20b.colors
TAB20C = matplotlib.cm.tab20c.colors

models = {
    'cornet_s_custom': {
        'cornet-s, pretrained': {
            'architecture': 'cornet_s',
            'path': 'cornet_s/pretrained',
            'color': TAB20B[11],
        },
        'supervised, unoccluded': {
            'architecture': 'cornet_s_custom',
            'path': 'cornet_s_custom/base-model',
            'color': TAB20C[0],
        },
        'supervised, unoccluded (more features': {
            'architecture': 'cornet_s_custom',
            'path': 'cornet_s_custom/feat-512',
            'color': TAB20B[16],
        },
        'supervised, unoccluded (larger kernel': {
            'architecture': 'cornet_s_custom',
            'path': 'cornet_s_custom/kern-5',
            'color': TAB20B[17],
        },
        'supervised, unoccluded (no recurrence': {
            'architecture': 'cornet_s_custom',
            'path': 'cornet_s_custom/rec-0',
            'color': TAB20B[18],
        },
        'supervised, unoccluded (double recurrence': {
            'architecture': 'cornet_s_custom',
            'path': 'cornet_s_custom/rec-2x',
            'color': TAB20B[19],
        },
        'supervised, horz bar occluders': {
            'architecture': 'cornet_s_custom',
            'path': 'cornet_s_custom/occ-fmri',
            'color': TAB20B[10],
        },
        'supervised, behav. occluders': {
            'architecture': 'cornet_s_custom',
            'path': 'cornet_s_custom/occ-beh',
            'color': TAB20C[4],
        },
        'supervised, natural occluders': {
            'architecture': 'cornet_s_custom',
            'path': 'cornet_s_custom/occ-nat',
            'color': TAB20C[8],
        },
        'finetune_mixed, behav. occluders': {
            'architecture': 'cornet_s_custom',
            'path': 'cornet_s_custom/base-model/'
                    'finetune_task-class-cont_occ-beh_high-vis',
            'color': TAB20B[14],
        },
        'finetune_mixed, natural occluders': {
            'architecture': 'cornet_s_custom',
            'path': 'cornet_s_custom/base-model/'
                    'finetune_task-class-cont_occ-nat_high-vis',
            'color': TAB20B[15],
        },
    },
}
