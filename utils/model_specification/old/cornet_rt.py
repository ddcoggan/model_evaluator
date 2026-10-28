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
    'CORnet-RT+\nclassification': {
        'supervised, unoccluded, contrastive xform': {
            'architecture': 'cornet_rt_hw3',
            'path': 'cornet_rt_hw3/xform-cont',
            'color': 'k',
            'xpos': 0
        },
        'supervised, natural occluders (high-vis), contrastive xform': {
            'architecture': 'cornet_rt_hw3',
            'path': 'cornet_rt_hw3/occ-nat_xform-cont',
            'color': COLS[2],
            'xpos': 3
        },
        'supervised, natural occluders (all-vis), contrastive xform': {
            'architecture': 'cornet_rt_hw3',
            'path': 'cornet_rt_hw3/occ-nat_vis-beh_xform-cont',
            'color': COLS[2],
            'xpos': 4
        },
        'supervised, behavioral occluders (high-vis), contrastive xform': {
            'architecture': 'cornet_rt_hw3',
            'path': 'cornet_rt_hw3/occ-beh_xform-cont',
            'color': COLS[0],
            'xpos': 1
        },
        'supervised, behavioral occluders (all-vis), contrastive xform': {
            'architecture': 'cornet_rt_hw3',
            'path': 'cornet_rt_hw3/occ-beh_vis-beh_xform-cont',
            'color': COLS[0],
            'xpos': 2
        }
    },
    'CORnet-RT+\nSimCLR': {
        'self-supervised, unoccluded, contrastive xform': {
            'architecture': 'cornet_rt_hw3',
            'path': 'cornet_rt_hw3/task-cont',
            'color': 'k',
            'xpos': 0
        },
        'self-sup, natural occluders (high-vis), contrastive xform': {
            'architecture': 'cornet_rt_hw3',
            'path': 'cornet_rt_hw3/occ-nat_task-cont/transfer_unocc',
            'color': COLS[2],
            'xpos': 3
        },
        'self-sup, natural occluders (all-vis), contrastive xform': {
            'architecture': 'cornet_rt_hw3',
            'path': 'cornet_rt_hw3/occ-nat_vis-beh_task-cont/transfer_unocc',
            'color': COLS[2],
            'xpos': 4
        },
        'self-sup, behavioral occluders (high-vis), contrastive xform': {
            'architecture': 'cornet_rt_hw3',
            'path': 'cornet_rt_hw3/occ-beh_task-cont/transfer_unocc',
            'color': COLS[0],
            'xpos': 1
        },
        'self-sup, behavioral occluders (all-vis), contrastive xform': {
            'architecture': 'cornet_rt_hw3',
            'path': 'cornet_rt_hw3/occ-beh_vis-beh_task-cont/transfer_unocc',
            'color': COLS[0],
            'xpos': 2
        }
    }
}