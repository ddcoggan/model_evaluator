# /usr/bin/python
"""
Created by David Coggan on 2023 02 14
"""

import matplotlib
import numpy as np
TAB20B = matplotlib.cm.tab20b.colors
TAB20C = matplotlib.cm.tab20c.colors
cols = [matplotlib.cm.viridis.colors[int(i)] for i in np.linspace(0, 255, 4)]


models = {
    'supervised CNNs': {
        'CORnet-S+, unoccluded, classification': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/xform-cont',
            'color': 'k',  # cols[0],#TAB20C[0],
            'xpos': 0,
        },
        'CORnet-S+, artificial shape, classification': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-art-weak_xform-cont',
            'color': cols[1],  # TAB20C[4],
            'xpos': 1,
        },
        'CORnet-S+, natural shape, classification': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-nat-untex-weak_xform-cont',
            'color': cols[2],  # TAB20C[4],
            'xpos': 2,
        },
        'CORnet-S+, natural shape and texture, classification': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-nat-weak_xform-cont',
            'color': cols[3],  # TAB20C[4],
            'xpos': 3,
        },
    },
    'self-supervised CNNs': {
        'CORnet-S+, unoccluded, contrastive': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/task-cont',
            'color': 'k',  # cols[0],#TAB20C[0],
            'xpos': 0,
        },
        'CORnet-S+, artificial shape, contrastive': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-art-weak_task-cont',
            'color': cols[1],  # TAB20C[4],
            'xpos': 1,
        },
        'CORnet-S+, natural shape, contrastive': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-nat-untex-weak_task-cont',
            'color': cols[2],  # TAB20C[4],
            'xpos': 2,
        },
        'CORnet-S+, natural shape and texture, contrastive': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s_plus/occ-nat-weak_task-cont',
            'color': cols[3],  # TAB20C[4],
            'xpos': 3,
        },
    },
}