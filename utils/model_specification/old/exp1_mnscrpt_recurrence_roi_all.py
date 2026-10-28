"""
Created by David Coggan on 2023 02 14
"""

import matplotlib
import numpy as np
TAB20B = matplotlib.cm.tab20b.colors
TAB20C = matplotlib.cm.tab20c.colors
cols = [matplotlib.cm.viridis.colors[int(i)] for i in np.linspace(0, 255, 3)]

properties = {
    'all': {'architecture': 'cognet_v25'},
    'level1': {
        'V1': {'readout_layer': 'V1.f', 'marker': 'o'},
        'V2': {'readout_layer': 'V2.f', 'marker': '^'},
        'V4': {'readout_layer': 'V4.f', 'marker': 's'},
        'IT': {'readout_layer': 'IT.f', 'marker': 'p'},
    },
    'level2': {
        'no occlusion': {
            'path': 'cognet/v25',
            'xpos': 0,
            'color': 'w',
            'edgecolor': 'k',
            'markercolor': 'w',
            'markeredgecolor': 'k',
            'linestyle': 'solid',
            'linecolor': 'k',
        },
        'artificial shape': {
            'path': 'cognet/v26_occ-art3_all-cycles',
            'xpos': 1,
            'color': cols[0],
            'edgecolor': 'None',
            'markercolor': cols[0],
            'markeredgecolor': 'None',
            'linestyle': 'solid',
            'linecolor': cols[0],
        },
        'natural shape': {
            'path': 'cognet/v26_occ-nat-untex_all-cycles',
            'xpos': 2,
            'color': cols[1],
            'edgecolor': 'None',
            'markercolor': cols[1],
            'markeredgecolor': 'None',
            'linestyle': 'solid',
            'linecolor': cols[1],
        },
        'natural shape and texture': {
            'path': 'cognet/v25_natural_occluders',
            'xpos': 3,
            'color': cols[2],
            'edgecolor': 'None',
            'markercolor': cols[2],
            'markeredgecolor': 'None',
            'linestyle': 'solid',
            'linecolor': cols[2],
        },
    }}

models = {k1: {k2: {**properties['all'], **v1, **v2} for k2, v2 in properties[
    'level2'].items()} for k1, v1 in properties['level1'].items()}
