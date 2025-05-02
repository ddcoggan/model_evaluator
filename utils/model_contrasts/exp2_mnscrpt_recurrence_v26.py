# /usr/bin/python
"""
Created by David Coggan on 2023 02 14
"""

import matplotlib
import numpy as np
TAB20B = matplotlib.cm.tab20b.colors
TAB20C = matplotlib.cm.tab20c.colors
cols = [matplotlib.cm.viridis.colors[int(i)] for i in np.linspace(0, 255, 3)]

properties = {
    'all': {'readout_layer': 'output', 'marker': 'o'},
    'level1': {},
    'level2': {}}

models = {
    'FLaBnet v26': {
        'no occlusion - clear': {
            'architecture': 'cognet_v25',
            'path': 'cognet/v25',
            'xpos': 0,
            'color': 'w',
            'edgecolor': 'k',
            'markercolor': 'w',
            'markeredgecolor': 'k',
            'linestyle': 'solid',
            'linecolor': 'k',
        },
        'no occlusion - clear to blurry': {
            'architecture': 'cognet_v25',
            'path': 'cognet/v26_clear2blurry',
            'xpos': 1,
            'color': 'tab:red',
            'edgecolor': 'k',
            'markercolor': 'w',
            'markeredgecolor': 'k',
            'linestyle': 'solid',
            'linecolor': 'k',
        },
        'no occlusion - blurry to clear': {
            'architecture': 'cognet_v25',
            'path': 'cognet/v26_blurry2clear',
            'xpos': 2,
            'color': 'tab:green',
            'edgecolor': 'k',
            'markercolor': 'w',
            'markeredgecolor': 'k',
            'linestyle': 'solid',
            'linecolor': 'k',
        },
        'artificial shape': {
            'architecture': 'cognet_v25',  # use v25 to output all cycles
            'path': 'cognet/v26_occ-art3_all-cycles',
            'xpos': 4,
            'color': cols[0],
            'edgecolor': 'None',
            'markercolor': cols[0],
            'markeredgecolor': 'None',
            'linestyle': 'solid',
            'linecolor': cols[0],
        },
        'natural shape': {
            'architecture': 'cognet_v25',  # use v25 to output all cycles
            'path': 'cognet/v26_occ-nat-untex_all-cycles',
            'xpos': 5,
            'color': cols[1],
            'edgecolor': 'None',
            'markercolor': cols[1],
            'markeredgecolor': 'None',
            'linestyle': 'solid',
            'linecolor': cols[1],
        },
        'natural shape and texture': {
            'architecture': 'cognet_v25',
            'path': 'cognet/v25_natural_occluders',
            'xpos': 6,
            'color': cols[2],
            'edgecolor': 'None',
            'markercolor': cols[2],
            'markeredgecolor': 'None',
            'linestyle': 'solid',
            'linecolor': cols[2],
        },
    },
}

for level1, level2s in models.items():
    for level2, config in level2s.items():
        models[level1][level2] = {
            **config,
            **properties['all'],
            #**properties['level1'][level1],
            #**properties['level2'][level2]
        }