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
    'level2': {},
}

models = {
    'FLaBnet': {
        'clear': {
            'architecture': 'cognet_v27',
            'path': 'cognet/v27_noblur',
            'xpos': 0,
            'color': 'w',
            'edgecolor': 'k',
            'markercolor': 'w',
            'markeredgecolor': 'k',
            'linestyle': 'solid',
            'linecolor': 'k',
        },
        'blurry to clear': {
            'architecture': 'cognet_v27',  # use v25 to output all cycles
            'path': 'cognet/v27_blurry2clear',
            'xpos': 1,
            'color': cols[0],
            'edgecolor': 'None',
            'markercolor': cols[0],
            'markeredgecolor': 'None',
            'linestyle': 'solid',
            'linecolor': cols[0],
        },
        'clear to blurry': {
            'architecture': 'cognet_v27',  # use v25 to output all cycles
            'path': 'cognet/v27_clear2blurry',
            'xpos': 2,
            'color': cols[1],
            'edgecolor': 'None',
            'markercolor': cols[1],
            'markeredgecolor': 'None',
            'linestyle': 'solid',
            'linecolor': cols[1],
        },
    },
}

#for level1, level2s in models.items():
#    for level2, config in level2s.items():
#        models[level1][level2] = {
#            **config,
#            **properties['all'],
#            **properties['level1'][level1],
#            **properties['level2'][level2]}