# /usr/bin/python
"""
Created by David Coggan on 2023 02 14
contrasts for publically available pretrained models
"""

import sys
import os.path as op
import matplotlib
import numpy as np
TAB20B = matplotlib.cm.tab20b.colors
TAB20C = matplotlib.cm.tab20c.colors
COLS = [matplotlib.cm.viridis.colors[int(i)] for i in np.linspace(0, 255, 3)]

models = {
    'PredNet\nprediction': {
        'PredNet': {
            'path': 'prednet/pretrained',
            'color': 'k',
            'xpos': 0,
        },
    },
    'Pix2Pix\nreconstruction': {
        'Pix2Pix, artificial shapes, reconstruction': {
            'path': 'pix2pix/occ-art2',
            'color': COLS[0],#TAB20C[4],
            'xpos': 0,
        },
        'Pix2Pix, natural shapes, reconstruction': {
            'path': 'pix2pix/occ-nat-untex',
            'color': COLS[1],#TAB20C[8],
            'xpos': 1,
        },
        'Pix2Pix, natural shapes and textures, reconstruction': {
            'path': 'pix2pix/occ-nat',
            'color': COLS[2],#TAB20C[8],
            'xpos': 2,
        },
    },
}
