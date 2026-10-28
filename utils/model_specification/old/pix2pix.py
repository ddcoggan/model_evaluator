# /usr/bin/python
"""
Created by David Coggan on 2023 02 14
"""

import matplotlib
import numpy as np
TAB20B = matplotlib.cm.tab20b.colors
TAB20C = matplotlib.cm.tab20c.colors
cols = [matplotlib.cm.viridis.colors[int(i)] for i in np.linspace(0, 255, 3)]

models = {
    'Pix2Pix\nreconstruction': {
        'Pix2Pix, artificial shapes, reconstruction': {
            'path': 'pix2pix/occ-art2',
            'dataset': 'artificial shapes',
            'task': 'reconstruction',
            'color': cols[0],#TAB20C[4],
            'xpos': 0,
        },
        'Pix2Pix, natural shapes, reconstruction': {
            'path': 'pix2pix/occ-nat-untex',
            'dataset': 'natural shapes',
            'task': 'reconstruction',
            'color': cols[1],#TAB20C[8],
            'xpos': 1,
        },
        'Pix2Pix, natural shapes and textures, reconstruction': {
            'path': 'pix2pix/occ-nat',
            'dataset': 'natural shapes and textures',
            'task': 'reconstruction',
            'color': cols[2],#TAB20C[8],
            'xpos': 2,
        },
    }
}
