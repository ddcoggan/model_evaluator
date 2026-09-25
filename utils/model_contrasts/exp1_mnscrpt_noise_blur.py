# Created by David Coggan on 2025 03 13
import matplotlib
import numpy as np
TAB20 = matplotlib.cm.tab20.colors
TAB20B = matplotlib.cm.tab20b.colors
TAB20C = matplotlib.cm.tab20c.colors
#cols = [matplotlib.cm.viridis.colors[int(i)] for i in np.linspace(0, 255, 3)]
#cols_light = [list(c) for c in 1 - ((1 - np.array(cols)) * .8)]
#cols_dark = [list(c) for c in np.array(cols) * .8]

cols = dict(
    blue=(0.122, 0.467, 0.706),
    red=(0.839, 0.153, 0.157),
    brown=(0.549, 0.337, 0.294),
    green=(0.172, 0.627, 0.172),
)


properties = {
    'all': {'architecture': 'resnet101', 'readout_layer': 'output'},
    'level1': {},
    'level2': {},
}

models = {
    'No augmentation': {
        'ResNet101': {
            'path': 'resnet101/xform-cont-no-blur-weak-resize',
            'color': 'w', 'edgecolor': 'k',
            'xpos': 0,
        },
    },
    #'Noise': {
    #    'Fourier': {
    #        'path': 'resnet101/noise-fourier_xform-weak-resize',
    #        'color': 'tab:blue',
    #        'xpos': 0,
    #    },
    #    'Gaussian': {
    #        'path': 'resnet101/noise-gaussian_xform-weak-resize',
    #        'color': 'tab:red',
    #        'xpos': 1,
    #    },
    #    'Fourier and Gaussian': {
    #        'path': 'resnet101/noise-fourier-gaussian_xform-weak-resize',
    #        'color': 'tab:purple',
    #        'xpos': 2,
    #    },
    #},
    #'Blur': {
    #    'Weak': {
    #        'path': 'resnet101/blur-weak_xform-weak-resize',
    #        'color': TAB20[5],#'tab:yellow',
    #        'xpos': 0,
    #    },
    #    'Strong': {
    #        'path': 'resnet101/blur-strong_xform-weak-resize',
    #        'color': TAB20[4],#'tab:green',
    #        'xpos': 1,
    #    },
    #},
}

for level1, level2s in models.items():
    for level2, config in level2s.items():
        models[level1][level2] = {
            **config, **properties['all'],
            #**properties['level1'][level1],
            #**properties['level2'][level2]
        }