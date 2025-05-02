# /usr/bin/python
"""
Created by David Coggan on 2023 02 14
contrasts for base model plus minor architectural changes
"""

import matplotlib
TAB20 = matplotlib.cm.tab20.colors

models = {
    'CORnet-S': {
        'base model': {
            'architecture': 'cornet_s',
            'path': 'cornet_s/pretrained',
            'color': 'k',#cols[0],#TAB20C[0],
            'xpos': 0,
        },
        'foveal lattice': {
            'architecture': 'cornet_s_FovealBlock',
            'path': 'FovealBlock/cornet_s_no-dilation',
            'color': 'r',#cols[0],#TAB20C[0],
            'xpos': 1,
        },
        'foveal dilation': {
            'architecture': 'cornet_s_FovealBlock',
            'path': 'FovealBlock/cornet_s_uniform-lattice',
            'color': 'g',  # cols[0],#TAB20C[0],
            'xpos': 2,
        },
        'foveal lattice + dilation': {
            'architecture': 'cornet_s_FovealBlock',
            'path': 'FovealBlock/cornet_s',
            'color': 'b',  # cols[0],#TAB20C[0],
            'xpos': 3,
        },
    },
}
