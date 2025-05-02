# /usr/bin/python
"""
Created by David Coggan on 2023 02 14
contrasts for larger model developed for better contrastive learning
"""

import matplotlib
TAB20B = matplotlib.cm.tab20b.colors
TAB20C = matplotlib.cm.tab20c.colors

models = {
    'CORnet-S+': {
        'supervised, unoccluded': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s/xform-cont',
            'color': TAB20C[0],
            'xpos': 0
        },
        'supervised, natural occluders with textures': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s/occ-nat_tex-nat_vis-beh_xform-cont',
            'color': TAB20B[4],
            'xpos': 3
        },
        'supervised, natural occluders without textures': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s/occ-nat_tex-uni_vis-beh_xform-cont',
            'color': TAB20C[8],
            'xpos': 2
        },
        'supervised, artificial occluders without textures': {
            'architecture': 'cornet_s_plus',
            'path': 'cornet_s/occ-art_vis-beh_xform-cont',
            'color': TAB20C[4],
            'xpos': 1
        },
    },
}
