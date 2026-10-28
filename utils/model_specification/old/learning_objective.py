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
    'CORnet-RT+\nunoccluded': {
        'supervised, unoccluded, contrastive xform': {
            'path': 'cornet_rt_hw3/xform-cont',
            'color': 'tab:orange',
            'xpos': 0
        },
        'self-supervised, unoccluded, contrastive xform': {
            'path': 'cornet_rt_hw3/task-cont',
            'color': 'tab:blue',
            'xpos': 1
        }
    },
    'CORnet-RT+\nart. weak': {
        'supervised, behavioral occluders (high-vis), contrastive xform': {
            'path': 'cornet_rt_hw3/occ-beh_xform-cont',
            'color': 'tab:orange',
            'xpos': 0
        },
        'self-sup, behavioral occluders (high-vis), contrastive xform': {
            'path': 'cornet_rt_hw3/occ-beh_task-cont/transfer_unocc',
            'color': 'tab:blue',
            'xpos': 1
        }
    },
    'CORnet-RT+\nart. strong': {
        'supervised, behavioral occluders (all-vis), contrastive xform': {
            'path': 'cornet_rt_hw3/occ-beh_vis-beh_xform-cont',
            'color': 'tab:orange',
            'xpos': 0
        },
        'self-sup, behavioral occluders (all-vis), contrastive xform': {
            'path': 'cornet_rt_hw3/occ-beh_vis-beh_task-cont/transfer_unocc',
            'color': 'tab:blue',
            'xpos': 1
        }
    },
    'CORnet-RT+\nnat. weak': {
        'supervised, natural occluders (high-vis), contrastive xform': {
            'path': 'cornet_rt_hw3/occ-nat_xform-cont',
            'color': 'tab:orange',
            'xpos': 0
        },
        'self-sup, natural occluders (high-vis), contrastive xform': {
            'path': 'cornet_rt_hw3/occ-nat_task-cont/transfer_unocc',
            'color': 'tab:blue',
            'xpos': 1
        }
    },
    'CORnet-RT+\nnat. strong': {
        'supervised, natural occluders (all-vis), contrastive xform': {
            'path': 'cornet_rt_hw3/occ-nat_vis-beh_xform-cont',
            'color': 'tab:orange',
            'xpos': 0
        },
        'self-sup, natural occluders (all-vis), contrastive xform': {
            'path': 'cornet_rt_hw3/occ-nat_vis-beh_task-cont/transfer_unocc',
            'color': 'tab:blue',
            'xpos': 1
        }
    },
    'CORnet-S+\nunoccluded': {
        'CORnet-S+, unoccluded, classification': {
            'path': 'cornet_s_plus/xform-cont',
            'color': 'tab:orange',
            'xpos': 0
        },
        'CORnet-S+, unoccluded, SimCLR': {
            'path': 'cornet_s_plus/task-cont/transfer_unocc',
            'color': 'tab:blue',
            'xpos': 1
        }
    },
    'CORnet-S+\nart. exp 1': {
        'CORnet-S+, artificial shapes, classification': {
            'path': 'cornet_s_plus/occ-art-weak_xform-cont',
            'color': 'tab:orange',
            'xpos': 0
        },
        'CORnet-S+, artificial shapes, SimCLR': {
            'path': 'cornet_s_plus/occ-art-weak_task-cont/transfer_unocc',
            'color': 'tab:blue',
            'xpos': 1
        }
    },
    'CORnet-S+\nart. exp 2': {
        'CORnet-S+, artificial shapes 2, classification': {
            'path': 'cornet_s_plus/occ-art2-weak_xform-cont',
            'color': 'tab:orange',
            'xpos': 0
        },
        'CORnet-S+, artificial shapes 2, SimCLR': {
            'path': 'cornet_s_plus/occ-art2-weak_task-cont/transfer_unocc',
            'color': 'tab:blue',
            'xpos': 1
        }
    },
    'CORnet-S+\nnat. shp': {
        'CORnet-S+, natural shapes, classification': {
            'path': 'cornet_s_plus/occ-nat-untex-weak_xform-cont',
            'color': 'tab:orange',
            'xpos': 0
        },
        'CORnet-S+, natural shapes, SimCLR': {
            'path': 'cornet_s_plus/occ-nat-untex-weak_task-cont/transfer_unocc',
            'color': 'tab:blue',
            'xpos': 1
        }
    },
    'CORnet-S+\nnat. shp+tx': {
        'CORnet-S+, natural shapes and textures, classification': {
            'path': 'cornet_s_plus/occ-nat-weak_xform-cont',
            'color': 'tab:orange',
            'xpos': 0
        },
        'CORnet-S+, natural shapes and textures, SimCLR': {
            'path': 'cornet_s_plus/occ-nat-weak_task-cont/transfer_unocc',
            'color': 'tab:blue',
            'xpos': 1
        }
    }
}