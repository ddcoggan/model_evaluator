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
    'all': {},
    'level1': {
        'no occlusion': {
            'color': 'w', 'edgecolor': 'k', 'xpos': 0},
        'natural': {
            'color': 'tab:green', 'edgecolor': 'tab:green', 'xpos': 1},
        'natural_mix': {
            'color': 'tab:orange', 'edgecolor': 'tab:orange', 'xpos': 2},
        'natural_silhouette': {
            'color': 'tab:brown', 'edgecolor': 'tab:brown', 'xpos': 3},
        'artificial_1': {
            'color': 'tab:blue', 'edgecolor': 'tab:blue', 'xpos': 4},
        'artificial_2': {
            'color': 'tab:red', 'edgecolor': 'tab:red', 'xpos': 5},
    },
    'level2': {
        'CORnet-S+\nclassification': {'architecture': 'cornet_s_plus'},
        'CORnet-S+\nSimCLR': {'architecture': 'cornet_s_plus',
                              'readout_layer': 'IT.output'},
        'ResNet101\nclassification': {'architecture': 'resnet101'},
        'EfficientNet-B1\nclassification': {'architecture': 'efficientnet_b1'},
        'ViT-B/16\nclassification': {'architecture': 'vit_b_16'},
    },
}

models = {
    #'no occlusion': {
    #    'CORnet-S+\nclassification': {
    #        'path': 'cornet_s_plus/xform-cont-weak-resize',
    #    },
    #    'CORnet-S+\nSimCLR': {
    #        'path': 'cornet_s_plus/task-cont-weak-resize',
    #    },
    #    'ResNet101\nclassification': {
    #        'path': 'resnet101/xform-cont-weak-resize',
    #    },
    #    'EfficientNet-B1\nclassification': {
    #        'path': 'efficientnet_b1/xform-cont-weak-resize',
    #    },
        #'ViT-B/16\nclassification': {
        #    'path': 'vit_b_16/xform-cont-weak-resize',
        #},
    #},
    'natural_mix_unocc50': {
        'CORnet-S+\nclassification': {
            'path': 'cornet_s_plus/occ-nat-mix-unocc50',
        }
    },
    'natural_mix_unocc75': {
        'CORnet-S+\nclassification': {
            'path': 'cornet_s_plus/occ-nat-mix-unocc75',
        }
    },
    'natural_mix_unocc88': {
        'CORnet-S+\nclassification': {
            'path': 'cornet_s_plus/occ-nat-mix-unocc88',
        },
    },
    'natural_mix_unocc94': {
        'CORnet-S+\nclassification': {
            'path': 'cornet_s_plus/occ-nat-mix-unocc94',
        },
    },
    'artificial_3_pink_noise': {
        'CORnet-S+\nclassification': {
            'path': 'cornet_s_plus/occ-art3-pink_xform-cont-weak-resize',
        }
    },
    'natural_unocc50': {
        'CORnet-S+\nclassification': {
            'path': 'cornet_s_plus/occ-nat-unocc50',
        }
    }
}

"""
    'natural': {
        'CORnet-S+\nclassification': {
            'path': 'cornet_s_plus/occ-nat_xform-cont-weak-resize',
        },
        'CORnet-S+\nSimCLR': {
            'path': 'cornet_s_plus/occ-nat_task-cont-weak-resize',
        },
        'ResNet101\nclassification': {
            'path': 'resnet101/occ-nat_xform-cont-weak-resize',
        },
        'EfficientNet-B1\nclassification': {
            'path': 'efficientnet_b1/occ-nat_xform-cont-weak-resize',
        },
        'ViT-B/16\nclassification': {
            'path': 'vit_b_16/occ-nat_xform-cont-weak-resize',
        },
    },
    'natural_mix': {
        'CORnet-S+\nclassification': {
            'path': 'cornet_s_plus/occ-nat-mix_xform-cont-weak-resize'
                    '-mix',
        },
        'CORnet-S+\nSimCLR': {
            'path': 'cornet_s_plus/occ-nat-mix_task-cont-weak-resize',
        },
        'ResNet101\nclassification': {
            'path': 'resnet101/occ-nat-mix_xform-cont-weak-resize',
        },
        'EfficientNet-B1\nclassification': {
            'path': 'efficientnet_b1/occ-nat-mix_xform-cont-weak-resize',
        },
        'ViT-B/16\nclassification': {
            'path': 'vit_b_16/occ-nat-mix_xform-cont-weak-resize',
        },
    },
    'natural_silhouette': {
        'CORnet-S+\nclassification': {
            'path': 'cornet_s_plus/occ-nat-untex_xform-cont-weak-resize',
        },
        'CORnet-S+\nSimCLR': {
            'path': 'cornet_s_plus/occ-nat-untex_task-cont-weak-resize',
        },
        'ResNet101\nclassification': {
            'path': 'resnet101/occ-nat-untex_xform-cont-weak-resize',
        },
        'EfficientNet-B1\nclassification': {
            'path': 'efficientnet_b1/occ-nat-untex_xform-cont-weak-resize',
        },
        'ViT-B/16\nclassification': {
            'path': 'vit_b_16/occ-nat-untex_xform-cont-weak-resize',
        },
    },
    'artificial_1': {
        'CORnet-S+\nclassification': {
            'path': 'cornet_s_plus/occ-art_xform-cont-weak-resize',
        },
        'CORnet-S+\nSimCLR': {
            'path': 'cornet_s_plus/occ-art_task-cont-weak-resize',
        },
        'ResNet101\nclassification': {
            'path': 'resnet101/occ-art_xform-cont-weak-resize',
        },
        'EfficientNet-B1\nclassification': {
            'path': 'efficientnet_b1/occ-art_xform-cont-weak-resize',
        },
        'ViT-B/16\nclassification': {
            'path': 'vit_b_16/occ-art_xform-cont-weak-resize',
        },
    },
    'artificial_2': {
        'CORnet-S+\nclassification': {
            'path': 'cornet_s_plus/occ-art3_xform-cont-weak-resize',
        },
        'CORnet-S+\nSimCLR': {
            'path': 'cornet_s_plus/occ-art3_task-cont-weak-resize',
        },
        'ResNet101\nclassification': {
            'path': 'resnet101/occ-art3_xform-cont-weak-resize',
        },
        'EfficientNet-B1\nclassification': {
            'path': 'efficientnet_b1/occ-art3_xform-cont-weak-resize',
        },
        'ViT-B/16\nclassification': {
            'path': 'vit_b_16/occ-art3_xform-cont-weak-resize',
        },
    },
}
"""
for level1, level2s in models.items():
    for level2, config in level2s.items():
        try:
            models[level1][level2] = {
                **config, **properties['all'],
                **properties['level1'][level1],
                **properties['level2'][level2]
            }
        except:
            pass