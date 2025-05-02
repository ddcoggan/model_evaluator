"""
controller script for running the evaluation pipeline
"""

import os
import os.path as op
import glob
import time
import datetime
from joblib import Parallel, delayed
from utils.model_contrasts import all_models


# hardware
num_procs = 8
gpu = 0
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu}'

# optionally split model list across 2 GPUs
#models = {k: v for i, (k, v) in enumerate(all_models.items()) if i % 2 == gpu}
models = all_models

# some benchmarks require models trained to classify imagenet
supervised_models = {k: v for k, v in models.items() if 'task-cont' not in k}

benchmarks = [
    #'make_legends',
    #'plot_filters',
    #'collate_training_inputs',
    #'coco_instaorder_cls',
    'model_vs_human',
    'cocoa_cls',
    'coco_occluded_vehicles',
    'pascal3d_occluded_objects',
    'texture_versus_shape_bias',
    'imagenet',
    'imagenet_noise',
    'imagenet_C',
    'imagenet_occluded',
    'imagenet_cutmix',
    'occlusion_behavioral_exp1',
    'occlusion_behavioral_exp2',
    'occlusion_fmri',
    'brainscore',
]

# start running benchmarks
start = time.time()

# get master scripts
for script in ['seconds_to_text', 'plot_utils', 'math_functions', 'Occlude',
               'Noise', 'get_activations', 'accuracy', 'plot_conv_filters',
               'save_images', 'predict', 'get_transforms', 'CustomDataset',
               'image_processing', 'get_model', 'load_params',
               'AverageMeter', 'calculate_batch_size']:
    if not op.exists(f'utils/{script}.py'):
        script_orig = glob.glob(op.expanduser(
            f'~/david/master_scripts/**/{script}.py'), recursive=True)
        assert len(script_orig) == 1
        os.system(f'ln -s {script_orig[0]} utils/{script}.py')

if 'make_legends' in benchmarks:
    print('Making model contrast legends...')
    from utils.make_legends import make_legends
    make_legends(overwrite=False)

if 'plot_filters' in benchmarks:
    print('Plot convolutional filters...')
    from utils.plot_filters import plot_filters
    plot_filters(models, overwrite=False)

if 'collate_training_inputs' in benchmarks:
    print('Collating training inputs...')
    from utils.collate_training_inputs import collate_training_inputs
    collate_training_inputs(overwrite=False)

if 'coco_instaorder_cls' in benchmarks:
    print('COCO Instaorder benchmark...')
    from utils.benchmarks.coco_instaorder_cls import (
        make_coco_instaorder_cls_dataset, score_model)
    make_coco_instaorder_cls_dataset(num_procs=num_procs)
    for m, (model_dir, info) in enumerate(supervised_models.items()):
        score_model(
            model_dir=model_dir,
            architecture=info['architecture'],
            batch_size=info['batch_size'],
            m=m,
            total_models=len(models),
            overwrite=False,
            num_procs=num_procs)

if 'model_vs_human' in benchmarks:
    print('Model-vs-Human benchmark...')
    from utils.benchmarks.model_vs_human import score_model
    for m, (model_dir, info) in enumerate(supervised_models.items()):
        score_model(
            model_dir=model_dir,
            architecture=info['architecture'],
            batch_size=64, #info['batch_size'],
            m=m,
            total_models=len(models),
            overwrite=False,
            num_procs=num_procs)

if 'cocoa_cls' in benchmarks:
    print('COCOA cls benchmark...')
    from utils.benchmarks.cocoa_cls import (
        make_cocoa_cls_dataset, score_model)
    #make_cocoa_cls_dataset()
    for m, (model_dir, info) in enumerate(supervised_models.items()):
        score_model(
            model_dir=model_dir,
            architecture=info['architecture'],
            batch_size=64, #info['batch_size'],
            m=m,
            total_models=len(models),
            overwrite=False,
            num_procs=num_procs)

if 'coco_occluded_vehicles' in benchmarks:
    print('COCO Occluded Vehicles benchmark...')
    from utils.benchmarks.coco_occluded_vehicles import score_model
    for m, (model_dir, info) in enumerate(supervised_models.items()):
        score_model(
            model_dir=model_dir,
            architecture=info['architecture'],
            batch_size=64,#info['batch_size'],
            m=m,
            total_models=len(supervised_models),
            overwrite=False,
            num_procs=num_procs)

if 'pascal3d_occluded_objects' in benchmarks:
    print('PASCAL3D+ Occluded Objects benchmark...')
    from utils.benchmarks.pascal3d_occluded_objects import score_model
    for m, (model_dir, info) in enumerate(supervised_models.items()):
        score_model(
            model_dir=model_dir,
            architecture=info['architecture'],
            batch_size=64,#info['batch_size'],
            m=m,
            total_models=len(supervised_models),
            overwrite=False,
            num_procs=num_procs)

if 'texture_versus_shape_bias' in benchmarks:
    print('Texture versus shape bias benchmark...')
    from utils.benchmarks.texture_versus_shape_bias import score_model
    for m, (model_dir, info) in enumerate(supervised_models.items()):
        score_model(
            model_dir=model_dir,
            architecture=info['architecture'],
            batch_size=64, #info['batch_size'],
            m=m,
            total_models=len(supervised_models),
            overwrite=False,
            num_procs=num_procs)

if 'imagenet' in benchmarks:
    print('ImageNet benchmark...')
    from utils.benchmarks.imagenet import score_model
    for m, (model_dir, info) in enumerate(supervised_models.items()):
        architecture = info['architecture']
        score_model(
            model_dir=model_dir,
            architecture=architecture,
            batch_size=64, #info['batch_size'],
            m=m,
            total_models=len(supervised_models),
            overwrite=False,
            num_procs=num_procs)

if 'imagenet_noise' in benchmarks:
    print('ImageNet-Noise benchmark...')
    from utils.benchmarks.imagenet_noise import (
        score_model)
    #make_imagenet_noise_dataset(overwrite=False, num_procs=num_procs)
    for m, (model_dir, info) in enumerate(supervised_models.items()):
        score_model(
            model_dir=model_dir,
            architecture=info['architecture'],
            batch_size=64, #info['batch_size'],
            m=m,
            total_models=len(supervised_models),
            overwrite=False,
            num_procs=num_procs)

if 'imagenet_C' in benchmarks:
    print('ImageNet-C benchmark...')
    from utils.benchmarks.imagenet_C import score_model
    for m, (model_dir, info) in enumerate(supervised_models.items()):
        architecture = info['architecture']
        score_model(
            model_dir=model_dir,
            architecture=architecture,
            batch_size=64,#info['batch_size'],
            m=m,
            total_models=len(supervised_models),
            overwrite=False,
            num_procs=num_procs)

if 'imagenet_occluded' in benchmarks:
    print('ImageNet-Occluded benchmark...')
    from utils.benchmarks.imagenet_occluded import score_model, make_dataset
    #make_dataset(overwrite=False, num_procs=num_procs)
    for m, (model_dir, info) in enumerate(supervised_models.items()):
        score_model(
            model_dir=model_dir,
            architecture=info['architecture'],
            batch_size=64, #info['batch_size'],
            m=m,
            total_models=len(supervised_models),
            overwrite=False,
            num_procs=num_procs)

if 'imagenet_cutmix' in benchmarks:
    print('ImageNet-CutMix benchmark...')
    from utils.benchmarks.imagenet_cutmix import score_model
    for m, (model_dir, info) in enumerate(supervised_models.items()):
        score_model(
            model_dir=model_dir,
            architecture=info['architecture'],
            batch_size=64,  # info['batch_size'],
            m=m,
            total_models=len(supervised_models),
            overwrite=False,
            num_procs=num_procs)

if any([f'occlusion_behavioral_exp{i}' in benchmarks for i in [1, 2]]):
    print('Behavioral benchmarks...')
    from utils.benchmarks.occlusion_behavioral_exp1 import (
        make_pca_dataset, make_svc_dataset, train_svc)
    make_svc_dataset(overwrite=False)
    make_pca_dataset(overwrite=False)
    for m, (model_dir, info) in enumerate(models.items()):
        architecture = info['architecture']
        readout_layers = info['readout_layers']
        for layer in readout_layers:
            kwargs = dict(
                model_dir=model_dir,
                architecture=architecture,
                batch_size=64,#info['batch_size'],
                m=m,
                total_models=len(models),
                layer=layer,
                overwrite=True,
                num_procs=num_procs)

            # ensure transfer learning used in both experiments is performed
            if layer != 'output':
                kwargs['overwrite'] = train_svc(**kwargs)

    # Experiment 1
    if 'occlusion_behavioral_exp1' in benchmarks:
        from utils.benchmarks.occlusion_behavioral_exp1 import (
            get_responses, analyse_performance)
        parallel_analysis = True
        kwargs_list = []  # for parallel analysis
        for m, (model_dir, info) in enumerate(models.items()):
            architecture = info['architecture']
            readout_layers = info['readout_layers']
            for layer in readout_layers:
                kwargs = dict(
                    model_dir=model_dir,
                    architecture=architecture,
                    batch_size=64,#info['batch_size'],
                    m=m,
                    total_models=len(models),
                    layers=[layer],
                    overwrite=True,
                    num_procs=num_procs)
                kwargs['overwrite'] = get_responses(**kwargs)
                kwargs_performance = {**kwargs, **dict(remake_plots=False)}
                for k in ['batch_size', 'architecture', 'num_procs']:
                    del kwargs_performance[k]
                #kwargs_performance['overwrite'] = True  # force performance analysis
                if parallel_analysis:
                    kwargs_list.append(kwargs_performance)
                else:
                    kwargs['overwrite'] = analyse_performance(**kwargs_performance)
        if parallel_analysis:
            overwrite_analyses = Parallel(n_jobs=num_procs)(
                delayed(analyse_performance)(**kwargs) for kwargs in kwargs_list)
        from utils.behavioral_compare_models_exp1 import compare_models
        compare_models(overwrite=False)

    # Experiment 2
    if 'occlusion_behavioral_exp2' in benchmarks:
        from utils.benchmarks.occlusion_behavioral_exp2 import (
            get_responses, analyse_performance)
        parallel_analysis = True
        kwargs_list = []  # for parallel analysis
        for m, (model_dir, info) in enumerate(models.items()):
            architecture = info['architecture']
            readout_layers = info['readout_layers']
            for layer in readout_layers:
                kwargs = dict(
                    model_dir=model_dir,
                    architecture=architecture,
                    batch_size=64,  # info['batch_size'],
                    m=m,
                    total_models=len(models),
                    layers=[layer],
                    overwrite=True,
                    num_procs=num_procs)
                kwargs['overwrite'] = get_responses(**kwargs)
                kwargs_performance = {**kwargs, **dict(remake_plots=False)}
                for k in ['batch_size', 'architecture', 'num_procs']:
                    del kwargs_performance[k]
                # kwargs_performance['overwrite'] = True  # force performance analysis
                if parallel_analysis:
                    kwargs_list.append(kwargs_performance)
                else:
                    kwargs['overwrite'] = analyse_performance(**kwargs_performance)
        if parallel_analysis:
            overwrite_analyses = Parallel(n_jobs=num_procs)(
                delayed(analyse_performance)(**kwargs) for kwargs in kwargs_list)
        from utils.behavioral_compare_models_exp2 import compare_models
        compare_models(overwrite=True)

    """
    print('pixel attribution...')
    from utils import (pixel_attribution, evaluate_salience,
                       plot_pixel_attribution, compare_models_pixel)
    for m, model_dir in enumerate(model_dirs):
        pixel_attribution(model_dir, m, len(model_dirs), num_procs=num_procs,
                          overwrite=False)
        evaluate_salience(model_dir, m, len(model_dirs), overwrite=False)
        plot_pixel_attribution(model_dir, m, len(model_dirs), overwrite=False)
    compare_models_pixel(overwrite=False)
    
    print('fMRI benchmark...')
    from utils.fMRI_benchmark import (get_model_responses, RSA_fMRI,
                                      get_prednet_responses,
                                      generate_reconstructions)
    recompare_models = False
    for m, model_dir in enumerate(model_dirs):
        overwrite = False
        if 'prednet' in model_dir:
            overwrite = get_prednet_responses(model_dir, overwrite=overwrite)
            generate_reconstructions(model_dir, overwrite)
        else:
            overwrite = get_model_responses(
                model_dir, m, len(model_dirs), overwrite=overwrite)
        #if 'pix2pix' in model_dir:
        #    generate_reconstructions(model_dir, overwrite)
        overwrite = RSA_fMRI(
            model_dir, m, len(model_dirs), overwrite=overwrite)
        if overwrite:
            recompare_models = True
    from utils.fMRI_compare_models import compare_models
    compare_models(overwrite=recompare_models)
    
    Brainscore is in the middle of an upgrade, nothing works right now
    print('BrainScore benchmark...')
    recompare_models = True
    for m, model_dir in enumerate(model_dirs):
        overwrite = False
        overwrite = measure_scores(
            model_dir, m, len(model_dirs), overwrite=overwrite)
        if overwrite:
            recompare_models = True
    compare_models_brainscore(overwrite=recompare_models)
    """

finish = time.time()
print(f'Done. Total time: {str(datetime.timedelta(seconds=finish-start))}')

