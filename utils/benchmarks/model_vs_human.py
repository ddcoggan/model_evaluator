# Created by David Coggan on 2025 04 22
'''
This scripts tests models on the model-vs-human package associated with
Geirhos et al. 2021 https://arxiv.org/pdf/2106.07411
'''

import pandas as pd
import torch
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from utils import now
from utils.get_trained_model import get_trained_model
from utils.load_benchmark_scores import load_benchmark_scores
from modelvshuman import Plot, Evaluate
from modelvshuman.datasets.registry import list_datasets
from modelvshuman.models.wrappers.pytorch import PytorchModel

BENCHMARK = 'model-vs-human'
DATASETS = list_datasets()
del DATASETS['imagenet_validation']
del DATASETS['original']
del DATASETS['greyscale']
del DATASETS['texture']

@torch.no_grad()
def score_model(model_dir, architecture, batch_size, image_size=224, m=0,
                total_models=0, num_procs=1, overwrite=False):

    results, out_path = load_benchmark_scores(
        model_dir, BENCHMARK, overwrite)

    if results.empty:
        subsets_to_run = DATASETS
    else:
        res = results[results.benchmark == BENCHMARK]
        subsets_to_run = [i for i in DATASETS if i not in res.level_1.unique()]
    if not len(subsets_to_run):
        return False

    print(f'{now()} | Measuring performance for {BENCHMARK}, '
          f'model: {m + 1}/{total_models} at {model_dir}')

    model = get_trained_model(model_dir, architecture, True, ['output'])
    model_dir_fmt = model_dir.replace('/', '--').replace('_', '-')
    wrapped_model = PytorchModel(model, model_dir_fmt)
    params = {
        'loaded_model': wrapped_model,
        'framework': 'pytorch',
        'batch_size': batch_size,
        'image_size': image_size,
        'print_predictions': True,
        'num_workers': num_procs}
    Evaluate()([model_dir_fmt], subsets_to_run, **params)

    # load results, calculate condition-wise a
    for dataset in subsets_to_run:
        raw_results = pd.read_csv(
            f'utils/benchmarks/model-vs-human/raw-data'
            f'/{dataset}/{dataset}_{model_dir_fmt}_session-1.csv')
        raw_results['score'] = pd.Series(
            raw_results.object_response == raw_results.category, dtype=int)
        if len(raw_results.condition.unique()) == 1:
            new_results = pd.DataFrame({'score': [raw_results.score.mean()]})
        else:
            new_results = (raw_results
                .groupby('condition')
                .agg({'score': 'mean'})
                .reset_index()
                [['condition', 'score']]
                .rename(columns={'condition': 'level_2'}))
        new_results['benchmark'] = BENCHMARK
        new_results['level_1'] = dataset
        new_results['cycle'] = -1
        new_results['metric'] = 'accuracy'
        results = pd.concat([results, new_results]).reset_index(drop=True)

    results.to_csv(out_path, index=False)
