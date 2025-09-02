import numpy as np
import json
import os
import pandas as pd
from utils import (
    PosthocAnalyzer, XAIDistanceBenchmark, DistanceEvaluator,
    split_tcr_regions
)
from utils import build_eval_configs, brhr_configs


output_file = 'eval_brhrs.json'
tcrxai_path = 'data/TCRXAI'
model_type = 'TCRpMHC-MLM-CA-Tasks'
checkpoint_path = 'projects/representations/cdr3_only_binder'
posthoc_config = 'configs/posthocs/cdr3_only_binder.json'

def get_all_checkpoint_paths(path):
    ids = [
        int(i.split('-')[-1])
        for i
        in os.listdir(os.path.join(path, 'immrep23/'))
        if i.split('-')[0] == 'checkpoint'
    ]
    ids = sorted(ids)
    return {_id:os.path.join(path, 'immrep23/', f'checkpoint-{_id}') for _id in ids}

distance_dataset = XAIDistanceBenchmark(os.path.join(tcrxai_path, 'distance.json'))
models = get_all_checkpoint_paths(checkpoint_path)
brhrs = {}
for run_id, model_path in models.items():
    analyzer = PosthocAnalyzer.from_configs(
        model = model_type,
        model_weights = model_path,
        dataset = tcrxai_path,
        track_config = posthoc_config,
        losses = ['loss_binder']
    )
    split_tcr_regions(analyzer)
    
    evaluator = DistanceEvaluator(
        analyzer = analyzer,
        distance_dataset=distance_dataset,
        processor = lambda x:[np.nan_to_num(np.convolve(i, np.ones(3)/3, mode='same'), nan=0) for i in x]
    )
    df = pd.DataFrame(evaluator(build_eval_configs(brhr_configs))).T
    rel = {f'{k[0]}->{k[1].split("->")[-1]}':v for k,v in df.applymap(lambda x:x[0])['BRHR.25'].to_dict().items()}
    brhrs[run_id] = rel

with open(output_file, 'w') as f:
    json.dump(brhrs, f, indent=2)