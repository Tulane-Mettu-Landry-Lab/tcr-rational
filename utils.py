import os
import json
import re
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Callable, Union
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from imml.tokenizers import build_tokenizer
from imml.data import KFold, TrainTestSplit, TransformerDataset, DataFrameDataset
from imml.configs import IMMLConfigurationGroup, IMMLConfiguration
from imml.models import ModelLibrary
from imml import ModelExperiment
from posthoc import ModelToolKit, SolverLibrary, FlowCollector

# Performance Utils
def table_exps_rocauc(path):
    with open(os.path.join(path, 'models.json'), 'r') as f:
        models = json.load(f)
    _metrics = {}
    for _folder, _model_name in models.items():
        _model_path = os.path.join(path, _folder)
        vals = fetch_eval_states(_model_path).loc[('binder', 'binary_rocauc')].values
        _metrics[_model_name] = {
            '5-Fold (mean)': vals[:-1].mean(),
            '5-Fold (std)': vals[:-1].std(),
            'IMMREP23': vals[-1]
        }
    _metrics = pd.DataFrame(_metrics).T
    return _metrics

# Posthoc Utils
def flatten_input(data):
    _flatten_data = {}
    for key, val in data.items():
        if isinstance(val, dict):
            for k, v in val.items():
                _flatten_data[f'{key}_{k}'] = torch.tensor([v])
            _flatten_data[f'{key}_labels'] = _flatten_data[f'{key}_input_ids']
        else:
            _flatten_data[key] = torch.tensor([val])
    return _flatten_data

class CollectFuncLibrary(object):
    _lib = {
        'token': lambda x:x[:, 0],
        'flatten': lambda x: x.flatten(1),
    }
    
    def __class_getitem__(cls, item):
        return cls._lib[item]
    
    @classmethod
    def register(cls, name:str):
        def _register_wrapper(method):
            cls._lib[name] = method
            return method
        return _register_wrapper
    
    @classmethod
    def methods(cls):
        return cls._lib.keys()
    
def _setup_track_flow(node:ModelToolKit, configs:dict):
    _modules = [
        node.nodes.search_modules(_module)
        for _module in configs['modules']
    ]
    node.tracks.add(*_modules)
    track = node.tracks[-1]
    _solvers = [
        SolverLibrary[flow['method']](**{k:v for k,v in flow.items() if k!='method'})
        for flow in configs['flow']
    ]
    track.flow.set_flow(*_solvers)
    track.flow.set_collect(CollectFuncLibrary[configs['collect']])
    return track

def setup_track_flows(node:ModelToolKit, configs:dict):
    flows = {}
    for config in configs:
        flows[config['name']] = _setup_track_flow(node, config)
    return flows

class PosthocAnalyzer(object):
    
    def __init__(
        self,
        model:nn.Module,
        track_configs:IMMLConfiguration,
        dataset:DataFrameDataset,
        collate_func:Callable=lambda x:x[0][1:-1],
        losses:list[str]=['loss'],
        disable_tqdm:bool=False,
        positive_only:bool=False,
    ):
        
        self.model = model
        self.track_configs = track_configs
        self.dataset = dataset
        self.node = ModelToolKit(model)
        self.flows = setup_track_flows(self.node, configs=track_configs.to_list())
        self.collate_func = collate_func
        self.disable_tqdm = disable_tqdm
        self.losses = losses
        self.collects = None
        self.positive_only = positive_only
        self.recoreded_ids = []
        self.collect()
        
    def __repr__(self):
        return f'{len(self)} samples of {list(self.keys())} Analyzed'
        
    def keys(self):
        return self.collects.keys()
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        if isinstance(key, (tuple, list)):
            _key, _index = key
            return self.collects[_key][_index]
        return self.collects[key]
    
    @classmethod
    def from_configs(
        cls,
        model:str,
        model_weights:str,
        track_config:str,
        dataset:str,
        losses:list[str],
        disable_tqdm=False,
        positive_only:bool=False,
    ):
        _model = ModelLibrary[model].from_pretrained(model_weights)
        _config = IMMLConfiguration.from_config(track_config)
        _data = DataFrameDataset(dataset)
        return cls(
            model=_model,
            track_configs=_config,
            dataset=_data,
            collate_func=lambda x:(x[0]*len(x[0]))[1:-1],
            losses=losses,
            disable_tqdm=disable_tqdm,
            positive_only=positive_only
        )
        
        
    def _backward_loss(self, out):
        _loss = None
        for loss in self.losses:
            if _loss is None:
                _loss = out[loss]
            else:
                _loss += out[loss]
        _loss.backward()
        return _loss
    
    def collect(self, fix_l1=True):
        from scipy.special import softmax
        collector = FlowCollector(self.node)
        collect_map = lambda x:[self.collate_func(i) for i in x]
        _dataset = TransformerDataset(self.dataset, processor=self.model.processor)
        if not self.disable_tqdm:
            _dataset = tqdm(_dataset)
        self.recoreded_ids = []
        self.model = self.model.eval()
        for idx, sample in enumerate(_dataset):
            self.node.model_hook.clean()
            self.model.zero_grad()
            out = self.model(**flatten_input(sample))
            
            probs = out['predictions_binder'].detach().numpy()
            probs = softmax(probs, axis=1)[0][-1]
            prediction = np.array(probs >= 0.5, np.int32)
            
            self._backward_loss(out)
            if prediction or (not self.positive_only):
                collector.collect()
                self.recoreded_ids.append(idx)
        _collects = {
            flow_name:collect_map(collect)
            for flow_name, collect in
            zip(self.flows.keys(), collector.collects)
        }
        self.collects = _collects
        
    def __call__(self):
        return self.collect()

class XAIRegionBenchmark(object):
    def __init__(self, data_or_path:Union[str, list]):
        if isinstance(data_or_path, list):
            self._data = data_or_path
        else:
            with open(data_or_path, 'r') as _data_file:
                self._data = json.load(_data_file)
                
    def __len__(self):
        return len(self._data)
    
    def _fetch_index(self, source:str):
        return [_sample['index'][source] for _sample in self._data]
    
    def __getitem__(self, key):
        return self._fetch_index(key)
    

class XAIDistanceBenchmark(object):
    
    def __init__(self, data_or_path:Union[str, list]):
        if isinstance(data_or_path, list):
            self._data = data_or_path
        else:
            with open(data_or_path, 'r') as _data_file:
                self._data = json.load(_data_file)
    
    def __len__(self):
        return len(self._data)
    
    def _fetch_distance(self, source:str, targets:list[str]):
        _distances = [
            np.stack([
                _sample['distance'][source][_target]
                for _target in targets
            ]).min(axis=0)
            for _sample in self._data
        ]
        return _distances
    
    def _fetch_index(self, source:str):
        return [_sample['index'][source] for _sample in self._data]

    def _decode_distance_index(self, key):
        _source, _target = key.split('->')
        _targets = _target.split(',')
        _source = _source.strip()
        _targets = [_target.strip() for _target in _targets]
        return _source, _targets
    
    def __getitem__(self, key):
        if isinstance(key, str):
            _source, _targets = self._decode_distance_index(key)
            return self._fetch_distance(_source, _targets)
        else:
            _source, _targets = key
            return self._fetch_distance(_source, _targets)

# Posthoc Distance Metrics
def binding_region_hit_rate(y_trues, y_preds, threshold=0.8, aggregate='mean'):
    _hit_rates = []
    for pred, gt in zip(y_preds, y_trues):
        try:
            gt = np.array(gt)
            _threshold_num = int(np.ceil(threshold * len(pred)))
            _indices = np.argsort(pred)[::-1]
            _indices = _indices[:_threshold_num]
            _gt_quantile = np.quantile(gt, 1-threshold)
            _hit_rate = np.mean(gt[_indices] < _gt_quantile)
            _hit_rates.append(_hit_rate)
        except:
            _hit_rates.append(0)
    _hit_rates = np.array(_hit_rates)
    if aggregate is None or aggregate == 'none':
        return _hit_rates
    else:
        return np.mean(_hit_rates), np.std(_hit_rates)

def activate_level(y_trues, y_preds, method='mean', aggregate=False, fix_l1_norm=False):
    if fix_l1_norm:
        y_preds = [np.array(y_pred)*len(y_pred) for y_pred in y_preds]
    if method == 'mean':
        _vals =  np.array([np.mean(np.array(y_pred) * (np.array(y_pred) >= 0.75)) for y_pred in y_preds])
        # _vals = np.concatenate([np.array(y_pred) for y_pred in y_preds]).flatten()
        # _vals = _vals[_vals > 0.75]
        if aggregate: _vals = np.mean(_vals)
    elif method == 'min':
        _vals = np.array([np.min(y_pred) for y_pred in y_preds])
        if aggregate: _vals = np.min(_vals)
    elif method == 'max':
        _vals =  np.array([np.max(y_pred) for y_pred in y_preds])
        if aggregate: _vals = np.max(_vals)
    elif method == 'std':
        _vals = np.array([np.std(y_pred) for y_pred in y_preds])
        if aggregate: _vals = np.std(_vals)
    elif method == 'median':
        _vals = np.array([np.median(y_pred) for y_pred in y_preds])
        if aggregate: _vals = np.median(_vals)
    elif method == 'var':
        _vals = np.array([np.var(y_pred) for y_pred in y_preds])
        if aggregate: _vals = np.var(_vals)
    elif method == 'flatten':
        _vals = np.concatenate(y_preds)
    else:
        _vals =  np.array([np.mean(y_pred) for y_pred in y_preds])
        if aggregate: _vals = np.mean(_vals)
    return _vals

def threshold_flatten_metric(metric):
    
    def _wrap(y_trues, y_preds, y_trues_threshold=4.0, y_preds_threshold=None, **kwargs):
        y_trues = np.concatenate(y_trues)
        y_preds = np.concatenate(y_preds)
        if y_trues_threshold is not None:
            y_trues = (y_trues <= y_trues_threshold)
            num_y_trues = np.sum(y_trues)
        if y_preds_threshold is not None:
            if isinstance(y_preds_threshold, (float, int)):
                y_preds = (y_preds >= y_preds_threshold)
            elif y_preds_threshold == 'mean':
                y_preds = (y_preds >= np.mean(y_preds))
            elif y_preds_threshold == 'median':
                y_preds = (y_preds >= np.median(y_preds))
            elif y_preds_threshold == 'quantile':
                y_preds = (y_preds >= np.sort(y_preds)[num_y_trues])
        try:
            _val = metric(y_trues, y_preds, **kwargs)
        except:
            _val = np.nan
        return _val
    return _wrap
threshold_roc_auc_score = threshold_flatten_metric(roc_auc_score)
threshold_precision_score = threshold_flatten_metric(precision_score)
threshold_recall_score = threshold_flatten_metric(recall_score)
threshold_accuracy_score = threshold_flatten_metric(accuracy_score)
threshold_f1_score = threshold_flatten_metric(f1_score)

class DistanceMetricLibrary(object):
    _lib = {
        'hitrate': binding_region_hit_rate,
        'rocauc': threshold_roc_auc_score,
        'precision': threshold_precision_score,
        'recall': threshold_recall_score,
        'accuracy': threshold_accuracy_score,
        'f1': threshold_f1_score,
        'actlevel': activate_level,
    }
    
    def __class_getitem__(cls, item):
        return cls._lib[item]
    
    @classmethod
    def register(cls, name:str):
        def _register_wrapper(method):
            cls._lib[name] = method
            return method
        return _register_wrapper
    
    @classmethod
    def methods(cls):
        return cls._lib.keys()
    
    
class DistanceEvaluator(object):
    
    def __init__(
        self,
        analyzer:PosthocAnalyzer,
        distance_dataset:XAIDistanceBenchmark,
        disable_tqdm:bool=False,
        processor:Callable=lambda x:x,
    ):
        self.analyzer = analyzer
        self.dataset = distance_dataset
        self.disable_tqdm = disable_tqdm
        self.processor = processor
        
    
    def evaluate(self, metric:Union[Callable, str], true_key:str, pred_key:str, **kwargs):
        if isinstance(metric, str):
            metric = DistanceMetricLibrary[metric]
        if 'disable_processor' in kwargs and kwargs['disable_processor']:
            _preds = self.analyzer[pred_key]
            del kwargs['disable_processor']
        else:
            _preds = self.processor(self.analyzer[pred_key])
        _gts = self.dataset[true_key]
        _gts = [_gts[idx] for idx in self.analyzer.recoreded_ids]
        return metric(
            _gts,
            _preds,
            **kwargs
        )
    
    def forward(self, configs:dict):
        _values = {}
        # _total = sum([len(_preds) for _pred_key, _preds in configs.items() if _pred_key in self.analyzer.keys()])
        if not self.disable_tqdm:
            _bar = tqdm()
        for pred_key, pred_config in configs.items():
            if pred_key not in self.analyzer.keys():
                continue
            for true_key, true_config in pred_config.items():
                _values[(pred_key, true_key)] = {}
                for metric_name, metric_config in true_config.items():
                    if not self.disable_tqdm:
                        _bar.update(1)
                        _bar.set_description(f'{metric_name}({pred_key}, [{true_key}])')
                    _values[(pred_key, true_key)][metric_name] = \
                        self.evaluate(true_key=true_key, pred_key=pred_key, **metric_config)
        _bar.set_description('Evaluating done.')
        return _values
    
    def __call__(self, configs:dict):
        return self.forward(configs=configs)

# Posthoc Distance Metrics Configs
brhr_configs = {
    'BRHR.75': dict(metric='hitrate', threshold=0.75, aggregate='mean'),
    'BRHR.60': dict(metric='hitrate', threshold=0.6, aggregate='mean'),
    'BRHR.50': dict(metric='hitrate', threshold=0.5, aggregate='mean'),
    'BRHR.40': dict(metric='hitrate', threshold=0.4, aggregate='mean'),
    'BRHR.30': dict(metric='hitrate', threshold=0.3, aggregate='mean'),
    'BRHR.25': dict(metric='hitrate', threshold=0.25, aggregate='mean'),
}
precision_configs = {
    'Precision(3.0A)': dict(metric='precision', y_trues_threshold=3.0, y_preds_threshold='quantile'),
    'Precision(3.4A)': dict(metric='precision', y_trues_threshold=3.4, y_preds_threshold='quantile'),
    'Precision(3.5A)': dict(metric='precision', y_trues_threshold=3.5, y_preds_threshold='quantile'),
    'Precision(4.0A)': dict(metric='precision', y_trues_threshold=4.0, y_preds_threshold='quantile'),
    'Precision(4.5A)': dict(metric='precision', y_trues_threshold=4.5, y_preds_threshold='quantile'),
    'Precision(5.0A)': dict(metric='precision', y_trues_threshold=5.0, y_preds_threshold='quantile'),
    'Precision(5.5A)': dict(metric='precision', y_trues_threshold=5.5, y_preds_threshold='quantile'),
    'Precision(6.0A)': dict(metric='precision', y_trues_threshold=6.0, y_preds_threshold='quantile'),
    'Precision(6.5A)': dict(metric='precision', y_trues_threshold=6.5, y_preds_threshold='quantile'),
    'Precision(7.0A)': dict(metric='precision', y_trues_threshold=7.0, y_preds_threshold='quantile'),
}
recall_configs = {
    'Recall(3.0A)': dict(metric='recall', y_trues_threshold=3.0, y_preds_threshold='quantile'),
    'Recall(3.4A)': dict(metric='recall', y_trues_threshold=3.4, y_preds_threshold='quantile'),
    'Recall(3.5A)': dict(metric='recall', y_trues_threshold=3.5, y_preds_threshold='quantile'),
    'Recall(4.0A)': dict(metric='recall', y_trues_threshold=4.0, y_preds_threshold='quantile'),
    'Recall(4.5A)': dict(metric='recall', y_trues_threshold=4.5, y_preds_threshold='quantile'),
    'Recall(5.0A)': dict(metric='recall', y_trues_threshold=5.0, y_preds_threshold='quantile'),
    'Recall(5.5A)': dict(metric='recall', y_trues_threshold=5.5, y_preds_threshold='quantile'),
    'Recall(6.0A)': dict(metric='recall', y_trues_threshold=6.0, y_preds_threshold='quantile'),
    'Recall(6.5A)': dict(metric='recall', y_trues_threshold=6.5, y_preds_threshold='quantile'),
    'Recall(7.0A)': dict(metric='recall', y_trues_threshold=7.0, y_preds_threshold='quantile'),
}
rocauc_configs = {
    'ROCAUC(3.0A)': dict(metric='rocauc', y_trues_threshold=3.0, y_preds_threshold='quantile'),
    'ROCAUC(3.4A)': dict(metric='rocauc', y_trues_threshold=3.4, y_preds_threshold='quantile'),
    'ROCAUC(3.5A)': dict(metric='rocauc', y_trues_threshold=3.5, y_preds_threshold='quantile'),
    'ROCAUC(4.0A)': dict(metric='rocauc', y_trues_threshold=4.0, y_preds_threshold='quantile'),
    'ROCAUC(4.5A)': dict(metric='rocauc', y_trues_threshold=4.5, y_preds_threshold='quantile'),
    'ROCAUC(5.0A)': dict(metric='rocauc', y_trues_threshold=5.0, y_preds_threshold='quantile'),
    'ROCAUC(5.5A)': dict(metric='rocauc', y_trues_threshold=5.5, y_preds_threshold='quantile'),
    'ROCAUC(6.0A)': dict(metric='rocauc', y_trues_threshold=6.0, y_preds_threshold='quantile'),
    'ROCAUC(6.5A)': dict(metric='rocauc', y_trues_threshold=6.5, y_preds_threshold='quantile'),
    'ROCAUC(7.0A)': dict(metric='rocauc', y_trues_threshold=7.0, y_preds_threshold='quantile'),
}
accuracy_configs = {
    'Accuracy(3.0A)': dict(metric='accuracy', y_trues_threshold=3.0, y_preds_threshold='quantile'),
    'Accuracy(3.4A)': dict(metric='accuracy', y_trues_threshold=3.4, y_preds_threshold='quantile'),
    'Accuracy(3.5A)': dict(metric='accuracy', y_trues_threshold=3.5, y_preds_threshold='quantile'),
    'Accuracy(4.0A)': dict(metric='accuracy', y_trues_threshold=4.0, y_preds_threshold='quantile'),
    'Accuracy(4.5A)': dict(metric='accuracy', y_trues_threshold=4.5, y_preds_threshold='quantile'),
    'Accuracy(5.0A)': dict(metric='accuracy', y_trues_threshold=5.0, y_preds_threshold='quantile'),
    'Accuracy(5.5A)': dict(metric='accuracy', y_trues_threshold=5.5, y_preds_threshold='quantile'),
    'Accuracy(6.0A)': dict(metric='accuracy', y_trues_threshold=6.0, y_preds_threshold='quantile'),
    'Accuracy(6.5A)': dict(metric='accuracy', y_trues_threshold=6.5, y_preds_threshold='quantile'),
    'Accuracy(7.0A)': dict(metric='accuracy', y_trues_threshold=7.0, y_preds_threshold='quantile'),
}
f1_configs = {
    'F1(3.0A)': dict(metric='f1', y_trues_threshold=3.0, y_preds_threshold='quantile'),
    'F1(3.4A)': dict(metric='f1', y_trues_threshold=3.4, y_preds_threshold='quantile'),
    'F1(3.5A)': dict(metric='f1', y_trues_threshold=3.5, y_preds_threshold='quantile'),
    'F1(4.0A)': dict(metric='f1', y_trues_threshold=4.0, y_preds_threshold='quantile'),
    'F1(4.5A)': dict(metric='f1', y_trues_threshold=4.5, y_preds_threshold='quantile'),
    'F1(5.0A)': dict(metric='f1', y_trues_threshold=5.0, y_preds_threshold='quantile'),
    'F1(5.5A)': dict(metric='f1', y_trues_threshold=5.5, y_preds_threshold='quantile'),
    'F1(6.0A)': dict(metric='f1', y_trues_threshold=6.0, y_preds_threshold='quantile'),
    'F1(6.5A)': dict(metric='f1', y_trues_threshold=6.5, y_preds_threshold='quantile'),
    'F1(7.0A)': dict(metric='f1', y_trues_threshold=7.0, y_preds_threshold='quantile'),
}
agg_activate_level_configs = {
    'Activate Avg': dict(metric='actlevel', method='mean', disable_processor=True, aggregate=True),
    'Activate Mid': dict(metric='actlevel', method='median', disable_processor=True, aggregate=True),
    'Activate Min': dict(metric='actlevel', method='min', disable_processor=True, aggregate=True),
    'Activate Max': dict(metric='actlevel', method='max', disable_processor=True, aggregate=True),
    'Activate Std': dict(metric='actlevel', method='std', disable_processor=True, aggregate=True),
    'Activate Var': dict(metric='actlevel', method='var', disable_processor=True, aggregate=True),
}
activate_level_configs = {
    'Activate Avg': dict(metric='actlevel', method='mean', disable_processor=True),
    'Activate Mid': dict(metric='actlevel', method='median', disable_processor=True),
    'Activate Min': dict(metric='actlevel', method='min', disable_processor=True),
    'Activate Max': dict(metric='actlevel', method='max', disable_processor=True),
    'Activate Std': dict(metric='actlevel', method='std', disable_processor=True),
    'Activate Var': dict(metric='actlevel', method='var', disable_processor=True),
}
flatten_activate_level_configs = {
    'Activate': dict(metric='actlevel', method='flatten', disable_processor=True),
}


def build_eval_configs(metric_configs):
    eval_configs = dict(
        epitope = {
            'Peptide->TRA': metric_configs,
            'Peptide->TRB': metric_configs,
            'Peptide->CDR1A': metric_configs,
            'Peptide->CDR2A': metric_configs,
            'Peptide->CDR3A': metric_configs,
            'Peptide->CDR1A,CDR2A,CDR3A': metric_configs,
            'Peptide->CDR1B': metric_configs,
            'Peptide->CDR2B': metric_configs,
            'Peptide->CDR3B': metric_configs,
            'Peptide->CDR1B,CDR2B,CDR3B': metric_configs,
            'Peptide->TRA,TRB': metric_configs,
            'Peptide->CDR3A,CDR3B': metric_configs,
            'Peptide->CDR1B,CDR2B,CDR3B,CDR1A,CDR2A,CDR3A': metric_configs,
        },
        CDR1B = {
            'CDR1B->TRA': metric_configs,
            'CDR1B->Peptide': metric_configs,
            'CDR1B->Peptide,TRA': metric_configs,
        },
        CDR2B = {
            'CDR2B->TRA': metric_configs,
            'CDR2B->Peptide': metric_configs,
            'CDR2B->Peptide,TRA': metric_configs,
        },
        CDR3B = {
            'CDR3B->TRA': metric_configs,
            # 'CDR3B->CDR3A': metric_configs,
            # 'CDR3B->CDR1A,CDR2A,CDR3A': metric_configs,
            # 'CDR3B->CDR2B': metric_configs,
            'CDR3B->Peptide': metric_configs,
            'CDR3B->Peptide,TRA': metric_configs,
        },
        CDR1A = {
            'CDR1A->TRB': metric_configs,
            'CDR1A->Peptide': metric_configs,
            'CDR1A->Peptide,TRB': metric_configs,
        },
        CDR2A = {
            'CDR2A->TRB': metric_configs,
            'CDR2A->Peptide': metric_configs,
            'CDR2A->Peptide,TRB': metric_configs,
        },
        CDR3A = {
            'CDR3A->TRB': metric_configs,
            # 'CDR3A->CDR3B': metric_configs,
            # 'CDR3A->CDR1B,CDR2B,CDR3B': metric_configs,
            # 'CDR3A->CDR2A': metric_configs,
            'CDR3A->Peptide': metric_configs,
            'CDR3A->Peptide,TRB': metric_configs,
        },
        TRB = {
            'TRB->TRA': metric_configs,
            'TRB->Peptide': metric_configs,
        },
        TRA = {
            'TRA->TRB': metric_configs,
            'TRA->Peptide': metric_configs,
        },
    )
    return eval_configs

def build_activate_configs(metric_configs):
    eval_configs = dict(
        epitope = {
            'Peptide->TRA': metric_configs,
        },
        TRB = {
            'TRB->Peptide': metric_configs,
        },
        CDR1B = {
            'CDR1B->Peptide': metric_configs,
        },
        CDR2B = {
            'CDR2B->Peptide': metric_configs,
        },
        CDR3B = {
            'CDR3B->Peptide': metric_configs,
        },
        TRA = {
            'TRA->Peptide': metric_configs,
        },
        CDR1A = {
            'CDR1A->Peptide': metric_configs,
        },
        CDR2A = {
            'CDR2A->Peptide': metric_configs,
        },
        CDR3A = {
            'CDR3A->Peptide': metric_configs,
        },
        CDR123A = {
            'TRA->Peptide': metric_configs,
        },
        NCDRA = {
            'TRA->Peptide': metric_configs,
        },
        CDR123B = {
            'TRB->Peptide': metric_configs,
        },
        NCDRB = {
            'TRB->Peptide': metric_configs,
        },
    )
    return eval_configs

def fetch_activate_region(
    analyzer:PosthocAnalyzer,
    region_dataset:XAIRegionBenchmark,
    input_key:str,
    region_key:str,
):
    _region_vals = [
        _sample[_region[0]:_region[1]]
        for _region, _sample
        in zip(region_dataset[region_key], analyzer[input_key])
    ]
    analyzer.collects[region_key] = _region_vals

def fetch_regions(
    analyzer:PosthocAnalyzer,
    region_dataset:XAIRegionBenchmark,
    input_key:str,
    region_keys:list[str],
    name:str='region',
    reverse:bool=False
):
    _region_vals = []
    for _idx, _sample in enumerate(analyzer[input_key]):
        _indices = np.zeros_like(_sample, dtype=np.bool_)
        for _region_key in region_keys:
            _region = region_dataset[_region_key][_idx]
            _indices[_region[0]:_region[1]] = True
        if reverse: _indices = ~_indices
        _region_vals.append(_sample[_indices])
    analyzer.collects[name] = _region_vals

def split_tcr_regions(analyzer:PosthocAnalyzer):
    region_dataset = XAIRegionBenchmark('data/TCRXAI/distance.json')
    if 'TRA' in analyzer.keys():
        fetch_activate_region(analyzer, region_dataset, 'TRA', 'CDR3A')
        fetch_activate_region(analyzer, region_dataset, 'TRA', 'CDR2A')
        fetch_activate_region(analyzer, region_dataset, 'TRA', 'CDR1A')
        fetch_regions(analyzer, region_dataset, 'TRA', ['CDR1A','CDR2A','CDR3A'], name='CDR123A')
        fetch_regions(analyzer, region_dataset, 'TRA', ['CDR1A','CDR2A','CDR3A'], name='NCDRA', reverse=True)
    if 'TRB' in analyzer.keys():
        fetch_activate_region(analyzer, region_dataset, 'TRB', 'CDR3B')
        fetch_activate_region(analyzer, region_dataset, 'TRB', 'CDR2B')
        fetch_activate_region(analyzer, region_dataset, 'TRB', 'CDR1B')
        fetch_regions(analyzer, region_dataset, 'TRB', ['CDR1B','CDR2B','CDR3B'], name='CDR123B')
        fetch_regions(analyzer, region_dataset, 'TRB', ['CDR1B','CDR2B','CDR3B'], name='NCDRB', reverse=True)

def eval_posthoc(evaluator:DistanceEvaluator, output_dir:str):
    output_dir = os.path.join(output_dir, 'distance')
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(evaluator(build_eval_configs(brhr_configs))).T
    df.applymap(lambda x:x[0]).to_csv(os.path.join(output_dir, 'brhr.csv'))
    df.applymap(lambda x:x[1]).to_csv(os.path.join(output_dir, 'brhr_std.csv'))
    df = pd.DataFrame(evaluator(build_eval_configs(rocauc_configs))).T
    df.to_csv(os.path.join(output_dir, 'rocauc.csv'))
    df = pd.DataFrame(evaluator(build_eval_configs(precision_configs))).T
    df.to_csv(os.path.join(output_dir, 'precision.csv'))
    df = pd.DataFrame(evaluator(build_eval_configs(recall_configs))).T
    df.to_csv(os.path.join(output_dir, 'recall.csv'))
    df = pd.DataFrame(evaluator(build_eval_configs(f1_configs))).T
    df.to_csv(os.path.join(output_dir, 'f1.csv'))
    df = pd.DataFrame(evaluator(build_eval_configs(accuracy_configs))).T
    df.to_csv(os.path.join(output_dir, 'accuracy.csv'))
    df = pd.DataFrame(evaluator(build_activate_configs(agg_activate_level_configs))).T
    df.to_csv(os.path.join(output_dir, 'activation.csv'))
    with open(os.path.join(output_dir, 'weights.json'), 'w') as f:
        _collects = evaluator.analyzer.collects
        _collects = {k:[np.array(i).tolist() for i in v] for k,v in _collects.items()}
        json.dump(_collects, f)

# Experiment Extension
class TCRRepExperiment(ModelExperiment):
    def forward(self, tcrr=None, immrep23=None):
        super().forward()
        self.experiment(
            TrainTestSplit(
                trainset=tcrr,
                testset=immrep23,
            ),
            run_name = 'immrep23'
        )
        folds = KFold(tcrr, k=5)
        for i, fold in enumerate(folds):
            self.experiment(fold, run_name=f'fold_{i}')
        
        
class TCRIMMREPExperiment(ModelExperiment):
    def forward(self, tcrr=None, immrep23=None, select_folds=[1,2], skip_immrep=False):
        super().forward()
        if not skip_immrep:
            self.experiment(
                TrainTestSplit(
                    trainset=tcrr,
                    testset=immrep23,
                ),
                run_name = 'immrep23'
            )
        folds = KFold(tcrr, k=5)
        for i, fold in enumerate(folds):
            if select_folds is None or i in select_folds:
                self.experiment(fold, run_name=f'fold_{i}')

# Plot & Report Utils
def fetch_train_state_(path):
    records = os.listdir(path)
    ids = re.findall('checkpoint-([0-9]+)', '\n'.join(records))
    ids = sorted([int(i) for i in ids])
    best_epoch = max(ids)
    best_epoch_folder = f'checkpoint-{best_epoch}'
    trainer_state_path = os.path.join(path, best_epoch_folder, 'trainer_state.json')
    with open(trainer_state_path, 'r') as f:
        trainer_state = json.load(f)
    loss = [i['loss'] for i in trainer_state['log_history'] if 'loss' in i]
    lr = [i['learning_rate'] for i in trainer_state['log_history'] if 'learning_rate' in i]
    return {
        'loss': loss,
        'learning_rate': lr,
    }
    
def fetch_train_states(path):
    records = {}
    for _case in sorted(os.listdir(path)):
        _full_path = os.path.join(path, _case)
        if os.path.isdir(_full_path):
            records[_case] = fetch_train_state_(_full_path)
    return records

def plot_loss(states, ax=None, title=None):
    if ax is None:
        fig = plt.figure(figsize=(5,3))
        ax = fig.subplots()
    for key, val in states.items():
        ax.plot(val['loss'], label=key)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(linestyle=':')
    ax.legend()
    if title is not None:
        ax.set_title(title)
    return ax

def fetch_eval_states(path):
    records = {}
    for _case in sorted(os.listdir(path)):
        _full_path = os.path.join(path, _case)
        if os.path.isdir(_full_path):
            _full_path = os.path.join(_full_path, 'eval', f'{_case}.json')
            with open(_full_path, 'r') as f:
                eval_data = json.load(f)
            records[_case] = {(k,k_):v_ for k,v in eval_data.items() for k_, v_ in v.items()}
    return pd.DataFrame(records)

# generate_tokenizers(data:DataFrameDataset, tokenizer_build_configs)
def generate_tokenizers(data, configs):
    bar = tqdm(configs)
    for config in bar:
        bar.set_description(config['name'])
        tokenizer = build_tokenizer(
            data = data,
            **config
        )

tokenizer_build_configs = [
    dict(
        columns = ['TRA', 'TRB', 'epitope'],
        template = 'tokenizers/builders/aminoacid.json',
        path = 'tokenizers/aminoacid',
        remove_na = True,
        unique = True,
        name = 'Amino Acid Tokenizer',
        desc = 'Tokenize amino-acid sequence bit by bit.'
    ),
    dict(
        columns = ['TRA', 'TRB'],
        template = 'tokenizers/builders/aminoacid.json',
        path = 'tokenizers/tcr',
        remove_na = True,
        unique = True,
        name = 'T Cell Receptor Tokenizer',
        desc = 'Tokenize T cell receptor full sequence bit by bit.'
    ),
    dict(
        columns = ['epitope'],
        template = 'tokenizers/builders/aminoacid.json',
        path = 'tokenizers/epitope',
        remove_na = True,
        unique = True,
        name = 'Epitope Tokenizer',
        desc = 'Tokenize epitope bit by bit.'
    ),
    dict(
        columns = ['MHC'],
        template = 'tokenizers/builders/alleles.json',
        path = 'tokenizers/mhc',
        remove_na = True,
        unique = True,
        name = 'MHC Tokenizer',
        desc = 'Tokenize MHC alleles.'
    ),
    dict(
        columns = ['TRAV'],
        template = 'tokenizers/builders/alleles.json',
        path = 'tokenizers/trav',
        remove_na = True,
        unique = True,
        name = 'TRAV Tokenizer',
        desc = 'Tokenize TRAV alleles.'
    ),
    dict(
        columns = ['TRAJ'],
        template = 'tokenizers/builders/alleles.json',
        path = 'tokenizers/traj',
        remove_na = True,
        unique = True,
        name = 'TRAJ Tokenizer',
        desc = 'Tokenize TRAJ alleles.'
    ),
    dict(
        columns = ['TRBV'],
        template = 'tokenizers/builders/alleles.json',
        path = 'tokenizers/trbv',
        remove_na = True,
        unique = True,
        name = 'TRBV Tokenizer',
        desc = 'Tokenize TRBV alleles.'
    ),
    dict(
        columns = ['TRBJ'],
        template = 'tokenizers/builders/alleles.json',
        path = 'tokenizers/trbj',
        remove_na = True,
        unique = True,
        name = 'TRBJ Tokenizer',
        desc = 'Tokenize TRBJ alleles.'
    ),
    dict(
        columns = ['MHC'],
        template = 'tokenizers/builders/alleles_components.json',
        path = 'tokenizers/mhc-components',
        remove_na = True,
        unique = True,
        name = 'MHC Component Tokenizer',
        desc = 'Tokenize MHC alleles following components.'
    ),
    dict(
        columns = ['TRAV'],
        template = 'tokenizers/builders/alleles_components.json',
        path = 'tokenizers/trav-components',
        remove_na = True,
        unique = True,
        name = 'TRAV Component Tokenizer',
        desc = 'Tokenize TRAV alleles following components.'
    ),
    dict(
        columns = ['TRAJ'],
        template = 'tokenizers/builders/alleles_components.json',
        path = 'tokenizers/traj-components',
        remove_na = True,
        unique = True,
        name = 'TRAJ Component Tokenizer',
        desc = 'Tokenize TRAJ alleles following components.'
    ),
    dict(
        columns = ['TRBV'],
        template = 'tokenizers/builders/alleles_components.json',
        path = 'tokenizers/trbv-components',
        remove_na = True,
        unique = True,
        name = 'TRBV Component Tokenizer',
        desc = 'Tokenize TRBV alleles following components.'
    ),
    dict(
        columns = ['TRBJ'],
        template = 'tokenizers/builders/alleles_components.json',
        path = 'tokenizers/trbj-components',
        remove_na = True,
        unique = True,
        name = 'TRBJ Component Tokenizer',
        desc = 'Tokenize TRBJ alleles following components.'
    ),
]