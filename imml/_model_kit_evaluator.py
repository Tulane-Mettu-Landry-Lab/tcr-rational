import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Optional

from .data import TransformerDataset
from .metrics import MetricManager
from ._model_kit_trainer import ModelKitTrainer

class ModelKitEvaluator(object):
    
    def __init__(
        self,
        modelkit_trainer: ModelKitTrainer,
        evalset: Dataset,
        metrics: MetricManager,
        output_dir: Optional[str]=None,
        disable_tqdm: bool=False,
        eval_name: str='Evaluation',
    ):
        self.modelkit_trainer = modelkit_trainer
        self.evalset = evalset
        self.metrics = metrics
        self.output_dir = output_dir
        self.disable_tqdm = disable_tqdm
        self.eval_name = eval_name
        
    @torch.no_grad()
    def fetch(self):
        torch.cuda.empty_cache()
        _dataset = TransformerDataset(self.evalset, processor=self.modelkit_trainer.modelkit.processor)
        testloader = self.modelkit_trainer.trainer.get_test_dataloader(_dataset)
        if not self.disable_tqdm:
            testloader = tqdm(testloader, total=len(testloader))
        outputs, inputs = [], []
        model = self.modelkit_trainer.trainer.model
        model = self.modelkit_trainer.trainer._wrap_model(model, training=False, dataloader=testloader)
        model = model.eval()
        for batch in testloader:
            out = model(**batch)
            outputs.append({k:v.cpu().detach().numpy() for k,v in out.items()})
            inputs.append({k:v.cpu().detach().numpy() for k,v in batch.items()})
        return inputs, outputs

    def eval(self):
        inputs, outputs = self.fetch()
        metric_values = []
        for _input, _output in zip(inputs, outputs):
            metric_values.append(self.metrics(_input, _output))
        metric_value = None
        for metric in metric_values:
            if metric_value is None:
                metric_value = {k:{_k:[_v] for _k,_v in v.items()} for k, v in metric.items()}
            else:
                for k in metric_value:
                    for _k in metric_value[k]:
                        metric_value[k][_k].append(metric[k][_k])
        for k in metric_value:
            for _k in metric_value[k]:
                metric_value[k][_k] = np.mean(metric_value[k][_k])
        
        if self.output_dir is not None:
            os.makedirs(os.path.join(self.output_dir, 'eval'), exist_ok=True)
            with open(os.path.join(self.output_dir, 'eval', f'{self.eval_name}.json'), 'w') as f_:
                json.dump(metric_value, f_, indent=2)
        return metric_value
    
    def __call__(self):
        return self.eval()    