import numpy as np
from typing import Union

from ._metric import Metric

class MetricGroup(Metric):
    name = 'metric_group'
    def __repr__(self):
        return f'{self.name}{list(self.configs)}(y_true={self.label_key}, y_pred={self.pred_key})'
    
    def __len__(self):
        return len(self.configs)
    
    def __getitem__(self, key):
        return self.configs[key]
    
    def forward(self, labels:dict[np.ndarray], predictions:dict[np.ndarray]):
        return {
            key:metric(labels=labels, predictions=predictions)
            for key, metric in self.configs.items()
        }
    
    @classmethod
    def from_list(
        cls,
        pred_key:str,
        label_key:str,
        metrics:list[Union[str,Metric]],
        **kwargs
    ):
        return cls(
            pred_key, label_key,
            **{
                _metric
                if isinstance(_metric, str)
                else _metric.name:(
                    Metric[_metric](pred_key, label_key, **kwargs)
                    if isinstance(_metric, str)
                    else _metric(pred_key, label_key, **kwargs)
                )
                for _metric in metrics
            }
        )
    
    @classmethod
    def from_config(
        cls,
        config:dict
    ):
        return cls.from_list(
            pred_key=config['pred_key'],
            label_key=config['label_key'],
            metrics=config['metrics'],
            **config['parameters'],
        )