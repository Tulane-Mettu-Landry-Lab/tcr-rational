import numpy as np
from ._metric_group import MetricGroup

class MetricManager(object):
    def __init__(self, **kwargs):
        self.metrics = kwargs
        
    def __repr__(self):
        return f'Metrics for {list(self.metrics)}'
    
    def __len__(self):
        return len(self.metrics)
    
    def __getitem__(self, key):
        return self.metrics[key]
    
    def forward(self, labels:dict[np.ndarray], predictions:dict[np.ndarray]):
        return {
            name: metric(labels, predictions)
            for name, metric in self.metrics.items()
        }
    
    def __call__(self, labels:dict[np.ndarray], predictions:dict[np.ndarray]):
        return self.forward(labels, predictions)
    
    @classmethod
    def from_config(cls, configs:dict):
        return cls(**{name:MetricGroup.from_config(config) for name, config in configs.items()})