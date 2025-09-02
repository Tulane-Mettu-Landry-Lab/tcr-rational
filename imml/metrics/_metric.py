import numpy as np

class Metric(object):
    _global_metrics = {}
    name = 'base_metric'
    def __init__(self, pred_key:str, label_key:str, **kwargs):
        self.pred_key = pred_key
        self.label_key = label_key
        self.configs = kwargs
        
    def __repr__(self):
        return f'{self.name}(y_true={self.label_key}, y_pred={self.pred_key})'
    
    def metric(self, y_true, y_pred, **kwargs):
        raise NotImplemented
      
    def forward(self, labels:dict[np.ndarray], predictions:dict[np.ndarray]):
        y_pred = predictions[self.pred_key]
        y_true = labels[self.label_key]
        return self.metric(y_true=y_true, y_pred=y_pred, **self.configs)
    
    def __call__(self, labels:dict[np.ndarray], predictions:dict[np.ndarray]):
        return self.forward(predictions=predictions, labels=labels)
    
    def __class_getitem__(cls, item):
        return cls._global_metrics[item]
    
    @classmethod
    def register(cls):
        def _register_wrapper(metric):
            cls._global_metrics[metric.name] = metric
            return metric
        return _register_wrapper
    
    @classmethod
    def models(cls):
        return cls._global_metrics.keys()