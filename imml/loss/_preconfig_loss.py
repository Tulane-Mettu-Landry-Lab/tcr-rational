import torch.nn as nn
from ._forward import ForwardLoss

class PreConfigLoss(nn.Module):
    
    _loss_dict = {
        'cross_entropy': nn.CrossEntropyLoss,
        'l1': nn.L1Loss,
        'mse': nn.MSELoss,
        'nll': nn.NLLLoss,
        'bce': nn.BCELoss,
        'forward': ForwardLoss,
    }
    _act_dict = {
        'softmax': nn.Softmax,
        'softmin': nn.Softmin,
        'log_softmax': nn.LogSoftmax,
    }
    
    def __init__(self, config):
        super().__init__()
        _loss_name = config['loss']
        _act_name = config['act_func']
        self._mix_weight = config['mix_weight']
        _config = {k:v for k,v in config.items() if k not in ['loss', 'act_func', 'mix_weight']}
        self._act = self._act_dict.get(_act_name)
        if self._act is not None:
            self._act = self._act(dim=-1)
        self._loss = self._loss_dict[_loss_name](**_config)
    
    def forward(self, input, target=None):
        if self._act is not None:
            input = self._act(input)
        return self._loss(input, target) * self._mix_weight
    
    @classmethod
    def add(cls, name:str, loss:nn.Module):
        cls._loss_dict[name] = loss