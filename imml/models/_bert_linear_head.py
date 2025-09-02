import torch
import torch.nn as nn
from typing import Optional, Union
from transformers import BertPreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import ModelOutput, dataclass

@dataclass
class BertLinearHeadOutput(ModelOutput):
    predictions: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None
    
class BertLinearHead(nn.Module):
    
    def __init__(
        self,
        models:list[BertPreTrainedModel],
        loss:nn.Module,
        num_labels:int=2,
        layers:list[int]=[],
        bias:bool=True,
        act_func:Optional[Union[str, nn.Module]]='relu',
    ):
        super().__init__()
        self.loss = loss
        self.in_dim = sum([model.config.hidden_size for model in models])
        self.num_labels = num_labels
        _layers = [self.in_dim, *layers, self.num_labels]
        _modules = []
        for i in range(len(_layers)-1):
            _linear = nn.Linear(in_features=_layers[i], out_features=_layers[i+1], bias=bias)
            _modules.append(_linear)
            if _layers[i+1] != _layers[-1] and act_func is not None:
                _act = self.__get_act_func(act_func)
                _modules.append(_act)
        self._linear = nn.Sequential(*_modules)
    
    _act_func_dict = dict(
        relu = nn.ReLU,
        gelu = nn.GELU,
        elu = nn.ELU,
        leakyrelu = nn.LeakyReLU,
        sigmoid = nn.Sigmoid
    )
    def __get_act_func(
        self,
        act_func:Union[str, nn.Module]='relu'
    ) -> nn.Module:
        if isinstance(act_func, nn.Module):
            return act_func()
        else:
            return self._act_func_dict[act_func]()
    
    def __fetch_class_token(self, output):
        return output['hidden_states'][-1][:,0,:]
    
    def forward(self, *args, target=None):
        _out = torch.concat([
            self.__fetch_class_token(arg) for arg in args
        ], dim=-1)
        _out = self._linear(_out)
        _loss = None
        if target is not None:
            _loss = self.loss(_out, target)
        return BertLinearHeadOutput(
            predictions=_out,
            loss = _loss
        )
    
    @classmethod
    def preconfig(cls, config:dict):
        return lambda models, loss : cls(models=models, loss=loss, **config)