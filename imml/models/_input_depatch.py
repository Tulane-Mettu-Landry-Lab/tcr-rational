import torch.nn as nn
from transformers import BertPreTrainedModel, PretrainedConfig
from typing import Any

class InputDepatch(BertPreTrainedModel):
    def __init__(
        self,
        model:BertPreTrainedModel,
        prefix:str='a',
        keys:list[str]=['output_attentions', 'output_hidden_states', 'return_dict']
    ) -> None:
        super().__init__(config=PretrainedConfig())
        self.model = model
        self.prefix = prefix
        self.keys = keys
        self.config = self.model.config
    
    def __extract_inputs(
        self,
        params,
        prefix='a'
    ) -> dict[str, Any]:
        _selected_params = {}
        for param_key, param_val in params.items():
            if param_key.split('_')[0] == prefix:
                _selected_params['_'.join(param_key.split('_')[1:])] = param_val
            elif param_key in self.keys:
                _selected_params[param_key] = param_val
        return _selected_params
    
    def forward(self, **kwargs):
        params = self.__extract_inputs(kwargs, prefix=self.prefix)
        return self.model(**params)