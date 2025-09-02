import torch.nn as nn
from transformers import (
    BertConfig, AutoTokenizer,
    BertPreTrainedModel, BertForMaskedLM, BertModel, 
)
from ._input_depatch import InputDepatch
from ..data._rule_processor import RuleProcessor


class BertBuilder(object):
    
    model_dict = {
        'BertForMaskedLM': BertForMaskedLM,
        'BertModel': BertModel,
    }
    
    @classmethod
    def fetch_model(cls, model_name:str):
        return cls.model_dict[model_name]

    @classmethod
    def build_model(cls, model_name:str, tokenizer:AutoTokenizer, configs:dict, decoder:bool=False):
        _model = cls.fetch_model(model_name=model_name)
        _config = BertConfig(
            vocab_size=tokenizer.vocab_size,
            pad_token_id=tokenizer.pad_token_type_id,
            **configs
        )
        if decoder:
            _config.is_decoder = True
            _config.add_cross_attention=True
        return _model(_config)
    
    @classmethod
    def build(
        cls,
        config:dict,
        processor:RuleProcessor,
        keys:list[str]=['output_attentions', 'output_hidden_states', 'return_dict']
    ) -> nn.Module:
        prefix = config['prefix']
        model_name = config['model']
        tokenizer_name = config['tokenizer']
        tokenizer = processor._tokenizers[tokenizer_name]
        config = {k:v for k,v in config.items() if k not in ['prefix', 'model', 'tokenizer']}
        _model = cls.build_model(model_name=model_name, tokenizer=tokenizer, configs=config)
        _model = InputDepatch(_model, prefix=prefix, keys=keys)
        return _model
    
    @classmethod
    def add(cls, name:str, model:BertPreTrainedModel):
        cls.model_dict[name] = model