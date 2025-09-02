import torch
import torch.nn as nn
from transformers import BertPreTrainedModel
from transformers.modeling_outputs import ModelOutput
from transformers.models.bert.modeling_bert import BertOnlyMLMHead

from ._bert_builder import BertBuilder
from ..data._rule_processor import RuleProcessor

class BertCrossAttentionDecoder(BertPreTrainedModel):
    
    def __init__(
        self,
        prefix:str, 
        hidden_state_keys:list[str],
        bert:BertPreTrainedModel,
        forward_keys:list[str]=['output_attentions', 'output_hidden_states', 'return_dict'],
    ):
        super().__init__(bert.config)
        self.bert = bert
        self.input_key = prefix
        self.hidden_state_keys = hidden_state_keys
        self.forward_keys = forward_keys
        self.loss_fn = nn.CrossEntropyLoss()
        self.cls = BertOnlyMLMHead(bert.config)
        
    def __fetch_hidden_states(
        self,
        output:dict[str,ModelOutput],
        keys:list[str]
    ):
        _hidden_states = []
        for key in keys:
            _hidden_state = output[key]['hidden_states'][-1]
            _hidden_states.append(_hidden_state)
        _hidden_states = torch.concat(_hidden_states, dim=1)
        return _hidden_states
    
    
    def __fetch_attention_masks(
        self,
        inputs:dict[str,torch.Tensor],
        outputs:dict[str,torch.Tensor],
        keys:list[str]
    ):
        _attention_masks = []
        for key in keys:
            if f'{key}_attention_mask' in inputs:
                _attention_mask = inputs[f'{key}_attention_mask']
            else:
                _attention_mask = outputs[key]['attention_mask']
            _attention_masks.append(_attention_mask)
        _attention_masks = torch.concat(_attention_masks, dim=1)
        return _attention_masks
    
    def __extract_inputs(
        self,
        params:dict[str,torch.Tensor],
        prefix='a'
    ) -> dict[str, torch.Tensor]:
        _selected_params = {}
        for param_key, param_val in params.items():
            if param_key.split('_')[0] == prefix:
                _selected_params['_'.join(param_key.split('_')[1:])] = param_val
            elif param_key in self.forward_keys:
                _selected_params[param_key] = param_val
        return _selected_params
    
    def forward(
        self,
        inputs:dict[str,torch.Tensor],
        output:dict[str,ModelOutput],
    ):
        _input = self.__extract_inputs(params=inputs, prefix=self.input_key)
        _input['encoder_hidden_states'] = self.__fetch_hidden_states(output=output, keys=self.hidden_state_keys)
        _input['encoder_attention_mask'] = self.__fetch_attention_masks(inputs=inputs, outputs=output, keys=self.hidden_state_keys)
        _output = self.bert(**{k:v for k,v in _input.items() if k!='labels'})
        _logits = self.cls(_output[0])
        _output['logits'] = _logits
        _loss = self.loss_fn(_logits.view(-1, self.config.vocab_size), _input['labels'].view(-1))
        _output['loss'] = _loss
        _output['attention_mask'] = _input['attention_mask']
        return _output
    
    @classmethod
    def build(
        cls,
        config:dict,
        processor:RuleProcessor,
        keys:list[str]=['output_attentions', 'output_hidden_states', 'return_dict']
    ):
        prefix = config['prefix']
        model_name = config['model']
        tokenizer_name = config['tokenizer']
        hidden_state_keys = config['hidden_state_keys']
        tokenizer = processor._tokenizers[tokenizer_name]
        config = {k:v for k,v in config.items() if k not in ['prefix', 'model', 'tokenizer', 'hidden_state_keys']}
        bert = BertBuilder.build_model(
            model_name=model_name,
            tokenizer=tokenizer,
            configs=config,
            decoder=True
        )
        _model = cls(
            prefix=prefix,
            bert = bert,
            hidden_state_keys=hidden_state_keys,
            forward_keys=keys,
        )
        return _model