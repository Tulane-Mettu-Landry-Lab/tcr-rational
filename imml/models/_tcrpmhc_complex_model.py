import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import ModelOutput
from typing import OrderedDict, Optional

from ._tcrpmhc_model import TCRpMHCModel
from ._bert_builder import BertBuilder
from ._bert_crossatten_decoder import BertCrossAttentionDecoder
from ._bert_linear_head import BertLinearHead
from ..data._rule_processor import RuleProcessor
from ..loss._preconfig_loss import PreConfigLoss
from ._model_lib import ModelLibrary

@ModelLibrary.register('TCRpMHC-MLM-CA-Tasks')
class TCRpMHCModelComplex(TCRpMHCModel):
        
    def __init__(
        self,
        model_configs:dict,
        loss_configs:dict,
        processor_configs:dict,
        keys:list[str],
        mlm_loss_prefix:str='mlm',
        task_configs:list[dict]=[],
    ) -> None:
        super(BertPreTrainedModel, self).__init__(config=PretrainedConfig())
        
        self.keys=keys
        self.task_configs=task_configs
        self.model_configs = model_configs
        self.loss_configs = loss_configs
        self.processor_configs = processor_configs
        self.mlm_loss_prefix = mlm_loss_prefix
        processor = RuleProcessor(
            rules=processor_configs['rules'],
            tokenizers=processor_configs['tokenizers'],
            primary_tokenizer=processor_configs.get('primary_tokenizer', 0),
        )
        self.processor = processor
        
        self._dynamic_tied_weights_keys = []
        self.bert_mlms = nn.ModuleDict({
            key:BertBuilder.build(config=model_configs[key], processor=processor)
            for key in self.keys
        })
        for _key, _module in self.bert_mlms.items():
            if _module._tied_weights_keys is not None:
                self._dynamic_tied_weights_keys += [f'bert_mlms.{_key}.{key}' for key in _module._tied_weights_keys]
        self.loss_mlms = nn.ModuleDict({
            key:PreConfigLoss(config=loss_configs[f'{mlm_loss_prefix}_{key}'])
            for key in self.keys
        })
        
        self.loss_decoder = nn.ModuleDict({
            key:PreConfigLoss(config=loss_configs[key])
            for key in model_configs
            if str(key).split('_')[0] == 'decoder'
        })
        self.decoders = nn.ModuleDict(OrderedDict({
            key:BertCrossAttentionDecoder.build(config=model_configs[key], processor=self.processor)
            for key in model_configs
            if str(key).split('_')[0] == 'decoder'
        }))
        
        self.task_heads = nn.ModuleDict(dict())
        self.task_inputs = dict()
        self.task_targets = dict()
        
        for task_config in task_configs:
            self.add_task_head(**task_config)
            
    def add_task_head(
        self,
        name:str,
        keys:list[str],
        target:str,
        prefix:Optional[str]='header'
    ) -> None:
        if prefix is None:
            _key = name
        else:
            _key = f'{prefix}_{name}'
        _header_class = BertLinearHead.preconfig(config=self.model_configs[_key])
        _header = _header_class(
            models = [
                self.bert_mlms[k] for k in keys if k in self.bert_mlms
            ] + [
                self.decoders[k] for k in keys if k in self.decoders
            ],
            loss = PreConfigLoss(config=self.loss_configs[_key])
        )
        self.task_heads[name] = _header
        self.task_inputs[name] = keys
        self.task_targets[name] = target
        
    def _forward_decoder(self, inputs, output, model_output:ModelOutput):
        for key, decoder in self.decoders.items():
            output[key] = decoder(inputs=inputs, output=output)
            model_output[f'logits_{key}'] = output[key].logits
        return output, model_output

    def _loss_decoder(self, outputs, model_output:ModelOutput):
        losses = []
        for key in self.decoders:
            out = outputs[key]
            losses.append(self.loss_decoder[key](out.loss))
            model_output[f'loss_{key}'] = out.loss
        _loss = torch.stack(losses).sum()
        model_output['loss'] += _loss
        return outputs, model_output
    
    def forward_decoder(self, inputs, output, model_output:ModelOutput):
        outputs, model_output = self._forward_decoder(inputs=inputs, output=output, model_output=model_output)
        outputs, model_output = self._loss_decoder(outputs, model_output)
        return outputs, model_output
            
    def forward(self, full_outputs = False, **kwargs):
        kwargs['output_hidden_states'] = True
        outputs, model_output = self.forward_mlm(**kwargs)
        outputs, model_output = self.forward_decoder(inputs=kwargs, output=outputs, model_output=model_output)
        outputs, model_output = self.forward_tasks(inputs=kwargs, outputs=outputs, model_output=model_output)
        if full_outputs:
            return outputs, model_output
        else:
            return model_output