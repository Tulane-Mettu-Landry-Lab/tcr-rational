import os
import json
import torch
import torch.nn as nn
from transformers.modeling_outputs import ModelOutput, dataclass
from transformers import PretrainedConfig, BertPreTrainedModel
from typing import Optional

from ..configs._config import IMMLConfiguration

from ._bert_builder import BertBuilder
from ._bert_linear_head import BertLinearHead
from ..data._rule_processor import RuleProcessor
from ..loss._preconfig_loss import PreConfigLoss
from ._model_lib import ModelLibrary

@dataclass
class TCRpMHCModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    
    loss_mlm_cdr3a: Optional[torch.FloatTensor] = None
    loss_mlm_cdr2a: Optional[torch.FloatTensor] = None
    loss_mlm_cdr1a: Optional[torch.FloatTensor] = None
    loss_mlm_tra: Optional[torch.FloatTensor] = None
    
    loss_mlm_cdr3b: Optional[torch.FloatTensor] = None
    loss_mlm_cdr2b: Optional[torch.FloatTensor] = None
    loss_mlm_cdr1b: Optional[torch.FloatTensor] = None
    loss_mlm_trb: Optional[torch.FloatTensor] = None
    
    loss_mlm_epitope: Optional[torch.FloatTensor] = None
    
    loss_binder: Optional[torch.FloatTensor] = None
    loss_mhc: Optional[torch.FloatTensor] = None
    loss_mhclass: Optional[torch.FloatTensor] = None
    loss_trva: Optional[torch.FloatTensor] = None
    loss_trja: Optional[torch.FloatTensor] = None
    loss_trvb: Optional[torch.FloatTensor] = None
    loss_trjb: Optional[torch.FloatTensor] = None
    loss_species: Optional[torch.FloatTensor] = None
    
    logits_cdr3a: Optional[torch.FloatTensor] = None
    logits_cdr2a: Optional[torch.FloatTensor] = None
    logits_cdr1a: Optional[torch.FloatTensor] = None
    logits_tra: Optional[torch.FloatTensor] = None
    
    logits_cdr3b: Optional[torch.FloatTensor] = None
    logits_cdr2b: Optional[torch.FloatTensor] = None
    logits_cdr1b: Optional[torch.FloatTensor] = None
    logits_trb: Optional[torch.FloatTensor] = None
    
    logits_epitope: Optional[torch.FloatTensor] = None
    
    predictions_binder: Optional[torch.FloatTensor] = None
    predictions_mhc: Optional[torch.FloatTensor] = None
    predictions_mhclass: Optional[torch.FloatTensor] = None
    predictions_trva: Optional[torch.FloatTensor] = None
    predictions_trja: Optional[torch.FloatTensor] = None
    predictions_trvb: Optional[torch.FloatTensor] = None
    predictions_trjb: Optional[torch.FloatTensor] = None
    predictions_species: Optional[torch.FloatTensor] = None

@ModelLibrary.register('TCRpMHC-MLM-Tasks')
class TCRpMHCModel(BertPreTrainedModel):
    def __init__(
        self,
        model_configs:dict,
        loss_configs:dict,
        processor_configs:dict,
        keys:list[str],
        mlm_loss_prefix:str='mlm',
        task_configs:list[dict]=[],
    ) -> None:
        super().__init__(config=PretrainedConfig())
        
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
                self.bert_mlms[k] for k in keys
            ],
            loss = PreConfigLoss(config=self.loss_configs[_key])
        )
        self.task_heads[name] = _header
        self.task_inputs[name] = keys
        self.task_targets[name] = target
        
    def forward_tasks(self, inputs, outputs, model_output):
        
        for _task, _head in self.task_heads.items():
            _out = _head(
                *[outputs[i] for i in self.task_inputs[_task]],
                target = inputs[self.task_targets[_task]]
            )
            model_output = self.update_output(task_output=_out, output=model_output, task=_task)
        return outputs, model_output
        
        
        
    def _forward_mlm(self, **kwargs):
        outputs = {}
        model_output = TCRpMHCModelOutput()
        for key, mlm in self.bert_mlms.items():
            outputs[key] = mlm(**kwargs)
            model_output[f'logits_{key}'] = outputs[key].logits
        return outputs, model_output

    def _loss_mlm(self, outputs, model_output):
        losses = []
        for key, out in outputs.items():
            losses.append(out.loss)
            model_output[f'loss_mlm_{key}'] = out.loss
        model_output['loss'] = torch.stack(losses).sum()
        return outputs, model_output
    
    def forward_mlm(self, **kwargs):
        outputs, model_output = self._forward_mlm(**kwargs)
        outputs, model_output = self._loss_mlm(outputs, model_output)
        return outputs, model_output
    
    def update_output(self, task_output, output, task:str='binder'):
        output['loss'] = output['loss'] + task_output['loss']
        output[f'predictions_{task}'] = task_output['predictions']
        output[f'loss_{task}'] = task_output['loss']
        return output
    
    def forward(self, full_outputs:bool=False, **kwargs):
        kwargs['output_hidden_states'] = True
        outputs, model_output = self.forward_mlm(**kwargs)
        outputs, model_output = self.forward_tasks(inputs=kwargs, outputs=outputs, model_output=model_output)
        if full_outputs:
            return outputs, model_output
        else:
            return model_output
    
    def save_pretrained(self, save_directory, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        state_path = os.path.join(save_directory, 'model.pt')
        torch.save(self.state_dict(), state_path)
        config = {
            'model_configs': self.model_configs.to_dict(),
            'loss_configs': self.loss_configs.to_dict(),
            'processor_configs': {k:v.to_dict() if isinstance(v, IMMLConfiguration) else v for k,v in self.processor_configs.items()},
            'mlm_loss_prefix': self.mlm_loss_prefix,
            'keys': self.keys,
            'task_configs': self.task_configs,
        }
        config_path = os.path.join(save_directory, 'config.json')
        with open(config_path, 'w') as config_file_:
            json.dump(config, config_file_, indent=2)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config_path = os.path.join(pretrained_model_name_or_path, 'config.json')
        with open(config_path, 'r') as config_file_:
            config = json.load(config_file_)
        _model = cls(**config)
        state_path = os.path.join(pretrained_model_name_or_path, 'model.pt')
        _model.load_state_dict(torch.load(state_path, weights_only=True))
        return _model