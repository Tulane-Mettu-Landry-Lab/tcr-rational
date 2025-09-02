import torch.nn as nn
from typing import Union
from copy import deepcopy
from .models import ModelLibrary
from .configs import IMMLConfigurationGroup

class ModelKit(object):
    
    def __init__(
        self,
        model:Union[str,nn.Module],
        configs:IMMLConfigurationGroup,
        mlm_loss_prefix:str='mlm',
        primary_tokenizer:Union[int,str]=0,
    ) -> None:
        if isinstance(model, str):
            model = ModelLibrary[model]
        self.configs = configs
        self.mlm_loss_prefix = mlm_loss_prefix
        self.primary_tokenizer = primary_tokenizer
        self.model_template = model
        self.model = self.__init_model(
            model=self.model_template,
            configs=self.configs,
            mlm_loss_prefix=self.mlm_loss_prefix,
            primary_tokenizer=self.primary_tokenizer,
        )
        self.processor = self.model.processor
        if 'collator' in configs:
            self.collator = \
                self.processor.colletor(
                    **configs['collator']
                )
        else:
            self.collator = None
    
    def __init_model(
        self,
        model:nn.Module,
        configs:IMMLConfigurationGroup,
        mlm_loss_prefix:str='mlm',
        primary_tokenizer:Union[int,str]=0,
    ):
        return model(
            model_configs=configs['model'],
            loss_configs=configs['loss'],
            processor_configs=dict(
                rules=configs['rules'].to_list(),
                tokenizers=configs['tokenizers'].to_list(),
                primary_tokenizer=primary_tokenizer,
            ),
            mlm_loss_prefix=mlm_loss_prefix,
            task_configs=configs['model_tasks']['task_configs'],
            keys=configs['model_tasks']['keys'],
        )
        
    def clone(self):
        return ModelKit(
            model = self.model_template,
            configs = self.configs,
            mlm_loss_prefix = self.mlm_loss_prefix,
            primary_tokenizer = self.primary_tokenizer,
        )