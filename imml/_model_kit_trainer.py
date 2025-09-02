from torch.utils.data import Dataset
from transformers import TrainerCallback, Trainer, TrainingArguments

from ._model_kit import ModelKit
from .configs import IMMLConfigurationGroup
from .data import TransformerDataset
            
class SaveEveryNEpochsCallback(TrainerCallback):
    def __init__(self, n=10):
        self.n = n

    def on_epoch_end(self, args, state, control, **kwargs):
        if (int(state.epoch + 1e-5) % self.n) != 0:
            control.should_save = False 
        return control


class ModelKitTrainer:
    
    def __init__(
        self,
        modelkit:ModelKit,
        trainset:Dataset,
        configs:IMMLConfigurationGroup,
        output_dir:str='logs/',
        log_level:str='error',
        run_name:str='train',
        overwrite_output_dir:bool=True,
    ) -> None:
        self.modelkit = modelkit
        self.trainset = trainset
        self.configs = configs
        
        self.output_dir = output_dir
        self.log_level = log_level
        self.run_name = run_name
        self.overwrite_output_dir = overwrite_output_dir
        
        self.wrapped_trainset = \
            TransformerDataset(
                trainset,
                processor=modelkit.processor
            )
        self.trainer_args = self.__build_train_args()
        self.trainer = self.__build_trainer()
        
    
    def __build_train_args(self):
        train_configs = {k:v for k,v in self.configs['train'].items() if k not in ['save_n_epoch']}
        train_configs['log_level'] = self.log_level
        train_configs['output_dir'] = self.output_dir
        train_configs['run_name'] = self.run_name
        train_configs['report_to'] = 'none'
       
        train_configs['overwrite_output_dir'] = self.overwrite_output_dir
        return TrainingArguments(**train_configs)
    
    
    def __build_trainer(self):
        save_n_epoch = self.configs['train']['save_n_epoch']
        trainer = Trainer(
            model=self.modelkit.model,
            args=self.trainer_args,
            train_dataset=self.wrapped_trainset,
            data_collator=self.modelkit.collator,
            callbacks=[
                SaveEveryNEpochsCallback(save_n_epoch),
            ]
        )
        return trainer
    
    def train(self):
        return self.trainer.train()
    
    def __call__(self):
        return self.train()