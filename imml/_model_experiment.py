import os
import platform
from typing import Union, Optional

from ._model_kit import ModelKit
from ._model_kit_trainer import ModelKitTrainer
from ._model_kit_evaluator import ModelKitEvaluator
from .configs import IMMLConfigurationGroup
from .data import TrainTestSplit
from .metrics import MetricManager

class ModelExperiment(object):
    
    def __init__(
        self,
        modelkit:Union[ModelKit,str],
        configs:IMMLConfigurationGroup,
        output_dir:str='logs/',
        log_level:str='error',
        overwrite_output_dir:bool=True,
    ):
        if isinstance(modelkit, ModelKit):
            self.modelkit = modelkit
        else:
            self.modelkit = ModelKit(modelkit, configs=configs)
        self.configs = configs
        self.output_dir = output_dir
        self.log_level = log_level
        self.overwrite_output_dir = overwrite_output_dir
    
    def train_evaluate(
        self,
        modelkit:ModelKit,
        dataset:TrainTestSplit,
        configs:IMMLConfigurationGroup,
        output_dir:str='logs/',
        run_name:str='experiment',
        log_level:str='error',
        overwrite_output_dir:bool=True,
    ):
        modelkit = modelkit.clone()
        model_trainer = ModelKitTrainer(
            modelkit=modelkit,
            trainset=dataset.train,
            configs=configs,
            output_dir=output_dir,
            run_name=run_name,
            log_level=log_level,
            overwrite_output_dir=overwrite_output_dir
        )
        model_trainer()
        if isinstance(dataset.test, dict):
            eval_result = {}
            for eval_name, _data in dataset.test.items():
                evaluator = ModelKitEvaluator(
                    modelkit_trainer=model_trainer,
                    evalset=_data,
                    metrics=MetricManager.from_config(configs['metrics']),
                    output_dir=output_dir,
                    disable_tqdm=False,
                    eval_name=eval_name,
                )
                _eval_result = evaluator()
                print(_eval_result)
                eval_result[eval_name] = _eval_result
        else:
            evaluator = ModelKitEvaluator(
                modelkit_trainer=model_trainer,
                evalset=dataset.test,
                metrics=MetricManager.from_config(configs['metrics']),
                output_dir=output_dir,
                disable_tqdm=False,
                eval_name=run_name,
            )
            eval_result = evaluator()
            print(eval_result)
        return eval_result, model_trainer
    
    def experiment(
        self,
        dataset:TrainTestSplit,
        run_name:str='experiment',
    ):
        print(f'run {run_name}')
        _out = self.train_evaluate(
            modelkit=self.modelkit,
            dataset=dataset,
            configs=self.configs,
            output_dir=os.path.join(self.output_dir, run_name),
            run_name=run_name,
            log_level=self.log_level,
            overwrite_output_dir=self.overwrite_output_dir,
        )
        return _out
    
    def forward(self, **kwargs):
        os.makedirs(self.output_dir, exist_ok=True)
        self.configs.save_config(os.path.join(self.output_dir, 'config.json'))
    
    def __call__(self, **kwargs):
        return self.forward(**kwargs)