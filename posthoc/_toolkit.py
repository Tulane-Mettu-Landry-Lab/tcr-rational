import torch.nn as nn
from ._component import ModelComponents
from ._hook import ModuleHook
from ._stack_group import ModuleStackGroup
from .module_node import ModuleNode

class ModelToolKit(object):

    def __init__(self, model:nn.Module):
        self.model = model
        self.model_name = model._get_name()
        self.model_node = ModuleNode(module=self.model, name=self.model_name)
        self.model_hook = ModuleHook(module=self.model)
        self.modules_group = ModuleStackGroup(module_node=self.model_node, module_hook=self.model_hook)
    
    @property
    def components(self):
        return ModelComponents(self.model_node)
    
    @property
    def tracks(self):
        return self.modules_group
    
    @property
    def nodes(self):
        return self.model_node
    
    def __repr__(self):
        return f'Model:{self.model_name}'
    
    def _repr_html_(self):
        return self.components._repr_html_()