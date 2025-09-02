import torch.nn as nn
from typing import Optional
class ModuleInfo(object):

    def __init__(
            self,
            module:nn.Module,
            name:Optional[str]=None,
            path:Optional[str]=None
        ) -> None:
        if name is None:
            self.name = module._get_name()
        else:
            self.name = name

        _type = type(module)
        self.module_name = module._get_name()
        self.module_bases = _type.__bases__
        self.module_path = _type.__module__
        self.module_type = _type.__name__

        self.id = id(module)
        self.path = path
    
    def __repr__(self):
        return f'{self.module_name}::{self.name} ({self.module_path})'