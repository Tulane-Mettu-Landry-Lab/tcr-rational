from typing import Optional
import pandas as pd
import numpy as np

from .module_node import ModuleNode
from ._hook import ModuleHook
from .methods import ModulePosthocReduce
from ._flow import ModelFlow


class ModuleStack(object):

    def __init__(self, module_node:ModuleNode, *args, module_hook:Optional[ModuleHook]=None):
        self.node = module_node
        self.hook = module_hook
        self.track_modules = []
        self.add(*args)
        self.flow = ModelFlow(self)

    def set_hook(self, module_hook:Optional[ModuleHook]=None):
        self.hook = module_hook
        if module_hook is not None:
            self.hook.add_modules(self.module_ids)

    def set_node(self, module_node:ModuleNode):
        self.node = module_node
    
    def __len__(self):
        return len(self.track_modules)
    
    def __iter__(self):
        self._iter_idx = 0
        return self
    
    def __repr__(self):
        return f'Track {len(self)} modules in {self.node.name}'
    
    def __next__(self):
        if self._iter_idx >= len(self):
            raise StopIteration
        else:
            _value = self.track_modules[self._iter_idx]
            self._iter_idx += 1
            return _value
    
    def add(self, *args):
        for arg in args:
            if isinstance(arg, list):
                self.track_modules += arg
            else:
                self.track_modules.append(arg)
        if self.hook is not None:
            self.hook.add_modules(self.module_ids)
    
    @property
    def module_ids(self):
        return [subnode.id for subnode in self.track_modules]
    
    @property
    def forward_ordered_module_ids(self):
        _module_ids = self.module_ids
        return [id for id in self.hook.forward_order if id in _module_ids]
    @property
    def backward_ordered_module_ids(self):
        _module_ids = self.module_ids
        return [id for id in self.hook.backward_order if id in _module_ids]

    @property
    def table(self):
        module_id_map = {module.id:module for module in self}
        _module_ids = self.module_ids
        module_info = []
        for id in self.hook.forward_order:
            if id in _module_ids:
                module_name = module_id_map[id].module_name
                name = module_id_map[id].name
                path = module_id_map[id].path
                try: b_size = 'x'.join(['B'] + [str(i) for i in self.hook['b', 'i', id].shape[1:]])
                except: b_size = None
                try: f_size = 'x'.join(['B'] + [str(i) for i in self.hook['f', 'o', id].shape[1:]])
                except: f_size = None
                module_info.append([module_name, name, b_size, f_size, path, id])
        columns = ['Module', 'Name', 'Gradient Shape', 'Output Shape', 'Path', 'ID']
        _df =  pd.DataFrame(module_info, columns=columns)
        _df['Forward Order'] = np.arange(1, len(_df)+1, dtype=int)
        _df['Backward Order'] = np.arange(len(_df), 0, -1, dtype=int)
        return _df
    
    def _repr_html_(self):
        if self.hook is not None:
            return self.table._repr_html_()
        else:
            return repr(self)
  