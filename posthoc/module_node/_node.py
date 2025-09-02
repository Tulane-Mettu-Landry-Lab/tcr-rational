import torch.nn as nn
from typing import Optional, Union
from ._info import ModuleInfo

class ModuleNode(object):

    def __init__(
            self,
            module:nn.Module,
            name:Optional[str]=None,
            path:Optional[str]=None
        ) -> None:
        self.module_info = \
            ModuleInfo(
                module=module,
                name=name,
                path=path
            )
        self.module = module
        self.submodules = self._enum_submodules(module=module)
    
    def _enum_submodules(
            self,
            module:nn.Module
        ) -> list:
        _submodules = []
        _next_path = self.name if self.path is None else '.'.join([self.path, self.name])
        for _name, _module in module.named_children():
            _submodules.append(
                (
                    _name,
                    ModuleNode(
                        module=_module,
                        name=_name,
                        path=_next_path
                    )
                )
            )
        return _submodules
    
    def __len__(self) -> int:
        return len(self.submodules)
    
    def _search_name(self, name:str) -> list:
        _search_results = []
        if self.module_name == name or self.name == name:
            _search_results.append(self)
        for _submodule in self.values():
            _search_results += _submodule._search_name(name=name)
        return _search_results
    
    def search_modules(self, query_path=''):
        queries = [i.strip() for i in query_path.split('->') if len(i) > 0]
        if len(queries) <= 0:
            return [self]
        else:
            query = queries[0]
            next_query = '->'.join(queries[1:])
            module_list = []
            for submodule in self[query]:
                module_list += submodule.search_modules(query_path=next_query)
            return module_list

    def __getitem__(self, key:Union[int, str]) -> list:
        if isinstance(key, int):
            return self.submodules[key][-1]
        elif isinstance(key, str):
            return self._search_name(name=key)
        else:
            raise TypeError('Index key type unsopport.')
    
    def __iter__(self):
        self._iter_idx = 0
        return self

    def __next__(self):
        if self._iter_idx >= len(self):
            raise StopIteration
        else:
            _value = self.submodules[self._iter_idx][-1]
            self._iter_idx += 1
            return _value
        
    def __repr__(self):
        return f'{self.module_name}::{self.name} -> {len(self)} submodules'
    
    def keys(self):
        return [_m[0] for _m in self.submodules]
    
    def values(self):
        return [_m[1] for _m in self.submodules]

    def items(self):
        return self.submodules
    
    @property
    def id(self):
        return self.module_info.id
    @property
    def name(self):
        return self.module_info.name
    @property
    def module_name(self):
        return self.module_info.module_name
    @property
    def path(self):
        return self.module_info.path
    

    @property
    def leaves(self):
        if len(self) == 0:
            return [self]
        else:
            _nodes = []
            for _node in self.values():
                _nodes += _node.leaves
            return _nodes