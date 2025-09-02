import torch.nn as nn
from typing import List
from functools import wraps

class ModuleHook(object):

    def __init__(self, module:nn.Module, track_ids:List[int]=[]):
        self.track_ids = list(set(track_ids))
        self.module = module
        self.forward_hooks = {}
        self.backward_hooks = {}
        self.set_forward_cache()
        self.set_backward_cache()
        self.set_order_cache()
        self.hook()

    def add_modules(self, track_ids:List[int]):
        self.unhook()
        self.track_ids = list(set(self.track_ids + track_ids))
        self.set_forward_cache()
        self.set_backward_cache()
        self.set_order_cache()
        self.hook()

    def set_modules(self, track_ids:List[int]):
        self.unhook()
        self.track_ids = list(set(track_ids))
        self.set_forward_cache()
        self.set_backward_cache()
        self.set_order_cache()
        self.hook()

    def hook(self):
        for _name, _module in self.module.named_modules():
            _module_id = id(_module)
            if _module_id in self.track_ids:
                self.forward_hooks[_module_id] = _module.register_forward_hook(self.forward_hook)
                self.backward_hooks[_module_id] = _module.register_full_backward_hook(self.backward_hook)

    def unhook(self):
        for _hook in self.backward_hooks.values():
            _hook.remove()
        for _hook in self.forward_hooks.values():
            _hook.remove()
        self.backward_hooks = {}
        self.forward_hooks = {}
    
    def set_forward_cache(self):
        self.forward_in = {}
        self.forward_out = {}
    
    def set_backward_cache(self):
        self.backward_in = {}
        self.backward_out = {}

    def set_order_cache(self):
        self.forward_order = []
        self.backward_order = []

    def forward_hook(self, module, input, output):
        _module_id = id(module)
        self.forward_in[_module_id] = input
        self.forward_out[_module_id] = output
        self.forward_order.append(_module_id)
        
    def backward_hook(self, module, input, output):
        _module_id = id(module)
        self.backward_in[_module_id] = input[0]
        self.backward_out[_module_id] = output[0]
        self.backward_order.append(_module_id)

    def keys(self):
        return self.track_ids
    
    def __getitem__(self, key):
        _dirct, _io, _idx = None, None, None
        if isinstance(key, tuple):
            if len(key) == 3:
                _dirct, _io, _idx = key
            elif len(key) == 2:
                _dirct, _io = key
        else:
            if isinstance(key, str):
                _dirct = key
            elif isinstance(key, int):
                return (
                    self.forward_in[key],
                    self.forward_out[key],
                    self.backward_in[key],
                    self.backward_out[key],
                )
            else:
                raise IndexError('The index is invalid.')

        if _dirct == 'forward' or _dirct == 'f':
            _data = (self.forward_in, self.forward_out)
        elif _dirct == 'backward' or _dirct == 'b':
            _data = (self.backward_in, self.backward_out)
        else:
            raise IndexError('The first index should be forward(f) or backward(b).')

        if _io == 'input' or _io == 'in' or _io == 'i':
            _data = _data[0]
        elif _io == 'output' or _io == 'out' or _io == 'o':
            _data = _data[1]
        else:
            return _data
        
        if _idx is None:
            return _data
        else:
            return _data[_idx]
    
    def __len__(self):
        return len(self.track_ids)
    
    def __repr__(self):
        return f'{len(self)} hooks on {self.module._get_name()}'
    
    def clean(self):
        self.set_forward_cache()
        self.set_backward_cache()
        self.set_order_cache()
    
    def collect(self, func):
        @wraps
        def wrapper(*args, **kwargs):
            self.clean()
            return func(*args, **kwargs)
        return wrapper