import torch
from typing import Callable
from .solvers._base import ModulePosthoc

class ModulePosthocReduce(object):
    _reduce_methods = {}
    def __init__(self, next_:ModulePosthoc, prev_:ModulePosthoc):
        next_ = next_.__qualname__
        prev_ = prev_.__qualname__
        if next_ not in self._reduce_methods:
            raise LookupError(f'Reduce from {next_} to {prev_} doesn\'t exist')
        if prev_ not in self._reduce_methods[next_]:
            raise LookupError(f'Reduce from {next_} to {prev_} doesn\'t exist')
        self.reduce_func = self._reduce_methods[next_][prev_]
        self.next_ = next_
        self.prev_ = prev_
    
    def __repr__(self):
        return f'Reduce: {self.next_} -> {self.prev_}'

    def reduce(self, x_next_:torch.Tensor, x_prev_:torch.Tensor):
        return self.reduce_func(x_next_, x_prev_)
    
    def __call__(self, x_next_:torch.Tensor, x_prev_:torch.Tensor):
        return self.reduce(x_next_=x_next_, x_prev_=x_prev_)
    
    @classmethod
    def register(cls, next_:ModulePosthoc, prev_:ModulePosthoc, method:Callable=torch.bmm, direction:bool=True):
        next_ = next_.__qualname__
        prev_ = prev_.__qualname__
        if next_ not in cls._reduce_methods:
            cls._reduce_methods[next_] = {}
        cls._reduce_methods[next_][prev_] = method
        if next_ != prev_ and not direction:
            if prev_ not in cls._reduce_methods:
                cls._reduce_methods[prev_] = {}
            cls._reduce_methods[prev_][next_] = method