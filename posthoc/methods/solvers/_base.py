from typing import Callable

class ModulePosthoc(object):
    _target_layer = 'Any'
    _target_mechanism = 'Any'
    def __repr__(self):
        return f'{self.__qualname__}: {self._target_mechanism} -> {self._target_layer}'
    def solve(self, forward_in, forward_out, backward_in, backward_out):
        raise NotImplementedError
    def __call__(self, forward_in, forward_out, backward_in, backward_out):
        return self.solve(forward_in, forward_out, backward_in, backward_out)