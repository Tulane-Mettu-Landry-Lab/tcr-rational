import torch
from ._base import ModulePosthoc
from .attn_grad_rollout import GradAttnRollout, GradAttn
from .key_decompose import KeyDecomposeQuantifyQuery
from .._reducer import ModulePosthocReduce
from ._lib import SolverLibrary

@SolverLibrary.register('quantify_query')
class QuantifyQuery(ModulePosthoc):
    def __init__(self, reduce_method='sum', norm=True):
        if callable(reduce_method):
            self.reduce_method = reduce_method
        elif reduce_method == 'mean':
            self.reduce_method = lambda x: torch.mean(x, dim=-1)
        elif reduce_method == 'max':
            self.reduce_method = lambda x: torch.max(x, dim=-1).values
        elif reduce_method == 'sum':
            self.reduce_method = lambda x: torch.sum(x, dim=-1)
        else:
            raise TypeError('reduce_method is not supported')
        self.norm = norm
        
    def solve(self, forward_in, forward_out, backward_in, backward_out):
        h = forward_out
        dh = backward_in
        dhh = (dh * h)
        # dhh = h

        dhh = torch.relu(dhh)
        dhh = self.reduce_method(dhh)
        # dhh = self.reduce_method(backward_in)
        if self.norm:
            dhh = dhh / dhh.sum(dim=-1, keepdim=True)
            # dhh = (dhh - dhh.min(dim=-1, keepdim=True).values) / dhh.max(dim=-1, keepdim=True).values
        
        return dhh[:, :, None]
ModulePosthocReduce.register(KeyDecomposeQuantifyQuery, QuantifyQuery, method=lambda a,b:torch.concat([a, b], dim=-1).max(dim=-1, keepdim=True).values)
ModulePosthocReduce.register(QuantifyQuery, GradAttnRollout, method=lambda a,b: b @ a)
ModulePosthocReduce.register(QuantifyQuery, GradAttn, method=lambda a,b: b @ a)

@SolverLibrary.register('grad_attn_rollout_quantify_query')
class GradAttnRolloutQuantifyQuery(GradAttnRollout):
    pass

ModulePosthocReduce.register(GradAttnRollout, GradAttnRolloutQuantifyQuery, method=lambda a, b:  (a, b))
ModulePosthocReduce.register(GradAttnRolloutQuantifyQuery, KeyDecomposeQuantifyQuery, method=lambda a, b:  (a[1] @ b).max(dim=-1, keepdim=True).values * a[0])
ModulePosthocReduce.register(GradAttn, GradAttnRolloutQuantifyQuery, method=lambda a, b:  (a, b))