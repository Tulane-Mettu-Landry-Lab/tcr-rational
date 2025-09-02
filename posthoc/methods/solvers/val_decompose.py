import torch
from ._base import ModulePosthoc
from .._reducer import ModulePosthocReduce
from .attn_grad_rollout import GradAttnRollout, GradAttn

class ValueDecompose(ModulePosthoc):
    _target_layer = 'Linear'
    _target_mechanism = 'Value'

    def __init__(self):
        self.errs = None

    def cal_err(self, x, xi):
        return torch.linalg.norm(x @ xi @ x - x, dim=[1, 2])
    
    def solve(self, forward_in, forward_out, backward_in, backward_out):
        V = forward_out
        VVT = V @ V.permute([0, 2, 1])
        VVTi = torch.linalg.pinv(VVT)
        VT_VVTi = V.permute([0, 2, 1]) @ VVTi
        self.errs = self.cal_err(VVT, VVTi)
        return VT_VVTi

def _merge_value_decomp(x_key_attn, x_val):
    _x = torch.relu(x_key_attn[0] @ x_val)
    _x = _x / _x.sum(dim=-1, keepdim=True)
    _x = _x @ x_key_attn[1]
    return _x
ModulePosthocReduce.register(GradAttnRollout, ValueDecompose, method=_merge_value_decomp)
ModulePosthocReduce.register(ValueDecompose, GradAttnRollout, method=torch.bmm)

ModulePosthocReduce.register(GradAttn, ValueDecompose, method=_merge_value_decomp)
ModulePosthocReduce.register(ValueDecompose, GradAttn, method=torch.bmm)