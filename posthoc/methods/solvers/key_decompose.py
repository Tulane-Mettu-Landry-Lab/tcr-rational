import torch
from ._base import ModulePosthoc
from .._reducer import ModulePosthocReduce
from .attn_grad_rollout import GradAttnRollout, GradAttn
from ._lib import SolverLibrary

class KeyDecompose(ModulePosthoc):
    _target_layer = 'Linear'
    _target_mechanism = 'Key'

    def __init__(self):
        self.errs = None

    def cal_err(self, x, xi):
        return torch.linalg.norm(x @ xi @ x - x, dim=[1,2])
    
    def solve(self, forward_in, forward_out, backward_in, backward_out):
        K = forward_out
        KTK = K.permute([0, 2, 1]) @ K
        KTKi = torch.linalg.pinv(KTK)
        K_KTKi = K @ KTKi
        self.errs = self.cal_err(KTK, KTKi)
        return K_KTKi

ModulePosthocReduce.register(GradAttnRollout, KeyDecompose, method=torch.bmm)
ModulePosthocReduce.register(KeyDecompose, GradAttnRollout, method=lambda x_key, x_attn: (x_key, x_attn))
ModulePosthocReduce.register(GradAttn, KeyDecompose, method=torch.bmm)
ModulePosthocReduce.register(KeyDecompose, GradAttn, method=lambda x_key, x_attn: (x_key, x_attn))

@SolverLibrary.register('key_decompose_quantify_query')
class KeyDecomposeQuantifyQuery(KeyDecompose):
    pass
def _kq_merge(a, b):
    ab = torch.bmm(a, b)
    ab_max = ab.max(dim=-1, keepdim=True).values
    ab_max = ab_max / ab_max.sum(dim=-1, keepdim=True)
    # ab_max = (ab_max - ab_max.min(dim=-1, keepdim=True).values) / ab_max.max(dim=-1, keepdim=True).values
    return ab_max
ModulePosthocReduce.register(GradAttnRollout, KeyDecomposeQuantifyQuery, method=_kq_merge)
ModulePosthocReduce.register(KeyDecomposeQuantifyQuery, GradAttnRollout, method=lambda a, b: b @ a)
ModulePosthocReduce.register(GradAttn, KeyDecomposeQuantifyQuery, method=_kq_merge)
ModulePosthocReduce.register(KeyDecomposeQuantifyQuery, GradAttn, method=lambda a, b: b @ a)