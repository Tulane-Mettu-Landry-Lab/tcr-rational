import torch

from .attn_grad_rollout import GradAttnRollout, GradAttn
from .quantify_query import QuantifyQuery, GradAttnRolloutQuantifyQuery
from .._reducer import ModulePosthocReduce
from ._lib import SolverLibrary

@SolverLibrary.register('grad_crossattn_rollout')
class GradCrossAttnRollout(GradAttn):
    def __init__(self, discard_ratio=0.9, multihead_reduce='mean', residual_connect=True, norm=True,
                 clip=(0, -1), compress_reduce=lambda x, dim=1:x[:, 0]):
        super().__init__(discard_ratio, multihead_reduce, residual_connect, norm)
        self.compress_reduce = compress_reduce
        self.clip=clip
        
    def _search_catbackward(self, var):
        _queue = [i[0] for i in var.grad_fn.next_functions]
        _func = _queue.pop(0)
        while _func.name() != 'CatBackward0':
            _queue += [i[0] for i in _func.next_functions if i[0] is not None]
            _func = _queue.pop(0)
        return _func
    
    def solve_cat_inshapes(self, conout_grad_fn, has_batch=True):
        _in_funcs = conout_grad_fn.next_functions
        _shapes = []
        for _in_func in _in_funcs:
            if _in_func[0] is not None:
                _shape = _in_func[0]._input_metadata[0].shape
                _shapes.append(_shape)
        if has_batch:
            _shapes = [_shape[1] for _shape in _shapes]
        else:
            _shapes = [_shape[0] for _shape in _shapes]
        return _shapes
    
    def solve(self, forward_in, forward_out, backward_in, backward_out):
        _r = super().solve(forward_in, forward_in[0], backward_in, backward_out)
        _r = self.compress_reduce(_r, dim=1)
        
        if isinstance(self.clip, (tuple, list)):
            return _r[:, self.clip[0]:self.clip[1]][:, None, :]
        else:
            _grad_fn = self._search_catbackward(forward_in[0])
            _shapes = self.solve_cat_inshapes(_grad_fn)
            _out = torch.split(_r, _shapes, dim=1)
            _out = _out[self.clip]
            return _out[:, None, :]

@SolverLibrary.register('quantify_queryin')
class QuantifyQueryIn(QuantifyQuery):
    def __init__(self, reduce_method='sum', norm=True, clip=(0, -1)):
        super().__init__(reduce_method, norm)
        self.clip=clip
        
    def solve_cat_inshapes(self, conout, has_batch=True):
        _in_funcs = conout.grad_fn.next_functions[0][0].next_functions
        _shapes = []
        for _in_func in _in_funcs:
            if _in_func[0] is not None:
                _shape = _in_func[0]._input_metadata[0].shape
                _shapes.append(_shape)
        if has_batch:
            _shapes = [_shape[1] for _shape in _shapes]
        else:
            _shapes = [_shape[0] for _shape in _shapes]
        return _shapes
    
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
        
        if isinstance(self.clip, (tuple, list)):
            return dhh[:, self.clip[0]:self.clip[1]][:, None, :]
        else:
            _shapes = self.solve_cat_inshapes(forward_in[0])
            _out = torch.split(dhh, _shapes, dim=1)[self.clip]
            return _out[:, None, :]

def _max_pos_merge(_x_next, _x_prev):
    _r = torch.concat([_x_next, _x_prev], dim=1)
    # _r = _r.sum(dim=1, keepdim=True)
    _r = _r.max(dim=1, keepdim=True).values
    return _r

def _max_pos_merge_t(_x_next, _x_prev):
    _r = torch.concat([_x_next.permute([0, 2, 1]), _x_prev], dim=1)
    # _r = _r.sum(dim=1, keepdim=True)
    _r = _r.max(dim=1, keepdim=True).values
    return _r


ModulePosthocReduce.register(GradAttn, GradCrossAttnRollout, method=_max_pos_merge_t)
ModulePosthocReduce.register(GradCrossAttnRollout, GradCrossAttnRollout, method=_max_pos_merge)
ModulePosthocReduce.register(QuantifyQueryIn, GradCrossAttnRollout, method=_max_pos_merge)
ModulePosthocReduce.register(GradCrossAttnRollout, QuantifyQueryIn, method=_max_pos_merge)
ModulePosthocReduce.register(GradCrossAttnRollout, GradAttnRollout, method=lambda c, a: a @ (c / c.sum(dim=-1, keepdim=True)).permute([0, 2, 1]))
ModulePosthocReduce.register(GradCrossAttnRollout, GradAttn, method=lambda c, a: a @ (c / c.sum(dim=-1, keepdim=True)).permute([0, 2, 1]))
# ModulePosthocReduce.register(GradAttn, GradCrossAttnRollout, method=lambda c, a: a @ (c / c.sum(dim=-1, keepdim=True)).permute([0, 2, 1]))
ModulePosthocReduce.register(QuantifyQueryIn, GradAttnRollout, method=lambda c, a: a @ (c / c.sum(dim=-1, keepdim=True)).permute([0, 2, 1]))
ModulePosthocReduce.register(QuantifyQueryIn, GradAttn, method=lambda c, a: a @ (c / c.sum(dim=-1, keepdim=True)).permute([0, 2, 1]))
ModulePosthocReduce.register(QuantifyQueryIn, GradAttnRolloutQuantifyQuery, method=lambda a, b:  (a, b))