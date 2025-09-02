import torch
from typing import Callable
from ._base import ModulePosthoc
from .._reducer import ModulePosthocReduce
from ._lib import SolverLibrary

@SolverLibrary.register('grad_attn_rollout')
class GradAttnRollout(ModulePosthoc):
    _target_layer = 'Dropout'
    _target_mechanism = 'Attention'
    def __init__(self, discard_ratio=0.9, multihead_reduce='mean', residual_connect=True, norm=True):
        self.discard_ratio = discard_ratio
        self.residual_connect = residual_connect
        self.norm=norm
        if callable(multihead_reduce):
            self.multihead_reduce = multihead_reduce
        elif multihead_reduce == 'mean':
            self.multihead_reduce = lambda x: torch.mean(x, dim=1)
        elif multihead_reduce == 'max':
            self.multihead_reduce = lambda x: torch.max(x, dim=1).values
        else:
            raise TypeError('multihead_reduce is not supported')
    
    def grad_rollout(
            self,
            attention:torch.Tensor,
            gradient:torch.Tensor,
            discard_ratio:float=0.9,
            multihead_reduce:Callable=lambda x: torch.mean(x, dim=1),
            residual_connect:bool=True,
            norm:bool=True
        ) -> torch.Tensor:
        batch_size, n_heads, n_tokens, n_dim = attention.shape
        device = attention.device
        attn_fused_layers = attention * gradient
        attn_fused_layers = multihead_reduce(attn_fused_layers)
        attn_fused_layers = torch.relu(attn_fused_layers)
        flats = attn_fused_layers.view(batch_size, -1)
        k = int(flats.size(-1) * discard_ratio)
        _, indices = flats.topk(k, dim=-1, largest=False)
        _b_indices = torch.arange(batch_size, device=device).reshape(-1, 1)
        flats[_b_indices, indices] = 0
        if residual_connect:
            I = torch.eye(max(n_tokens, n_dim), device=device)
            I = I[:n_tokens, :n_dim]
            a = (attn_fused_layers + 1.0 * I) / 2
        else:
            a = attn_fused_layers
        if norm:
            a /= a.sum(dim=-1, keepdim=True)
        return a
        

    def solve(self, forward_in, forward_out, backward_in, backward_out):
        attention = forward_out
        gradient = backward_in
        return self.grad_rollout(
            attention=attention,
            gradient=gradient,
            discard_ratio=self.discard_ratio,
            multihead_reduce=self.multihead_reduce,
            residual_connect=self.residual_connect,
            norm=self.norm,
        )

ModulePosthocReduce.register(GradAttnRollout, GradAttnRollout, method=lambda a, b: b @ a)

@SolverLibrary.register('grad_attn')
class GradAttn(ModulePosthoc):
    _target_layer = 'Dropout'
    _target_mechanism = 'Attention'
    def __init__(self, discard_ratio=0.9, multihead_reduce='mean', residual_connect=True, norm=True):
        self.discard_ratio = discard_ratio
        self.residual_connect = residual_connect
        self.norm=norm
        if callable(multihead_reduce):
            self.multihead_reduce = multihead_reduce
        elif multihead_reduce == 'mean':
            self.multihead_reduce = lambda x: torch.mean(x, dim=1)
        elif multihead_reduce == 'max':
            self.multihead_reduce = lambda x: torch.max(x, dim=1).values
        else:
            raise TypeError('multihead_reduce is not supported')
    
    def grad_rollout(
            self,
            attention:torch.Tensor,
            gradient:torch.Tensor,
            discard_ratio:float=0.9,
            multihead_reduce:Callable=lambda x: torch.mean(x, dim=1),
            residual_connect:bool=True,
            norm:bool=True
        ) -> torch.Tensor:
        batch_size, n_heads, n_tokens, n_dim = attention.shape
        device = attention.device
        attn_fused_layers = attention * gradient
        attn_fused_layers = torch.relu(attn_fused_layers)
        attn_fused_layers = multihead_reduce(attn_fused_layers)
        # attn_fused_layers = torch.relu(attn_fused_layers)
        flats = attn_fused_layers.view(batch_size, -1)
        if residual_connect:
            I = torch.eye(max(n_tokens, n_dim), device=device)
            I = I[:n_tokens, :n_dim]
            a = (attn_fused_layers + 1.0 * I) / 2
        else:
            a = attn_fused_layers
        if norm:
            a /= a.sum(dim=-1, keepdim=True)
        return a
        

    def solve(self, forward_in, forward_out, backward_in, backward_out):
        attention = forward_out
        gradient = backward_in
        return self.grad_rollout(
            attention=attention,
            gradient=gradient,
            discard_ratio=self.discard_ratio,
            multihead_reduce=self.multihead_reduce,
            residual_connect=self.residual_connect,
            norm=self.norm,
        )

ModulePosthocReduce.register(GradAttn, GradAttn, method=lambda a, b: b @ a)