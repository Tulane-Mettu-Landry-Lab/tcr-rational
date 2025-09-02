from ._base import ModulePosthoc
from .._reducer import ModulePosthocReduce
from .attn_grad_rollout import GradAttn
from ._lib import SolverLibrary
import torch
from typing import Callable

@SolverLibrary.register('attn_lrp')
class AttentionAwareLayerRelevancePropagation(GradAttn):
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
        # attn_fused_layers = torch.relu(attn_fused_layers)
        attn_fused_layers = multihead_reduce(attn_fused_layers)
        attn_fused_layers = torch.relu(attn_fused_layers)
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
ModulePosthocReduce.register(AttentionAwareLayerRelevancePropagation, AttentionAwareLayerRelevancePropagation, method=lambda a, b: b@a)