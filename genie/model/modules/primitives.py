import math
from typing import Optional, Callable
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F  # Optimization: Import functional
from scipy.stats import truncnorm

from genie.utils.tensor_utils import (
    permute_final_dims,
    flatten_final_dims,
)


# ... [Keep initializers: _calculate_fan, trunc_normal_init_, etc. unchanged] ...
def _calculate_fan(linear_weight_shape, fan_mode):
    fan_in, fan_out = linear_weight_shape
    return fan_in if fan_mode == 'fan_in' else fan_out


def trunc_normal_init_(tensor, mean=0., std=1., a=-2., b=2.):
    # From https://github.com/aqlaboratory/openfold/blob/main/openfold/model/primitives.py
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def ipa_point_weights_init_(tensor):
    trunc_normal_init_(tensor, std=0.1, a=-0.2, b=0.2)


class Linear(nn.Linear):
    """
    A Linear layer with built-in initialization options.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, init: str = 'default'):
        self.init = init
        super().__init__(in_features, out_features, bias=bias)

    def reset_parameters(self):
        if self.init == 'default':
            super().reset_parameters()
        elif self.init == 'relu':
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)
        elif self.init == 'glorot':
            nn.init.xavier_uniform_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)
        elif self.init == 'gating':
            nn.init.zeros_(self.weight)
            if self.bias is not None:
                nn.init.ones_(self.bias)
        elif self.init == 'final':
            nn.init.zeros_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)
        elif self.init == 'normal':
            nn.init.normal_(self.weight, mean=0.0, std=0.02)
            if self.bias is not None:
                nn.init.zeros_(self.bias)
        else:
            raise ValueError(f"Unknown initialization: {self.init}")


class Attention(nn.Module):
    """ 
        Standard multi-head attention optimized with PyTorch SDPA (FlashAttention).
    """

    def __init__(self,
                 c_q: int,
                 c_k: int,
                 c_v: int,
                 c_hidden: int,
                 no_heads: int,
                 gating: bool = True,
                 ):
        super(Attention, self).__init__()

        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        self.linear_q = Linear(
            self.c_q, self.c_hidden * self.no_heads, bias=False, init="glorot"
        )
        self.linear_k = Linear(
            self.c_k, self.c_hidden * self.no_heads, bias=False, init="glorot"
        )
        self.linear_v = Linear(
            self.c_v, self.c_hidden * self.no_heads, bias=False, init="glorot"
        )
        self.linear_o = Linear(
            self.c_hidden * self.no_heads, self.c_q, init="final"
        )

        if (self.gating):
            self.linear_g = Linear(self.c_q, self.c_hidden * self.no_heads, init="gating")

        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=-1) # Optimization: Removed, SDPA handles this

    def forward(self,
                q_x: torch.Tensor,
                k_x: torch.Tensor,
                v_x: torch.Tensor,
                biases: list = None,
                ) -> torch.Tensor:
        """
            Args:
                q_x: [*, Q, C_q]
                k_x: [*, K, C_k]
                v_x: [*, V, C_v]
            Returns
                [*, Q, C_q] attention update
        """
        # [*, Q/K/V, H * C_hidden]
        q = self.linear_q(q_x)
        k = self.linear_k(k_x)
        v = self.linear_v(v_x)

        # [*, Q/K, H, C_hidden] -> [*, H, Q/K, C_hidden]
        q = q.view(*q.shape[:-1], self.no_heads, -1).transpose(-2, -3)
        k = k.view(*k.shape[:-1], self.no_heads, -1).transpose(-2, -3)
        v = v.view(*v.shape[:-1], self.no_heads, -1).transpose(-2, -3)

        # Optimization: Use PyTorch Scaled Dot Product Attention (SDPA)
        # This enables FlashAttention-2 if available on the GPU
        attn_mask = None
        if biases is not None:
            # Combine all additive biases into a single mask
            attn_mask = sum(biases)

        # SDPA handles scaling (1/sqrt(dim)) internally if we don't scale q/k manually.
        # However, OpenFold/Genie logic scales manually before adding bias. 
        # SDPA expects unscaled inputs usually, but we need to match the logic:
        # Score = (Q @ K.T) / sqrt(d) + Bias

        # [*, H, Q, C_hidden]
        o = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False
        )

        # [*, Q, H, C_hidden]
        o = o.transpose(-2, -3)

        if (self.gating):
            g = self.sigmoid(self.linear_g(q_x))
            g = g.view(*g.shape[:-1], self.no_heads, -1)
            o = o * g

        # [*, Q, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, Q, C_q]
        o = self.linear_o(o)

        return o