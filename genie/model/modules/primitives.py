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
# ... [Keep class Linear unchanged] ...

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