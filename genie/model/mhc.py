"""
Manifold-Constrained Hyper-Connections (mHC) Module

Based on: "mHC: Manifold-Constrained Hyper-Connections" (arXiv:2512.24880)
Authors: Zhenda Xie, Yixuan Wei, et al. (DeepSeek-AI)

Key Ideas:
1. Expand residual stream width from C to n*C dimensions
2. Three learnable mappings: H_pre, H_post, H_res
3. Project H_res onto Birkhoff polytope (doubly stochastic matrices) via Sinkhorn-Knopp
4. Maintains identity mapping property for training stability

This implementation adapts mHC for protein structure prediction networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def sinkhorn_knopp(M: torch.Tensor, n_iters: int = 20, eps: float = 1e-8) -> torch.Tensor:
    """
    Sinkhorn-Knopp algorithm to project a matrix onto the Birkhoff polytope.
    
    The Birkhoff polytope is the set of doubly stochastic matrices:
    - All entries are non-negative
    - All rows sum to 1
    - All columns sum to 1
    
    Args:
        M: Input matrix of shape [..., n, n] (will be exponentiated first)
        n_iters: Number of iterations (default 20 as in paper)
        eps: Small value for numerical stability
    
    Returns:
        Doubly stochastic matrix of the same shape
    """
    # Ensure positivity via exp
    M_pos = torch.exp(M)
    
    for _ in range(n_iters):
        # Row normalization
        M_pos = M_pos / (M_pos.sum(dim=-1, keepdim=True) + eps)
        # Column normalization
        M_pos = M_pos / (M_pos.sum(dim=-2, keepdim=True) + eps)
    
    return M_pos


class ManifoldConstrainedHyperConnections(nn.Module):
    """
    Manifold-Constrained Hyper-Connections (mHC) layer.
    
    This module wraps any layer function F and adds hyper-connections
    with manifold constraints for training stability.
    
    Formula:
        x_{l+1} = H_res @ x_l + H_post^T @ F(H_pre @ x_l)
    
    Where:
        - H_res is projected onto Birkhoff polytope (doubly stochastic)
        - H_pre and H_post use sigmoid for non-negativity
    
    Args:
        c_in: Input channel dimension
        expansion_rate: Width expansion factor n (default: 4)
        n_sinkhorn_iters: Sinkhorn-Knopp iterations during training (default: 20)
        n_sinkhorn_iters_inference: Sinkhorn-Knopp iterations during inference (default: 5)
                                     Fewer iterations during inference for speed since weights are already trained
        alpha_init: Initial value for gating factors (default: 0.01)
    """
    
    def __init__(
        self,
        c_in: int,
        expansion_rate: int = 4,
        n_sinkhorn_iters: int = 20,
        n_sinkhorn_iters_inference: int = 5,  # Fewer iterations during inference for speed
        alpha_init: float = 0.01,
    ):
        super().__init__()

        self.c_in = c_in
        self.n = expansion_rate
        self.n_sinkhorn_iters = n_sinkhorn_iters
        self.n_sinkhorn_iters_inference = n_sinkhorn_iters_inference
        
        # Dimension of flattened hidden state: n * C
        c_hidden = self.n * c_in
        
        # RMSNorm for normalization (use weight in same dtype as input for better compatibility)
        self.rms_norm = nn.RMSNorm(c_hidden, elementwise_affine=False)
        
        # Linear projections for dynamic mappings
        # phi_pre, phi_post: [n*C, n]
        # phi_res: [n*C, n*n]
        self.phi_pre = nn.Linear(c_hidden, self.n, bias=False)
        self.phi_post = nn.Linear(c_hidden, self.n, bias=False)
        self.phi_res = nn.Linear(c_hidden, self.n * self.n, bias=False)
        
        # Static biases
        # Initialize b_res so that exp(b_res) is close to identity
        # Diagonal elements large (2.0), off-diagonal small (-2.0)
        # This ensures initial H_res ≈ identity after Sinkhorn-Knopp
        self.b_pre = nn.Parameter(torch.zeros(1, self.n))
        self.b_post = nn.Parameter(torch.zeros(1, self.n))
        b_res_init = torch.full((self.n, self.n), -2.0)  # Off-diagonal: exp(-2) ≈ 0.14
        b_res_init.fill_diagonal_(2.0)  # Diagonal: exp(2) ≈ 7.4
        self.b_res = nn.Parameter(b_res_init)
        
        # Gating factors (initialized small as in paper)
        self.alpha_pre = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_post = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_res = nn.Parameter(torch.tensor(alpha_init))
        
    def compute_mappings(self, x: torch.Tensor):
        """
        Compute the three learnable mappings H_pre, H_post, H_res.
        
        Args:
            x: Hidden state [B, L, n, C] or [B, L, n*C]
        
        Returns:
            H_pre: [B, L, 1, n] - aggregates n streams to single input
            H_post: [B, L, 1, n] - distributes output to n streams  
            H_res: [B, L, n, n] - residual stream mixing (doubly stochastic)
        """
        B, L = x.shape[:2]
        
        # Flatten to [B, L, n*C]
        if x.dim() == 4:
            x_flat = x.reshape(B, L, -1)
        else:
            x_flat = x
            
        # Normalize
        x_norm = self.rms_norm(x_flat)
        
        # Compute dynamic components with tanh activation (as per paper Eq. 5)
        # H_l = α_l · tanh(θ_l · x̃_l^T) + b_l
        H_pre_dyn = torch.tanh(self.phi_pre(x_norm))  # [B, L, n]
        H_post_dyn = torch.tanh(self.phi_post(x_norm))  # [B, L, n]
        H_res_dyn = torch.tanh(self.phi_res(x_norm)).view(B, L, self.n, self.n)  # [B, L, n, n]
        
        # Combine dynamic and static with gating
        H_pre_raw = self.alpha_pre * H_pre_dyn + self.b_pre  # [B, L, n]
        H_post_raw = self.alpha_post * H_post_dyn + self.b_post  # [B, L, n]
        H_res_raw = self.alpha_res * H_res_dyn + self.b_res  # [B, L, n, n]
        
        # Apply constraints:
        # H_pre, H_post: sigmoid for non-negativity
        H_pre = torch.sigmoid(H_pre_raw).unsqueeze(-2)  # [B, L, 1, n]
        H_post = 2 * torch.sigmoid(H_post_raw).unsqueeze(-2)  # [B, L, 1, n] (scale by 2 as in paper)
        
        # H_res: Sinkhorn-Knopp for doubly stochastic
        # Use fewer iterations during inference for speed (weights are already trained)
        n_iters = self.n_sinkhorn_iters if self.training else self.n_sinkhorn_iters_inference
        H_res = sinkhorn_knopp(H_res_raw, n_iters=n_iters)  # [B, L, n, n]
        
        return H_pre, H_post, H_res
    
    def expand_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expand input from [B, L, C] to [B, L, n, C] by repeating.
        
        The first stream gets the original input, others are initialized
        with the same values (will diverge during training).
        """
        # [B, L, C] -> [B, L, n, C]
        return x.unsqueeze(-2).expand(-1, -1, self.n, -1).clone()
    
    def contract_output(self, x: torch.Tensor) -> torch.Tensor:
        """
        Contract output from [B, L, n, C] to [B, L, C] by taking mean.
        """
        # [B, L, n, C] -> [B, L, C]
        return x.mean(dim=-2)
    
    def forward(
        self,
        x: torch.Tensor,
        layer_fn,
        layer_inputs: tuple = None,
        is_first_layer: bool = False,
        is_last_layer: bool = False,
    ):
        """
        Apply mHC wrapped around a layer function.
        
        Args:
            x: Current hidden state
               - If is_first_layer: [B, L, C]
               - Otherwise: [B, L, n, C]
            layer_fn: The layer function F(input, *layer_inputs) -> output
            layer_inputs: Additional inputs to layer_fn (e.g., pair repr, transforms, mask)
            is_first_layer: If True, expand input from [B, L, C] to [B, L, n, C]
            is_last_layer: If True, contract output from [B, L, n, C] to [B, L, C]
        
        Returns:
            Updated hidden state
        """
        # Expand on first layer
        if is_first_layer:
            x = self.expand_input(x)  # [B, L, C] -> [B, L, n, C]
        
        B, L, n, C = x.shape
        assert n == self.n, f"Expected n={self.n}, got n={n}"
        
        # Compute mappings
        H_pre, H_post, H_res = self.compute_mappings(x)
        
        # Aggregate streams for layer input: H_pre @ x
        # H_pre: [B, L, 1, n], x: [B, L, n, C]
        # Result: [B, L, 1, C] -> squeeze to [B, L, C]
        layer_input = torch.matmul(H_pre, x).squeeze(-2)  # [B, L, C]
        
        # Apply layer function
        if layer_inputs is not None:
            layer_output = layer_fn(layer_input, *layer_inputs)
        else:
            layer_output = layer_fn(layer_input)
        
        # layer_output: [B, L, C]
        # Distribute to streams: H_post^T @ layer_output
        # H_post: [B, L, 1, n], layer_output: [B, L, C]
        # H_post^T: [B, L, n, 1]
        layer_output_expanded = H_post.transpose(-1, -2) * layer_output.unsqueeze(-2)  # [B, L, n, C]
        
        # Residual mixing: H_res @ x
        # H_res: [B, L, n, n], x: [B, L, n, C]
        residual = torch.matmul(H_res, x)  # [B, L, n, C]
        
        # Combine
        x_out = residual + layer_output_expanded  # [B, L, n, C]
        
        # Contract on last layer
        if is_last_layer:
            x_out = self.contract_output(x_out)  # [B, L, n, C] -> [B, L, C]
        
        return x_out


class mHCResidualWrapper(nn.Module):
    """
    A simpler mHC wrapper that can be applied between existing layers.
    
    This version applies mHC connections around a single layer,
    handling the expand/contract automatically based on layer position.
    """
    
    def __init__(
        self,
        c_in: int,
        expansion_rate: int = 4,
        n_sinkhorn_iters: int = 20,
        alpha_init: float = 0.01,
    ):
        super().__init__()
        self.mhc = ManifoldConstrainedHyperConnections(
            c_in=c_in,
            expansion_rate=expansion_rate,
            n_sinkhorn_iters=n_sinkhorn_iters,
            alpha_init=alpha_init,
        )
        self.expansion_rate = expansion_rate
        self.c_in = c_in
        
    def forward(
        self,
        x: torch.Tensor,
        layer_fn,
        layer_inputs: tuple = None,
        is_first_layer: bool = False,
        is_last_layer: bool = False,
    ):
        return self.mhc(x, layer_fn, layer_inputs, is_first_layer, is_last_layer)


class mHCStructureLayerWrapper(nn.Module):
    """
    Wrapper that applies mHC to structure network layers.
    
    This wrapper handles the specific interface of StructureLayer:
    - Input: (s, p, t, mask)
    - IPA operates on single representation s
    - Returns updated (s, p, t, mask)
    """
    
    def __init__(
        self,
        c_s: int,
        expansion_rate: int = 4,
        n_sinkhorn_iters: int = 20,
        alpha_init: float = 0.01,
    ):
        super().__init__()
        self.mhc = ManifoldConstrainedHyperConnections(
            c_in=c_s,
            expansion_rate=expansion_rate,
            n_sinkhorn_iters=n_sinkhorn_iters,
            alpha_init=alpha_init,
        )
        
    def compute_mappings(self, s: torch.Tensor):
        """Expose mapping computation for the structure layer."""
        return self.mhc.compute_mappings(s)
    
    def expand_input(self, s: torch.Tensor) -> torch.Tensor:
        """Expand single representation for mHC."""
        return self.mhc.expand_input(s)
    
    def contract_output(self, s: torch.Tensor) -> torch.Tensor:
        """Contract mHC streams back to single representation."""
        return self.mhc.contract_output(s)
