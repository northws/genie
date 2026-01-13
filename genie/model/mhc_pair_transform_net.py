import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

# Optimization: Try to import NVIDIA's cuEquivariance
try:
    import cuequivariance_torch
    HAS_CUEQUIVARIANCE = True
except ImportError:
    HAS_CUEQUIVARIANCE = False

from genie.model.modules.pair_transition import PairTransition
from genie.model.modules.triangular_attention import (
    TriangleAttentionStartingNode,
    TriangleAttentionEndingNode,
)
from genie.model.modules.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
)
from genie.model.modules.dropout import (
    DropoutRowwise,
    DropoutColumnwise
)
from genie.model.mhc import sinkhorn_knopp


class PairManifoldConstrainedHyperConnections(nn.Module):
    """
    mHC Adapter for Pair Features ([B, L, L, C]).
    
    Adapted from genie.model.mhc.ManifoldConstrainedHyperConnections to handle
    5D Tensors [B, L, L, n, C] correctly.
    """
    
    def __init__(
        self,
        c_in: int,
        expansion_rate: int = 4,
        n_sinkhorn_iters: int = 20,
        n_sinkhorn_iters_inference: int = 5,  # Fewer iterations during inference
        alpha_init: float = 0.01,
    ):
        super().__init__()

        self.c_in = c_in
        self.n = expansion_rate
        self.n_sinkhorn_iters = n_sinkhorn_iters
        self.n_sinkhorn_iters_inference = n_sinkhorn_iters_inference
        
        # Dimension of flattened hidden state: n * C
        c_hidden = self.n * c_in
        
        # RMSNorm for normalization
        self.rms_norm = nn.RMSNorm(c_hidden, elementwise_affine=False)
        
        # Linear projections for dynamic mappings
        self.phi_pre = nn.Linear(c_hidden, self.n, bias=False)
        self.phi_post = nn.Linear(c_hidden, self.n, bias=False)
        self.phi_res = nn.Linear(c_hidden, self.n * self.n, bias=False)
        
        # Static biases
        self.b_pre = nn.Parameter(torch.zeros(1, self.n))
        self.b_post = nn.Parameter(torch.zeros(1, self.n))
        b_res_init = torch.full((self.n, self.n), -2.0)
        b_res_init.fill_diagonal_(2.0)
        self.b_res = nn.Parameter(b_res_init)
        
        # Gating factors
        self.alpha_pre = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_post = nn.Parameter(torch.tensor(alpha_init))
        self.alpha_res = nn.Parameter(torch.tensor(alpha_init))
        
    def compute_mappings(self, x: torch.Tensor):
        """
        Compute H_pre, H_post, H_res for input x.
        
        Args:
            x: Pair hidden state [B, L, L, n, C]
        """
        # Flatten input to [B, L, L, n*C] for the controller MLPs
        # x shape: [B, L, L, n, C] -> [B, L, L, n*C]
        x_flat = x.flatten(start_dim=-2) 
            
        # Normalize
        x_norm = self.rms_norm(x_flat)
        
        # Compute dynamic components
        H_pre_dyn = torch.tanh(self.phi_pre(x_norm))         # [B, L, L, n]
        H_post_dyn = torch.tanh(self.phi_post(x_norm))       # [B, L, L, n]
        
        # [B, L, L, n*n] -> [B, L, L, n, n]
        H_res_dyn = torch.tanh(self.phi_res(x_norm)).unflatten(-1, (self.n, self.n))
        
        # Combine dynamic and static
        H_pre_raw = self.alpha_pre * H_pre_dyn + self.b_pre
        H_post_raw = self.alpha_post * H_post_dyn + self.b_post
        H_res_raw = self.alpha_res * H_res_dyn + self.b_res
        
        # Apply constraints
        H_pre = torch.sigmoid(H_pre_raw).unsqueeze(-2)       # [B, L, L, 1, n]
        H_post = 2 * torch.sigmoid(H_post_raw).unsqueeze(-2) # [B, L, L, 1, n]
        
        # Sinkhorn-Knopp for H_res
        # Use fewer iterations during inference for speed
        n_iters = self.n_sinkhorn_iters if self.training else self.n_sinkhorn_iters_inference
        H_res = sinkhorn_knopp(H_res_raw, n_iters=n_iters) # [B, L, L, n, n]
        
        return H_pre, H_post, H_res
    
    def expand_input(self, x: torch.Tensor) -> torch.Tensor:
        """[B, L, L, C] -> [B, L, L, n, C]"""
        return x.unsqueeze(-2).expand(-1, -1, -1, self.n, -1).clone()
    
    def contract_output(self, x: torch.Tensor) -> torch.Tensor:
        """[B, L, L, n, C] -> [B, L, L, C]"""
        return x.mean(dim=-2)


class mHCPairTransformLayer(nn.Module):
    """
    Pair Transform Layer wrapped with mHC.
    """
    
    def __init__(self,
                 c_p,
                 include_mul_update,
                 include_tri_att,
                 c_hidden_mul,
                 c_hidden_tri_att,
                 n_head_tri,
                 tri_dropout,
                 pair_transition_n,
                 mhc_expansion_rate=4,
                 mhc_sinkhorn_iters=20,
                 use_optimized_kernel=True,
                 use_grad_checkpoint=False,
                 is_first_layer=False,
                 is_last_layer=False
                 ):
        super(mHCPairTransformLayer, self).__init__()

        self.is_first_layer = is_first_layer
        self.is_last_layer = is_last_layer
        self.use_grad_checkpoint = use_grad_checkpoint
        
        self.mhc = PairManifoldConstrainedHyperConnections(
            c_in=c_p,
            expansion_rate=mhc_expansion_rate,
            n_sinkhorn_iters=mhc_sinkhorn_iters
        )

        # --- Standard Pair Transform Components (Optimized Mode) ---
        self.use_optimized_kernel = use_optimized_kernel and HAS_CUEQUIVARIANCE
        self.include_mul_update = include_mul_update

        if self.use_optimized_kernel and include_mul_update:
            pass # functional API used
        else:
            self.tri_mul_out = TriangleMultiplicationOutgoing(
                c_p, c_hidden_mul, use_grad_checkpoint=use_grad_checkpoint
            ) if include_mul_update else None

            self.tri_mul_in = TriangleMultiplicationIncoming(
                c_p, c_hidden_mul, use_grad_checkpoint=use_grad_checkpoint
            ) if include_mul_update else None

        self.tri_att_start = TriangleAttentionStartingNode(
            c_p, c_hidden_tri_att, n_head_tri
        ) if include_tri_att else None

        self.tri_att_end = TriangleAttentionEndingNode(
            c_p, c_hidden_tri_att, n_head_tri
        ) if include_tri_att else None

        self.pair_transition = PairTransition(c_p, pair_transition_n)

        self.dropout_row_layer = DropoutRowwise(tri_dropout)
        self.dropout_col_layer = DropoutColumnwise(tri_dropout)

    def _run_standard_transform(self, p, p_mask):
        """Runs the standard sequence of pair updates on contracted tensor p([B, L, L, C])."""
        
        # 1. Triangular Multiplicative Update
        if self.use_optimized_kernel and self.include_mul_update:
             p = cuequivariance_torch.triangle_multiplicative_update(
                p, mask=p_mask, add_outgoing=True, add_incoming=True
            )
        else:
            if getattr(self, 'tri_mul_out', None) is not None:
                p = p + self.dropout_row_layer(self.tri_mul_out(p, p_mask))
                p = p + self.dropout_row_layer(self.tri_mul_in(p, p_mask))

        # 2. Triangular Attention
        if getattr(self, 'tri_att_start', None) is not None:
            p = p + self.dropout_row_layer(self.tri_att_start(p, p_mask))
            p = p + self.dropout_col_layer(self.tri_att_end(p, p_mask))

        # 3. Transition
        p = p + self.pair_transition(p, p_mask)
        
        return p

    def forward(self, inputs):
        """
        Args:
            inputs: tuple (p, p_mask) where:
                    p: [B, L, L, C] (if first) or [B, L, L, n, C] (if internal)
        Returns:
            tuple (p_out, p_mask)
        """
        p, p_mask = inputs
        
        # Expand
        if self.is_first_layer:
            p = self.mhc.expand_input(p)
            
        # Compute Mappings
        H_pre, H_post, H_res = self.mhc.compute_mappings(p)
        
        # Contract Input: H_pre @ p
        # H_pre: [B, L, L, 1, n]
        # p:     [B, L, L, n, C]
        # Matmul broadcasts over B, L, L. (1,n) @ (n,C) -> (1,C)
        p_contracted = torch.matmul(H_pre, p).squeeze(-2) # [B, L, L, C]
        
        # Run Standard Transform
        if self.training and self.use_grad_checkpoint and p_contracted.requires_grad:
             p_updated = checkpoint(self._run_standard_transform, p_contracted, p_mask, use_reentrant=False)
        else:
             p_updated = self._run_standard_transform(p_contracted, p_mask)
             
        # Expand Output: H_post^T @ p_updated
        # H_post: [B, L, L, 1, n]
        # p_updated: [B, L, L, C] -> unsqueeze -> [B, L, L, 1, C] (Wait, need matchingdims)
        
        # H_post.transpose: [B, L, L, n, 1]
        # p_updated.unsqueeze(-2): [B, L, L, 1, C]
        # We want [B, L, L, n, C].
        # (n,1) * (1,C) -> (n,C). This is outer product logic.
        p_expanded = H_post.transpose(-1, -2) * p_updated.unsqueeze(-2) 
        
        # Residual: H_res @ p
        # H_res: [B, L, L, n, n]
        # p:     [B, L, L, n, C]
        # (n,n) @ (n,C) -> (n,C)
        p_residual = torch.matmul(H_res, p)
        
        # Combine
        p_out = p_residual + p_expanded
        
        # Contract Final
        if self.is_last_layer:
            p_out = self.mhc.contract_output(p_out)
            
        return (p_out, p_mask)


class mHCPairTransformNet(nn.Module):
    """
    PairTransformNet with mHC (Manifold-Constrained Hyper-Connections).

    MEMORY WARNING:
    ===============
    Pair features have dimension L² × C. With mHC expansion rate n, this becomes L² × n × C.
    For long sequences (e.g., L=1024), this can quickly exhaust GPU memory.

    Memory usage examples:
    - L=256, C=128, n=4: ~134 MB per batch
    - L=512, C=128, n=4: ~536 MB per batch
    - L=1024, C=128, n=4: ~2.1 GB per batch

    RECOMMENDATIONS:
    1. Use smaller expansion rates (n=2) for pair features
    2. Apply mHC only to critical layers, not all layers
    3. For very long sequences (L>512), consider disabling mHC on pair features
    4. Use gradient checkpointing to trade computation for memory
    """

    def __init__(self,
                 c_p,
                 n_pair_transform_layer,
                 include_mul_update,
                 include_tri_att,
                 c_hidden_mul,
                 c_hidden_tri_att,
                 n_head_tri,
                 tri_dropout,
                 pair_transition_n,
                 mhc_expansion_rate=4,
                 mhc_sinkhorn_iters=20,
                 use_optimized_kernel=True,
                 use_grad_checkpoint=False
                 ):
        super(mHCPairTransformNet, self).__init__()

        # Memory warning for large expansion rates on pair features
        if mhc_expansion_rate > 2:
            print(f"========================================================")
            print(f"WARNING: mHCPairTransformNet Memory Usage")
            print(f"========================================================")
            print(f"  Expansion rate: {mhc_expansion_rate}")
            print(f"  Pair features have dimension L² × C")
            print(f"  With mHC, this becomes L² × {mhc_expansion_rate} × C")
            print(f"  ")
            print(f"  For L=512, this uses ~{0.536 * mhc_expansion_rate / 4:.1f}GB per batch")
            print(f"  For L=1024, this uses ~{2.1 * mhc_expansion_rate / 4:.1f}GB per batch")
            print(f"  ")
            print(f"  RECOMMENDATION: Consider using expansion_rate=2 for")
            print(f"                  pair features to reduce memory usage")
            print(f"========================================================")

        self.layers = nn.ModuleList()
        for i in range(n_pair_transform_layer):
            self.layers.append(mHCPairTransformLayer(
                c_p,
                include_mul_update,
                include_tri_att,
                c_hidden_mul,
                c_hidden_tri_att,
                n_head_tri,
                tri_dropout,
                pair_transition_n,
                mhc_expansion_rate=mhc_expansion_rate,
                mhc_sinkhorn_iters=mhc_sinkhorn_iters,
                use_optimized_kernel=use_optimized_kernel,
                use_grad_checkpoint=use_grad_checkpoint,
                is_first_layer=(i == 0),
                is_last_layer=(i == n_pair_transform_layer - 1)
            ))

    def forward(self, p, p_mask):
        for layer in self.layers:
            p, p_mask = layer((p, p_mask))
        return p
