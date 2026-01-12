"""
mHC (Manifold-Constrained Hyper-Connections) Structure Network

This module implements the Structure Network with mHC connections
for improved training stability and performance.

Based on: "mHC: Manifold-Constrained Hyper-Connections" (arXiv:2512.24880)
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from genie.model.modules.invariant_point_attention import InvariantPointAttention
from genie.model.modules.structure_transition import StructureTransition
from genie.model.modules.backbone_update import BackboneUpdate
from genie.model.mhc import ManifoldConstrainedHyperConnections


class mHCStructureLayer(nn.Module):
    """
    Structure Layer with mHC (Manifold-Constrained Hyper-Connections).
    
    This layer wraps the standard IPA with mHC for improved training stability
    at large scales. The mHC framework:
    1. Expands the residual stream width by factor n
    2. Uses learnable mappings (H_pre, H_post, H_res) for information flow
    3. Projects H_res onto Birkhoff polytope for stability
    
    The layer follows the formula:
        s_{l+1} = H_res @ s_l + H_post^T @ IPA(H_pre @ s_l, p, t, mask)
    """
    
    def __init__(
        self,
        c_s: int,
        c_p: int,
        c_hidden_ipa: int,
        n_head: int,
        n_qk_point: int,
        n_v_point: int,
        ipa_dropout: float,
        n_structure_transition_layer: int,
        structure_transition_dropout: float,
        mhc_expansion_rate: int = 4,
        mhc_sinkhorn_iters: int = 20,
        mhc_alpha_init: float = 0.01,
        use_grad_checkpoint: bool = False,
        is_first_layer: bool = False,
        is_last_layer: bool = False,
    ):
        super().__init__()
        
        self.c_s = c_s
        self.is_first_layer = is_first_layer
        self.is_last_layer = is_last_layer
        self.use_grad_checkpoint = use_grad_checkpoint
        self.mhc_expansion_rate = mhc_expansion_rate
        
        # mHC module for residual stream management
        self.mhc = ManifoldConstrainedHyperConnections(
            c_in=c_s,
            expansion_rate=mhc_expansion_rate,
            n_sinkhorn_iters=mhc_sinkhorn_iters,
            alpha_init=mhc_alpha_init,
        )
        
        # Standard IPA (operates on contracted single representation)
        self.ipa = InvariantPointAttention(
            c_s,
            c_p,
            c_hidden_ipa,
            n_head,
            n_qk_point,
            n_v_point,
            use_checkpointing=use_grad_checkpoint,
        )
        
        self.ipa_dropout = nn.Dropout(ipa_dropout)
        self.ipa_layer_norm = nn.LayerNorm(c_s)
        
        # Structure transition (operates on contracted representation)
        self.transition = StructureTransition(
            c_s,
            n_structure_transition_layer,
            structure_transition_dropout
        )
        
        # Backbone update
        self.bb_update = BackboneUpdate(c_s)
    
    def _run_ipa(self, s_contracted, p, t, mask):
        """Run IPA on contracted single representation."""
        # IPA forward
        s_out = s_contracted + self.ipa(s_contracted, p, t, mask)
        s_out = self.ipa_dropout(s_out)
        s_out = self.ipa_layer_norm(s_out)
        s_out = self.transition(s_out)
        return s_out
    
    def forward(self, inputs):
        """
        Forward pass with mHC connections.
        
        Args:
            inputs: Tuple of (s, p, t, mask)
                - s: Single representation
                    - If is_first_layer: [B, L, C]
                    - Otherwise: [B, L, n, C] (mHC expanded)
                - p: Pair representation [B, L, L, C_p]
                - t: Rigid transform
                - mask: Sequence mask [B, L]
        
        Returns:
            Tuple of (s_out, p, t_out, mask)
        """
        s, p, t, mask = inputs
        
        # Expand on first layer
        if self.is_first_layer:
            s = self.mhc.expand_input(s)  # [B, L, C] -> [B, L, n, C]
        
        # Compute mHC mappings
        H_pre, H_post, H_res = self.mhc.compute_mappings(s)
        
        # Contract for IPA: H_pre @ s -> [B, L, C]
        # H_pre: [B, L, 1, n], s: [B, L, n, C]
        s_contracted = torch.matmul(H_pre, s).squeeze(-2)  # [B, L, C]
        
        # Run IPA on contracted representation
        if self.training and s_contracted.requires_grad and self.use_grad_checkpoint:
            s_ipa_out = checkpoint(
                self._run_ipa, s_contracted, p, t, mask,
                use_reentrant=False
            )
        else:
            s_ipa_out = self._run_ipa(s_contracted, p, t, mask)
        
        # Expand IPA output: H_post^T @ s_ipa_out -> [B, L, n, C]
        # H_post: [B, L, 1, n], s_ipa_out: [B, L, C]
        s_expanded = H_post.transpose(-1, -2) * s_ipa_out.unsqueeze(-2)  # [B, L, n, C]
        
        # Residual mixing: H_res @ s -> [B, L, n, C]
        s_residual = torch.matmul(H_res, s)  # [B, L, n, C]
        
        # Combine
        s_out = s_residual + s_expanded  # [B, L, n, C]
        
        # Contract on last layer
        if self.is_last_layer:
            s_out = self.mhc.contract_output(s_out)  # [B, L, C]
        
        # Update backbone (use contracted representation for coordinate update)
        s_for_bb = s_out if self.is_last_layer else self.mhc.contract_output(s_out)
        t = t.compose(self.bb_update(s_for_bb))
        
        return (s_out, p, t, mask)


class mHCStructureNet(nn.Module):
    """
    Structure Network with mHC (Manifold-Constrained Hyper-Connections).
    
    This network stacks multiple mHCStructureLayer modules to form the
    complete structure prediction module with mHC for training stability.
    
    Key Features:
    - Expanded residual stream (n times wider internally)
    - Doubly stochastic residual mixing via Sinkhorn-Knopp
    - Preserves identity mapping property for stable gradient flow
    - Compatible with standard IPA (no FlashIPA dependency)
    """
    
    def __init__(
        self,
        c_s: int,
        c_p: int,
        n_structure_layer: int,
        n_structure_block: int,
        c_hidden_ipa: int,
        n_head_ipa: int,
        n_qk_point: int,
        n_v_point: int,
        ipa_dropout: float,
        n_structure_transition_layer: int,
        structure_transition_dropout: float,
        mhc_expansion_rate: int = 4,
        mhc_sinkhorn_iters: int = 20,
        mhc_alpha_init: float = 0.01,
        use_grad_checkpoint: bool = False,
    ):
        super().__init__()
        
        self.n_structure_block = n_structure_block
        self.n_structure_layer = n_structure_layer
        self.use_grad_checkpoint = use_grad_checkpoint
        self.mhc_expansion_rate = mhc_expansion_rate
        
        print(f"============================================================")
        print(f"mHCStructureNet: Manifold-Constrained Hyper-Connections")
        print(f"============================================================")
        print(f"  n_structure_layer: {n_structure_layer}")
        print(f"  n_structure_block: {n_structure_block}")
        print(f"  mhc_expansion_rate: {mhc_expansion_rate} (residual stream {mhc_expansion_rate}x wider)")
        print(f"  mhc_sinkhorn_iters: {mhc_sinkhorn_iters}")
        print(f"  mhc_alpha_init: {mhc_alpha_init}")
        print(f"============================================================")
        
        # Build layers with proper first/last layer flags
        layers = []
        for i in range(n_structure_layer):
            is_first = (i == 0)
            is_last = (i == n_structure_layer - 1)
            
            layer = mHCStructureLayer(
                c_s=c_s,
                c_p=c_p,
                c_hidden_ipa=c_hidden_ipa,
                n_head=n_head_ipa,
                n_qk_point=n_qk_point,
                n_v_point=n_v_point,
                ipa_dropout=ipa_dropout,
                n_structure_transition_layer=n_structure_transition_layer,
                structure_transition_dropout=structure_transition_dropout,
                mhc_expansion_rate=mhc_expansion_rate,
                mhc_sinkhorn_iters=mhc_sinkhorn_iters,
                mhc_alpha_init=mhc_alpha_init,
                use_grad_checkpoint=use_grad_checkpoint,
                is_first_layer=is_first,
                is_last_layer=is_last,
            )
            layers.append(layer)
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, s, p, t, mask):
        """
        Forward pass through mHC structure network.
        
        Args:
            s: Single representation [B, L, C]
            p: Pair representation [B, L, L, C_p]
            t: Rigid transform
            mask: Sequence mask [B, L]
        
        Returns:
            t: Updated rigid transform
        """
        for block_idx in range(self.n_structure_block):
            # Run through all layers
            for layer in self.layers:
                if self.training and self.use_grad_checkpoint:
                    s, p, t, mask = checkpoint(
                        layer, (s, p, t, mask),
                        use_reentrant=False
                    )
                else:
                    s, p, t, mask = layer((s, p, t, mask))
        
        return t
