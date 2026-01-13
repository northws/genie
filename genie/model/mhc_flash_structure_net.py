"""
mHC with Flash-IPA Structure Network

This module combines mHC (Manifold-Constrained Hyper-Connections) with Flash-IPA
for both training stability AND memory efficiency.

Features:
- mHC expanded residual stream for stability (from mHC paper)
- Flash-IPA for memory-efficient attention (from Flash-IPA paper)
- Best of both worlds: stable training on long sequences

Based on:
- mHC: "Manifold-Constrained Hyper-Connections" (arXiv:2512.24880)
- Flash-IPA: "Flash Invariant Point Attention" (Liu et al., 2025)
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from genie.flash_ipa.ipa import InvariantPointAttention as FlashIPA, IPAConfig
from genie.model.modules.structure_transition import StructureTransition
from genie.model.modules.backbone_update import BackboneUpdate
from genie.model.mhc import ManifoldConstrainedHyperConnections


class mHCFlashStructureLayer(nn.Module):
    """
    Structure Layer combining mHC with Flash-IPA.
    
    This layer provides:
    1. mHC expanded residual stream for training stability
    2. Flash-IPA for memory-efficient attention on long sequences
    3. Compatible with sequences up to 1024+ residues
    
    The layer follows:
        s_{l+1} = H_res @ s_l + H_post^T @ FlashIPA(H_pre @ s_l, p, z_factors, t, mask)
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
        max_n_res: int,
        z_factor_rank: int = 2,
        k_neighbors: int = 10,
        mhc_expansion_rate: int = 4,
        mhc_sinkhorn_iters: int = 20,
        mhc_alpha_init: float = 0.01,
        use_grad_checkpoint: bool = False,
        use_flash_attn_3: bool = True,
        is_first_layer: bool = False,
        is_last_layer: bool = False,
    ):
        super().__init__()
        
        self.c_s = c_s
        self.c_p = c_p
        self.max_n_res = max_n_res
        self.z_factor_rank = z_factor_rank
        self.k_neighbors = k_neighbors
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
        
        # Calculate headdim_eff to determine optimal attn_dtype
        headdim_eff = max(
            c_hidden_ipa + 5 * n_qk_point + (z_factor_rank * n_head),
            c_hidden_ipa + 3 * n_v_point + (z_factor_rank * c_p // 4),
        )
        attn_dtype = "fp16" if headdim_eff > 256 else "bf16"
        
        # Flash IPA configuration
        ipa_conf = IPAConfig(
            use_flash_attn=True,
            use_flash_attn_3=use_flash_attn_3,
            attn_dtype=attn_dtype,
            c_s=c_s,
            c_z=c_p,
            c_hidden=c_hidden_ipa,
            no_heads=n_head,
            z_factor_rank=z_factor_rank,
            no_qk_points=n_qk_point,
            no_v_points=n_v_point,
        )
        self.ipa = FlashIPA(ipa_conf)
        
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
        
    def _run_flash_ipa(self, s_contracted, z_factor_1, z_factor_2, t, mask):
        """Run Flash-IPA on contracted single representation."""
        # Flash-IPA forward (no pair features, uses z_factors instead)
        s_out = s_contracted + self.ipa(
            s_contracted,
            z=None,  # Not used in flash mode
            z_factor_1=z_factor_1,
            z_factor_2=z_factor_2,
            r=t,
            mask=mask
        )
        s_out = self.ipa_dropout(s_out)
        s_out = self.ipa_layer_norm(s_out)
        s_out = self.transition(s_out)
        return s_out
    
    def forward(self, inputs):
        """
        Forward pass with mHC + Flash-IPA.
        
        Args:
            inputs: Tuple of (s, z_factor_1, z_factor_2, t, mask)
                - s: Single representation
                    - If is_first_layer: [B, L, C]
                    - Otherwise: [B, L, n, C] (mHC expanded)
                - z_factor_1: [B, L, rank, C_z] (from factorizer)
                - z_factor_2: [B, L, n_head, rank, C_z//4] (from factorizer)
                - t: Rigid transform
                - mask: Sequence mask [B, L]
        
        Returns:
            Tuple of (s_out, z_factor_1, z_factor_2, t_out, mask)
        """
        s, z_factor_1, z_factor_2, t, mask = inputs
        
        # Expand on first layer
        if self.is_first_layer:
            s = self.mhc.expand_input(s)  # [B, L, C] -> [B, L, n, C]
        
        # Compute mHC mappings
        H_pre, H_post, H_res = self.mhc.compute_mappings(s)
        
        # Contract for IPA: H_pre @ s -> [B, L, C]
        s_contracted = torch.matmul(H_pre, s).squeeze(-2)  # [B, L, C]
        
        # Run Flash-IPA on contracted representation
        if self.training and s_contracted.requires_grad and self.use_grad_checkpoint:
            s_ipa_out = checkpoint(
                self._run_flash_ipa, s_contracted, z_factor_1, z_factor_2, t, mask,
                use_reentrant=False
            )
        else:
            s_ipa_out = self._run_flash_ipa(s_contracted, z_factor_1, z_factor_2, t, mask)
        
        # Expand IPA output: H_post^T @ s_ipa_out -> [B, L, n, C]
        s_expanded = H_post.transpose(-1, -2) * s_ipa_out.unsqueeze(-2)  # [B, L, n, C]
        
        # Residual mixing: H_res @ s -> [B, L, n, C]
        s_residual = torch.matmul(H_res, s)  # [B, L, n, C]
        
        # Combine
        s_out = s_residual + s_expanded  # [B, L, n, C]
        
        # Contract on last layer
        if self.is_last_layer:
            # Average over expanded dimension: [B, L, n, C] -> [B, L, C]
            s_out = s_out.mean(dim=-2)
        
        # Update backbone (use genie T.compose method, not Flash-IPA's compose_q_update_vec)
        t = t.compose(self.bb_update(s_ipa_out))
        
        return s_out, z_factor_1, z_factor_2, t, mask


class mHCFlashStructureNet(nn.Module):
    """
    Structure Network combining mHC with Flash-IPA.
    
    Stacks multiple mHCFlashStructureLayer blocks with skip connections.
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
        n_structure_layer: int,
        n_structure_block: int,
        max_n_res: int,
        z_factor_rank: int = 2,
        k_neighbors: int = 10,
        mhc_expansion_rate: int = 4,
        mhc_sinkhorn_iters: int = 20,
        mhc_alpha_init: float = 0.01,
        use_grad_checkpoint: bool = False,
        use_flash_attn_3: bool = True,
    ):
        super().__init__()
        
        print(f"============================================================")
        print(f"mHCFlashStructureNet: Combining mHC with Flash-IPA")
        print(f"============================================================")
        print(f"  max_n_res: {max_n_res}")
        print(f"  z_factor_rank: {z_factor_rank}")
        print(f"  k_neighbors: {k_neighbors}")
        print(f"  mhc_expansion_rate: {mhc_expansion_rate}")
        print(f"  mhc_sinkhorn_iters: {mhc_sinkhorn_iters}")
        print(f"  mhc_alpha_init: {mhc_alpha_init}")
        print(f"  n_structure_layer: {n_structure_layer}")
        print(f"  n_structure_block: {n_structure_block}")
        print(f"  use_flash_attn_3: {use_flash_attn_3}")
        print(f"============================================================")
        
        self.n_structure_layer = n_structure_layer
        self.n_structure_block = n_structure_block
        
        # Build structure network
        self.net = nn.ModuleDict()
        for b in range(n_structure_block):
            for i in range(n_structure_layer):
                layer_idx = b * n_structure_layer + i
                is_first = (layer_idx == 0)
                is_last = (layer_idx == n_structure_block * n_structure_layer - 1)
                
                self.net[f'layer_{b}_{i}'] = mHCFlashStructureLayer(
                    c_s=c_s,
                    c_p=c_p,
                    c_hidden_ipa=c_hidden_ipa,
                    n_head=n_head,
                    n_qk_point=n_qk_point,
                    n_v_point=n_v_point,
                    ipa_dropout=ipa_dropout,
                    n_structure_transition_layer=n_structure_transition_layer,
                    structure_transition_dropout=structure_transition_dropout,
                    max_n_res=max_n_res,
                    z_factor_rank=z_factor_rank,
                    k_neighbors=k_neighbors,
                    mhc_expansion_rate=mhc_expansion_rate,
                    mhc_sinkhorn_iters=mhc_sinkhorn_iters,
                    mhc_alpha_init=mhc_alpha_init,
                    use_grad_checkpoint=use_grad_checkpoint,
                    use_flash_attn_3=use_flash_attn_3,
                    is_first_layer=is_first,
                    is_last_layer=is_last,
                )
    
    def forward(self, s, z_factor_1, z_factor_2, t, mask):
        """
        Forward pass through all structure layers.
        
        Args:
            s: Single representation [B, L, C]
            z_factor_1: Factorized pair features [B, L, rank, C_z]
            z_factor_2: Factorized pair features [B, L, n_head, rank, C_z//4]
            t: Rigid transform
            mask: Sequence mask [B, L]
        
        Returns:
            Tuple of (s_out, t_out)
        """
        inputs = (s, z_factor_1, z_factor_2, t, mask)
        
        for b in range(self.n_structure_block):
            # Store input for skip connection
            s_skip, _, _, t_skip, _ = inputs
            
            # Forward through structure layers
            for i in range(self.n_structure_layer):
                inputs = self.net[f'layer_{b}_{i}'](inputs)
            
            # Extract outputs
            s_out, z_factor_1, z_factor_2, t_out, mask = inputs
            
            # Skip connection (only add after first layer)
            if b > 0 or True:  # Always apply skip
                # Check the state of both s_skip and s_out
                is_skip_expanded = (len(s_skip.shape) == 4)
                is_out_expanded = (len(s_out.shape) == 4)

                if is_out_expanded:
                    if is_skip_expanded:
                        # Both are [B, L, n, C], direct addition
                        s_out = s_out + s_skip
                    else:
                        # s_skip is [B, L, C], needs broadcasting
                        s_out = s_out + s_skip.unsqueeze(-2)
                else:
                    # s_out is contracted (only happens on last layer of last block)
                    if is_skip_expanded:
                        # Contract s_skip first (should rarely happen)
                        s_out = s_out + s_skip.mean(dim=-2)
                    else:
                        # Both contracted [B, L, C]
                        s_out = s_out + s_skip
            
            # Update inputs for next block
            inputs = (s_out, z_factor_1, z_factor_2, t_out, mask)
        
        # Final output (s should be contracted [B, L, C] from last layer)
        s_final, _, _, t_final, _ = inputs
        return s_final, t_final
