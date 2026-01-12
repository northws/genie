"""
mHC with Flash-IPA Denoiser

This module combines mHC (Manifold-Constrained Hyper-Connections) with Flash-IPA
for a denoiser that provides both training stability AND memory efficiency.

Features:
- mHC expanded residual stream for training stability
- Flash-IPA for memory-efficient long sequences
- Best of both worlds: stable training on 512+ residue proteins

Based on:
- mHC: "Manifold-Constrained Hyper-Connections" (arXiv:2512.24880)
- Flash-IPA: "Flash Invariant Point Attention" (Liu et al., 2025)
"""

import torch
from torch import nn

from genie.model.single_feature_net import SingleFeatureNet
from genie.model.pair_feature_net import PairFeatureNet
from genie.model.pair_transform_net import PairTransformNet
from genie.model.mhc_flash_structure_net import mHCFlashStructureNet
from genie.flash_ipa.factorizer import LinearFactorizer


class mHCFlashDenoiser(nn.Module):
    """
    Denoiser combining mHC with Flash-IPA.
    
    This denoiser provides:
    1. Training stability from mHC expanded residual stream
    2. Memory efficiency from Flash-IPA factorized attention
    3. Support for very long sequences (512-1024 residues)
    4. Compatible with large batch training
    
    Architecture:
        Input (noisy frames) -> SingleFeatureNet -> PairFeatureNet 
        -> PairTransformNet -> LinearFactorizer -> mHCFlashStructureNet 
        -> Output (denoised frames)
    """
    
    def __init__(
        self,
        c_s: int,
        c_p: int,
        n_timestep: int,
        c_pos_emb: int,
        c_timestep_emb: int,
        relpos_k: int,
        template_type: str,
        n_pair_transform_layer: int,
        include_mul_update: bool,
        include_tri_att: bool,
        c_hidden_mul: int,
        c_hidden_tri_att: int,
        n_head_tri: int,
        tri_dropout: float,
        pair_transition_n: int,
        n_structure_layer: int,
        n_structure_block: int,
        c_hidden_ipa: int,
        n_head_ipa: int,
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
    ):
        super().__init__()
        
        print(f"============================================================")
        print(f"mHCFlashDenoiser: Combining mHC with Flash-IPA")
        print(f"============================================================")
        print(f"  max_n_res: {max_n_res}")
        print(f"  Flash-IPA parameters:")
        print(f"    - z_factor_rank: {z_factor_rank}")
        print(f"    - k_neighbors: {k_neighbors}")
        print(f"    - use_flash_attn_3: {use_flash_attn_3}")
        print(f"  mHC parameters:")
        print(f"    - expansion_rate: {mhc_expansion_rate}")
        print(f"    - sinkhorn_iters: {mhc_sinkhorn_iters}")
        print(f"    - alpha_init: {mhc_alpha_init}")
        print(f"  Structure layers: {n_structure_layer} x {n_structure_block}")
        print(f"  Pair transform layers: {n_pair_transform_layer}")
        print(f"============================================================")
        
        self.max_n_res = max_n_res
        self.z_factor_rank = z_factor_rank
        
        # Single feature network
        self.single_feature_net = SingleFeatureNet(
            c_s,
            n_timestep,
            c_pos_emb,
            c_timestep_emb
        )
        
        # Pair feature network
        self.pair_feature_net = PairFeatureNet(
            c_s,
            c_p,
            relpos_k,
            template_type
        )
        
        # Pair transform network (if needed)
        if n_pair_transform_layer > 0:
            self.pair_transform_net = PairTransformNet(
                c_p,
                n_pair_transform_layer,
                include_mul_update,
                include_tri_att,
                c_hidden_mul,
                c_hidden_tri_att,
                n_head_tri,
                tri_dropout,
                pair_transition_n
            )
        else:
            self.pair_transform_net = None
        
        # Linear factorizer to convert pair features to z_factors for Flash-IPA
        self.factorizer = LinearFactorizer(
            in_L=max_n_res,
            in_D=c_p,
            target_rank=z_factor_rank,
            target_inner_dim=c_p,
        )
        
        # mHC + Flash-IPA Structure Network
        self.structure_net = mHCFlashStructureNet(
            c_s=c_s,
            c_p=c_p,
            c_hidden_ipa=c_hidden_ipa,
            n_head=n_head_ipa,
            n_qk_point=n_qk_point,
            n_v_point=n_v_point,
            ipa_dropout=ipa_dropout,
            n_structure_transition_layer=n_structure_transition_layer,
            structure_transition_dropout=structure_transition_dropout,
            n_structure_layer=n_structure_layer,
            n_structure_block=n_structure_block,
            max_n_res=max_n_res,
            z_factor_rank=z_factor_rank,
            k_neighbors=k_neighbors,
            mhc_expansion_rate=mhc_expansion_rate,
            mhc_sinkhorn_iters=mhc_sinkhorn_iters,
            mhc_alpha_init=mhc_alpha_init,
            use_grad_checkpoint=use_grad_checkpoint,
            use_flash_attn_3=use_flash_attn_3,
        )
    
    def forward(self, ts, timesteps, mask):
        """
        Forward pass of the mHC + Flash-IPA Denoiser.
        
        Args:
            ts: rigid transforms (T object)
            timesteps: diffusion timesteps [B]
            mask: sequence mask [B, L]
        
        Returns:
            ts: denoised rigid transforms
        """
        # Compute single features (same as flash_denoiser)
        s = self.single_feature_net(ts, timesteps, mask)
        
        # Compute pair features
        p_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
        p = self.pair_feature_net(s, ts, p_mask)
        
        # Pair transform network (optional)
        if self.pair_transform_net is not None:
            p = self.pair_transform_net(p, p_mask)
        
        # Factorize pair features for Flash-IPA
        z_factor_1, z_factor_2 = self.factorizer(p, mask)
        
        # Structure network with mHC + Flash-IPA
        # Returns (s_final, t_final), we only need t_final
        _, ts = self.structure_net(s, z_factor_1, z_factor_2, ts, mask)
        
        return ts
