"""
mHC (Manifold-Constrained Hyper-Connections) Denoiser

This module implements the Denoiser with mHC-based Structure Network
for improved training stability at large scales.

Based on: "mHC: Manifold-Constrained Hyper-Connections" (arXiv:2512.24880)
"""

import torch
from torch import nn

from genie.model.single_feature_net import SingleFeatureNet
from genie.model.pair_feature_net import PairFeatureNet
from genie.model.pair_transform_net import PairTransformNet
from genie.model.mhc_pair_transform_net import mHCPairTransformNet
from genie.model.mhc_structure_net import mHCStructureNet


class mHCDenoiser(nn.Module):
    """
    Denoiser with mHC (Manifold-Constrained Hyper-Connections).
    
    This denoiser uses mHCStructureNet instead of the standard StructureNet
    or FlashStructureNet for improved training stability at large scales.
    
    Key Features:
    - Standard IPA (no FlashIPA dependency, works on all GPUs)
    - mHC expanded residual stream for better expressivity
    - Doubly stochastic residual mixing for stable training
    - Compatible with large batch training
    
    Architecture:
        Input (noisy frames) -> SingleFeatureNet -> PairFeatureNet 
        -> PairTransformNet -> mHCStructureNet -> Output (denoised frames)
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
        max_n_res: int = None,
        use_grad_checkpoint: bool = False,
        # mHC specific parameters
        mhc_expansion_rate: int = 4,
        mhc_sinkhorn_iters: int = 20,
        mhc_alpha_init: float = 0.01,
    ):
        super().__init__()
        
        print(f"============================================================")
        print(f"mHCDenoiser: Using Manifold-Constrained Hyper-Connections")
        print(f"============================================================")
        print(f"  max_n_res: {max_n_res}")
        print(f"  mhc_expansion_rate: {mhc_expansion_rate}")
        print(f"  mhc_sinkhorn_iters: {mhc_sinkhorn_iters}")
        print(f"  mhc_alpha_init: {mhc_alpha_init}")
        print(f"  n_structure_layer: {n_structure_layer}")
        print(f"  n_structure_block: {n_structure_block}")
        print(f"  n_pair_transform_layer: {n_pair_transform_layer}")
        print(f"============================================================")
        
        # Single feature network (same as standard Denoiser)
        self.single_feature_net = SingleFeatureNet(
            c_s,
            n_timestep,
            c_pos_emb,
            c_timestep_emb
        )
        
        # Pair feature network (same as standard Denoiser)
        self.pair_feature_net = PairFeatureNet(
            c_s,
            c_p,
            relpos_k,
            template_type
        )
        
        # Pair transform network with mHC (Manifold-Constrained Hyper-Connections)
        # Uses mHCPairTransformNet for improved training stability
        self.pair_transform_net = mHCPairTransformNet(
            c_p,
            n_pair_transform_layer,
            include_mul_update,
            include_tri_att,
            c_hidden_mul,
            c_hidden_tri_att,
            n_head_tri,
            tri_dropout,
            pair_transition_n,
            mhc_expansion_rate=mhc_expansion_rate,
            mhc_sinkhorn_iters=mhc_sinkhorn_iters,
            use_grad_checkpoint=use_grad_checkpoint
        ) if n_pair_transform_layer > 0 else None
        
        # mHC Structure Network
        self.structure_net = mHCStructureNet(
            c_s=c_s,
            c_p=c_p,
            n_structure_layer=n_structure_layer,
            n_structure_block=n_structure_block,
            c_hidden_ipa=c_hidden_ipa,
            n_head_ipa=n_head_ipa,
            n_qk_point=n_qk_point,
            n_v_point=n_v_point,
            ipa_dropout=ipa_dropout,
            n_structure_transition_layer=n_structure_transition_layer,
            structure_transition_dropout=structure_transition_dropout,
            mhc_expansion_rate=mhc_expansion_rate,
            mhc_sinkhorn_iters=mhc_sinkhorn_iters,
            mhc_alpha_init=mhc_alpha_init,
            use_grad_checkpoint=use_grad_checkpoint,
        )
    
    def forward(self, ts, timesteps, mask):
        """
        Forward pass through mHC denoiser.
        
        Args:
            ts: Noisy frames (rigid transforms)
            timesteps: Diffusion timesteps [B]
            mask: Sequence mask [B, L]
        
        Returns:
            ts: Denoised frames (updated rigid transforms)
        """
        # Pair mask for pair features
        p_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
        
        # Single features from positions and timesteps
        s = self.single_feature_net(ts, timesteps, mask)
        
        # Pair features from single features and frames
        p = self.pair_feature_net(s, ts, p_mask)
        
        # Pair transform (if enabled)
        if self.pair_transform_net is not None:
            p = self.pair_transform_net(p, p_mask)
        
        # Structure prediction with mHC
        ts = self.structure_net(s, p, ts, mask)
        
        return ts
