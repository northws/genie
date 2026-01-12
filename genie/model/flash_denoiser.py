"""
Flash Denoiser - Memory Efficient Model for Long Sequences

This module provides a memory-efficient alternative to the standard Denoiser
by using Flash IPA, which avoids materializing O(L²) pair embeddings.

Key differences from standard Denoiser:
1. Skips PairTransformNet (which requires O(L²) memory)
2. Uses FlashStructureNet with 1D bias mode
3. Significantly reduced memory footprint for long sequences

Use this model when:
- Processing sequences longer than 512 residues
- Memory is a constraint
- Triangular attention is not strictly required

Trade-offs:
- No triangular attention/multiplication (may affect quality for some tasks)
- Edge features are computed per-layer (slight compute overhead)
"""

import torch
from torch import nn

from genie.model.single_feature_net import SingleFeatureNet
from genie.model.pair_feature_net import PairFeatureNet

# Conditional import for Flash components
try:
    from genie.model.flash_structure_net import FlashStructureNet, HAS_FLASH_IPA
except ImportError:
    HAS_FLASH_IPA = False


class FlashDenoiser(nn.Module):
    """
    Memory-efficient Denoiser using Flash IPA.
    
    This model is designed for long sequences where memory is a constraint.
    It skips the O(L²) pair transform layers and uses Flash IPA's 1D bias
    mode to compute edge features on-the-fly.
    
    Architecture:
        SingleFeatureNet -> PairFeatureNet -> FlashStructureNet
                                              (no PairTransformNet)
    """

    def __init__(self,
                 c_s, c_p, n_timestep,
                 c_pos_emb, c_timestep_emb,
                 relpos_k, template_type,
                 # PairTransform params (kept for config compatibility, but ignored)
                 n_pair_transform_layer=0,
                 include_mul_update=False,
                 include_tri_att=False,
                 c_hidden_mul=128,
                 c_hidden_tri_att=32,
                 n_head_tri=4,
                 tri_dropout=0.0,
                 pair_transition_n=2,
                 # Structure params
                 n_structure_layer=8,
                 n_structure_block=1,
                 c_hidden_ipa=16,
                 n_head_ipa=8,
                 n_qk_point=4,
                 n_v_point=8,
                 ipa_dropout=0.0,
                 n_structure_transition_layer=1,
                 structure_transition_dropout=0.0,
                 # Flash-specific params
                 max_n_res=512,
                 z_factor_rank=2,
                 k_neighbors=10,
                 use_grad_checkpoint=False,
                 use_flash_attn_3=True  # Use FA3 on Hopper GPUs if available
                 ):
        super(FlashDenoiser, self).__init__()
        
        if not HAS_FLASH_IPA:
            raise RuntimeError(
                "FlashDenoiser requires flash_ipa to be installed. "
                "Install with: pip install flash_ipa"
            )
        
        self.max_n_res = max_n_res
        
        # Log configuration
        print("="*60)
        print("FlashDenoiser: Memory-Efficient Mode Enabled")
        print("="*60)
        print(f"  max_n_res: {max_n_res}")
        print(f"  z_factor_rank: {z_factor_rank}")
        print(f"  k_neighbors: {k_neighbors}")
        print(f"  n_structure_layer: {n_structure_layer}")
        print(f"  n_structure_block: {n_structure_block}")
        print(f"  use_flash_attn_3: {use_flash_attn_3}")
        if n_pair_transform_layer > 0:
            print(f"  WARNING: n_pair_transform_layer={n_pair_transform_layer} is ignored in Flash mode")
        print("  PairTransformNet: DISABLED (O(L²) -> O(L) memory)")
        print("="*60)
        
        self.single_feature_net = SingleFeatureNet(
            c_s,
            n_timestep,
            c_pos_emb,
            c_timestep_emb
        )
        
        # PairFeatureNet is still used for initial pair features
        # but we don't run PairTransformNet
        self.pair_feature_net = PairFeatureNet(
            c_s,
            c_p,
            relpos_k,
            template_type
        )
        
        # No PairTransformNet - this is where the memory savings come from
        self.pair_transform_net = None
        
        # Flash Structure Net
        self.structure_net = FlashStructureNet(
            c_s,
            c_p,
            n_structure_layer,
            n_structure_block,
            c_hidden_ipa,
            n_head_ipa,
            n_qk_point,
            n_v_point,
            ipa_dropout,
            n_structure_transition_layer,
            structure_transition_dropout,
            max_n_res=max_n_res,
            z_factor_rank=z_factor_rank,
            k_neighbors=k_neighbors,
            use_grad_checkpoint=use_grad_checkpoint,
            use_flash_attn_3=use_flash_attn_3
        )

    def forward(self, ts, timesteps, mask):
        """
        Forward pass of the Flash Denoiser.
        
        Args:
            ts: rigid transforms (T object)
            timesteps: diffusion timesteps [B]
            mask: sequence mask [B, L]
        
        Returns:
            ts: denoised rigid transforms
        """
        # Compute single features
        s = self.single_feature_net(ts, timesteps, mask)
        
        # Compute pair features (still O(L²) but no transform layers)
        # Note: In a fully optimized version, we could skip this entirely
        # and let FlashStructureNet compute everything from s
        p_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
        p = self.pair_feature_net(s, ts, p_mask)
        
        # Structure net (p is passed but ignored in flash mode)
        ts = self.structure_net(s, p, ts, mask)
        
        return ts


def create_flash_denoiser_from_config(config):
    """
    Factory function to create FlashDenoiser from a Genie config.
    
    Args:
        config: Genie Config object
    
    Returns:
        FlashDenoiser instance
    """
    return FlashDenoiser(
        c_s=config.model['c_s'],
        c_p=config.model['c_p'],
        n_timestep=config.diffusion['n_timestep'],
        c_pos_emb=config.model['c_pos_emb'],
        c_timestep_emb=config.model['c_timestep_emb'],
        relpos_k=config.model['relpos_k'],
        template_type=config.model['template_type'],
        n_pair_transform_layer=config.model.get('n_pair_transform_layer', 0),
        include_mul_update=config.model.get('include_mul_update', False),
        include_tri_att=config.model.get('include_tri_att', False),
        c_hidden_mul=config.model.get('c_hidden_mul', 128),
        c_hidden_tri_att=config.model.get('c_hidden_tri_att', 32),
        n_head_tri=config.model.get('n_head_tri', 4),
        tri_dropout=config.model.get('tri_dropout', 0.0),
        pair_transition_n=config.model.get('pair_transition_n', 2),
        n_structure_layer=config.model['n_structure_layer'],
        n_structure_block=config.model['n_structure_block'],
        c_hidden_ipa=config.model['c_hidden_ipa'],
        n_head_ipa=config.model['n_head_ipa'],
        n_qk_point=config.model['n_qk_point'],
        n_v_point=config.model['n_v_point'],
        ipa_dropout=config.model['ipa_dropout'],
        n_structure_transition_layer=config.model['n_structure_transition_layer'],
        structure_transition_dropout=config.model['structure_transition_dropout'],
        max_n_res=config.io['max_n_res'],
        z_factor_rank=config.model.get('z_factor_rank', 2),
        k_neighbors=config.model.get('k_neighbors', 10),
        use_grad_checkpoint=config.training.get('use_grad_checkpoint', False),
        use_flash_attn_3=config.model.get('use_flash_attn_3', True)
    )
