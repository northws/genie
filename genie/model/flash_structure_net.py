"""
Flash IPA Structure Network - Memory Efficient Implementation

This module provides a memory-efficient alternative to the standard StructureNet
by using Flash IPA with 1D bias mode, which avoids materializing O(L²) pair embeddings.

Key differences from standard StructureNet:
1. Uses EdgeEmbedder with flash_1d_bias mode instead of full pair embeddings
2. Computes z_factors directly from node embeddings (O(L) memory)
3. Uses Flash Attention for IPA computation

Memory savings: O(L²) -> O(L) for pair embeddings
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from genie.model.modules.structure_transition import StructureTransition
from genie.model.modules.backbone_update import BackboneUpdate

# Import Flash IPA components (from genie.flash_ipa, modified for PyTorch 2.9)
try:
    from genie.flash_ipa.ipa import InvariantPointAttention as FlashIPA, IPAConfig
    from genie.flash_ipa.edge_embedder import EdgeEmbedder, EdgeEmbedderConfig
    from genie.flash_ipa.rigid import create_rigid
    from genie.flash_ipa.utils import ANG_TO_NM_SCALE
    HAS_FLASH_IPA = True
except ImportError:
    HAS_FLASH_IPA = False
    print("Warning: genie.flash_ipa not available. FlashStructureNet will not be available.")


class FlashStructureLayer(nn.Module):
    """
    Structure layer using Flash IPA with 1D bias mode.
    
    Unlike the standard StructureLayer which takes pre-computed pair embeddings (p),
    this layer computes edge features on-the-fly from node embeddings,
    never materializing the full O(L²) pair embedding matrix.
    """

    def __init__(self,
                 c_s,
                 c_p,
                 c_hidden_ipa,
                 n_head,
                 n_qk_point,
                 n_v_point,
                 ipa_dropout,
                 n_structure_transition_layer,
                 structure_transition_dropout,
                 max_n_res,
                 z_factor_rank=2,
                 k_neighbors=10,
                 use_grad_checkpoint=False,
                 use_flash_attn_3=True
                 ):
        super(FlashStructureLayer, self).__init__()
        
        if not HAS_FLASH_IPA:
            raise RuntimeError("flash_ipa is required for FlashStructureLayer but not installed.")
        
        self.c_s = c_s
        self.c_p = c_p
        self.max_n_res = max_n_res
        self.z_factor_rank = z_factor_rank
        self.use_grad_checkpoint = use_grad_checkpoint
        
        # Calculate headdim_eff to determine optimal attn_dtype
        # When headdim_eff > 256, Flash Attention requires fp16 (FFPA mode)
        headdim_eff = max(
            c_hidden_ipa + 5 * n_qk_point + (z_factor_rank * n_head),
            c_hidden_ipa + 3 * n_v_point + (z_factor_rank * c_p // 4),
        )
        attn_dtype = "fp16" if headdim_eff > 256 else "bf16"
        
        # Flash IPA configuration
        ipa_conf = IPAConfig(
            use_flash_attn=True,
            use_flash_attn_3=use_flash_attn_3,  # Use FA3 on Hopper GPUs if available
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
        
        # Edge embedder with flash_1d_bias mode - NEVER materializes 2D matrix
        edge_conf = EdgeEmbedderConfig(
            z_factor_rank=z_factor_rank,
            c_s=c_s,
            c_p=c_p,
            mode="flash_1d_bias",
            k=k_neighbors,
            max_len=max_n_res,
        )
        self.edge_embedder = EdgeEmbedder(edge_conf)
        
        self.ipa_dropout = nn.Dropout(ipa_dropout)
        self.ipa_layer_norm = nn.LayerNorm(c_s)
        
        self.transition = StructureTransition(
            c_s,
            n_structure_transition_layer,
            structure_transition_dropout
        )
        
        self.bb_update = BackboneUpdate(c_s)
        
        # Unit conversion: Angstrom to nanometer (Flash IPA expects nm)
        self.ang_to_nm = ANG_TO_NM_SCALE

    def forward(self, inputs):
        """
        Args:
            inputs: tuple of (s, t, mask)
                s: node embeddings [B, L, c_s]
                t: rigid transforms (genie T object with .rots and .trans)
                mask: sequence mask [B, L]
        
        Returns:
            tuple of (s, t, mask) - updated node embeddings and transforms
        """
        s, t, mask = inputs
        
        # Get translations for edge embedding computation
        translations = t.trans  # [B, L, 3]
        
        # Compute z_factors using 1D bias mode (no 2D matrix materialized)
        # trans_sc is for side-chain, we use zeros since Genie only models backbone
        trans_sc = torch.zeros_like(translations)
        
        _, z_factor_1, z_factor_2, _ = self.edge_embedder(
            s, translations, trans_sc, mask, None, None
        )
        
        # Convert genie rigid to Flash IPA rigid format
        # Flash IPA expects positions in nanometers
        curr_rigids = create_rigid(t.rots, t.trans * self.ang_to_nm)
        
        # Apply Flash IPA
        s = s + self.ipa(s, None, z_factor_1, z_factor_2, curr_rigids, mask)
        
        s = self.ipa_dropout(s)
        s = self.ipa_layer_norm(s)
        s = self.transition(s)
        t = t.compose(self.bb_update(s))
        
        return (s, t, mask)


class FlashStructureNet(nn.Module):
    """
    Memory-efficient Structure Network using Flash IPA.
    
    This network replaces the standard StructureNet when memory efficiency
    is critical for long sequences. It avoids materializing O(L²) pair
    embeddings by using Flash IPA's 1D bias mode.
    
    Note: This implementation skips the pair transform layers (triangular
    attention/multiplication) since they require full 2D pair embeddings.
    The edge features are computed on-the-fly in each structure layer.
    """

    def __init__(self,
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
                 max_n_res,
                 z_factor_rank=2,
                 k_neighbors=10,
                 use_grad_checkpoint=False,
                 use_flash_attn_3=True
                 ):
        super(FlashStructureNet, self).__init__()
        
        if not HAS_FLASH_IPA:
            raise RuntimeError("flash_ipa is required for FlashStructureNet but not installed.")
        
        self.n_structure_block = n_structure_block
        self.use_grad_checkpoint = use_grad_checkpoint
        
        print(f"FlashStructureNet: Initializing with {n_structure_layer} layers, "
              f"{n_structure_block} blocks, max_n_res={max_n_res}")
        print(f"FlashStructureNet: Using flash_1d_bias mode (O(L) memory for edge features)")
        if use_flash_attn_3:
            print(f"FlashStructureNet: FA3 mode enabled (will use on Hopper GPUs if available)")
        
        layers = [
            FlashStructureLayer(
                c_s, c_p,
                c_hidden_ipa, n_head_ipa, n_qk_point, n_v_point, ipa_dropout,
                n_structure_transition_layer, structure_transition_dropout,
                max_n_res=max_n_res,
                z_factor_rank=z_factor_rank,
                k_neighbors=k_neighbors,
                use_grad_checkpoint=use_grad_checkpoint,
                use_flash_attn_3=use_flash_attn_3
            )
            for _ in range(n_structure_layer)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, s, p, t, mask):
        """
        Args:
            s: node embeddings [B, L, c_s]
            p: pair embeddings [B, L, L, c_p] - IGNORED in flash mode
            t: rigid transforms (genie T object)
            mask: sequence mask [B, L]
        
        Returns:
            t: updated rigid transforms
        
        Note: The pair embeddings `p` are ignored in this implementation.
        Edge features are computed on-the-fly from node embeddings.
        """
        # p is ignored - we compute edge features on-the-fly
        for block_idx in range(self.n_structure_block):
            if self.training and self.use_grad_checkpoint:
                s, t, mask = checkpoint(
                    self.run_net, s, t, mask, use_reentrant=False
                )
            else:
                s, t, mask = self.net((s, t, mask))
        return t

    def run_net(self, s, t, mask):
        return self.net((s, t, mask))
