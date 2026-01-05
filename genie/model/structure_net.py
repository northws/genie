import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from genie.model.modules.invariant_point_attention import InvariantPointAttention
from genie.model.modules.structure_transition import StructureTransition
from genie.model.modules.backbone_update import BackboneUpdate

# Optimization: Try to import FlashIPA
try:
    from flash_ipa.ipa import InvariantPointAttention as FlashIPA, IPAConfig
    from flash_ipa.factorizer import LinearFactorizer
    from flash_ipa.rigid import create_rigid
    from flash_ipa.utils import ANG_TO_NM_SCALE

    HAS_FLASH_IPA = True
except ImportError:
    HAS_FLASH_IPA = False


class StructureLayer(nn.Module):

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
                 use_flash_ipa=False,  # Optimization Flag
                 max_n_res=None
                 ):
        super(StructureLayer, self).__init__()

        self.use_flash_ipa = use_flash_ipa and HAS_FLASH_IPA and (max_n_res is not None)
        if use_flash_ipa and not self.use_flash_ipa:
             if not HAS_FLASH_IPA:
                 print("Warning: use_flash_ipa=True but flash_ipa not installed. Fallback to standard IPA.")
             if max_n_res is None:
                 print("Warning: use_flash_ipa=True but max_n_res is None. Fallback to standard IPA.")

        # Optimization: Conditional FlashIPA initialization
        if self.use_flash_ipa:
            # FlashIPA Config
            # Assuming z_factor_rank=2 as per README example, or we could make it configurable
            self.z_factor_rank = 2
            ipa_conf = IPAConfig(
                use_flash_attn=True,
                attn_dtype="bf16", # Assuming Ampere+ GPU as per context
                c_s=c_s,
                c_z=c_p,
                c_hidden=c_hidden_ipa,
                no_heads=n_head,
                z_factor_rank=self.z_factor_rank,
                no_qk_points=n_qk_point,
                no_v_points=n_v_point,
            )
            self.ipa = FlashIPA(ipa_conf)
            
            # Factorizer to convert full pair embeddings (p) to z_factors
            self.factorizer = LinearFactorizer(
                in_L=max_n_res,
                in_D=c_p,
                target_rank=self.z_factor_rank,
                target_inner_dim=c_p,
            )
        else:
            self.ipa = InvariantPointAttention(
                c_s,
                c_p,
                c_hidden_ipa,
                n_head,
                n_qk_point,
                n_v_point
            )

        self.ipa_dropout = nn.Dropout(ipa_dropout)
        self.ipa_layer_norm = nn.LayerNorm(c_s)

        # Built-in dropout and layer norm
        self.transition = StructureTransition(
            c_s,
            n_structure_transition_layer,
            structure_transition_dropout
        )

        # backbone update
        self.bb_update = BackboneUpdate(c_s)

    def forward(self, inputs):
        s, p, t, mask = inputs

        # Apply IPA (Standard or Flash)
        if self.use_flash_ipa:
            # Convert t (genie T object) to FlashIPA Rigid
            curr_rigids = create_rigid(t.rots, t.trans)
            
            # Factorize p (B, L, L, C_p) -> z_factor_1, z_factor_2
            z_factor_1, z_factor_2 = self.factorizer(p, mask)
            
            s = self.ipa(s, None, z_factor_1, z_factor_2, curr_rigids, mask)
        else:
            s = s + self.ipa(s, p, t, mask)

        s = self.ipa_dropout(s)
        s = self.ipa_layer_norm(s)
        s = self.transition(s)
        t = t.compose(self.bb_update(s))
        outputs = (s, p, t, mask)
        return outputs


class StructureNet(nn.Module):

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
                 use_flash_ipa=False,  # Pass optimization flag
                 max_n_res=None
                 ):
        super(StructureNet, self).__init__()

        self.n_structure_block = n_structure_block

        layers = [
            StructureLayer(
                c_s, c_p,
                c_hidden_ipa, n_head_ipa, n_qk_point, n_v_point, ipa_dropout,
                n_structure_transition_layer, structure_transition_dropout,
                use_flash_ipa=use_flash_ipa,
                max_n_res=max_n_res
            )
            for _ in range(n_structure_layer)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, s, p, t, mask):
        for block_idx in range(self.n_structure_block):
            # Optimization: Optional gradient checkpointing for the whole block
            # to save memory during training deep networks
            if self.training:
                s, p, t, mask = checkpoint(
                    self.run_net, s, p, t, mask, use_reentrant=False
                )
            else:
                s, p, t, mask = self.net((s, p, t, mask))
        return t

    def run_net(self, s, p, t, mask):
        return self.net((s, p, t, mask))