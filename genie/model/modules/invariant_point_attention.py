import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint  # Optimization

from genie.model.modules.primitives import Linear, ipa_point_weights_init_
from genie.utils.affine_utils import T
from genie.utils.tensor_utils import (
    permute_final_dims,
    flatten_final_dims,
)


class InvariantPointAttention(nn.Module):
    # ... [Init method remains largely unchanged] ...
    def __init__(self, c_s, c_z, c_hidden, no_heads, no_qk_points, no_v_points, inf=1e5, eps=1e-8, use_checkpointing=True):
        super(InvariantPointAttention, self).__init__()
        # ... [Copy init code from uploaded file] ...
        self.c_s = c_s
        self.c_z = c_z
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.no_qk_points = no_qk_points
        self.no_v_points = no_v_points
        self.inf = inf
        self.eps = eps
        self.use_checkpointing = use_checkpointing

        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)
        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)
        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv)
        self.linear_b = Linear(self.c_z, self.no_heads)
        self.head_weights = nn.Parameter(torch.zeros((no_heads)))
        ipa_point_weights_init_(self.head_weights)
        concat_out_dim = self.no_heads * (self.c_z + self.c_hidden + self.no_v_points * 4)
        self.linear_out = Linear(concat_out_dim, self.c_s, init="final")
        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    # Optimization: Encapsulate the heavy logic
    def _run_ipa(self, s, z, t_trans, t_rots, mask):
        # Reconstruct T object inside checkpoint (T is not a Tensor, so can't pass directly to checkpoint)
        t = T(t_rots, t_trans)

        # ... [Paste the entire forward pass logic from the original file here] ...
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)
        q = q.view(*q.shape[:-1], self.no_heads, -1)
        kv = kv.view(*kv.shape[:-1], self.no_heads, -1)
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        q_pts = self.linear_q_points(s)
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = t[..., None].apply(q_pts)
        q_pts = q_pts.view(*q_pts.shape[:-2], self.no_heads, self.no_qk_points, 3)

        kv_pts = self.linear_kv_points(s)
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = t[..., None].apply(kv_pts)
        kv_pts = kv_pts.view(*kv_pts.shape[:-2], self.no_heads, -1, 3)
        k_pts, v_pts = torch.split(kv_pts, [self.no_qk_points, self.no_v_points], dim=-2)

        b = self.linear_b(z)
        a = torch.matmul(permute_final_dims(q, 1, 0, 2), permute_final_dims(k, 1, 2, 0))
        a *= math.sqrt(1. / (3 * self.c_hidden))
        a += math.sqrt(1. / 3) * permute_final_dims(b, 2, 0, 1)

        pt_att = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        pt_att = pt_att ** 2
        pt_att = torch.sum(pt_att, dim=-1)
        head_weights = self.softplus(self.head_weights).view(*((1,) * len(pt_att.shape[:-2]) + (-1, 1)))
        head_weights = head_weights * math.sqrt(1. / (3 * (self.no_qk_points * 9. / 2)))
        pt_att = pt_att * head_weights
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)

        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)
        pt_att = permute_final_dims(pt_att, 2, 0, 1)
        a += pt_att
        a += square_mask.unsqueeze(-3)
        a = self.softmax(a)

        o = torch.matmul(a, v.transpose(-2, -3)).transpose(-2, -3)
        o = flatten_final_dims(o, 2)
        o_pt = torch.matmul(a.unsqueeze(-3), permute_final_dims(v_pts, 1, 3, 0, 2))
        o_pt = permute_final_dims(o_pt, 2, 0, 3, 1)
        o_pt = t[..., None, None].invert_apply(o_pt)
        o_pt_norm = flatten_final_dims(torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps), 2)
        o_pt = o_pt.view(*o_pt.shape[:-3], -1, 3)
        o_pair = torch.matmul(a.transpose(-2, -3), z)
        o_pair = flatten_final_dims(o_pair, 2)

        s_out = self.linear_out(torch.cat((o, *torch.unbind(o_pt, dim=-1), o_pt_norm, o_pair), dim=-1))
        return s_out

    def forward(self, s, z, t, mask):
        # Optimization: Apply checkpointing
        if self.training and s.requires_grad and self.use_checkpointing:
            # We must decompose 't' (AffineUtils object) into tensors to pass through checkpoint
            return checkpoint(self._run_ipa, s, z, t.trans, t.rots, mask, use_reentrant=False)
        else:
            return self._run_ipa(s, z, t.trans, t.rots, mask)