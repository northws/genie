from functools import partialmethod
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint  # Optimization

from genie.model.modules.primitives import Linear
from genie.utils.tensor_utils import permute_final_dims


class TriangleMultiplicativeUpdate(nn.Module):
    """
        Implements Algorithms 11 and 12 with Memory Optimizations.
    """

    def __init__(self, c_z, c_hidden, _outgoing=True, use_grad_checkpoint=False):
        """
            Args:
                c_z: Input channel dimension
                c_hidden: Hidden channel dimension
        """
        super(TriangleMultiplicativeUpdate, self).__init__()
        self.c_z = c_z
        self.c_hidden = c_hidden
        self._outgoing = _outgoing
        self.use_grad_checkpoint = use_grad_checkpoint

        self.linear_a_p = Linear(self.c_z, self.c_hidden)
        self.linear_a_g = Linear(self.c_z, self.c_hidden, init="gating")
        self.linear_b_p = Linear(self.c_z, self.c_hidden)
        self.linear_b_g = Linear(self.c_z, self.c_hidden, init="gating")
        self.linear_g = Linear(self.c_z, self.c_z, init="gating")
        self.linear_z = Linear(self.c_hidden, self.c_z, init="final")

        self.layer_norm_in = nn.LayerNorm(self.c_z)
        self.layer_norm_out = nn.LayerNorm(self.c_hidden)

        self.sigmoid = nn.Sigmoid()

        cp = self._outgoing_matmul if self._outgoing else self._incoming_matmul
        self.combine_projections = cp

    def _outgoing_matmul(self, a, b):
        p = torch.matmul(
            permute_final_dims(a, 2, 0, 1),
            permute_final_dims(b, 2, 1, 0),
        )
        return permute_final_dims(p, 1, 2, 0)

    def _incoming_matmul(self, a, b):
        p = torch.matmul(
            permute_final_dims(a, 2, 1, 0),
            permute_final_dims(b, 2, 0, 1),
        )
        return permute_final_dims(p, 1, 2, 0)

    # Optimization: Function to wrap in checkpoint
    def _run_block(self, z, mask):
        z = self.layer_norm_in(z)
        a = self.linear_a_p(z) * self.sigmoid(self.linear_a_g(z))
        a = a * mask
        b = self.linear_b_p(z) * self.sigmoid(self.linear_b_g(z))
        b = b * mask
        x = self.combine_projections(a, b)
        x = self.layer_norm_out(x)
        x = self.linear_z(x)
        g = self.sigmoid(self.linear_g(z))
        return x * g

    def forward(self, z, mask=None):
        if (mask is None):
            mask = z.new_ones(z.shape[:-1], requires_grad=False)

        mask = mask.unsqueeze(-1)

        # Optimization: Apply gradient checkpointing during training
        # This prevents OOM errors on long sequences for O(N^3) ops
        if self.training and z.requires_grad and self.use_grad_checkpoint:
            z_out = checkpoint(self._run_block, z, mask, use_reentrant=False)
            # Note: We need to reconstruct 'z' from input + update outside standard residual
            # But here the residual connection is usually handled by the parent layer (PairTransformLayer)
            # The class returns the update quantity, not the residual sum.
            return z_out
        else:
            return self._run_block(z, mask)


class TriangleMultiplicationOutgoing(TriangleMultiplicativeUpdate):
    __init__ = partialmethod(
        TriangleMultiplicativeUpdate.__init__, _outgoing=True,
    )


class TriangleMultiplicationIncoming(TriangleMultiplicativeUpdate):
    __init__ = partialmethod(
        TriangleMultiplicativeUpdate.__init__, _outgoing=False,
    )