import torch
import numpy as np


def distance(p, eps=1e-10):
    # [*, 2, 3]
    return (eps + torch.sum((p[..., 0, :] - p[..., 1, :]) ** 2, dim=-1)) ** 0.5


def dihedral(p, eps=1e-10):
    # p: [*, 4, 3]

    # [*, 3]
    u1 = p[..., 1, :] - p[..., 0, :]
    u2 = p[..., 2, :] - p[..., 1, :]
    u3 = p[..., 3, :] - p[..., 2, :]

    # [*, 3]
    u1xu2 = torch.cross(u1, u2, dim=-1)
    u2xu3 = torch.cross(u2, u3, dim=-1)

    # [*]
    u2_norm = (eps + torch.sum(u2 ** 2, dim=-1)) ** 0.5
    u1xu2_norm = (eps + torch.sum(u1xu2 ** 2, dim=-1)) ** 0.5
    u2xu3_norm = (eps + torch.sum(u2xu3 ** 2, dim=-1)) ** 0.5

    # [*]
    cos_enc = torch.einsum('...d,...d->...', u1xu2, u2xu3) / (u1xu2_norm * u2xu3_norm)
    sin_enc = torch.einsum('...d,...d->...', u2, torch.cross(u1xu2, u2xu3, dim=-1)) / (
                u2_norm * u1xu2_norm * u2xu3_norm)

    return torch.stack([cos_enc, sin_enc], dim=-1)


def compute_frenet_frames(x, mask, eps=1e-10):
    """
    Vectorized computation of Frenet-Serret frames.
    x: [b, n_res, 3]
    mask: [b, n_res]
    """

    # Vectorized calculation of tangent, binormal, normal
    # Note: t is calculated on the whole sequence, so t[i] corresponds to vector x[i]->x[i+1]
    t = x[:, 1:] - x[:, :-1]
    t_norm = torch.sqrt(eps + torch.sum(t ** 2, dim=-1))
    t = t / t_norm.unsqueeze(-1)

    b = torch.cross(t[:, :-1], t[:, 1:])
    b_norm = torch.sqrt(eps + torch.sum(b ** 2, dim=-1))
    b = b / b_norm.unsqueeze(-1)

    n = torch.cross(b, t[:, 1:])

    # tbn shape: [B, N-2, 3, 3]
    # Corresponding to residues 1 to N-2 (0-indexed)
    tbn = torch.stack([t[:, 1:], b, n], dim=-1)

    B, N, _ = x.shape
    device = x.device

    # Optimization: Remove Python loop over batch dimension
    # Initialize with Identity
    rots = torch.eye(3, device=device, dtype=x.dtype).view(1, 1, 3, 3).repeat(B, N, 1, 1)

    # 1. Fill the middle body (indices 1 to N-2)
    # tbn corresponds to valid frames for indices [1, ..., N-2] relative to the full sequence
    # Slicing rots[:, 1:-1] gives shape [B, N-2, 3, 3], matching tbn
    rots[:, 1:-1] = tbn

    # 2. Handle N-terminus (Index 0)
    # Copy from Index 1
    rots[:, 0] = rots[:, 1]

    # 3. Handle C-terminus (Index length-1)
    # We need to copy from (length-2) to (length-1) for each batch item
    lengths = mask.sum(dim=1).long()  # [B]

    batch_indices = torch.arange(B, device=device)

    # Clamp indices to ensure we don't access -1 (though min length should be > 2)
    src_indices = (lengths - 2).clamp(min=0)
    tgt_indices = (lengths - 1).clamp(min=0)

    # Gather frames from src (length-2)
    c_term_frames = rots[batch_indices, src_indices]  # [B, 3, 3]

    # Scatter to tgt (length-1)
    rots[batch_indices, tgt_indices] = c_term_frames

    # 4. Cleanup Padded Regions
    # The vectorized assignment `rots[:, 1:-1] = tbn` might have written garbage
    # into the padded regions (where x was 0).
    # We reset padded regions to Identity using the mask.

    # mask: [B, N] -> [B, N, 1, 1]
    mask_expanded = mask.view(B, N, 1, 1)
    identity = torch.eye(3, device=device, dtype=x.dtype).view(1, 1, 3, 3)

    rots = rots * mask_expanded + identity * (1 - mask_expanded)

    return rots