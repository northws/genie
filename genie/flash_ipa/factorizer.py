import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple
from torch import nn, Tensor
import math


class LinearFactorizer(nn.Module):

    def __init__(self, in_L, in_D, target_rank=4, target_inner_dim=8):
        super().__init__()
        self.target_rank = target_rank
        self.target_inner_dim = target_inner_dim
        # self.linear_col = nn.Linear(in_L * in_D, target_rank * target_inner_dim, bias=False)
        # self.linear_row = nn.Linear(in_L * in_D, target_rank * target_inner_dim, bias=False)
        self.length_compressor = nn.Linear(in_L, target_rank, bias=False)
        self.inner_compressor = nn.Linear(in_D, target_inner_dim, bias=False)
        self.in_L = in_L
        self.length_norm = nn.LayerNorm(target_rank)
        self.inner_norm = nn.LayerNorm(target_inner_dim)

        nn.init.kaiming_uniform_(self.length_compressor.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.inner_compressor.weight, a=math.sqrt(5))

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Input:
            x: (B, L, L, D)
            mask: (B, L), optional length mask
        Output:
            U: (B, L, target_rank, target_inner_dim)
            V: (B, L, target_rank, target_inner_dim)
        """
        # Compress along length
        L_orig = x.shape[1]
        x = F.pad(x, (0, 0, 0, self.in_L - x.shape[2], 0, self.in_L - x.shape[1]), value=0.0)
        
        # Replace einops with permute for torch.compile compatibility
        # x: (B, R, C, D) -> (B, C, D, R)
        row_embed = self.length_compressor(x.permute(0, 2, 3, 1))[:, :L_orig, :, :]
        # x: (B, R, C, D) -> (B, R, D, C)
        col_embed = self.length_compressor(x.permute(0, 1, 3, 2))[:, :L_orig, :, :]

        row_embed = self.length_norm(row_embed)
        col_embed = self.length_norm(col_embed)

        # row_embed: (B, C, D, R) -> (B, C, R, D)
        row_embed = self.inner_compressor(row_embed.permute(0, 1, 3, 2))
        # col_embed: (B, R, D, C) -> (B, R, C, D)
        col_embed = self.inner_compressor(col_embed.permute(0, 1, 3, 2))

        row_embed = self.inner_norm(row_embed)
        col_embed = self.inner_norm(col_embed)
        
        if mask is not None:
            # Apply mask to row_embed and col_embed
            row_embed = row_embed * mask[:, :, None, None]
            col_embed = col_embed * mask[:, :, None, None]

        # Return (B, L, rank, inner_dim)
        return row_embed, col_embed
