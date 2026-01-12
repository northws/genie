"""
mHC Loss Regularization for Residual Connections

This module implements mHC-inspired constraints as a regularization loss,
focusing on the core insight: stabilizing residual connections.

Key insight from mHC paper (arXiv:2512.24880):
- Doubly stochastic matrices have spectral radius = 1
- This prevents gradient explosion/vanishing in residual connections
- x_{l+1} = H_res @ x_l + F(x_l), where H_res is doubly stochastic

This loss-based approach applies soft constraints to:
1. Residual update ratios (keep balanced)
2. Layer-wise gradient flow (keep stable)
3. Feature norm preservation (prevent explosion)

Note: This is a LIGHTWEIGHT alternative to full mHC architecture.
For maximum stability, use the architectural mHC in mhc_structure_net.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def residual_balance_loss(
    x_input: torch.Tensor,
    x_output: torch.Tensor,
    residual_output: torch.Tensor,
    mask: torch.Tensor,
    target_ratio: float = 0.5,
) -> torch.Tensor:
    """
    mHC核心思想的损失函数版本：平衡残差连接。
    
    mHC通过双随机矩阵确保：残差分支和主分支的贡献是平衡的。
    这个损失函数软约束这种平衡。
    
    Args:
        x_input: 层输入 [B, L, C]
        x_output: 层输出 (含残差) [B, L, C]  
        residual_output: 残差分支输出 F(x) [B, L, C]
        mask: 序列掩码 [B, L]
        target_ratio: 目标残差比例 (0.5 表示平衡)
    
    Returns:
        平衡损失
    """
    mask_expanded = mask.unsqueeze(-1)  # [B, L, 1]
    
    # 计算残差更新量
    update = x_output - x_input  # 应该约等于 residual_output
    
    # 计算残差比例: ||F(x)|| / (||x|| + ||F(x)||)
    input_norm = (x_input ** 2 * mask_expanded).sum(dim=(1, 2))  # [B]
    residual_norm = (residual_output ** 2 * mask_expanded).sum(dim=(1, 2))  # [B]
    
    actual_ratio = residual_norm / (input_norm + residual_norm + 1e-8)
    
    # 惩罚偏离目标比例
    ratio_loss = ((actual_ratio - target_ratio) ** 2).mean()
    
    return ratio_loss


def gradient_norm_preservation_loss(
    x_input: torch.Tensor,
    x_output: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    mHC的梯度流保持约束：输出范数不应显著偏离输入范数。
    
    双随机矩阵的谱半径为1，这意味着 ||H_res @ x|| ≈ ||x||。
    这个损失函数软约束这种性质。
    
    Args:
        x_input: 层输入 [B, L, C]
        x_output: 层输出 [B, L, C]
        mask: 序列掩码 [B, L]
    
    Returns:
        范数保持损失
    """
    mask_expanded = mask.unsqueeze(-1)
    
    # 计算输入输出的 L2 范数
    input_norm = torch.sqrt((x_input ** 2 * mask_expanded).sum(dim=(1, 2)) + 1e-8)
    output_norm = torch.sqrt((x_output ** 2 * mask_expanded).sum(dim=(1, 2)) + 1e-8)
    
    # 范数比例应接近 1
    norm_ratio = output_norm / (input_norm + 1e-8)
    
    # 惩罚偏离 1 的情况 (允许 0.8-1.2 的范围)
    deviation = F.relu(torch.abs(norm_ratio - 1.0) - 0.2)
    
    return (deviation ** 2).mean()


def doubly_stochastic_penalty(
    weight_matrix: torch.Tensor,
    n_sinkhorn_iters: int = 5,
) -> torch.Tensor:
    """
    直接惩罚权重矩阵偏离双随机的程度。
    
    如果模型有可学习的残差权重矩阵，可以用这个损失
    软约束它接近双随机矩阵。
    
    Args:
        weight_matrix: 权重矩阵 [n, n] 或 [B, n, n]
        n_sinkhorn_iters: Sinkhorn迭代次数
    
    Returns:
        双随机偏离损失
    """
    # 确保非负
    W = torch.exp(weight_matrix)
    
    # 计算行和与列和
    row_sums = W.sum(dim=-1)  # [..., n]
    col_sums = W.sum(dim=-2)  # [..., n]
    
    # 目标：行和列和都为1
    row_loss = ((row_sums - 1.0) ** 2).mean()
    col_loss = ((col_sums - 1.0) ** 2).mean()
    
    return row_loss + col_loss


def representation_stability_loss(
    s_hidden: torch.Tensor,
    mask: torch.Tensor,
    target_rank: int = 4,
) -> torch.Tensor:
    """
    特征稳定性损失（辅助正则化，非 mHC 核心）。
    
    通过限制激活值范围来辅助训练稳定：
    1. 惩罚极端值
    2. 鼓励特征多样性
    
    Args:
        s_hidden: 单表示 [B, L, C]
        mask: 序列掩码 [B, L]
        target_rank: 未使用，保留接口
    
    Returns:
        稳定性损失
    """
    B, L, C = s_hidden.shape
    mask_expanded = mask.unsqueeze(-1)  # [B, L, 1]
    
    # 激活幅度惩罚
    s_masked = s_hidden * mask_expanded
    magnitude_loss = (s_masked ** 2).sum() / (mask.sum() * C + 1e-8)
    
    return 0.01 * magnitude_loss


def gradient_flow_loss(
    pred_trans: torch.Tensor,
    target_trans: torch.Tensor, 
    mask: torch.Tensor,
    alpha: float = 0.01,
) -> torch.Tensor:
    """
    梯度流平滑损失（空间正则化）。
    
    Args:
        pred_trans: 预测平移 [B, L, 3]
        target_trans: 目标平移 [B, L, 3]
        mask: 序列掩码 [B, L]
        alpha: 权重
    
    Returns:
        平滑损失
    """
    residual = pred_trans - target_trans
    diff = residual[:, 1:, :] - residual[:, :-1, :]
    mask_diff = mask[:, 1:] * mask[:, :-1]
    
    smoothness = (diff ** 2).sum(dim=-1) * mask_diff
    smoothness_loss = smoothness.sum() / (mask_diff.sum() + 1e-8)
    
    return alpha * smoothness_loss


class mHCResidualLoss(nn.Module):
    """
    mHC 风格的残差连接优化损失。
    
    核心思想：软约束残差连接的平衡性和范数保持性，
    这是 mHC 双随机约束的损失函数近似。
    
    使用场景：
    - 不想修改模型架构
    - 需要轻量级的训练稳定化
    - 与 Flash-IPA 结合使用
    """
    
    def __init__(
        self,
        balance_weight: float = 0.1,
        norm_weight: float = 0.1,
        target_residual_ratio: float = 0.5,
    ):
        super().__init__()
        self.balance_weight = balance_weight
        self.norm_weight = norm_weight
        self.target_residual_ratio = target_residual_ratio
        
    def forward(
        self,
        x_input: torch.Tensor,
        x_output: torch.Tensor,
        residual_output: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 mHC 风格残差优化损失。
        
        Args:
            x_input: 层输入 [B, L, C]
            x_output: 层输出 [B, L, C]
            residual_output: 残差分支输出 [B, L, C]
            mask: 序列掩码 [B, L]
        
        Returns:
            总损失
        """
        loss = torch.tensor(0.0, device=mask.device)
        
        if self.balance_weight > 0:
            loss = loss + self.balance_weight * residual_balance_loss(
                x_input, x_output, residual_output, mask, self.target_residual_ratio
            )
            
        if self.norm_weight > 0:
            loss = loss + self.norm_weight * gradient_norm_preservation_loss(
                x_input, x_output, mask
            )
            
        return loss


def compute_mhc_regularization(
    noise_pred: torch.Tensor,
    noise_target: torch.Tensor,
    mask: torch.Tensor,
    weight: float = 0.01,
) -> torch.Tensor:
    """
    简化的 mHC 正则化（用于扩散训练）。
    
    这是一个轻量级版本，适用于只想添加简单正则化的场景。
    对于完整的 mHC 效果，建议使用架构级实现 (mhc_structure_net.py)。
    
    Args:
        noise_pred: 预测噪声 [B, L, 3]
        noise_target: 目标噪声 [B, L, 3]
        mask: 序列掩码 [B, L]
        weight: 正则化权重
        
    Returns:
        正则化损失
    """
    residual = noise_pred - noise_target
    
    # 1. 范数保持：预测和目标的范数应该接近
    pred_norm = torch.sqrt((noise_pred ** 2).sum(dim=-1) + 1e-8)  # [B, L]
    target_norm = torch.sqrt((noise_target ** 2).sum(dim=-1) + 1e-8)  # [B, L]
    norm_ratio = pred_norm / (target_norm + 1e-8)
    norm_loss = ((norm_ratio - 1.0) ** 2 * mask).sum() / (mask.sum() + 1e-8)
    
    # 2. 幅度约束：残差不应过大
    residual_norm = (residual ** 2).sum(dim=-1)  # [B, L]
    mag_loss = (residual_norm * mask).sum() / (mask.sum() + 1e-8)
    
    return weight * (0.5 * norm_loss + 0.5 * mag_loss)
