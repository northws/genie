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


# ============================================================
# 长序列训练梯度稳定化函数
# ============================================================

def sequence_length_adaptive_norm_loss(
    noise_pred: torch.Tensor,
    noise_target: torch.Tensor,
    mask: torch.Tensor,
    base_seq_len: int = 128,
) -> torch.Tensor:
    """
    序列长度自适应范数损失 - 防止长序列梯度爆炸。
    
    长序列的累积误差更大，需要更强的范数约束。
    损失会根据序列长度自适应调整。
    
    Args:
        noise_pred: 预测噪声 [B, L, 3]
        noise_target: 目标噪声 [B, L, 3]
        mask: 序列掩码 [B, L]
        base_seq_len: 基准序列长度（用于归一化）
    
    Returns:
        自适应范数损失
    """
    B, L, _ = noise_pred.shape
    
    # 序列长度因子：长序列需要更强的约束
    seq_len_factor = (L / base_seq_len) ** 0.5
    
    # 计算每个位置的范数
    pred_norm = torch.sqrt((noise_pred ** 2).sum(dim=-1) + 1e-8)  # [B, L]
    target_norm = torch.sqrt((noise_target ** 2).sum(dim=-1) + 1e-8)  # [B, L]
    
    # 范数比例约束（更严格的约束用于长序列）
    norm_ratio = pred_norm / (target_norm + 1e-8)
    
    # 对于长序列，允许的范数偏差更小
    tolerance = max(0.1, 0.2 / seq_len_factor)
    deviation = F.relu(torch.abs(norm_ratio - 1.0) - tolerance)
    
    norm_loss = (deviation ** 2 * mask).sum() / (mask.sum() + 1e-8)
    
    return norm_loss * seq_len_factor


def gradient_magnitude_clipping_loss(
    noise_pred: torch.Tensor,
    noise_target: torch.Tensor,
    mask: torch.Tensor,
    max_magnitude: float = 10.0,
) -> torch.Tensor:
    """
    梯度幅度软裁剪损失 - 防止预测值过大导致梯度爆炸。
    
    软约束预测噪声的幅度不超过阈值。
    
    Args:
        noise_pred: 预测噪声 [B, L, 3]
        noise_target: 目标噪声 [B, L, 3]
        mask: 序列掩码 [B, L]
        max_magnitude: 最大允许幅度
    
    Returns:
        幅度裁剪损失
    """
    pred_magnitude = torch.sqrt((noise_pred ** 2).sum(dim=-1) + 1e-8)  # [B, L]
    
    # 惩罚超过阈值的预测
    excess = F.relu(pred_magnitude - max_magnitude)
    clip_loss = (excess ** 2 * mask).sum() / (mask.sum() + 1e-8)
    
    return clip_loss


def local_consistency_loss(
    noise_pred: torch.Tensor,
    mask: torch.Tensor,
    window_size: int = 5,
) -> torch.Tensor:
    """
    局部一致性损失 - 鼓励预测在局部窗口内平滑。
    
    对于蛋白质结构，相邻残基的预测不应有剧烈变化。
    这有助于防止长序列中的梯度爆炸。
    
    Args:
        noise_pred: 预测噪声 [B, L, 3]
        mask: 序列掩码 [B, L]
        window_size: 局部窗口大小
    
    Returns:
        局部一致性损失
    """
    B, L, C = noise_pred.shape
    
    total_loss = torch.tensor(0.0, device=noise_pred.device)
    count = 0
    
    # 计算相邻位置的差异
    for offset in range(1, min(window_size, L)):
        diff = noise_pred[:, offset:, :] - noise_pred[:, :-offset, :]
        diff_mask = mask[:, offset:] * mask[:, :-offset]
        
        # 差异平方和，按偏移距离加权（距离越远权重越小）
        weight = 1.0 / offset
        diff_loss = ((diff ** 2).sum(dim=-1) * diff_mask).sum()
        diff_count = diff_mask.sum()
        
        if diff_count > 0:
            total_loss = total_loss + weight * diff_loss / diff_count
            count += 1
    
    if count > 0:
        return total_loss / count
    return total_loss


def spectral_norm_regularization(
    noise_pred: torch.Tensor,
    noise_target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    谱范数正则化 - mHC 核心思想的另一种实现。
    
    mHC 的核心是保持谱半径为 1。这个损失函数通过约束
    预测和目标之间的关系来间接实现这一点。
    
    Args:
        noise_pred: 预测噪声 [B, L, 3]
        noise_target: 目标噪声 [B, L, 3]
        mask: 序列掩码 [B, L]
    
    Returns:
        谱范数正则化损失
    """
    B, L, C = noise_pred.shape
    mask_expanded = mask.unsqueeze(-1)  # [B, L, 1]
    
    # 将预测和目标视为向量变换
    # 理想情况下：noise_pred ≈ I @ noise_target（恒等映射）
    
    # 计算 "能量" 比
    pred_energy = ((noise_pred ** 2) * mask_expanded).sum()
    target_energy = ((noise_target ** 2) * mask_expanded).sum()
    
    # 谱范数约束：能量比应接近 1
    energy_ratio = pred_energy / (target_energy + 1e-8)
    spectral_loss = (energy_ratio - 1.0) ** 2
    
    # 同时约束每个 batch 的能量比
    batch_pred_energy = ((noise_pred ** 2) * mask_expanded).sum(dim=(1, 2))
    batch_target_energy = ((noise_target ** 2) * mask_expanded).sum(dim=(1, 2))
    batch_ratio = batch_pred_energy / (batch_target_energy + 1e-8)
    batch_loss = ((batch_ratio - 1.0) ** 2).mean()
    
    return 0.5 * spectral_loss + 0.5 * batch_loss


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


class AdvancedMHCLoss(nn.Module):
    """
    高级 mHC 损失模块 - 整合所有 mHC 相关的正则化损失函数。
    
    该模块整合了 10 个损失函数，用于稳定训练并防止梯度爆炸/消失：
    
    基础 mHC 损失（6 个核心函数，始终计算）：
    1. compute_mhc_regularization: 基础 mHC 正则化 - 范数保持和幅度约束
    2. gradient_norm_preservation_loss: 范数保持损失 - 防止梯度爆炸/消失
    3. gradient_flow_loss: 梯度流平滑损失 - 空间正则化
    4. residual_balance_loss: 残差平衡损失 - 确保残差分支和主分支贡献平衡
    5. representation_stability_loss: 特征稳定性损失 - 限制激活值范围
    6. doubly_stochastic_penalty: 双随机惩罚 - 约束隐式权重矩阵（仅对 L<=256 计算）
    
    长序列训练损失（4 个额外函数，long_seq_mode=True 时计算）：
    7. sequence_length_adaptive_norm_loss: 序列长度自适应范数损失
    8. gradient_magnitude_clipping_loss: 梯度幅度软裁剪损失
    9. local_consistency_loss: 局部一致性损失
    10. spectral_norm_regularization: 谱范数正则化
    
    使用示例:
        adv_mhc_loss = AdvancedMHCLoss(
            # 基础损失权重
            balance_weight=0.1,
            norm_weight=0.1,
            stability_weight=0.05,
            flow_weight=0.01,
            ds_penalty_weight=0.01,
            base_reg_weight=0.01,
            target_residual_ratio=0.5,
            # 长序列训练参数
            long_seq_mode=True,
            adaptive_norm_weight=0.05,
            magnitude_clip_weight=0.02,
            local_consistency_weight=0.01,
            spectral_norm_weight=0.02,
        )
        
        # 在训练循环中
        loss_dict = adv_mhc_loss.compute_diffusion_loss(
            noise_pred=noise_pred,
            noise_target=noise_target,
            mask=mask,
        )
        total_loss = loss_dict['total_loss']
    
    WandB 监控指标（当 useAdvMHCLoss=True 时）：
    - train/adv_mhc_total: 高级 mHC 总损失
    - train/adv_mhc_base_reg: 基础正则化损失
    - train/adv_mhc_norm: 范数保持损失
    - train/adv_mhc_flow: 梯度流平滑损失
    - train/adv_mhc_balance: 残差平衡损失
    - train/adv_mhc_stability: 特征稳定性损失
    - train/adv_mhc_ds_penalty: 双随机惩罚损失
    - train/adv_mhc_adaptive_norm: 自适应范数损失（长序列模式）
    - train/adv_mhc_magnitude_clip: 幅度裁剪损失（长序列模式）
    - train/adv_mhc_local_consistency: 局部一致性损失（长序列模式）
    - train/adv_mhc_spectral_norm: 谱范数损失（长序列模式）
    """
    
    def __init__(
        self,
        balance_weight: float = 0.1,
        norm_weight: float = 0.1,
        stability_weight: float = 0.05,
        flow_weight: float = 0.01,
        ds_penalty_weight: float = 0.0,  # 默认禁用，因为需要权重矩阵
        base_reg_weight: float = 0.01,
        target_residual_ratio: float = 0.5,
        # 长序列训练参数
        long_seq_mode: bool = False,
        adaptive_norm_weight: float = 0.05,
        magnitude_clip_weight: float = 0.02,
        local_consistency_weight: float = 0.01,
        spectral_norm_weight: float = 0.02,
        base_seq_len: int = 128,
        max_magnitude: float = 10.0,
        consistency_window: int = 5,
    ):
        """
        初始化高级 mHC 损失模块。
        
        Args:
            balance_weight: 残差平衡损失权重 (residual_balance_loss)
            norm_weight: 范数保持损失权重 (gradient_norm_preservation_loss)
            stability_weight: 特征稳定性损失权重 (representation_stability_loss)
            flow_weight: 梯度流平滑损失权重 (gradient_flow_loss)
            ds_penalty_weight: 双随机惩罚权重 (doubly_stochastic_penalty)
            base_reg_weight: 基础正则化权重 (compute_mhc_regularization)
            target_residual_ratio: 目标残差比例，用于平衡损失
            
            长序列训练参数（当 long_seq_mode=True 时启用）:
            long_seq_mode: 是否启用长序列训练模式
            adaptive_norm_weight: 序列长度自适应范数损失权重
            magnitude_clip_weight: 梯度幅度软裁剪损失权重
            local_consistency_weight: 局部一致性损失权重
            spectral_norm_weight: 谱范数正则化权重
            base_seq_len: 基准序列长度（用于自适应计算）
            max_magnitude: 最大允许预测幅度
            consistency_window: 局部一致性窗口大小
        """
        super().__init__()
        self.balance_weight = balance_weight
        self.norm_weight = norm_weight
        self.stability_weight = stability_weight
        self.flow_weight = flow_weight
        self.ds_penalty_weight = ds_penalty_weight
        self.base_reg_weight = base_reg_weight
        self.target_residual_ratio = target_residual_ratio
        
        # 长序列训练参数
        self.long_seq_mode = long_seq_mode
        self.adaptive_norm_weight = adaptive_norm_weight
        self.magnitude_clip_weight = magnitude_clip_weight
        self.local_consistency_weight = local_consistency_weight
        self.spectral_norm_weight = spectral_norm_weight
        self.base_seq_len = base_seq_len
        self.max_magnitude = max_magnitude
        self.consistency_window = consistency_window
        
    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor,
        mask: torch.Tensor,
        x_input: torch.Tensor = None,
        x_output: torch.Tensor = None,
        residual_output: torch.Tensor = None,
        s_hidden: torch.Tensor = None,
        pred_trans: torch.Tensor = None,
        target_trans: torch.Tensor = None,
        weight_matrix: torch.Tensor = None,
    ) -> dict:
        """
        计算所有 mHC 相关损失。
        
        Args:
            noise_pred: 预测噪声 [B, L, 3]
            noise_target: 目标噪声 [B, L, 3]
            mask: 序列掩码 [B, L]
            x_input: 层输入 [B, L, C]，用于残差平衡和范数保持损失
            x_output: 层输出 [B, L, C]，用于残差平衡和范数保持损失
            residual_output: 残差分支输出 [B, L, C]，用于残差平衡损失
            s_hidden: 单表示隐藏状态 [B, L, C]，用于特征稳定性损失
            pred_trans: 预测平移 [B, L, 3]，用于梯度流平滑损失
            target_trans: 目标平移 [B, L, 3]，用于梯度流平滑损失
            weight_matrix: 权重矩阵 [n, n] 或 [B, n, n]，用于双随机惩罚
        
        Returns:
            包含各项损失及总损失的字典:
            {
                'total_loss': 总损失,
                'base_reg_loss': 基础正则化损失,
                'balance_loss': 残差平衡损失,
                'norm_loss': 范数保持损失,
                'stability_loss': 特征稳定性损失,
                'flow_loss': 梯度流平滑损失,
                'ds_penalty_loss': 双随机惩罚损失,
            }
        """
        device = mask.device
        loss_dict = {
            'total_loss': torch.tensor(0.0, device=device),
            'base_reg_loss': torch.tensor(0.0, device=device),
            'balance_loss': torch.tensor(0.0, device=device),
            'norm_loss': torch.tensor(0.0, device=device),
            'stability_loss': torch.tensor(0.0, device=device),
            'flow_loss': torch.tensor(0.0, device=device),
            'ds_penalty_loss': torch.tensor(0.0, device=device),
        }
        
        # 1. 基础 mHC 正则化（总是计算）
        if self.base_reg_weight > 0:
            base_reg = compute_mhc_regularization(
                noise_pred, noise_target, mask, weight=self.base_reg_weight
            )
            loss_dict['base_reg_loss'] = base_reg
            loss_dict['total_loss'] = loss_dict['total_loss'] + base_reg
        
        # 2. 残差平衡损失（需要 x_input, x_output, residual_output）
        if self.balance_weight > 0 and x_input is not None and x_output is not None:
            if residual_output is None:
                # 如果没有提供 residual_output，使用 noise_pred 作为替代
                residual_output = noise_pred
            balance_loss = self.balance_weight * residual_balance_loss(
                x_input, x_output, residual_output, mask, self.target_residual_ratio
            )
            loss_dict['balance_loss'] = balance_loss
            loss_dict['total_loss'] = loss_dict['total_loss'] + balance_loss
        
        # 3. 范数保持损失（需要 x_input, x_output）
        if self.norm_weight > 0 and x_input is not None and x_output is not None:
            norm_loss = self.norm_weight * gradient_norm_preservation_loss(
                x_input, x_output, mask
            )
            loss_dict['norm_loss'] = norm_loss
            loss_dict['total_loss'] = loss_dict['total_loss'] + norm_loss
        
        # 4. 特征稳定性损失（需要 s_hidden）
        if self.stability_weight > 0 and s_hidden is not None:
            stability_loss = self.stability_weight * representation_stability_loss(
                s_hidden, mask
            )
            loss_dict['stability_loss'] = stability_loss
            loss_dict['total_loss'] = loss_dict['total_loss'] + stability_loss
        
        # 5. 梯度流平滑损失（需要 pred_trans, target_trans）
        if self.flow_weight > 0:
            # 如果没有提供 pred_trans/target_trans，使用 noise_pred/noise_target
            _pred_trans = pred_trans if pred_trans is not None else noise_pred
            _target_trans = target_trans if target_trans is not None else noise_target
            flow_loss = gradient_flow_loss(
                _pred_trans, _target_trans, mask, alpha=self.flow_weight
            )
            loss_dict['flow_loss'] = flow_loss
            loss_dict['total_loss'] = loss_dict['total_loss'] + flow_loss
        
        # 6. 双随机惩罚（需要 weight_matrix）
        if self.ds_penalty_weight > 0 and weight_matrix is not None:
            ds_loss = self.ds_penalty_weight * doubly_stochastic_penalty(weight_matrix)
            loss_dict['ds_penalty_loss'] = ds_loss
            loss_dict['total_loss'] = loss_dict['total_loss'] + ds_loss
        
        return loss_dict
    
    def compute_diffusion_loss(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict:
        """
        完整版：使用所有 mHC 损失函数进行扩散训练正则化。
        
        这个方法整合了所有 mHC 损失函数：
        
        基础损失（始终计算）：
        1. compute_mhc_regularization: 基础 mHC 正则化
        2. gradient_norm_preservation_loss: 范数保持损失 - 防止梯度爆炸/消失
        3. gradient_flow_loss: 梯度流平滑损失 - 空间正则化
        4. residual_balance_loss: 残差平衡损失 - 确保残差分支和主分支贡献平衡
        5. representation_stability_loss: 特征稳定性损失 - 限制激活值范围
        6. doubly_stochastic_penalty: 双随机惩罚 - 约束隐式权重矩阵
        
        长序列损失（long_seq_mode=True 时计算）：
        7. sequence_length_adaptive_norm_loss: 序列长度自适应范数损失
        8. gradient_magnitude_clipping_loss: 梯度幅度软裁剪损失
        9. local_consistency_loss: 局部一致性损失
        10. spectral_norm_regularization: 谱范数正则化
        
        注意：由于扩散训练中没有完整的模型内部状态（如 x_input, x_output），
        使用 noise_target 和 noise_pred 作为代理来计算这些损失。
        
        Args:
            noise_pred: 预测噪声 [B, L, 3]
            noise_target: 目标噪声 [B, L, 3]
            mask: 序列掩码 [B, L]
        
        Returns:
            包含各项损失及总损失的字典
        """
        device = mask.device
        loss_dict = {
            'total_loss': torch.tensor(0.0, device=device),
            # 基础损失
            'base_reg_loss': torch.tensor(0.0, device=device),
            'norm_preservation_loss': torch.tensor(0.0, device=device),
            'flow_loss': torch.tensor(0.0, device=device),
            'balance_loss': torch.tensor(0.0, device=device),
            'stability_loss': torch.tensor(0.0, device=device),
            'ds_penalty_loss': torch.tensor(0.0, device=device),
            # 长序列训练损失
            'adaptive_norm_loss': torch.tensor(0.0, device=device),
            'magnitude_clip_loss': torch.tensor(0.0, device=device),
            'local_consistency_loss': torch.tensor(0.0, device=device),
            'spectral_norm_loss': torch.tensor(0.0, device=device),
        }
        
        # 计算残差（用于多个损失函数）
        residual = noise_pred - noise_target
        
        # ============================================================
        # 基础 mHC 损失（所有 6 个核心函数）
        # ============================================================
        
        # 1. 基础 mHC 正则化 (compute_mhc_regularization)
        if self.base_reg_weight > 0:
            base_reg = compute_mhc_regularization(
                noise_pred, noise_target, mask, weight=self.base_reg_weight
            )
            loss_dict['base_reg_loss'] = base_reg
            loss_dict['total_loss'] = loss_dict['total_loss'] + base_reg
        
        # 2. 范数保持损失 (gradient_norm_preservation_loss)
        #    使用 noise_target 作为"输入"，noise_pred 作为"输出"
        if self.norm_weight > 0:
            norm_loss = self.norm_weight * gradient_norm_preservation_loss(
                noise_target,  # 作为层输入
                noise_pred,    # 作为层输出
                mask
            )
            loss_dict['norm_preservation_loss'] = norm_loss
            loss_dict['total_loss'] = loss_dict['total_loss'] + norm_loss
        
        # 3. 梯度流平滑损失 (gradient_flow_loss)
        if self.flow_weight > 0:
            flow_loss = gradient_flow_loss(
                noise_pred, noise_target, mask, alpha=self.flow_weight
            )
            loss_dict['flow_loss'] = flow_loss
            loss_dict['total_loss'] = loss_dict['total_loss'] + flow_loss
        
        # 4. 残差平衡损失 (residual_balance_loss)
        #    使用 noise_target 作为"输入"，noise_pred 作为"输出"，residual 作为"残差分支输出"
        if self.balance_weight > 0:
            balance_loss = self.balance_weight * residual_balance_loss(
                x_input=noise_target,      # 层输入（目标噪声）
                x_output=noise_pred,       # 层输出（预测噪声）
                residual_output=residual,  # 残差分支输出
                mask=mask,
                target_ratio=self.target_residual_ratio
            )
            loss_dict['balance_loss'] = balance_loss
            loss_dict['total_loss'] = loss_dict['total_loss'] + balance_loss
        
        # 5. 特征稳定性损失 (representation_stability_loss)
        #    使用 noise_pred 作为"隐藏表示"
        if self.stability_weight > 0:
            stability_loss = self.stability_weight * representation_stability_loss(
                s_hidden=noise_pred,  # 预测噪声作为隐藏表示的代理
                mask=mask
            )
            loss_dict['stability_loss'] = stability_loss
            loss_dict['total_loss'] = loss_dict['total_loss'] + stability_loss
        
        # 6. 双随机惩罚 (doubly_stochastic_penalty)
        #    构造隐式权重矩阵：基于 noise_pred 和 noise_target 的相关性
        if self.ds_penalty_weight > 0:
            # 构造隐式权重矩阵：W[i,j] 表示位置 i 和 j 的预测相关性
            # 使用归一化的点积作为权重
            B, L, C = noise_pred.shape
            
            # 只对较短序列计算（避免 O(L²) 内存问题）
            if L <= 256:
                # 归一化 noise_pred
                pred_norm = noise_pred / (torch.norm(noise_pred, dim=-1, keepdim=True) + 1e-8)
                # 计算相似度矩阵 [B, L, L]
                similarity = torch.bmm(pred_norm, pred_norm.transpose(1, 2))
                # 将相似度转换为 [0, 1] 范围并归一化
                similarity = (similarity + 1.0) / 2.0  # 从 [-1, 1] 转换到 [0, 1]
                # 应用 mask
                mask_2d = mask.unsqueeze(-1) * mask.unsqueeze(-2)  # [B, L, L]
                similarity = similarity * mask_2d
                
                # 对每个 batch 计算双随机惩罚
                ds_loss = torch.tensor(0.0, device=device)
                for b in range(B):
                    valid_len = int(mask[b].sum().item())
                    if valid_len > 1:
                        W_b = similarity[b, :valid_len, :valid_len]
                        # 对 W_b 进行行列归一化，使其更接近双随机矩阵
                        W_b = W_b / (W_b.sum(dim=-1, keepdim=True) + 1e-8)
                        # 计算与理想双随机矩阵的差距（行和列和都应为1）
                        row_sums = W_b.sum(dim=-1)
                        col_sums = W_b.sum(dim=-2)
                        row_loss = ((row_sums - 1.0) ** 2).mean()
                        col_loss = ((col_sums - 1.0) ** 2).mean()
                        ds_loss = ds_loss + (row_loss + col_loss)
                ds_loss = self.ds_penalty_weight * ds_loss / B
                
                loss_dict['ds_penalty_loss'] = ds_loss
                loss_dict['total_loss'] = loss_dict['total_loss'] + ds_loss
        
        # ============================================================
        # 长序列训练特有损失（防止梯度爆炸）
        # ============================================================
        if self.long_seq_mode:
            # 7. 序列长度自适应范数损失
            if self.adaptive_norm_weight > 0:
                adaptive_loss = self.adaptive_norm_weight * sequence_length_adaptive_norm_loss(
                    noise_pred, noise_target, mask, self.base_seq_len
                )
                loss_dict['adaptive_norm_loss'] = adaptive_loss
                loss_dict['total_loss'] = loss_dict['total_loss'] + adaptive_loss
            
            # 8. 梯度幅度软裁剪损失
            if self.magnitude_clip_weight > 0:
                clip_loss = self.magnitude_clip_weight * gradient_magnitude_clipping_loss(
                    noise_pred, noise_target, mask, self.max_magnitude
                )
                loss_dict['magnitude_clip_loss'] = clip_loss
                loss_dict['total_loss'] = loss_dict['total_loss'] + clip_loss
            
            # 9. 局部一致性损失
            if self.local_consistency_weight > 0:
                consistency_loss = self.local_consistency_weight * local_consistency_loss(
                    noise_pred, mask, self.consistency_window
                )
                loss_dict['local_consistency_loss'] = consistency_loss
                loss_dict['total_loss'] = loss_dict['total_loss'] + consistency_loss
            
            # 10. 谱范数正则化
            if self.spectral_norm_weight > 0:
                spectral_loss = self.spectral_norm_weight * spectral_norm_regularization(
                    noise_pred, noise_target, mask
                )
                loss_dict['spectral_norm_loss'] = spectral_loss
                loss_dict['total_loss'] = loss_dict['total_loss'] + spectral_loss
        
        return loss_dict
