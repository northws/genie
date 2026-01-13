"""
Adaptive Configuration Utilities

This module provides utilities for dynamically adjusting model configurations
based on sequence length to enable efficient long-sequence training.

Key Features:
1. Adaptive mHC expansion rates (降低长序列的 expansion)
2. Dynamic batch sizes (保持恒定内存占用)
3. Adaptive factorization ranks (长序列使用低 rank)
4. Memory estimation utilities
"""

import torch
import math


class AdaptiveMHCConfig:
    """
    Dynamically adjust mHC configuration based on sequence length.

    Strategy:
        - Short sequences (<256): Use full mHC (expansion=4) for both structure and pair
        - Medium sequences (256-512): Reduce pair expansion, keep structure expansion
        - Long sequences (512-1024): Reduce structure expansion, disable pair mHC
        - Very long sequences (>1024): Minimal expansion for memory efficiency
    """

    @staticmethod
    def get_config(seq_len):
        """
        Get adaptive mHC configuration for given sequence length.

        Args:
            seq_len: Sequence length

        Returns:
            dict with keys:
                - structure_expansion: Expansion rate for structure module
                - pair_expansion: Expansion rate for pair module (1 = disabled)
                - sinkhorn_iters: Sinkhorn-Knopp iterations
                - use_pair_mhc: Whether to use mHC on pair features
        """
        if seq_len < 256:
            return {
                'structure_expansion': 4,
                'pair_expansion': 2,  # Reduced from 4 to save memory
                'sinkhorn_iters': 20,
                'sinkhorn_iters_inference': 5,
                'use_pair_mhc': True,
            }
        elif seq_len < 512:
            return {
                'structure_expansion': 4,
                'pair_expansion': 1,  # Disabled
                'sinkhorn_iters': 15,
                'sinkhorn_iters_inference': 3,
                'use_pair_mhc': False,
            }
        elif seq_len < 1024:
            return {
                'structure_expansion': 2,  # Reduced
                'pair_expansion': 1,
                'sinkhorn_iters': 10,
                'sinkhorn_iters_inference': 2,
                'use_pair_mhc': False,
            }
        else:  # >= 1024
            return {
                'structure_expansion': 2,  # Minimal
                'pair_expansion': 1,
                'sinkhorn_iters': 10,
                'sinkhorn_iters_inference': 2,
                'use_pair_mhc': False,
            }

    @staticmethod
    def print_config(seq_len):
        """Print adaptive configuration for given sequence length."""
        config = AdaptiveMHCConfig.get_config(seq_len)
        print(f"Adaptive mHC Config for L={seq_len}:")
        print(f"  Structure expansion: {config['structure_expansion']}x")
        print(f"  Pair expansion: {config['pair_expansion']}x {'(disabled)' if config['pair_expansion'] == 1 else ''}")
        print(f"  Sinkhorn iterations: {config['sinkhorn_iters']} (train) / {config['sinkhorn_iters_inference']} (inference)")
        print(f"  Use pair mHC: {config['use_pair_mhc']}")


class DynamicBatchSize:
    """
    Dynamically adjust batch size based on sequence length to maintain
    constant memory usage.

    Strategy:
        - Memory usage ∝ batch_size × L²
        - Keep: batch_size × L² ≈ constant
        - Therefore: batch_size = base_batch × (base_len / L)²
    """

    @staticmethod
    def compute_batch_size(seq_len, base_batch=32, base_len=128, min_batch=1, max_batch=64):
        """
        Compute batch size for given sequence length.

        Args:
            seq_len: Current sequence length
            base_batch: Batch size for base_len
            base_len: Base sequence length
            min_batch: Minimum batch size
            max_batch: Maximum batch size

        Returns:
            batch_size: Adjusted batch size

        Examples:
            L=128:  batch=32 (base)
            L=256:  batch=8  (1/4 of base, since L² is 4x)
            L=512:  batch=2
            L=1024: batch=1  (minimum, use gradient accumulation)
        """
        ratio = (base_len / seq_len) ** 2
        batch = int(base_batch * ratio)
        batch = max(min_batch, min(max_batch, batch))
        return batch

    @staticmethod
    def compute_accumulation_steps(seq_len, base_batch=32, base_len=128, effective_batch=32):
        """
        Compute gradient accumulation steps to maintain effective batch size.

        Args:
            seq_len: Current sequence length
            base_batch: Base batch size
            base_len: Base sequence length
            effective_batch: Target effective batch size

        Returns:
            accumulation_steps: Number of steps to accumulate

        Example:
            L=1024, batch=1, target_effective=32 → accumulate 32 steps
        """
        actual_batch = DynamicBatchSize.compute_batch_size(seq_len, base_batch, base_len)
        steps = max(1, effective_batch // actual_batch)
        return steps

    @staticmethod
    def print_batch_config(seq_len, base_batch=32, base_len=128):
        """Print batch configuration for given sequence length."""
        batch = DynamicBatchSize.compute_batch_size(seq_len, base_batch, base_len)
        accum_steps = DynamicBatchSize.compute_accumulation_steps(seq_len, base_batch, base_len, base_batch)
        effective_batch = batch * accum_steps

        print(f"Batch Config for L={seq_len}:")
        print(f"  Batch size: {batch}")
        print(f"  Accumulation steps: {accum_steps}")
        print(f"  Effective batch: {effective_batch}")


class AdaptiveFactorizationRank:
    """
    Dynamically adjust factorization rank for pair features based on
    sequence length and memory constraints.

    Strategy:
        - Short sequences: Higher rank for better quality
        - Long sequences: Lower rank for memory efficiency
    """

    @staticmethod
    def compute_rank(seq_len, base_rank=2, max_rank=8):
        """
        Compute factorization rank based on sequence length.

        Args:
            seq_len: Sequence length
            base_rank: Minimum rank (default: 2)
            max_rank: Maximum rank (default: 8)

        Returns:
            rank: Factorization rank

        Strategy:
            L < 256:  rank = 8 (high quality)
            256-512:  rank = 4
            512-1024: rank = 2 (memory efficient)
            > 1024:   rank = 2 (minimum)
        """
        if seq_len < 256:
            return max_rank
        elif seq_len < 512:
            return max(base_rank * 2, base_rank)
        else:
            return base_rank

    @staticmethod
    def print_rank(seq_len):
        """Print factorization rank for given sequence length."""
        rank = AdaptiveFactorizationRank.compute_rank(seq_len)
        print(f"Factorization Rank for L={seq_len}: {rank}")


class MemoryEstimator:
    """
    Estimate memory usage for different sequence lengths and configurations.
    """

    @staticmethod
    def estimate_pair_memory(seq_len, c_p=128, use_factorization=False, rank=2, dtype_bytes=4):
        """
        Estimate pair feature memory usage.

        Args:
            seq_len: Sequence length
            c_p: Pair feature dimension
            use_factorization: Whether using factorized pairs
            rank: Factorization rank
            dtype_bytes: Bytes per element (4 for FP32, 2 for FP16/BF16)

        Returns:
            memory_mb: Memory usage in MB
        """
        if use_factorization:
            # Factorized: [2, L, rank, C]
            memory = 2 * seq_len * rank * c_p * dtype_bytes
        else:
            # Full: [L, L, C]
            memory = seq_len * seq_len * c_p * dtype_bytes

        return memory / (1024 ** 2)  # Convert to MB

    @staticmethod
    def estimate_total_memory(seq_len, batch_size, c_s=128, c_p=128, use_factorization=False, use_mhc=False, mhc_expansion=4):
        """
        Estimate total GPU memory usage.

        Args:
            seq_len: Sequence length
            batch_size: Batch size
            c_s: Single feature dimension
            c_p: Pair feature dimension
            use_factorization: Using factorized pairs
            use_mhc: Using mHC
            mhc_expansion: mHC expansion rate

        Returns:
            dict with memory breakdown
        """
        dtype_bytes = 4  # FP32

        # Single features: [B, L, C]
        single_mem = batch_size * seq_len * c_s * dtype_bytes / (1024 ** 2)

        # Single features with mHC: [B, L, n, C]
        if use_mhc:
            single_mem *= mhc_expansion

        # Pair features
        pair_mem = batch_size * MemoryEstimator.estimate_pair_memory(
            seq_len, c_p, use_factorization, rank=2, dtype_bytes=dtype_bytes
        )

        # Structure module (IPA activations)
        # Approximate: ~3x single feature size
        structure_mem = single_mem * 3

        # Gradients (approximately 2x parameters + activations)
        activation_mem = single_mem + pair_mem + structure_mem
        gradient_mem = activation_mem * 2

        total_mem = activation_mem + gradient_mem

        return {
            'single': single_mem,
            'pair': pair_mem,
            'structure': structure_mem,
            'activations': activation_mem,
            'gradients': gradient_mem,
            'total': total_mem,
        }

    @staticmethod
    def print_memory_comparison(seq_lengths=[128, 256, 512, 1024], batch_size=32, base_len=128):
        """
        Print memory comparison for different configurations.
        """
        print("=" * 80)
        print("Memory Usage Comparison (MB per batch)")
        print("=" * 80)
        print(f"{'Length':<10} {'Batch':<10} {'Standard':<15} {'Factorized':<15} {'Reduction':<15}")
        print("-" * 80)

        for seq_len in seq_lengths:
            # Compute dynamic batch size
            batch = DynamicBatchSize.compute_batch_size(seq_len, batch_size, base_len)

            # Standard (full pairs)
            mem_standard = MemoryEstimator.estimate_total_memory(
                seq_len, batch, use_factorization=False, use_mhc=False
            )['total']

            # Factorized pairs
            mem_factorized = MemoryEstimator.estimate_total_memory(
                seq_len, batch, use_factorization=True, use_mhc=False
            )['total']

            reduction = mem_standard / mem_factorized if mem_factorized > 0 else 0

            print(f"{seq_len:<10} {batch:<10} {mem_standard:<15.1f} {mem_factorized:<15.1f} {reduction:<15.1f}x")

        print("=" * 80)


def print_adaptive_configs():
    """
    Print all adaptive configurations for different sequence lengths.
    """
    print("=" * 80)
    print("Adaptive Configurations for Long Sequences")
    print("=" * 80)
    print()

    lengths = [128, 256, 384, 512, 768, 1024, 1536, 2048]

    for seq_len in lengths:
        print(f"{'=' * 80}")
        print(f"Sequence Length: {seq_len}")
        print(f"{'=' * 80}")

        # mHC Config
        AdaptiveMHCConfig.print_config(seq_len)
        print()

        # Batch Config
        DynamicBatchSize.print_batch_config(seq_len, base_batch=32, base_len=128)
        print()

        # Factorization Rank
        AdaptiveFactorizationRank.print_rank(seq_len)
        print()

        # Memory Estimate
        batch = DynamicBatchSize.compute_batch_size(seq_len, base_batch=32, base_len=128)
        mem = MemoryEstimator.estimate_total_memory(
            seq_len, batch, use_factorization=True, use_mhc=True, mhc_expansion=4
        )
        print(f"Memory Estimate:")
        print(f"  Activations: {mem['activations']:.1f} MB")
        print(f"  Total (with gradients): {mem['total']:.1f} MB")
        print()


if __name__ == "__main__":
    print_adaptive_configs()
    print()
    print()
    MemoryEstimator.print_memory_comparison()
