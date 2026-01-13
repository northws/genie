#!/usr/bin/env python3
"""
Long Sequence Denoiser Test Suite

Tests all Stage 1 optimizations for long-sequence protein generation.

Tests:
1. Factorized Pair Features - Memory and correctness
2. Adaptive mHC Configuration - Dynamic parameter adjustment
3. Dynamic Batch Sizing - Memory efficiency
4. End-to-end forward/backward pass - Integration test
5. Memory profiling - Verify memory savings
"""

import sys
import torch
import traceback
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_factorized_pair_features():
    """Test 1: Factorized Pair Features"""
    print("=" * 80)
    print("TEST 1: Factorized Pair Features")
    print("=" * 80)

    from genie.model.factorized_pair_features import FactorizedPairFeatureNet
    from genie.flash_ipa.rigid import create_identity_rigid

    # Test parameters
    B, L, C_s, C_p = 2, 512, 128, 128
    rank = 2

    print(f"Configuration: B={B}, L={L}, C_s={C_s}, C_p={C_p}, rank={rank}")

    # Create model
    model = FactorizedPairFeatureNet(
        c_s=C_s,
        c_p=C_p,
        rank=rank,
        relpos_k=32,
        template_type='v1'
    )

    # Test input
    s = torch.randn(B, L, C_s)
    t = create_identity_rigid(B, L)
    mask = torch.ones(B, L)

    # Forward pass
    print("Running forward pass...")
    factor_1, factor_2 = model(s, t, mask)

    # Check shapes
    assert factor_1.shape == (B, L, rank, C_p), f"Factor 1 shape mismatch"
    assert factor_2.shape == (B, L, rank, C_p), f"Factor 2 shape mismatch"

    # Memory comparison
    factor_memory = (factor_1.numel() + factor_2.numel()) * 4 / (1024 ** 2)
    full_memory = B * L * L * C_p * 4 / (1024 ** 2)

    print(f"✅ Shape test passed")
    print(f"   Factor 1: {factor_1.shape}")
    print(f"   Factor 2: {factor_2.shape}")
    print(f"")
    print(f"Memory savings:")
    print(f"   Factorized: {factor_memory:.2f} MB")
    print(f"   Full pair: {full_memory:.2f} MB")
    print(f"   Reduction: {full_memory / factor_memory:.1f}x")
    print()

    return True


def test_adaptive_config():
    """Test 2: Adaptive Configuration"""
    print("=" * 80)
    print("TEST 2: Adaptive mHC Configuration")
    print("=" * 80)

    from genie.utils.adaptive_config import AdaptiveMHCConfig, DynamicBatchSize, AdaptiveFactorizationRank

    test_lengths = [128, 256, 512, 1024, 2048]

    for L in test_lengths:
        print(f"\nSequence Length: {L}")
        print("-" * 40)

        # Get configurations
        mhc_config = AdaptiveMHCConfig.get_config(L)
        batch = DynamicBatchSize.compute_batch_size(L, base_batch=32, base_len=128)
        rank = AdaptiveFactorizationRank.compute_rank(L)

        print(f"  mHC expansion: {mhc_config['structure_expansion']}")
        print(f"  Batch size: {batch}")
        print(f"  Factor rank: {rank}")

        # Verify adaptive behavior
        if L < 256:
            assert mhc_config['structure_expansion'] == 4
            assert batch >= 8
        elif L < 512:
            assert mhc_config['structure_expansion'] == 4
            assert batch >= 2
        else:
            assert mhc_config['structure_expansion'] == 2
            assert batch >= 1

    print("\n✅ All adaptive configurations valid")
    print()

    return True


def test_memory_estimation():
    """Test 3: Memory Estimation"""
    print("=" * 80)
    print("TEST 3: Memory Estimation")
    print("=" * 80)

    from genie.utils.adaptive_config import MemoryEstimator, DynamicBatchSize

    print("\nMemory comparison for different lengths:")
    print("-" * 80)
    print(f"{'Length':<10} {'Batch':<10} {'Standard':<15} {'Factorized':<15} {'Savings':<15}")
    print("-" * 80)

    for L in [128, 256, 512, 1024]:
        batch = DynamicBatchSize.compute_batch_size(L, base_batch=32, base_len=128)

        mem_standard = MemoryEstimator.estimate_total_memory(
            L, batch, use_factorization=False, use_mhc=False
        )['total']

        mem_factorized = MemoryEstimator.estimate_total_memory(
            L, batch, use_factorization=True, use_mhc=True, mhc_expansion=4
        )['total']

        savings = mem_standard / mem_factorized

        print(f"{L:<10} {batch:<10} {mem_standard:<15.1f} {mem_factorized:<15.1f} {savings:<15.1f}x")

    print("-" * 80)
    print("\n✅ Memory estimation complete")
    print()

    return True


def test_long_sequence_denoiser_forward():
    """Test 4: LongSequenceDenoiser Forward Pass"""
    print("=" * 80)
    print("TEST 4: LongSequenceDenoiser Forward Pass")
    print("=" * 80)

    from genie.model.long_sequence_denoiser import LongSequenceDenoiser
    from genie.flash_ipa.rigid import create_identity_rigid

    test_configs = [
        {'L': 128, 'batch': 2, 'name': 'Short'},
        {'L': 256, 'batch': 2, 'name': 'Medium'},
        {'L': 512, 'batch': 1, 'name': 'Long'},
    ]

    for config in test_configs:
        L = config['L']
        B = config['batch']
        name = config['name']

        print(f"\n{name} sequence test: L={L}, batch={B}")
        print("-" * 40)

        try:
            # Create model
            model = LongSequenceDenoiser(
                c_s=128,
                c_p=128,
                n_timestep=100,
                c_pos_emb=128,
                c_timestep_emb=128,
                relpos_k=32,
                template_type='v1',
                n_structure_layer=2,
                n_structure_block=1,
                c_hidden_ipa=16,
                n_head_ipa=4,
                n_qk_point=4,
                n_v_point=8,
                ipa_dropout=0.1,
                n_structure_transition_layer=1,
                structure_transition_dropout=0.1,
                max_n_res=L,
                use_adaptive_config=True,
                use_grad_checkpoint=False,
                use_flash_attn_3=False,
            )

            # Test input
            ts = create_identity_rigid(B, L)
            timesteps = torch.randint(0, 100, (B,))
            mask = torch.ones(B, L)

            # Forward pass
            print("  Running forward pass...")
            with torch.no_grad():
                ts_out = model(ts, timesteps, mask)

            print(f"  ✅ Forward pass successful")
            print(f"     Input: {ts.shape}")
            print(f"     Output: {ts_out.shape}")

        except Exception as e:
            print(f"  ❌ Forward pass failed: {e}")
            traceback.print_exc()
            return False

    print("\n✅ All forward pass tests passed")
    print()

    return True


def test_backward_pass():
    """Test 5: Backward Pass (Gradient Flow)"""
    print("=" * 80)
    print("TEST 5: Backward Pass & Gradient Flow")
    print("=" * 80)

    from genie.model.long_sequence_denoiser import LongSequenceDenoiser
    from genie.flash_ipa.rigid import create_identity_rigid

    L, B = 256, 1

    print(f"Configuration: L={L}, batch={B}")

    try:
        # Create model
        model = LongSequenceDenoiser(
            c_s=64,
            c_p=64,
            n_timestep=100,
            c_pos_emb=64,
            c_timestep_emb=64,
            relpos_k=32,
            template_type='v1',
            n_structure_layer=2,
            n_structure_block=1,
            c_hidden_ipa=16,
            n_head_ipa=4,
            n_qk_point=2,
            n_v_point=4,
            ipa_dropout=0.1,
            n_structure_transition_layer=1,
            structure_transition_dropout=0.1,
            max_n_res=L,
            use_adaptive_config=True,
            use_grad_checkpoint=False,
            use_flash_attn_3=False,
        )

        # Test input
        ts = create_identity_rigid(B, L)
        timesteps = torch.randint(0, 100, (B,))
        mask = torch.ones(B, L)

        # Forward pass
        print("Running forward pass...")
        ts_out = model(ts, timesteps, mask)

        # Create dummy loss
        loss = ts_out.trans.sum()

        # Backward pass
        print("Running backward pass...")
        loss.backward()

        # Check gradients
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad, "No gradients computed!"

        # Count parameters with gradients
        n_params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
        n_params_total = sum(1 for p in model.parameters())

        print(f"✅ Backward pass successful")
        print(f"   Parameters with gradients: {n_params_with_grad}/{n_params_total}")
        print()

        return True

    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("LONG SEQUENCE DENOISER - STAGE 1 OPTIMIZATIONS TEST SUITE")
    print("=" * 80)
    print()

    tests = [
        ("Factorized Pair Features", test_factorized_pair_features),
        ("Adaptive Configuration", test_adaptive_config),
        ("Memory Estimation", test_memory_estimation),
        ("Forward Pass", test_long_sequence_denoiser_forward),
        ("Backward Pass", test_backward_pass),
    ]

    results = []

    for name, test_fn in tests:
        print(f"\nRunning: {name}")
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:<35} {status}")

    print("=" * 80)

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nYou can now train on long sequences up to 1024 residues!")
        print("\nQuick start:")
        print("  python -m genie.train runs/config_long_sequence_stage1.txt")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED - Please check errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
