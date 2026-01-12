#!/usr/bin/env python3
"""
测试 mHC + Flash-IPA 组合模式的导入和基本功能

运行此脚本以验证新模块是否正确安装和工作。
"""

import sys
import torch

def test_imports():
    """测试所有必需的导入"""
    print("=" * 60)
    print("测试模块导入...")
    print("=" * 60)
    
    try:
        from genie.model.mhc_flash_structure_net import mHCFlashStructureLayer, mHCFlashStructureNet
        print("✅ mHCFlashStructureNet 导入成功")
    except Exception as e:
        print(f"❌ mHCFlashStructureNet 导入失败: {e}")
        return False
    
    try:
        from genie.model.mhc_flash_denoiser import mHCFlashDenoiser
        print("✅ mHCFlashDenoiser 导入成功")
    except Exception as e:
        print(f"❌ mHCFlashDenoiser 导入失败: {e}")
        return False
    
    try:
        from genie.diffusion.diffusion import Diffusion
        print("✅ Diffusion (更新版) 导入成功")
    except Exception as e:
        print(f"❌ Diffusion 导入失败: {e}")
        return False
    
    return True


def test_basic_functionality():
    """测试基本功能"""
    print("\n" + "=" * 60)
    print("测试基本功能...")
    print("=" * 60)
    
    try:
        from genie.model.mhc_flash_structure_net import mHCFlashStructureLayer
        from genie.flash_ipa.ipa import IPAConfig
        
        # 创建一个小型测试层
        layer = mHCFlashStructureLayer(
            c_s=64,
            c_p=64,
            c_hidden_ipa=16,
            n_head=4,
            n_qk_point=2,
            n_v_point=4,
            ipa_dropout=0.1,
            n_structure_transition_layer=1,
            structure_transition_dropout=0.1,
            max_n_res=128,
            z_factor_rank=2,
            k_neighbors=8,
            mhc_expansion_rate=4,
            mhc_sinkhorn_iters=10,
            mhc_alpha_init=0.01,
            use_grad_checkpoint=False,
            use_flash_attn_3=False,
            is_first_layer=True,
            is_last_layer=False,
        )
        print("✅ mHCFlashStructureLayer 实例化成功")
        
        # 检查参数数量
        num_params = sum(p.numel() for p in layer.parameters())
        print(f"   参数数量: {num_params:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ 基本功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_parsing():
    """测试配置文件解析"""
    print("\n" + "=" * 60)
    print("测试配置解析...")
    print("=" * 60)
    
    try:
        from genie.config import Config
        import os
        
        # 检查示例配置是否存在
        config_path = "runs/config_mhc_flash_combined.txt"
        if os.path.exists(config_path):
            config = Config(config_path)
            
            # 验证关键配置
            use_mhc = config.training.get('use_mhc_mode', False)
            use_flash = config.training.get('use_flash_mode', False)
            
            print(f"   use_mhc_mode: {use_mhc}")
            print(f"   use_flash_mode: {use_flash}")
            
            if use_mhc and use_flash:
                print("✅ 配置解析成功 - mHC + Flash-IPA 同时启用")
                return True
            else:
                print("⚠️  配置文件未同时启用两种模式")
                return True
        else:
            print(f"⚠️  示例配置文件不存在: {config_path}")
            return True
            
    except Exception as e:
        print(f"❌ 配置解析失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_flash_attention():
    """检查 Flash Attention 可用性"""
    print("\n" + "=" * 60)
    print("检查 Flash Attention 可用性...")
    print("=" * 60)
    
    try:
        from flash_attn import flash_attn_func
        print("✅ Flash Attention 2 可用")
        
        try:
            from flash_attn import flash_attn_func_fa3
            print("✅ Flash Attention 3 可用 (Hopper GPU)")
        except:
            print("ℹ️  Flash Attention 3 不可用 (需要 Hopper GPU)")
            
        return True
    except ImportError:
        print("❌ Flash Attention 未安装")
        print("   安装命令: pip install flash-attn --no-build-isolation")
        return False


def main():
    print("\n" + "=" * 60)
    print("mHC + Flash-IPA 组合模式测试")
    print("=" * 60)
    
    print(f"\nPython 版本: {sys.version}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    results = []
    
    # 运行测试
    results.append(("模块导入", test_imports()))
    results.append(("基本功能", test_basic_functionality()))
    results.append(("配置解析", test_config_parsing()))
    results.append(("Flash Attention", check_flash_attention()))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 所有测试通过！可以使用 mHC + Flash-IPA 组合模式。")
        print("\n快速开始:")
        print("  python -m genie.train runs/config_mhc_flash_combined.txt")
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查错误信息。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
