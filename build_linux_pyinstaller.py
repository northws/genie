"""
Linux/Mac PyInstaller打包脚本
"""
import PyInstaller.__main__
import os
import sys
from pathlib import Path


def build_linux():
    """构建Linux/Mac可执行文件"""
    
    root_dir = Path(__file__).parent.absolute()
    
    # 准备打包参数
    args = [
        'genie_gui_enhanced.py',  # 使用增强版
        '--name=Genie',
        '--windowed',  # GUI模式
        '--onedir',  # 目录模式
        '--noconfirm',
        
        # 添加数据文件和目录
        f'--add-data={root_dir}/genie:genie',
        f'--add-data={root_dir}/evaluations:evaluations',
        f'--add-data={root_dir}/weights:weights',
        f'--add-data={root_dir}/packages:packages',
        f'--add-data={root_dir}/genie_backend.py:.',
        f'--add-data={root_dir}/task_queue.py:.',
        f'--add-data={root_dir}/environment.yml:.',
        
        # 隐藏导入
        '--hidden-import=PyQt6',
        '--hidden-import=PyQt6.QtCore',
        '--hidden-import=PyQt6.QtGui',
        '--hidden-import=PyQt6.QtWidgets',
        '--hidden-import=torch',
        '--hidden-import=pytorch_lightning',
        '--hidden-import=matplotlib',
        '--hidden-import=numpy',
        '--hidden-import=pandas',
        '--hidden-import=scipy',
        '--hidden-import=sklearn',
        '--hidden-import=seaborn',
        '--hidden-import=wandb',
        '--hidden-import=tensorboard',
        '--hidden-import=genie_backend',
        '--hidden-import=task_queue',
        '--hidden-import=genie',
        '--hidden-import=yaml',
        '--hidden-import=tqdm',
        
        # 收集子模块
        '--collect-submodules=torch',
        '--collect-submodules=pytorch_lightning',
        '--collect-submodules=genie',
        
        # 排除不需要的模块
        '--exclude-module=tkinter',
        '--exclude-module=test',
        '--exclude-module=unittest',
        
        # 输出目录
        '--distpath=dist',
        '--workpath=build',
        '--specpath=.',
    ]
    
    print("="*70)
    print(" "*20 + "Genie Linux/Mac 打包工具")
    print("="*70)
    print("\n开始打包Genie应用...")
    print(f"工作目录: {root_dir}")
    print("\n打包配置:")
    print("  - 平台: Linux/Mac")
    print("  - 模式: 目录模式 (onedir)")
    print("  - 界面: GUI模式")
    print("  - 版本: 增强版")
    print("\n")
    
    # 运行PyInstaller
    try:
        PyInstaller.__main__.run(args)
        
        print("\n" + "="*70)
        print("✓ 打包完成！")
        print("="*70)
        print(f"\n可执行文件位置: {root_dir}/dist/Genie/Genie")
        print(f"完整目录位置: {root_dir}/dist/Genie/")
        
        # 设置可执行权限
        exe_path = root_dir / "dist" / "Genie" / "Genie"
        if exe_path.exists():
            os.chmod(exe_path, 0o755)
            print("\n✓ 已设置可执行权限")
        
        print("\n使用说明:")
        print("1. 进入目录: cd dist/Genie")
        print("2. 运行应用: ./Genie")
        print("3. 分发应用: tar -czf Genie-Linux-x64.tar.gz -C dist Genie")
        
        print("\n目录结构:")
        print("  Genie/")
        print("    ├── Genie              # 主程序")
        print("    ├── genie/             # 核心模块")
        print("    ├── evaluations/       # 评估工具")
        print("    ├── weights/           # 预训练模型")
        print("    ├── packages/          # 第三方工具")
        print("    └── _internal/         # 运行时依赖")
        
        print("\n" + "="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ 打包失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    build_linux()
