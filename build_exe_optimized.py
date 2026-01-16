"""
优化的PyInstaller打包脚本
将Genie GUI应用打包为目录形式（更小、更快）
"""
import PyInstaller.__main__
import os
import sys
import shutil
from pathlib import Path


def build_exe():
    """构建exe文件（目录模式）"""
    
    # 获取项目根目录
    root_dir = Path(__file__).parent.absolute()
    
    # 准备打包参数
    args = [
        'genie_gui.py',  # 主入口文件
        '--name=Genie',  # 应用名称
        '--windowed',  # 窗口模式（不显示控制台）
        '--onedir',  # 打包为目录（推荐，更小更快）
        '--noconfirm',  # 覆盖输出目录
        
        # 添加数据文件和目录
        f'--add-data={root_dir}/genie;genie',
        f'--add-data={root_dir}/evaluations;evaluations',
        f'--add-data={root_dir}/weights;weights',
        f'--add-data={root_dir}/packages;packages',
        f'--add-data={root_dir}/genie_backend.py;.',
        f'--add-data={root_dir}/environment.yml;.',
        
        # 隐藏导入
        '--hidden-import=PyQt6',
        '--hidden-import=PyQt6.QtCore',
        '--hidden-import=PyQt6.QtGui',
        '--hidden-import=PyQt6.QtWidgets',
        '--hidden-import=torch',
        '--hidden-import=pytorch_lightning',
        '--hidden-import=matplotlib',
        '--hidden-import=matplotlib.backends.backend_qt5agg',
        '--hidden-import=numpy',
        '--hidden-import=pandas',
        '--hidden-import=scipy',
        '--hidden-import=sklearn',
        '--hidden-import=seaborn',
        '--hidden-import=wandb',
        '--hidden-import=tensorboard',
        '--hidden-import=genie_backend',
        '--hidden-import=genie',
        '--hidden-import=genie.train',
        '--hidden-import=genie.sample',
        '--hidden-import=genie.config',
        '--hidden-import=genie.diffusion',
        '--hidden-import=genie.model',
        '--hidden-import=genie.utils',
        '--hidden-import=yaml',
        '--hidden-import=tqdm',
        
        # 收集子模块
        '--collect-submodules=torch',
        '--collect-submodules=pytorch_lightning',
        '--collect-submodules=genie',
        
        # 排除不需要的模块以减小体积
        '--exclude-module=tkinter',
        '--exclude-module=test',
        '--exclude-module=unittest',
        
        # 输出目录
        '--distpath=dist',
        '--workpath=build',
        '--specpath=.',
    ]
    
    print("="*70)
    print(" "*20 + "Genie 打包工具")
    print("="*70)
    print("\n开始打包Genie应用...")
    print(f"工作目录: {root_dir}")
    print("\n打包配置:")
    print("  - 模式: 目录模式 (onedir)")
    print("  - 界面: 窗口模式 (无控制台)")
    print("  - 包含: genie模块、评估工具、预训练模型")
    print("\n")
    
    # 运行PyInstaller
    try:
        PyInstaller.__main__.run(args)
        
        print("\n" + "="*70)
        print("✓ 打包完成！")
        print("="*70)
        print(f"\n可执行文件位置: {root_dir}/dist/Genie/Genie.exe")
        print(f"完整目录位置: {root_dir}/dist/Genie/")
        
        print("\n使用说明:")
        print("1. 将 dist/Genie 整个文件夹复制到目标计算机")
        print("2. 双击 Genie.exe 启动应用")
        print("3. 首次运行需要CUDA环境（如使用GPU）")
        print("4. weights目录包含预训练模型")
        
        print("\n目录结构:")
        print("  Genie/")
        print("    ├── Genie.exe           # 主程序")
        print("    ├── genie/              # 核心模块")
        print("    ├── evaluations/        # 评估工具")
        print("    ├── weights/            # 预训练模型")
        print("    ├── packages/           # 第三方工具")
        print("    └── _internal/          # 运行时依赖")
        
        print("\n" + "="*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ 打包失败: {e}")
        sys.exit(1)


if __name__ == '__main__':
    build_exe()
