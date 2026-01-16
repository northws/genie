"""
PyInstaller打包脚本
将Genie GUI应用打包为独立的exe文件
"""
import PyInstaller.__main__
import os
import sys
import shutil
from pathlib import Path


def build_exe():
    """构建exe文件"""
    
    # 获取项目根目录
    root_dir = Path(__file__).parent.absolute()
    
    # 准备打包参数
    args = [
        'genie_gui.py',  # 主入口文件
        '--name=Genie',  # 应用名称
        '--windowed',  # 窗口模式（不显示控制台）
        '--onefile',  # 打包为单个exe
        '--noconfirm',  # 覆盖输出目录
        
        # 添加数据文件和目录
        f'--add-data={root_dir}/genie;genie',
        f'--add-data={root_dir}/evaluations;evaluations',
        f'--add-data={root_dir}/weights;weights',
        f'--add-data={root_dir}/packages;packages',
        f'--add-data={root_dir}/environment.yml;.',
        
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
        '--hidden-import=genie',
        '--hidden-import=genie.train',
        '--hidden-import=genie.sample',
        '--hidden-import=genie.config',
        '--hidden-import=genie.diffusion',
        '--hidden-import=genie.model',
        '--hidden-import=genie.utils',
        
        # 收集子模块
        '--collect-submodules=torch',
        '--collect-submodules=pytorch_lightning',
        '--collect-submodules=genie',
        
        # 收集数据
        '--collect-data=torch',
        '--collect-data=pytorch_lightning',
        
        # 图标（如果有的话）
        # '--icon=icon.ico',
        
        # 输出目录
        '--distpath=dist',
        '--workpath=build',
        '--specpath=.',
    ]
    
    print("开始打包Genie应用...")
    print("="*60)
    
    # 运行PyInstaller
    PyInstaller.__main__.run(args)
    
    print("="*60)
    print("打包完成！")
    print(f"可执行文件位置: {root_dir}/dist/Genie.exe")
    print("\n注意事项:")
    print("1. 首次运行可能需要安装CUDA驱动（用于GPU加速）")
    print("2. weights目录包含预训练模型")
    print("3. 确保有足够的磁盘空间（模型文件较大）")


if __name__ == '__main__':
    build_exe()
