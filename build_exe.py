"""
PyInstaller打包脚本
将Genie GUI应用打包为独立的exe文件
"""
import PyInstaller.__main__
import os
import sys
from pathlib import Path


def build_exe():
    """构建exe文件"""
    
    root_dir = Path(__file__).parent.absolute()
    
    print("="*60)
    print("Genie 打包脚本")
    print("="*60)
    print(f"当前Python: {sys.executable}")
    print(f"项目目录: {root_dir}")
    print()
    
    args = [
        'genie_gui.py',
        '--name=Genie',
        '--windowed',
        '--onefile',
        '--noconfirm',
        
        # 数据文件
        f'--add-data={root_dir}/genie;genie',
        f'--add-data={root_dir}/evaluations;evaluations',
        f'--add-data={root_dir}/weights;weights',
        f'--add-data={root_dir}/packages;packages',
        f'--add-data={root_dir}/environment.yml;.',
        
        # PyQt6
        '--hidden-import=PyQt6',
        '--hidden-import=PyQt6.QtCore',
        '--hidden-import=PyQt6.QtGui',
        '--hidden-import=PyQt6.QtWidgets',
        '--hidden-import=PyQt6.QtSvg',
        '--hidden-import=PyQt6.QtOpenGL',
        '--hidden-import=PyQt6.QtPrintSupport',
        
        # 机器学习
        '--hidden-import=torch',
        '--hidden-import=torch.nn',
        '--hidden-import=torch.optim',
        '--hidden-import=torch.utils',
        '--hidden-import=pytorch_lightning',
        '--hidden-import=pytorch_lightning.core',
        '--hidden-import=pytorch_lightning.callbacks',
        
        # 数据科学
        '--hidden-import=matplotlib',
        '--hidden-import=matplotlib.pyplot',
        '--hidden-import=matplotlib.backends',
        '--hidden-import=numpy',
        '--hidden-import=numpy.random',
        '--hidden-import=pandas',
        '--hidden-import=scipy',
        '--hidden-import=sklearn',
        '--hidden-import=seaborn',
        
        # 日志和监控
        '--hidden-import=wandb',
        '--hidden-import=tensorboard',
        '--hidden-import=tensorboardX',
        
        # Genie模块
        '--hidden-import=genie',
        '--hidden-import=genie.train',
        '--hidden-import=genie.sample',
        '--hidden-import=genie.config',
        '--hidden-import=genie.diffusion',
        '--hidden-import=genie.diffusion.diffusion',
        '--hidden-import=genie.diffusion.genie',
        '--hidden-import=genie.diffusion.schedule',
        '--hidden-import=genie.model',
        '--hidden-import=genie.model.model',
        '--hidden-import=genie.model.modules',
        '--hidden-import=genie.utils',
        '--hidden-import=genie.utils.data_io',
        '--hidden-import=genie.utils.encoding',
        '--hidden-import=genie.utils.geo_utils',
        '--hidden-import=genie.utils.loss',
        '--hidden-import=genie.utils.model_io',
        
        # 收集子模块
        '--collect-submodules=torch',
        '--collect-submodules=pytorch_lightning',
        '--collect-submodules=genie',
        '--collect-data=torch',
        
        # 输出目录
        '--distpath=dist',
        '--workpath=build',
        '--specpath=.',
    ]
    
    print("开始打包...")
    print("="*60)
    
    PyInstaller.__main__.run(args)
    
    print("="*60)
    print("打包完成！")
    print(f"\n可执行文件位置: {root_dir}/dist/Genie.exe")


if __name__ == '__main__':
    build_exe()
