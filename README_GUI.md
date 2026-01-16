# Genie GUI 应用 - 打包说明

## 概述

本项目将Genie蛋白质结构生成工具打包为一个独立的Windows GUI应用程序，包含四个核心模块：

1. **训练模块** - 训练新模型或继续训练
2. **采样模块** - 使用预训练模型或自定义模型生成蛋白质结构
3. **评估模块** - 评估生成的结构（折叠预测、逆向折叠、新颖性评估等）
4. **绘图模块** - 可视化结果（分析图、MDS图、结构图等）

## 项目结构

```
genie/
├── genie_gui.py              # GUI主程序
├── genie_backend.py          # 后端逻辑
├── build_exe.py              # 打包脚本（单文件模式）
├── build_exe_optimized.py    # 打包脚本（目录模式，推荐）
├── requirements_gui.txt      # GUI依赖
├── build.bat                 # Windows打包批处理脚本
├── genie/                    # 核心模块
├── evaluations/              # 评估工具
├── weights/                  # 预训练模型
├── packages/                 # 第三方工具
└── environment.yml           # Conda环境配置
```

## 安装依赖

### ⚠️ 重要：首次使用前必须先安装依赖！

### 方法1: 使用安装脚本（最简单）

```bash
# 双击运行（Windows）
install_dependencies.bat

# 这会自动：
# 1. 检查Python版本
# 2. 升级pip
# 3. 安装所有GUI依赖
# 4. 验证安装
```

### 方法2: 使用Conda

```bash
# 创建并激活conda环境
conda env create -f environment.yml
conda activate genie

# 安装额外的GUI依赖
pip install -r requirements_gui.txt
```

### 方法3: 使用pip

```bash
# 创建虚拟环境（可选）
python -m venv venv
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements_gui.txt
pip install -e .
```

### 验证安装

```bash
python -c "import PyQt6; print('PyQt6安装成功！')"
```

## 运行GUI应用（开发模式）

```bash
python genie_gui.py
```

## 打包为EXE

### 选项1: 使用批处理脚本（最简单）

```bash
build.bat
```

### 选项2: 使用Python脚本

#### 目录模式（推荐，体积小，启动快）

```bash
python build_exe_optimized.py
```

生成的应用位于: `dist/Genie/`

#### 单文件模式（便于分发）

```bash
python build_exe.py
```

生成的应用位于: `dist/Genie.exe`

## 打包后的文件结构

### 目录模式输出

```
dist/Genie/
├── Genie.exe              # 主程序
├── genie/                 # 核心模块
│   ├── __init__.py
│   ├── config.py
│   ├── train.py
│   ├── sample.py
│   └── ...
├── evaluations/           # 评估工具
│   ├── plot.py
│   ├── visualize.py
│   └── pipeline/
├── weights/               # 预训练模型
│   ├── scope_l_128/
│   ├── scope_l_256/
│   └── swissprot_l_256/
├── packages/              # 第三方工具
│   ├── ProteinMPNN/
│   └── TMscore/
└── _internal/             # 运行时库（DLL等）
```

## 使用说明

### 1. 训练模块

- 选择数据集（SCOPE、SwissProt或自定义）
- 配置训练参数（轮数、批次大小、GPU等）
- 可选择从检查点恢复训练
- 实时查看训练日志

### 2. 采样模块

- 选择预训练模型或加载自定义模型
- 设置采样参数：
  - 长度范围（最小/最大）
  - 批次大小和数量
  - 噪声比例
  - 是否保存轨迹
- 指定输出目录
- 查看采样进度和日志

### 3. 评估模块

- 选择输入目录（包含采样结果）
- 选择输出目录
- 配置评估选项：
  - ESMFold折叠预测
  - ProteinMPNN逆向折叠
  - 新颖性评估
  - TM分数计算
- 查看评估结果

### 4. 绘图模块

- 选择绘图类型：
  - 分析图（pLDDT vs scTM等）
  - MDS图（设计空间可视化）
  - 结构图（3D可视化）
  - 单个结构可视化
  - 轨迹可视化
- 选择输入目录或文件
- 指定输出目录
- 双击列表中的图片直接打开

## 系统要求

### 最低配置
- OS: Windows 10/11
- RAM: 8GB
- 磁盘: 10GB可用空间

### 推荐配置（GPU加速）
- OS: Windows 10/11
- GPU: NVIDIA GPU (CUDA兼容)
- RAM: 16GB+
- 磁盘: 20GB+可用空间
- CUDA: 11.8+

## 常见问题

### 1. 打包时出错

**问题**: 缺少模块或依赖

**解决**:
```bash
# 确保所有依赖已安装
pip install -r requirements_gui.txt
pip install pyinstaller
```

### 2. 运行时找不到模块

**问题**: ImportError

**解决**: 检查 `build_exe_optimized.py` 中的 `--hidden-import` 参数，确保包含所有必要模块

### 3. GPU不可用

**问题**: CUDA错误或GPU未检测到

**解决**:
- 安装NVIDIA驱动
- 安装CUDA Toolkit (11.8+)
- 确保PyTorch是GPU版本: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

### 4. 打包文件太大

**解决**:
- 使用目录模式而不是单文件模式
- 在 `build_exe_optimized.py` 中添加更多 `--exclude-module` 选项排除不需要的模块
- 删除不需要的预训练模型

### 5. 应用启动慢

**原因**: 单文件模式需要解压
**解决**: 使用目录模式 (`build_exe_optimized.py`)

## 分发应用

### 目录模式

1. 将整个 `dist/Genie/` 文件夹压缩为zip
2. 分发zip文件
3. 用户解压后直接运行 `Genie.exe`

### 单文件模式

1. 直接分发 `dist/Genie.exe`
2. 首次运行会较慢（需要解压）

## 开发者说明

### 修改GUI

编辑 `genie_gui.py` 中的对应标签页类：
- `TrainingTab` - 训练界面
- `SamplingTab` - 采样界面
- `EvaluationTab` - 评估界面
- `PlottingTab` - 绘图界面

### 添加新功能

1. 在 `genie_backend.py` 中添加后端逻辑
2. 在 `genie_gui.py` 中添加UI组件
3. 连接信号和槽
4. 重新打包

### 调试

开发模式运行（显示控制台输出）：
```bash
python genie_gui.py
```

## 性能优化建议

1. **使用GPU**: 设置GPU参数以启用CUDA加速
2. **批次大小**: 根据GPU内存调整批次大小
3. **多GPU**: 使用多个GPU（如 "0,1"）进行训练
4. **混合精度**: 训练时自动启用bf16/fp16混合精度

## 许可证

遵循原项目的许可证。

## 联系方式

如有问题或建议，请在GitHub仓库提issue。

## 更新日志

### v1.0.0 (2026-01-17)
- 初始版本
- 实现四个核心模块的GUI
- 支持预训练模型
- 完整的评估和可视化功能
