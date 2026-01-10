# Genie: 蛋白质从头设计 (De Novo Protein Design)

Genie 是一个基于扩散模型的蛋白质从头设计工具，通过对定向残基云进行等变扩散来实现。

## 安装

1.  **克隆仓库：**
    ```bash
    git clone https://github.com/a-lab-i/genie.git
    cd genie
    ```

2.  **安装依赖：**
    建议使用虚拟环境（如 Conda 或 venv）。
    ```bash
    pip install -e .
    ```

3.  **设置数据（可选）：**
    如果是为了训练，你需要下载并预处理 SCOPe 数据集。
    ```bash
    bash scripts/install_dataset.sh
    ```

4.  **外部工具：**
    本仓库在 `packages/TMscore/` 目录下包含了 `TMscore` 和 `TMalign` 的二进制文件。请确保它们具有执行权限：
    ```bash
    chmod +x packages/TMscore/TMscore packages/TMscore/TMalign
    ```
    如果遇到问题，你可能需要使用同一目录下的 C++ 源文件重新编译它们：
    ```bash
    g++ -static -O3 -ffast-math -lm -o packages/TMscore/TMscore packages/TMscore/TMscore.cpp
    g++ -static -O3 -ffast-math -lm -o packages/TMscore/TMalign packages/TMscore/TMalign.cpp
    ```

## 使用方法

### 1. 采样 (Sampling)

使用预训练模型生成蛋白质骨架。

**关于预训练权重的说明：**
提供的 `weights/` 目录包含检查点文件。采样脚本需要特定的目录结构（例如 `runs/<model_name>/version_<X>/checkpoints/`）。你可能需要调整权重文件的结构，或者直接使用提供的 Jupyter Notebook，它会自动处理这个问题。

标准命令：
```bash
python genie/sample.py \
    --rootdir runs \
    --model_name scope_l_128 \
    --model_version 0 \
    --model_epoch 49999 \
    --batch_size 5 \
    --num_batches 1 \
    --gpu 0
```

### 2. 训练 (Training)

从头开始训练新模型。

```bash
python genie/train.py \
    --config example_configuration \
    --gpus 0,1
```

配置文件定义了模型超参数和训练设置。详情请参考 `genie/config.py`。

### 3. 可视化 (Visualization)

你可以使用提供的脚本可视化生成的结构（保存为 `.npy` 坐标文件）。

```bash
python evaluations/visualize.py
```
（注意：你可能需要修改 `evaluations/visualize.py` 或使用 `evaluations/visualize_protein.py` 来指向你具体的输出文件）。

### 4. 分析与评估 (Analysis and Evaluation)

本项目包含用于评估生成设计的创新性（Novelty）以及可视化设计空间的脚本。

#### 质量评估 (Quality Evaluation)

为了评估生成骨架的可设计性，请使用评估流程。此步骤运行 ProteinMPNN (逆折叠) 和 ESMFold (折叠) 来计算自洽 TM-score (scTM) 和 pLDDT。

```bash
python evaluations/pipeline/evaluate.py \
    --input_dir runs/scope_l_128/version_0/samples/epoch_49999 \
    --output_dir runs/scope_l_128/version_0/samples/epoch_49999/evaluations
```
这将生成绘图脚本所需的 `info.csv` 文件。

#### 创新性评估 (Novelty Evaluation)

计算生成设计的创新性（即与 PDB 等参考数据库的 TM-score）：

*   **CPU 版本 (精确，暴力搜索):**
    ```bash
    python evaluations/Novelty_Evaluation_CPU.py \
        --input_dir runs/scope_l_128/version_0/samples/epoch_49999/evaluations \
        --ref_dir data/pdbstyle-2.08 \
        --num_workers 4
    ```

*   **GPU 版本 (混合方法，快速筛选):**
    使用 ProteinMPNN 嵌入在运行 TM-align 之前对候选者进行筛选。
    ```bash
    python evaluations/Novelty_Evaluation_GPU.py \
        --input_dir runs/scope_l_128/version_0/samples/epoch_49999/evaluations
    ```

#### 绘图分析 (Plotting Analysis)

*   **设计空间 MDS 图:**
    使用多维缩放 (MDS) 可视化生成样本的分布。
    ```bash
    python evaluations/plot_genie_mds_novelty.py \
        --input_dir runs/.../evaluations \
        --output_file mds_plot.png
    ```

*   **综合分析 (复现论文图2):**
    绘制 pLDDT 与 scTM 的关系图、SSE 分布以及可设计性统计。
    ```bash
    python evaluations/plot_genie_analysis.py \
        --input_dir runs/.../evaluations \
        --output_file analysis_plot.png
    ```

## 项目结构 (Project Structure)

-   `genie/`: 主要包源代码。
    -   `diffusion/`: 扩散模型实现。
    -   `model/`: 神经网络架构。
    -   `data/`: 数据加载和处理。
-   `evaluations/`: 评估流程组件。
-   `packages/`: 外部工具 (TMscore)。
-   `scripts/`: 设置用的实用脚本。
-   `weights/`: 预训练模型权重。
