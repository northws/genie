# Genie: 蛋白质从头设计 (De Novo Protein Design)

Genie 是一个基于扩散模型的蛋白质从头设计工具，通过对定向残基云进行等变扩散来实现。

本项目是对 [https://github.com/aqlaboratory/genie](https://github.com/aqlaboratory/genie) 的复现及优化。

## 安装

1.  **克隆仓库：**
    ```bash
    git clone https://github.com/northws/genie.git
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

### 1. 训练 (Training)

从头开始训练新模型。

```bash
python genie/train.py \
    --config example_configuration \
    --gpus 0,1
```

配置文件定义了模型超参数和训练设置。详情请参考 `genie/config.py`。

**参数说明 (genie/train.py)：**

- `-c, --config`（必选）：配置文件路径/名称。用于指定训练所需的模型结构与超参数配置。
- `-g, --gpus`：使用的 GPU 设备，例如 `"0"` 或 `"0,1"`，通常用于控制 `CUDA_VISIBLE_DEVICES` / 多卡选择。
- `-r, --resume`：断点续训的 checkpoint（`.ckpt`）文件路径。

### 2. 采样 (Sampling)

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

**参数说明 (genie/sample.py)：**

- `-n, --model_name`（必选）：模型名称（对应 `runs/<model_name>/...` 的目录名）。
- `-r, --rootdir`（默认：`runs`）：运行目录根路径（包含 `runs/<model_name>/...` 结构）。
- `-v, --model_version`：模型版本号（对应 `runs/<model_name>/version_<N>/...`）。
- `-e, --model_epoch`：加载的 checkpoint 对应 epoch（用于选择 checkpoint）。
- `-g, --gpu`：使用的 GPU 编号。注意该参数的值是“可选”的：写 `--gpu` 等价于 `--gpu 0`；写 `--gpu 1` 则使用 GPU 1。
- `--batch_size`（默认：`5`）：每个 batch 生成的样本数。
- `--num_batches`（默认：`2`）：生成的 batch 数，总样本数 = `batch_size * num_batches`。
- `--noise_scale`（默认：`0.6`）：采样噪声强度，影响随机性/多样性。
- `--min_length`（默认：`50`）：采样长度下限。
- `--max_length`（默认：`128`）：采样长度上限。
- `--save_trajectory`：是否保存扩散过程每个时间步的轨迹（`.npy`），用于生成动画可视化；会增加磁盘占用与耗时。

### 3. 可视化 (Visualization)

你可以使用提供的脚本可视化生成的结构（保存为 `.npy` 坐标文件）。

```bash
python evaluations/visualize.py <input_file> -o <output_dir>
```


**参数说明 (evaluations/visualize.py)：**

- `input_file`（位置参数）：输入坐标文件路径（通常为 `.npy`；脚本也会尝试按 CSV/文本读取）。
- `-o, --output_dir`（可选）：输出目录；不填则默认输出到 `input_file` 同目录。

**替代脚本（接口类似）：**

- `python evaluations/visualize_protein.py <input_file> -o <output_dir>`：更“蛋白质骨架风格”的可视化（平滑曲线 + N→C 渐变）。

**轨迹动画可视化（evaluations/visualize_trajectory.py）：**

```bash
python evaluations/visualize_trajectory.py <traj_npy> <output_gif>
```

**参数说明 (evaluations/visualize_trajectory.py)：**

- `traj_npy`（位置参数）：由 `genie/sample.py --save_trajectory` 生成的轨迹 `.npy` 文件。
- `output_gif`（位置参数）：输出 `.gif` 动画文件路径。

（注意：你也可以使用 `evaluations/visualize_protein.py` 来获得更平滑的骨架展示效果。）

### 4. 分析与评估 (Analysis and Evaluation)

本项目包含用于评估生成结构的质量与创新性，以及用于可视化分析结果的脚本。

#### 质量评估 (Quality Evaluation - scTM & pLDDT)

评估流程会运行 ProteinMPNN（逆折叠/序列设计）与 ESMFold（折叠/结构预测），计算自洽 TM-score（scTM）与 pLDDT，并生成后续绘图所需的 `info.csv`。

```bash
python evaluations/pipeline/evaluate.py \
    --input_dir runs/scope_l_128/version_0/samples/epoch_49999 \
    --output_dir runs/scope_l_128/version_0/samples/epoch_49999/evaluations
```

**参数说明 (evaluations/pipeline/evaluate.py)：**

- `--input_dir`（必选）：待评估样本所在目录。
- `--output_dir`（必选）：评估结果输出目录（会生成 `info.csv` 等）。
- `-g, --gpus`（可选）：使用的 GPU 设备，例如 `"0"` 或 `"0,1"`。
- `-c, --config`（可选）：为兼容保留，但脚本会忽略该参数。

#### 创新性评估 (Novelty Evaluation)

通过 TM-score 将生成的蛋白质与参考数据库（例如 PDB）进行比对，得到每个设计与数据库中最相似结构的最大 TM（越低通常越“新颖”）。

*   **CPU 版本（精确，暴力搜索）(evaluations/Novelty_Evaluation_CPU.py)：**

    ```bash
    python evaluations/Novelty_Evaluation_CPU.py \
        --input_dir runs/scope_l_128/version_0/samples/epoch_49999/evaluations \
        --ref_dir data/pdbstyle-2.08 \
        --num_workers 4
    ```

    **参数说明：**

    - `-i, --input_dir`：输入目录。可以指向包含 `info.csv` 的评估目录（若存在 `designs/` 子目录会自动识别）。
    - `-o, --output_csv`：输出 CSV 路径。默认：`<input_dir>/novelty.csv`。
    - `--ref_dir`：参考数据库目录（例如 `data/pdbstyle-2.08`）。
    - `--tmalign`：`TMalign` 可执行文件路径。
    - `--num_workers`：并行进程数。
    - `--length_tolerance`：长度预筛选容差（默认 `0.3` 表示 ±30%）。
    - `--early_stop_tm`：提前停止阈值（默认 `0.5`），当发现 TM 超过该值时可停止搜索（视为“不新颖”）。
    - `--no_early_stop`：关闭提前停止，改为精确搜索最大 TM。
    - `--no_length_filter`：关闭长度预筛选。

*   **GPU 版本（混合方法，快速筛选）(evaluations/Novelty_Evaluation_GPU.py)：**

    ```bash
    python evaluations/Novelty_Evaluation_GPU.py \
        --input_dir runs/scope_l_128/version_0/samples/epoch_49999/evaluations \
        --ref_dir data/pdbstyle-2.08
    ```

    **参数说明：**

    - `-i, --input_dir`：输入目录（包含 PDB 设计）。若目录下存在 `designs/` 子目录会自动切换到该子目录。
    - `-o, --output_csv`：输出 CSV 路径。默认：在评估目录（或 `designs/` 的父目录）生成 `novelty_hybrid.csv`。
    - `-r, --ref_dir`：参考数据库目录。

#### 绘图分析 (Plotting Analysis)

使用统一的 `evaluations/plot.py` 脚本来生成分析图表。该脚本整合了 MDS 图、综合分析（复现图2）和 3D 结构可视化的功能。

**命令行参数说明 (evaluations/plot.py)：**

- `-i, --input_dir`：输入目录（默认指向仓库内的一个示例 runs 目录）。通常应为评估目录，至少包含 `info.csv`。
- `-p, --plot`：生成哪种图表（默认 `all`）：
  - `analysis`：综合分析（复现论文图2风格：pLDDT vs scTM、SSE 分布、长度分布、统计柱状图）。
  - `mds`：设计空间 MDS 图（需要 `pair_info.csv`）。
  - `structures`：3D 结构示例（需要 PDB 设计文件与 novelty CSV）。
  - `all`：生成以上所有图。
- `-o, --output_dir`：输出目录（默认当前目录）。

**使用示例：**

```bash
# 生成所有图表
python evaluations/plot.py --input_dir runs/.../evaluations --output_dir outputs/plots --plot all

# 仅生成 MDS 图
python evaluations/plot.py -i runs/.../evaluations -p mds -o outputs/plots
```

**Python API（evaluations/plot.py）：**

- `get_default_run_dir()`：返回默认评估目录。
- `load_data(input_dir)`：
  - `input_dir`：评估目录（包含 `info.csv`）。
  - 返回：`(df, has_novelty)`。
- `parse_pdb_ca(filepath)`：
  - `filepath`：`.pdb` 文件路径。
  - 返回：`N x 3` 的 Cα 坐标数组。
- `plot_genie_analysis(input_dir, output_file='genie_analysis_figure2_repro_v2_hybrid.png')`：
  - `input_dir`：评估目录。
  - `output_file`：输出图片路径。
- `plot_genie_mds_novelty(input_dir, output_file='genie_design_space_mds_hybrid.png')`：
  - `input_dir`：评估目录（需要 `pair_info.csv`）。
  - `output_file`：输出图片路径。
- `plot_structures(input_dir, output_file='genie_structure_examples_novel.png')`：
  - `input_dir`：评估目录或 `designs/` 目录。
  - `output_file`：输出图片路径。
- `main()`：命令行入口（对应上述 `-i/-p/-o`）。

## 项目结构 (Project Structure)

-   `genie/`: 主要包源代码。
    -   `diffusion/`: 扩散模型实现。
    -   `model/`: 神经网络架构。
    -   `data/`: 数据加载和处理。
-   `evaluations/`: 评估流程组件。
-   `packages/`: 外部工具 (TMscore)。
-   `scripts/`: 设置用的实用脚本。
-   `weights/`: 预训练模型权重。

## 引用与致谢 (Citations and Acknowledgements)

本项目构建于多个优秀的开源项目成果之上。

### 核心算法与模型
*   **Genie (Original Implementation)**:
    Lin, Y. C., & AlQuraishi, M. (2023). Generating protein backbone structures with equivariant diffusion models. *arXiv preprint arXiv:2301.12485*.
    [[Paper]](https://arxiv.org/abs/2301.12485) [[Code]](https://github.com/aqlaboratory/genie)

*   **Flash-IPA (Optimization)**:
    Flagship Pioneering. (2023). Flash-IPA: Accelerated Invariant Point Attention. GitHub.
    [[Code]](https://github.com/flagshippioneering/flash_ipa)

### 评估流程 (Evaluation Pipeline)
*   **ProteinMPNN (Sequence Design)**:
    Dauparas, J., et al. (2022). Robust deep learning–based protein sequence design using ProteinMPNN. *Science*, 378(6615), 49-56.
    [[Paper]](https://www.science.org/doi/10.1126/science.add2187) [[Code]](https://github.com/dauparas/ProteinMPNN)

*   **ESMFold / ESM-2 (Structure Prediction)**:
    Lin, Z., et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*, 379(6637), 1123-1130.
    [[Paper]](https://www.science.org/doi/10.1126/science.ade2574) [[Code]](https://github.com/facebookresearch/esm)

*   **TM-score & TM-align (Structure Alignment)**:
    Zhang, Y., & Skolnick, J. (2005). TM-align: a protein structure alignment algorithm based on the TM-score. *Nucleic Acids Research*, 33(7), 2302-2309.
    [[Paper]](https://academic.oup.com/nar/article/33/7/2302/2401364) [[Code]](https://zhanggroup.org/TM-align/)

## 图库 (Gallery)

### 生成过程 (Generation Process)
![生成过程](process.gif)



## 优化结果

![优化对比](Training_process_parameters/optimization_comparison.png)

我们对比了原始实现与本优化版本的训练过程参数（数据位于 `Training_process_parameters/` 文件夹中）。

**硬件配置:**
*   **GPU:** RTX 5090 (32GB) * 1
*   **CPU:** 25 vCPU Intel(R) Xeon(R) Platinum 8470Q
*   **内存:** 90GB

**对比总结：**

| 指标 | 原始工作 | 本工作 (优化后) | 提升 |
| :--- | :--- | :--- | :--- |
| **训练时长 (500 Epochs)** | ~25.7 小时 | ~12.8 小时 | **~2.0倍 加速** |
| **最大 GPU 显存占用** | ~29.53 GB | ~25.92 GB | **降低约 12%** |
| **训练 Loss (最终 Epoch)** | ~0.758 | ~0.771 | 基本持平 |

优化后，训练速度提升了约 2 倍，同时显存占用降低了约 12%。对 Step Loss (平滑后) 的分析表明，最终 Epoch Loss 的微小差异源于随机波动，两者的收敛趋势在实际训练中表现一致。

### 生成质量对比 (Generative Quality Comparison)

我们将优化后的模型与原版数据在生成能力上进行了可视化对比。结果表明，优化后的模型保持了相当的生成质量。

**设计空间分析 (Design Space Analysis - MDS):**

| 原始工作 (Original Work) | 本工作 (This Work - Optimized) |
| :---: | :---: |
| ![Original MDS](Training_process_parameters/origin_work_mds_hybrid.png) | ![Optimized MDS](Training_process_parameters/this_work_mds_hybrid.png) |

**综合分析 (Comprehensive Analysis):**

| 原始工作 (Original Work) | 本工作 (This Work - Optimized) |
| :---: | :---: |
| ![Original Hybrid](Training_process_parameters/origin_work_hybrid.png) | ![Optimized Hybrid](Training_process_parameters/this_work_hybrid.png) |

**创新结构示例 (Novel Structure Examples - Optimized Work):**

![Novel Structures](Training_process_parameters/this_work_structure_examples_novel.png)



