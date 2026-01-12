# Genie: De Novo Protein Design

Genie is a tool for de novo protein design based on diffusion models, implemented through equivariant diffusion on oriented residue clouds.

## About This Project

This project is an **optimized reproduction** of the original [Genie implementation](https://github.com/aqlaboratory/genie) by Yeqing Lin and Mohammed AlQuraishi.

**Key Improvements:**
- ✨ Integrated Flash-IPA for memory-efficient generation of long sequences.
- 🔗 **Support for mHC + Flash-IPA combination**, balancing training stability and memory efficiency.
- ⚡ Flash Attention optimization, achieving up to 3.1x training speedup.
- 💾 Reduced GPU memory usage by 95% in Flash mode.
- 🚀 Large batch training optimizations (learning rate scaling, warmup, gradient accumulation).
- 🔧 Support for PyTorch 2.9+ and modern toolchains.

**Original Work:**
- Paper: [Generating Novel Protein Backbones with Equivariant Diffusion](https://arxiv.org/abs/2301.12485) (Lin & AlQuraishi, 2023).
- Original repo: https://github.com/aqlaboratory/genie
- License: Apache 2.0

**This Repository:**
- Original Genie code: Apache License 2.0
- New optimizations and features: MIT License
- See [LICENSE.md](LICENSE.md) for details.

---

**Other languages: [中文版](README.md)**

**Check the demo Notebook:** [genie_demo.ipynb](genie_demo.ipynb)

---

## Citations and Acknowledgments

This project is built upon several excellent open-source projects and academic research:

### Core Algorithms and Models

**Genie (Original Implementation)**  
Lin, Y. C., & AlQuraishi, M. (2023). Generating protein backbone structures with equivariant diffusion models. *arXiv preprint arXiv:2301.12485*.  
[[Paper]](https://arxiv.org/abs/2301.12485) [[Code]](https://github.com/aqlaboratory/genie)

**Flash-IPA (Optimization and Acceleration)**  
Flagship Pioneering. (2023). Flash-IPA: Accelerated Invariant Point Attention. GitHub.  
[[Code]](https://github.com/flagshippioneering/flash_ipa)

**mHC: Manifold-Constrained Hyper-Connections**  
Xie et al., DeepSeek-AI. (2025). mHC: Manifold-Constrained Hyper-Connections. *arXiv preprint arXiv:2301.12485*.  
[[Paper]](https://arxiv.org/abs/2512.24880) 

### Evaluation Pipeline Components

**ProteinMPNN (Sequence Design)**  
Dauparas, J., et al. (2022). Robust deep learning–based protein sequence design using ProteinMPNN. *Science*, 378(6615), 49-56.  
[[Paper]](https://www.science.org/doi/10.1126/science.add2187) [[Code]](https://github.com/dauparas/ProteinMPNN)

**ESMFold / ESM-2 (Structure Prediction)**  
Lin, Z., et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*, 379(6637), 1123-1130.  
[[Paper]](https://www.science.org/doi/10.1126/science.ade2574) [[Code]](https://github.com/facebookresearch/esm)

**TM-score & TM-align (Structure Alignment)**  
Zhang, Y., & Skolnick, J. (2005). TM-align: a protein structure alignment algorithm based on the TM-score. *Nucleic Acids Research*, 33(7), 2302-2309.  
[[Paper]](https://academic.oup.com/nar/article/33/7/2302/2401364) [[Code]](https://zhanggroup.org/TM-align/)

---

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/northws/genie.git
    cd genie
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment (e.g., Conda or venv).
    ```bash
    pip install -e .
    ```
    If you encounter issues setting up the environment, you can use our provided Docker image (ensure you re-clone the repo inside the container for the latest changes):
    ```bash
    docker pull ghcr.io/northws/genie:v1
    ```

4.  **Setup data (Optional):**
    For training, you need to download and preprocess the SCOPe dataset.
    ```bash
    bash scripts/install_dataset.sh
    ```

5.  **External tools:**
    This repository includes `TMscore` and `TMalign` binaries in the `packages/TMscore/` directory. Ensure they have execution permissions:
    
    ```bash
    chmod +x packages/TMscore/TMscore packages/TMscore/TMalign
    ```
    If you encounter issues, you may need to recompile them using the C++ source files in the same directory:
    ```bash
    g++ -static -O3 -ffast-math -lm -o packages/TMscore/TMscore packages/TMscore/TMscore.cpp
    g++ -static -O3 -ffast-math -lm -o packages/TMscore/TMalign packages/TMscore/TMalign.cpp
    ```

---

## Usage

## 1. Training

#### Training Objective

Genie uses the **Denoising Diffusion Probabilistic Model (DDPM)** framework, following the methodology described in [Lin & AlQuraishi, 2023](https://arxiv.org/abs/2301.12485). The model learns to denoise oriented residue clouds by predicting the noise added during the forward diffusion process.

**Forward Process (Diffusion):**

Given a protein backbone represented by $C_\alpha$ coordinates $\mathbf{x}_0$, the forward process gradually adds Gaussian noise over $T$ timesteps:

$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})$$

where $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$, $\alpha_t = 1 - \beta_t$, and $\beta_t$ is the noise schedule.

**Training Loss:**

The model $\epsilon_\theta$ is trained to predict the noise $\epsilon$ added at each timestep. The loss function is the **Root Mean Square Deviation (RMSD)** between predicted and actual noise:

$$\mathcal{L} = \mathbb{E}_{t, \mathbf{x}_0, \epsilon} \left[ \frac{1}{N}\sum_{i=1}^{N} \|\epsilon_\theta(\mathbf{x}_t, t)_i - \epsilon_i\|_2 \right]$$

where $N$ is the number of residues, and the expectation is taken over uniformly sampled timesteps $t \sim \mathcal{U}(1, T)$, data samples $\mathbf{x}_0$, and noise $\epsilon \sim \mathcal{N}(0, \mathbf{I})$.

**Reverse Process (Sampling):**

During generation, the model iteratively denoises starting from pure noise $\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$:

$$p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) = \mathcal{N}\left(\mathbf{x}_{t-1}; \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(\mathbf{x}_t, t)\right), \sigma_t^2\mathbf{I}\right)$$

---

#### Running Training

Train a new model from scratch.

```bash
python genie/train.py \
    -c example_configuration \
    -g 0,1
```

The configuration file defines model hyperparameters and training settings. Refer to `genie/config.py` for details.

**Arguments for `genie/train.py`:**
- `-c, --config` (Required): Path/name of the configuration file.
- `-g, --gpus`: GPU devices to use (e.g., `"0"` or `"0,1"`).
- `-r, --resume`: Path to a checkpoint (`.ckpt`) file for resuming training.

---

#### Model Architecture Hyperparameters Guide

Based on the [Genie paper](https://arxiv.org/abs/2301.12485) (Lin & AlQuraishi, 2023) and AlphaFold2 design principles, here is a guide for the four main network components.

##### Architecture Overview

The denoising network consists of:
1. **Single Feature Network**: Generates per-residue representations from positional and timestep encodings.
2. **Pair Feature Network**: Creates residue-pair representations from single features and relative positions.
3. **Pair Transform Network**: Refines pair representations using triangular operations (Evoformer-like).
4. **Structure Network**: Updates 3D coordinates using Invariant Point Attention (IPA).

```
Input (Noisy frames) → Single Feature Net → Pair Feature Net → Pair Transform Net → Structure Net → Output (Denoised frames)
```

---

##### 1. General Parameters

| Parameter | Config Key | Default | Description |
| :--- | :--- | :--- | :--- |
| Single Feature Dim | `singleFeatureDimension` | 128 | Channels for per-residue representation. |
| Pair Feature Dim | `pairFeatureDimension` | 128 | Channels for residue-pair representation. |

**Selection Guide:**
- These dimensions should be equal for optimal information flow.
- **Standard**: 128 (default, balances capacity and efficiency).
- **High Capacity**: 256 (stronger representational power, but high memory usage).
- **Memory Limited**: 64 (reduces capacity but saves significant memory).

---

##### 2. Single Feature Network

Combines positional and timestep encodings.

| Parameter | Config Key | Default | Description |
| :--- | :--- | :--- | :--- |
| Positional Emb Dim | `positionalEmbeddingDimension` | 128 | Dimension for sinusoidal positional encoding. |
| Timestep Emb Dim | `timestepEmbeddingDimension` | 128 | Dimension for diffusion timestep encoding. |

**Selection Guide:**
- Should match `singleFeatureDimension` for seamless integration.

---

##### 3. Pair Feature Network

Creates pair representations using outer products of single features, relative positions, and current structure estimations.

| Parameter | Config Key | Default | Description |
| :--- | :--- | :--- | :--- |
| Relative Position K | `relativePositionK` | 32 | Truncation range: $[-k, k]$. |
| Template Type | `templateType` | `v1` | Method for extracting template features. |

**Selection Guide:**
- **`relativePositionK`**: 32 is sufficient for short sequences (<128). Consider 64 for longer sequences (>256) to capture long-range info.
- **`templateType`**: `v1` is recommended for standard distance map features.

---

##### 4. Pair Transform Network

Refines pair representations via triangular operations ($O(L^2)$ memory).

| Parameter | Config Key | Default | Description |
| :--- | :--- | :--- | :--- |
| Transform Layers | `numPairTransformLayers` | 5 | Number of blocks. |
| Tri. Mult. Update | `includeTriangularMultiplicativeUpdate` | True | Enable triangular multiplication. |
| Tri. Attention | `includeTriangularAttention` | False | Enable triangular attention. |
| Dropout | `triangularDropout` | 0.25 | Dropout rate. |

**Selection Guide:**
- **Layers**: 5 is balanced. 2-3 for fast prototyping. 8-10 for high precision. Use 0 in Flash mode to skip.
- **Attention**: $O(L^3)$ computational cost. Enable only with ample GPU memory and high precision needs.

---

##### 5. Structure Network (IPA)

Updates 3D coordinates using Invariant Point Attention (IPA).

| Parameter | Config Key | Default | Description |
| :--- | :--- | :--- | :--- |
| Structure Layers | `numStructureLayers` | 5 | Number of IPA layers. |
| IPA Hidden Dim | `ipaHiddenDimension` | 16 | Hidden dimension per head. |
| IPA Heads | `ipaNumHeads` | 12 | Number of attention heads. |
| IPA Q/K Points | `ipaNumQkPoints` | 4 | 3D points for Q/K per head. |
| IPA V Points | `ipaNumVPoints` | 8 | 3D points for Value per head. |

**Selection Guide:**
- **IPA Parameters**: Default values (16/12/4/8) match AlphaFold2 and are highly recommended.
- **Layers**: 5 is standard. Increase for higher quality at the cost of memory.

---

##### Recommended Configurations

**Standard (Paper Default):**
5 layers for both Pair Transform and Structure networks, feature dimensions set to 128.

**Memory Efficient:**
Reduce layers to 3 and feature dimensions to 64.

**High Precision:**
Increase layers to 8+ and enable `includeTriangularAttention`.

---

### Flash-IPA Optimization

This implementation integrates **Flash-IPA**, modified to support PyTorch 2.9+. No separate installation is needed.

#### Mathematical Principles of Flash-IPA

Standard IPA has $O(L^2)$ complexity. Flash-IPA achieves $O(L)$ through:

1.  **Low-rank Decomposition for Edge Embeddings**:
    $$Z_{ij} \approx Z^{(1)}_i \cdot (Z^{(2)}_j)^T$$
    Reduces memory from $O(L^2 \cdot C_z)$ to $O(L \cdot r \cdot C_z)$.

2.  **Sparse k-NN Attention**:
    Computes attention only for the $k$ nearest neighbors in 3D space: $O(L \cdot k)$.

3.  **Flash Attention Fused Kernels**:
    Avoids storing the full attention matrix using tiling and recomputation.

**Flash-IPA Modes:**

- **Standard Flash-IPA** (`useFlashIPA True`): Enabled automatically for sequences > 512 residues.
- **Memory Efficient Flash Mode** (`useFlashMode True`): Skips PairTransformNet and computes edge features dynamically.

**Flash Attention 3 (Hopper GPUs Only):**
Supports **Flash Attention 3** for NVIDIA H100/H800 (SM90), providing further speedups via TMA and optimized kernels.

---

#### Flash-IPA Hyperparameters Detail

Referencing [Flash IPA paper](https://arxiv.org/abs/2505.11580) (Liu et al., 2025):

##### `zFactorRank` - Edge Embedding Decomposition Rank

**Principle**: Approximates the $L \times L$ pair embedding $Z$ with two 1D factors: $Z_{ij} \approx Z^{(1)}_i \cdot (Z^{(2)}_j)^T$.
- **Impact**: Reduces memory from $O(L^2)$ to $O(L \cdot r)$.
- **Recommendation**: 4-8 for short sequences; 2-4 for medium; 1-2 for long or memory-constrained.

> [!WARNING]
> **Hardware Constraint**: Flash Attention 2 kernels require `headdim` ≤ 256. If your effective `headdim` (calculated from IPA parameters and `zFactorRank`) exceeds 256, it will fallback to standard IPA. For default IPA settings, `zFactorRank` should be ≤ 7.

##### `kNeighbors` - Nearest Neighbors Count

**Principle**: Uses **sparse attention**. Only weights for the $k$ nearest spatial neighbors are computed.
- **Impact**: Complexity $O(L \cdot k)$. CAPTURES physical locality of proteins.
- **Recommendation**: 10-16 for standard training; 16-32 for high precision; 6-10 for extremely long sequences.

---

### Large Batch Training Optimization

When training with large batches (e.g., 512), training can be unstable. We provide:
1. **Automatic Learning Rate Scaling**: Square-root rule ($lr_{new} = lr_{base} \times \sqrt{batch/batch_{base}}$).
2. **LR Warmup**: Linear increase over the first $N$ epochs.
3. **Cosine Annealing**: Gradual decay after warmup.
4. **Gradient Clipping**: Essential for preventing gradient explosion.
5. **Gradient Accumulation**: Simulate large batches with less memory.

---

### mHC Mode: Manifold-Constrained Hyper-Connections

Based on [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880) (Xie et al., DeepSeek-AI, 2025).

**Key Features:**
- 🔄 Expanded residual flow (n-x width).
- 🎯 Doubly stochastic residual mixing via Sinkhorn-Knopp algorithm.
- ⚖️ Preserves identity mapping for smooth gradient flow.
- 🖥️ No Flash Attention dependency (works on all GPUs).

**mHC Workflow:**
1. **Flow Expansion**: Input flow $x$ is expanded to $n$ parallel flows (default $n=4$).
2. **Dynamic Mapping**: Compute PRE, POST, and RES mappings combining dynamic components (RMSNorm + Linear) and static learned gates.
3. **Constraints**: Apply Sigmoid for non-negativity (PRE/POST) and **Sinkhorn-Knopp** algorithm (20 iters) to project RES mapping to the Birkhoff polytope (doubly stochastic matrices).
4. **Forward Pass**: $x'_{l+1} = H_{\text{res}} \otimes x' + H_{\text{post}}^T \otimes F(H_{\text{pre}} \otimes x')$.
5. **Flow Contraction**: Final output is averaged across $n$ flows.

**mHC + Flash-IPA Combination:**
Combine both for maximum stability and memory efficiency. Recommended for very long sequences (512-1024+ residues).

---

### mHC Loss Regularization

A lightweight alternative to changing architecture. Adds mHC-inspired regularization to the loss function without extra memory cost.

$$\mathcal{L}_{total} = \mathcal{L}_{RMSD} + \lambda \cdot (0.5 \cdot \mathcal{L}_{norm} + 0.5 \cdot \mathcal{L}_{mag})$$

- **Norm Preservation**: Ensures $\|\hat{\epsilon}\| \approx \|\epsilon\|$.
- **Magnitude Penalty**: Prevents residual explosion.

---

## 2. Sampling

Generate protein backbones using a pre-trained model.

#### Standard Sampling
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

#### Flash Mode Sampling (Memory Efficient)
For long sequences (>256 residues):
```bash
python genie/sample.py --flash_mode ...
```
Or use the dedicated script:
```bash
python genie/flash_sample.py --flash_mode ...
```

---

## 3. Visualization

Visualize generated structures (.npy files) as PDB or GIF.

```bash
python evaluations/visualize.py <input_file> -o <output_dir>
python evaluations/visualize_trajectory.py <traj_npy> <output_gif>
```

---

## 4. Analysis and Evaluation

### Quality Evaluation (scTM & pLDDT)
Runs ProteinMPNN and ESMFold to calculate self-consistency TM-score and pLDDT.
```bash
python evaluations/pipeline/evaluate.py --input_dir ... --output_dir ...
```

### Novelty Evaluation
Compares design structures against reference databases (PDB) using TM-align.
- **CPU Version**: Precise, exhaustive search.
- **GPU Version**: Hybrid method for fast screening.

### Plotting
Unified script for MDS plots, quality analysis (reproducing paper figures), and 3D structure gallery.
```bash
python evaluations/plot.py --input_dir ... --output_dir ... --plot all
```

---

## 5. Project Structure

- `genie/`: Core package (diffusion, models, data).
- `evaluations/`: Evaluation pipeline components.
- `packages/`: External tools (TMscore, Flash Attention).
- `scripts/`: Setup and utility scripts.
- `weights/`: Pre-trained model checkpoints.

---

## 6. Gallery

### Generation Process
![Generation Process](process.gif)

---

## 7. Optimization Results

![Optimization Comparison](Training_process_parameters/optimization_comparison.png)

| Metric | Original Work | This Work (Optimized) | Full Flash Mode |
| :--- | :--- | :--- | :--- |
| **Training Time (500 Epochs)** | ~25.7 hours | ~12.8 hours | ~8.2 hours |
| **Max GPU Memory** | ~29.53 GB | ~25.92 GB | ~1.48 GB |
| **Speedup** | 1.0x | ~2.0x | **3.1x** |
| **Memory Reduction** | - | 12% | **95%** |

The standard optimization doubles training speed, while Full Flash Mode provides massive memory savings (95% reduction), enabling training of very long sequences on consumer GPUs.

---

**Generated Quality Comparison:**
Models maintain high quality across optimizations.

| Original | Optimized |
| :---: | :---: |
| ![Original](Training_process_parameters/origin_work_hybrid.png) | ![Optimized](Training_process_parameters/this_work_hybrid.png) |

**Novel Structure Examples:**
![Novel Structures](Training_process_parameters/this_work_structure_examples_novel.png)
