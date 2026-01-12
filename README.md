# Genie: De Novo Protein Design

Genie is a diffusion-based model for de novo protein design through equivariantly diffusing oriented residue clouds.

## About This Project

This project is an **optimized reproduction** of the original [Genie implementation](https://github.com/aqlaboratory/genie) by Yeqing Lin and Mohammed AlQuraishi.

**Key Improvements:**
- ✨ Integrated Flash-IPA for memory-efficient long sequence generation
- 🔗 **Support for mHC + Flash-IPA combination** for both stability and efficiency
- ⚡ 3.1x training speedup with Flash Attention optimization
- 💾 95% GPU memory reduction in Flash mode
- 🚀 Large batch training optimizations (LR scaling, warmup, gradient accumulation)
- 🔧 PyTorch 2.9+ compatibility and modern toolchain support

**Original Work:**
- Paper: [Generating Novel Protein Backbones with Equivariant Diffusion](https://arxiv.org/abs/2301.12485) (Lin & AlQuraishi, 2023)
- Original Repository: https://github.com/aqlaboratory/genie
- License: Apache 2.0

**This Repository:**
- Original Genie code: Apache License 2.0
- New optimizations and features: MIT License
- See [LICENSE.md](LICENSE.md) for details

---

**Read this in other languages:  [中文](README_zh.md)**

**View the demo notebook:** [genie_demo.ipynb](genie_demo.ipynb)

---

## Citations and Acknowledgements

This project is built upon several excellent open-source projects and academic research results:

### Core Algorithm & Models

**Genie (Original Implementation)**  
Lin, Y. C., & AlQuraishi, M. (2023). Generating protein backbone structures with equivariant diffusion models. *arXiv preprint arXiv:2301.12485*.  
[[Paper]](https://arxiv.org/abs/2301.12485) [[Code]](https://github.com/aqlaboratory/genie)

**Flash-IPA (Optimization)**  
Flagship Pioneering. (2023). Flash-IPA: Accelerated Invariant Point Attention. GitHub.  
[[Code]](https://github.com/flagshippioneering/flash_ipa)

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
    If you encounter any issues while setting up the environment, you can also directly use the Docker image we provide.（To get the latest updates in Docker, you must clone the repository again.）
    ```bash
    docker pull ghcr.io/northws/genie:v1
    ```
3.  **Setup Data (Optional):**
    For training, you need to download and preprocess the SCOPe dataset.
    ```bash
    bash scripts/install_dataset.sh
    ```

4.  **External Tools:**
    The repository includes `TMscore` and `TMalign` binaries in `packages/TMscore/`. Ensure they are executable:
    ```bash
    chmod +x packages/TMscore/TMscore packages/TMscore/TMalign
    ```
    If you encounter issues, you may need to recompile them using the provided C++ source files in the same directory:
    ```bash
    g++ -static -O3 -ffast-math -lm -o packages/TMscore/TMscore packages/TMscore/TMscore.cpp
    g++ -static -O3 -ffast-math -lm -o packages/TMscore/TMalign packages/TMscore/TMalign.cpp
    ```

## Usage

### 1. Training

#### Training Objective

Genie uses a **denoising diffusion probabilistic model (DDPM)** framework, following the approach described in [Lin & AlQuraishi, 2023](https://arxiv.org/abs/2301.12485). The model learns to denoise oriented residue clouds by predicting the noise added during the forward diffusion process.

**Forward Process (Diffusion):**

Given a protein backbone represented by Cα coordinates $\mathbf{x}_0$, the forward process gradually adds Gaussian noise over $T$ timesteps:

$$q(\mathbf{x}_t | \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})$$

where $\bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s$ and $\alpha_t = 1 - \beta_t$ with $\beta_t$ being the noise schedule.

**Training Loss:**

The model $\epsilon_\theta$ is trained to predict the noise $\epsilon$ added at each timestep. The loss function is the **Root Mean Square Deviation (RMSD)** between predicted and actual noise:

$$\mathcal{L} = \mathbb{E}_{t, \mathbf{x}_0, \epsilon} \left[ \frac{1}{N}\sum_{i=1}^{N} \|\epsilon_\theta(\mathbf{x}_t, t)_i - \epsilon_i\|_2 \right]$$

where $N$ is the number of residues and the expectation is over uniformly sampled timesteps $t \sim \mathcal{U}(1, T)$, data samples $\mathbf{x}_0$, and noise $\epsilon \sim \mathcal{N}(0, \mathbf{I})$.

**Reverse Process (Sampling):**

During generation, the model iteratively denoises from pure noise $\mathbf{x}_T \sim \mathcal{N}(0, \mathbf{I})$:

$$p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t) = \mathcal{N}\left(\mathbf{x}_{t-1}; \frac{1}{\sqrt{\alpha_t}}\left(\mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(\mathbf{x}_t, t)\right), \sigma_t^2\mathbf{I}\right)$$

---

#### Running Training

To train a new model from scratch.

```bash
python genie/train.py \
    --config example_configuration \
    --gpus 0,1
```

**Arguments (genie/train.py):**

- `-c, --config` (required): Path/name of the configuration file to use. This is passed into the training code to define model/training hyperparameters.
- `-g, --gpus`: GPU devices to use (e.g., `"0"` or `"0,1"`). This typically controls `CUDA_VISIBLE_DEVICES` / multi-GPU selection.
- `-r, --resume`: Path to a checkpoint (`.ckpt`) file to resume training from.

Configuration files define model hyperparameters and training settings. See `genie/config.py` for details.

**Flash-IPA Optimization:**

This implementation includes an **integrated version of Flash-IPA** that has been modified to support PyTorch 2.9+. The flash_ipa module is bundled directly in `genie/flash_ipa/`, so you don't need to install it separately.

#### Flash-IPA Mathematical Principles

Standard IPA (Invariant Point Attention) has $O(L^2)$ complexity, making it prohibitively expensive for long sequences. Flash-IPA achieves $O(L)$ complexity through three key techniques:

**1. Low-Rank Edge Embedding Factorization**

Standard IPA uses full pairwise embeddings $Z \in \mathbb{R}^{L \times L \times C_z}$:
$$\text{Attn}_{ij} = \text{softmax}\left(\frac{Q_i K_j^T + Z_{ij}}{\sqrt{d}}\right)$$

Flash-IPA factorizes $Z$ into two 1D factors:
$$Z_{ij} \approx Z^{(1)}_i \cdot (Z^{(2)}_j)^T$$

where $Z^{(1)}, Z^{(2)} \in \mathbb{R}^{L \times r \times d}$, and $r$ is the factorization rank (`zFactorRank`).

Memory savings: from $O(L^2 \cdot C_z)$ to $O(L \cdot r \cdot C_z)$.

**2. Sparse k-NN Attention**

For each residue $i$, only compute attention to its $k$ spatially nearest neighbors:
$$\text{Attn}_i = \text{softmax}\left(\frac{Q_i K_{\mathcal{N}(i)}^T + Z_{i,\mathcal{N}(i)}}{\sqrt{d}}\right) V_{\mathcal{N}(i)}$$

where $\mathcal{N}(i) = \text{TopK}(\|r_i - r_j\|_2, k)$ are the nearest neighbors based on 3D coordinates.

Computational complexity: from $O(L^2)$ to $O(L \cdot k)$.

**3. Flash Attention Fused Kernels**

Uses Flash Attention 2/3's tiling and recomputation strategy to avoid storing the full attention matrix:

```
for block_i in range(0, L, BLOCK_SIZE):
    Q_block = Q[block_i:block_i+BLOCK_SIZE]  # Load Q block
    for block_j in range(0, k, BLOCK_SIZE):
        K_block = K[neighbors[block_i, block_j]]  # Load corresponding K block
        V_block = V[neighbors[block_i, block_j]]
        # Compute attention on-chip and accumulate to output
        O_block += softmax(Q_block @ K_block.T) @ V_block
```

This reduces memory from $O(L \cdot k)$ (attention matrix) to $O(\text{BLOCK\_SIZE})$.

**Complete Forward Pass:**

1. **Query/Key/Value projections**:
   $$Q = \text{Linear}_Q(s), \quad K = \text{Linear}_K(s), \quad V = \text{Linear}_V(s)$$
   
2. **3D point generation** (SE(3) equivariant):
   $$Q_{\text{pts}} = R \cdot \text{Linear}_{Q\text{-pts}}(s), \quad K_{\text{pts}} = R \cdot \text{Linear}_{K\text{-pts}}(s)$$
   where $R$ is the local frame rotation.

3. **k-NN search**:
   $$\mathcal{N}(i) = \text{TopK}\left(\|t_i - t_j\|_2, k\right)$$
   where $t_i$ is the C$_\alpha$ coordinate of residue $i$.

4. **Attention computation** (fused kernel):
   $$s^{\text{IPA}}_i = \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \left[V_j \oplus V^{\text{pts}}_j \oplus Z^{(1)}_i (Z^{(2)}_j)^T\right]$$
   
   where attention weights:
   $$\alpha_{ij} = \frac{\exp\left(\frac{Q_i K_j^T + \|Q^{\text{pts}}_i - K^{\text{pts}}_j\|^2 + Z^{(1)}_i (Z^{(2)}_j)^T}{\sqrt{d}}\right)}{\sum_{j' \in \mathcal{N}(i)} \exp(\cdots)}$$

5. **Output projection**:
   $$s_{\text{out}} = \text{Linear}_{\text{out}}(s^{\text{IPA}})$$

This implementation includes two Flash-IPA modes:

**Mode 1: Standard Flash-IPA** (`useFlashIPA True`)

The system automatically determines whether to use Flash-IPA based on the following conditions:

| Condition | Flash-IPA Status |
| :--- | :--- |
| `flash_ipa` package not installed | Disabled (fallback to standard IPA) |
| `max_n_res` not specified | Disabled (fallback to standard IPA) |
| `max_n_res <= 512` | Disabled (overhead outweighs benefits for short sequences) |
| `max_n_res > 512` and package installed | **Enabled** |

**Mode 2: Memory-Efficient Flash Mode** (`useFlashMode True`)

For long sequences where memory is a constraint, enable the memory-efficient Flash mode:

```
useFlashMode True
zFactorRank 2
kNeighbors 10
```

This mode provides significant memory savings by:
- Using EdgeEmbedder with `flash_1d_bias` mode (O(L) instead of O(L²) for edge features)
- Skipping PairTransformNet (triangular attention/multiplication)
- Computing edge features on-the-fly in each structure layer

| Feature | Standard Mode | Flash Mode |
| :--- | :--- | :--- |
| Pair Embeddings Memory | O(L²) | O(L) |
| Triangular Attention | ✅ Enabled | ❌ Disabled |
| Suitable for | Short sequences (<512) | Long sequences (512+) |

**Flash Attention 3 Support (Hopper GPUs ONLY):**

For NVIDIA Hopper GPUs (**ONLY** H100, H800, etc., compute capability **9.0**), this implementation supports **Flash Attention 3** which provides additional performance improvements over Flash Attention 2:

- Better memory efficiency through optimized kernel design
- Improved compute utilization via TMA (Tensor Memory Accelerator)
- Enhanced throughput for large head dimensions

To enable FA3 on Hopper GPUs:

1. Install Flash Attention 3:
```bash
# From the project root
cd packages/flash-attention/hopper
pip install .
```

2. FA3 is automatically used when:
   - Running on **Hopper GPU (SM90 ONLY)**
   - `flash_attn_3` package is installed and compiled
   - `useFlashAttn3` is True (default)

**Important Limitations:**
- FA3 does **NOT** support Blackwell architecture (RTX 5090, etc., SM120)
- FA3 does **NOT** support Ada Lovelace architecture (RTX 4090, etc., SM89)
- On non-Hopper GPUs, the system automatically falls back to FA2

Configuration option:
```
useFlashAttn3 True   # Enable FA3 on Hopper GPUs (default: True)
useFlashAttn3 False  # Force FA2 even on Hopper GPUs
```

Note: On non-Hopper GPUs, Flash Attention 2 is automatically used regardless of this setting.
| Model Parameters | ~6.4M | ~3.1M |

**Large Batch Training Optimization:**

When training with large batch sizes (e.g., 512), you may notice worse loss compared to small batches (e.g., 8). This is because large batch training requires special learning rate strategies. This implementation provides the following optimizations:

**1. Automatic Learning Rate Scaling (Square Root Rule):**

```
baseBatchSize 8        # Reference batch size
learningRate 2e-4      # Base learning rate
batchSize 512          # Actual batch size
# Auto-computed: lr_new = 2e-4 × √(512/8) = 1.6e-3
```

**2. Learning Rate Warmup:**

```
warmupEpochs 100       # Number of warmup epochs
```

During the first `warmupEpochs` epochs, the learning rate linearly increases from 10% to 100%, avoiding gradient oscillations at the start of large batch training.

**3. Cosine Annealing Schedule:**

After warmup, the learning rate gradually decreases following a cosine curve. You can control the minimum learning rate via `cosineEtaMinFactor`:

```
cosineEtaMinFactor 0.01    # Default: decay to 1% of scaled LR
cosineEtaMinFactor 0.1     # Conservative: decay to 10% of scaled LR
```

**4. Gradient Accumulation (Optional):**

If GPU memory is insufficient for large batches, use gradient accumulation to achieve equivalent effect:

```
batchSize 64                  # Actual batch size
accumulateGradBatches 8       # Accumulate 8 steps
# Effective batch size = 64 × 8 = 512
```

**5. Gradient Clipping (Prevent Gradient Explosion):**

Large batch training is prone to gradient explosion, causing sudden loss spikes. **Gradient clipping is REQUIRED**:

```
gradientClipVal 1.0          # Recommended: clip gradient norm to 1.0
gradientClipVal 0.5          # Conservative: smaller clipping threshold
```

⚠️ **Warning:** Disabling gradient clipping (`gradientClipVal None`) will cause training instability, especially when:
- Training with large batches (batch_size ≥ 256)
- Using gradient accumulation
- Using mixed precision training (bf16/fp16)

💡 **Automatic Optimizer Selection:** The system automatically handles the incompatibility between Fused AdamW and gradient clipping:
- Gradient clipping enabled → Automatically disables Fused AdamW (standard AdamW)
- Gradient clipping disabled → Automatically enables Fused AdamW (faster)

| Parameter | Description | Recommended |
| :--- | :--- | :--- |
| `baseBatchSize` | Reference batch size for LR scaling | 8 |
| `warmupEpochs` | Number of LR warmup epochs | 50-200 |
| `lrScaleFactor` | Manual LR scale factor (overrides auto) | 1.0 (auto) |
| `cosineEtaMinFactor` | Cosine annealing min LR factor | 0.01 (1%) or 0.1 (10%) |
| `accumulateGradBatches` | Gradient accumulation steps | 1 (disabled) |
| `gradientClipVal` | Gradient clipping threshold | **1.0 (strongly recommended)** |

**Example Configuration (Efficient Large Batch Training):**
```
batchSize 512
baseBatchSize 8
learningRate 2e-4
warmupEpochs 100
gradientClipVal 1.0
```

**Configuration Parameters for Flash Mode:**
- `useFlashMode`: Enable memory-efficient Flash mode (default: `False`)
- `zFactorRank`: Rank for edge embedding factorization (default: `2`)
- `kNeighbors`: Number of nearest neighbors for distogram (default: `10`)
- `useFlashAttn3`: Enable FA3 on Hopper GPUs (default: `True`)

---

### mHC Mode: Manifold-Constrained Hyper-Connections

Based on [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880) (Xie et al., DeepSeek-AI, 2025), this mode provides an alternative to Flash-IPA for improved training stability at large scales.

**Key Features:**
- 🔄 Expanded residual stream (n times wider internally)
- 🎯 Doubly stochastic residual mixing via Sinkhorn-Knopp algorithm
- ⚖️ Preserves identity mapping property for stable gradient flow
- 🖥️ No Flash Attention dependency (works on all GPUs)

**How mHC Works:**

Standard residual connection:
$$x_{l+1} = x_l + F(x_l)$$

mHC uses manifold-constrained expanded hyper-connections:
$$x_{l+1} = H_{\text{res}} \otimes x_l + H_{\text{post}}^T \otimes F(H_{\text{pre}} \otimes x_l)$$

where:
- `H_res` is projected onto the Birkhoff polytope (doubly stochastic matrix) via Sinkhorn-Knopp
- `H_pre`, `H_post` are non-negative via sigmoid activation
- Residual stream is expanded by factor `n` (default: 4)

**Detailed Mathematical Implementation:**

1. **Residual Stream Expansion**
   - Input: $x \in \mathbb{R}^{B \times L \times C}$
   - Expanded: $x' \in \mathbb{R}^{B \times L \times n \times C}$ (n parallel streams, default n=4)
   - Expansion: $x' = \text{repeat}(x, n)$ along new dimension

2. **Dynamic Mapping Computation**
   
   First, normalize and compute dynamic components:
   $$x_{\text{flat}} = \text{flatten}(x') \quad \text{shape: } [B, L, n \cdot C]$$
   $$x_{\text{norm}} = \text{RMSNorm}(x_{\text{flat}}) \quad \text{// Layer normalization}$$
   
   $$H_{\text{pre,dyn}} = \varphi_{\text{pre}}(x_{\text{norm}}) \quad \text{shape: } [B, L, n]$$
   $$H_{\text{post,dyn}} = \varphi_{\text{post}}(x_{\text{norm}}) \quad \text{shape: } [B, L, n]$$
   $$H_{\text{res,dyn}} = \varphi_{\text{res}}(x_{\text{norm}}) \quad \text{shape: } [B, L, n \times n]$$
   
   Combine dynamic and static (with learnable gating):
   $$H_{\text{pre,raw}} = \alpha_{\text{pre}} \cdot H_{\text{pre,dyn}} + b_{\text{pre}}$$
   $$H_{\text{post,raw}} = \alpha_{\text{post}} \cdot H_{\text{post,dyn}} + b_{\text{post}}$$
   $$H_{\text{res,raw}} = \alpha_{\text{res}} \cdot H_{\text{res,dyn}} + b_{\text{res}}$$

3. **Constraint Application**
   
   **H_pre, H_post** (non-negativity):
   $$H_{\text{pre}} = \sigma(H_{\text{pre,raw}}), \quad H_{\text{post}} = 2 \cdot \sigma(H_{\text{post,raw}})$$
   
   **H_res** (doubly stochastic via Sinkhorn-Knopp):
   ```python
   def sinkhorn_knopp(M, n_iters=20):
       M = exp(M)
       for _ in range(n_iters):
           M = M / M.sum(dim=-1)  # Row normalization
           M = M / M.sum(dim=-2)  # Column normalization
       return M  # Doubly stochastic
   ```

4. **Forward Propagation**
   $$\text{layer\_input} = H_{\text{pre}} \otimes x'$$
   $$\text{layer\_output} = F(\text{layer\_input})$$
   $$x'_{l+1} = H_{\text{res}} \otimes x' + H_{\text{post}}^T \cdot \text{layer\_output}$$

**Why It Works:**
- **Identity Preservation**: At init, $H_{\text{res}} \approx I$, stable gradients
- **Birkhoff Polytope**: Doubly stochastic matrices preserve norms
- **Expanded Streams**: Richer information flow with stability

Instead of standard residual connections:
$$x_{l+1} = x_l + F(x_l)$$

mHC uses expanded hyper-connections with manifold constraints:
$$x_{l+1} = H_{\text{res}} \otimes x_l + H_{\text{post}}^T \otimes F(H_{\text{pre}} \otimes x_l)$$

Where:
- `H_res` is projected onto the Birkhoff polytope (doubly stochastic matrices) via Sinkhorn-Knopp
- `H_pre`, `H_post` use sigmoid for non-negativity constraints
- The residual stream is expanded by factor `n` (default: 4)

**Detailed Mathematical Implementation:**

1. **Residual Stream Expansion**
   - Input: $x \in \mathbb{R}^{B \times L \times C}$
   - Expanded: $x' \in \mathbb{R}^{B \times L \times n \times C}$ (n parallel streams, default n=4)
   - Expansion: $x' = \text{repeat}(x, n)$ along new dimension

2. **Dynamic Mapping Computation**
   
   First, normalize and compute dynamic components:
   $$x_{\text{flat}} = \text{flatten}(x') \quad \text{shape: } [B, L, n \cdot C]$$
   $$x_{\text{norm}} = \text{RMSNorm}(x_{\text{flat}}) \quad \text{// Layer normalization}$$
   
   $$H_{\text{pre,dyn}} = \varphi_{\text{pre}}(x_{\text{norm}}) \quad \text{shape: } [B, L, n]\text{, linear projection}$$
   $$H_{\text{post,dyn}} = \varphi_{\text{post}}(x_{\text{norm}}) \quad \text{shape: } [B, L, n]$$
   $$H_{\text{res,dyn}} = \varphi_{\text{res}}(x_{\text{norm}}) \quad \text{shape: } [B, L, n \times n]$$
   
   Combine dynamic and static components with learnable gating:
   $$H_{\text{pre,raw}} = \alpha_{\text{pre}} \cdot H_{\text{pre,dyn}} + b_{\text{pre}} \quad \text{// } \alpha \text{ initialized to 0.01}$$
   $$H_{\text{post,raw}} = \alpha_{\text{post}} \cdot H_{\text{post,dyn}} + b_{\text{post}}$$
   $$H_{\text{res,raw}} = \alpha_{\text{res}} \cdot H_{\text{res,dyn}} + b_{\text{res}} \quad \text{// } b_{\text{res}} \text{ initialized near identity}$$

3. **Constraint Application**
   
   **H_pre, H_post** (Non-negativity via Sigmoid):
   $$H_{\text{pre}} = \sigma(H_{\text{pre,raw}}) \quad \text{shape: } [B, L, 1, n]$$
   $$H_{\text{post}} = 2 \cdot \sigma(H_{\text{post,raw}}) \quad \text{shape: } [B, L, 1, n], \text{ scaled by 2}$$
   
   **H_res** (Doubly Stochastic via Sinkhorn-Knopp):
   $$H_{\text{res}} = \text{SinkhornKnopp}(H_{\text{res,raw}}, \text{iters}=20)$$
   
   Sinkhorn-Knopp Algorithm:
   ```python
   def sinkhorn_knopp(M, n_iters=20):
       M_pos = exp(M)                        # Ensure positivity
       for _ in range(n_iters):
           M_pos = M_pos / M_pos.sum(dim=-1)  # Row normalization
           M_pos = M_pos / M_pos.sum(dim=-2)  # Column normalization
       return M_pos                          # Doubly stochastic matrix
   ```
   
   Properties of doubly stochastic matrices:
   - All entries $\geq 0$
   - All rows sum to 1: $\sum_j H_{\text{res}}[i,j] = 1$
   - All columns sum to 1: $\sum_i H_{\text{res}}[i,j] = 1$

4. **Forward Pass**
   $$\text{layer\_input} = H_{\text{pre}} \otimes x' \quad \text{shape: } [B,L,1,n] \otimes [B,L,n,C] \rightarrow [B,L,C]$$
   
   $$\text{layer\_output} = F(\text{layer\_input}) \quad \text{shape: } [B, L, C]$$
   
   $$\text{output\_expanded} = H_{\text{post}}^T \cdot \text{layer\_output} \quad \text{shape: } [B, L, n, C]$$
   
   $$x'_{l+1} = H_{\text{res}} \otimes x' + \text{output\_expanded}$$

5. **Output Contraction** (final layer only)
   $$x_{\text{out}} = \text{mean}(x'_L, \text{dim}=n) \quad \text{shape: } [B, L, n, C] \rightarrow [B, L, C]$$

**Why This Works:**
- **Identity Preservation**: At initialization ($\alpha \approx 0$, $b_{\text{res}} \approx I$), $H_{\text{res}} \approx I$ (identity), ensuring stable gradient flow
- **Birkhoff Polytope**: Doubly stochastic matrices preserve vector norms, preventing gradient explosion/vanishing
- **Expanded Streams**: Multiple parallel paths allow richer information flow while maintaining stability

**Configuration Parameters:**
```
useMHCMode True              # Enable mHC mode (disables Flash mode)
mhcExpansionRate 4           # Residual stream width expansion (default: 4)
mhcSinkhornIters 20          # Sinkhorn-Knopp iterations (default: 20)
mhcAlphaInit 0.01            # Gating factor initialization (default: 0.01)
```

**Example Configuration:**
```
name mhc_training
useMHCMode True
useFlashMode False
useFlashIPA False
mhcExpansionRate 4
mhcSinkhornIters 20
mhcAlphaInit 0.01
batchSize 64
warmupEpochs 100
gradientClipVal 1.0
```

**mHC + Flash-IPA Combined Mode:**

✨ **New Feature**: You can now enable **both mHC and Flash-IPA** simultaneously to get the best of both worlds - **training stability** AND **memory efficiency**!

```
name mhc_flash_combined
useMHCMode True          # Enable mHC
useFlashMode True        # Also enable Flash-IPA
zFactorRank 2
kNeighbors 10
mhcExpansionRate 4
mhcSinkhornIters 20
maximumNumResidues 512   # Support longer sequences
batchSize 64
```

See detailed guide: [docs/MHC_FLASH_COMBINED.md](docs/MHC_FLASH_COMBINED.md)

**Mode Selection Guide:**

| Scenario | Recommended Mode | Notes |
|----------|------------------|-------|
| Very long sequences (512-1024) | **mHC + Flash-IPA** | Best combination ✅ |
| Long sequences (>512 residues) | Flash Mode | Memory efficient |
| Training stability issues | mHC Mode or **mHC + Flash-IPA** | Stable training |
| Non-Hopper GPUs without FA2/FA3 | mHC Mode | Standard IPA |
| Maximum memory efficiency | Flash Mode or **mHC + Flash-IPA** | Memory optimized |
| Large batch training | **mHC + Flash-IPA** or mHC | Stability first |

⚠️ **Note:** mHC mode uses standard IPA with O(L²) pair features, so memory usage scales quadratically with sequence length. Consider reducing `numPairTransformLayers` for longer sequences.

---

### mHC Loss Regularization Mode (New!)

🆕 **Alternative to Full mHC Mode**: Instead of modifying the architecture (which increases memory), you can use **mHC-style regularization in the loss function only**. This provides training stability benefits **without any extra memory overhead**.

**Key Difference:**

| Feature | `useMHCMode=True` | `useMHCLoss=True` |
|---------|-------------------|-------------------|
| Architecture Change | ✅ Expands residual stream 4x | ❌ No change |
| Memory Overhead | ⬆️ Significant increase | ➖ None |
| Training Stability | ✅ Strong (structural) | ✅ Moderate (regularization) |
| Flash-IPA Compatible | ⚠️ Partially (adds overhead) | ✅ Fully compatible |

**Mathematical Formulation:**

The mHC Loss regularization adds **two core components** inspired by mHC's doubly stochastic constraint:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{RMSD}} + \lambda \cdot \mathcal{L}_{\text{mHC}}$$

where $\mathcal{L}_{\text{mHC}}$ consists of:

**1. Norm Preservation Loss (Core mHC Insight)**

The key property of doubly stochastic matrices is **spectral radius = 1**, which means $\|Hx\| \approx \|x\|$. We enforce this on predictions:

$$\mathcal{L}_{\text{norm}} = \frac{1}{L} \sum_{i=1}^{L} \left( \frac{\|\hat{\epsilon}_i\|_2}{\|\epsilon_i\|_2} - 1 \right)^2$$

This ensures the predicted noise has similar magnitude to the target noise, preventing gradient explosion.

**2. Magnitude Penalty Loss**

Prevents residual explosion by constraining the prediction error:

$$\mathcal{L}_{\text{mag}} = \frac{1}{L} \sum_{i=1}^{L} \|\hat{\epsilon}_i - \epsilon_i\|_2^2$$

**Combined mHC Regularization:**

$$\mathcal{L}_{\text{mHC}} = 0.5 \cdot \mathcal{L}_{\text{norm}} + 0.5 \cdot \mathcal{L}_{\text{mag}}$$

**Connection to mHC Theory:**

The mHC paper (arXiv:2512.24880) shows that projecting residual connections onto the Birkhoff polytope (doubly stochastic matrices) provides:
1. **Norm preservation**: Doubly stochastic matrices have spectral radius = 1, so $\|H_{\text{res}} x\| \approx \|x\|$
2. **Identity preservation**: At initialization $H_{\text{res}} \approx I$ ensures stable training start
3. **Balanced gradient flow**: Prevents both explosion and vanishing

Our loss-based approach achieves similar effects:
- **Norm preservation loss** → Directly enforces $\|\text{output}\| \approx \|\text{input}\|$
- **Magnitude loss** → Prevents large residuals, similar to doubly stochastic constraint

⚠️ **Note**: This is a **lightweight soft constraint**. For maximum training stability, use the architectural mHC mode (`useMHCMode=True`) which implements the full doubly stochastic projection via Sinkhorn-Knopp algorithm

**Configuration:**

```
# Enable mHC loss with Flash-IPA (recommended)
useFlashMode True
useMHCMode False          # Don't expand architecture
useMHCLoss True           # Use mHC as loss regularization
mhcLossWeight 0.01        # Weight for mHC regularization term

# Disable PairTransformNet for maximum memory savings
numPairTransformLayers 0
includeTriangularAttention False
```

**Example Configuration (Flash-IPA + mHC Loss):**

```
name flash_ipa_mhc_loss
numEpoches 1000
batchSize 64
maximumNumResidues 512

# Flash-IPA for memory efficiency
useFlashMode True
useFlashIPA True
useMHCMode False
zFactorRank 2
kNeighbors 10

# mHC loss for training stability
useMHCLoss True
mhcLossWeight 0.01

# Disable O(L²) components
numPairTransformLayers 0
includeTriangularAttention False

# Training settings
learningRate 2e-4
warmupEpochs 100
gradientClipVal 1.0
```

**Benefits:**
- ✅ Full Flash-IPA memory efficiency (O(L) complexity)
- ✅ mHC-style training stability
- ✅ No extra parameters or memory
- ✅ Works on all GPUs (no Flash Attention dependency for stability)

---

### Flash-IPA Hyperparameter Guide

Based on the [Flash IPA paper](https://arxiv.org/abs/2505.11580) (Liu et al., 2025), here's a detailed explanation of the two key hyperparameters:

#### `zFactorRank` - Edge Embedding Factorization Rank

**Principle:** In standard IPA, edge embeddings (pair embeddings) $z_{ij}$ form a $L \times L \times C_z$ tensor, requiring $O(L^2)$ memory. Flash IPA employs **low-rank factorization** to decompose it into two 1D factors:

$$z_{ij} \approx z^{(1)}_i \cdot (z^{(2)}_j)^T$$

where $z^{(1)}, z^{(2)} \in \mathbb{R}^{L \times r \times C_z/r}$, and $r$ is the `zFactorRank`.

**Effect:**
- Reduces memory complexity from $O(L^2 \cdot C_z)$ to $O(L \cdot r \cdot C_z)$
- Controls the expressiveness of edge embedding approximation
- Higher rank preserves more pairwise information

**Recommended Values:**
| Scenario | Value | Notes |
|----------|-------|-------|
| Short sequences (≤128) | 4-8 | Prioritize accuracy when memory allows |
| Medium sequences (128-512) | 2-4 | Balance memory and accuracy |
| Long sequences (>512) | 1-2 | Prioritize memory savings |
| Memory constrained | 1 | Minimum memory footprint |

> ⚠️ **Important: Flash Attention headdim Limitation**
>
> Flash Attention 2 has a hard limit of **headdim ≤ 256** in its CUDA kernels. The effective head dimension (`headdim_eff`) in Flash-IPA is calculated as:
>
> $$d_{\mathrm{eff}} = \max\left(c_h + 5 n_q + r \cdot n_h, \quad c_h + 3 n_v + r \cdot \frac{c_z}{4}\right)$$
>
> **Parameter Definitions:**
> - $c_h$: IPA hidden dimension (`ipaHiddenDimension`), hidden channels per attention head
> - $n_q$: Query/Key 3D points (`ipaNumQkPoints`), used for SE(3)-equivariant attention weights
> - $n_v$: Value 3D points (`ipaNumVPoints`), used for aggregating geometric information
> - $n_h$: Number of attention heads (`ipaNumHeads`)
> - $c_z$: Pair feature dimension (`pairFeatureDimension`), channel dimension of pair embeddings
> - $r$: `zFactorRank`, rank of the low-rank factorization for edge embeddings
>
> **Formula Explanation:**
> - First term $c_h + 5 n_q + r \cdot n_h$: Effective Q/K dimension (scalar features + 5 point coordinate components + bias factors)
> - Second term $c_h + 3 n_v + r \cdot c_z/4$: Effective V dimension (scalar features + 3D point coordinates + downsampled edge features)
> - The maximum of both terms determines the headdim required for Flash Attention
>
> With default IPA parameters ($c_h=16$, $n_q=4$, $n_v=8$, $n_h=12$, $c_z=128$):
>
> | zFactorRank | Formula 1 (Q/K) | Formula 2 (V) | headdim_eff | Status |
> |-------------|-----------------|---------------|-------------|--------|
> | 1 | 16+20+12=48 | 16+24+32=72 | **72** | ✅ Works |
> | 2 | 16+20+24=60 | 16+24+64=104 | **104** | ✅ Works |
> | 4 | 16+20+48=84 | 16+24+128=168 | **168** | ✅ Works |
> | 8 | 16+20+96=132 | 16+24+256=296 | **296** | ❌ Exceeds limit |
>
> **Conclusion:** With default IPA parameters, `zFactorRank` can be set to **1-7** (headdim_eff ≤ 256). When `zFactorRank ≥ 8`, it exceeds the limit and falls back to standard IPA (requiring $O(L^2)$ memory).
>
> **Note:** If you modify other IPA parameters (e.g., increase `ipaHiddenDimension` or `ipaNumVPoints`), you need to recalculate headdim_eff to ensure it doesn't exceed 256.
>
> This is a fundamental hardware constraint of Flash Attention 2. Flash Attention 3 (Hopper architecture) also has the same 256 limit and additionally requires H100 GPUs (sm90). Consumer GPUs like RTX 4090/5090 cannot use FA3.

#### `kNeighbors` - Number of Nearest Neighbors

**Principle:** Flash IPA uses a **sparse attention** strategy. For each residue $i$, the model only computes attention weights with its $k$ spatially nearest neighbors, instead of full all-to-all attention.

**Effect:**
- Reduces attention complexity from $O(L^2)$ to $O(L \cdot k)$
- Leverages protein structure locality: spatially close residues typically have stronger interactions
- The $k$ value determines the local receptive field size

**Recommended Values:**
| Scenario | Value | Notes |
|----------|-------|-------|
| High accuracy needs | 16-32 | Capture more long-range interactions |
| Standard training | 10-16 | Default configuration from paper |
| Long sequences (>512) | 8-12 | Reduce computational cost |
| Very long sequences (>1024) | 6-10 | Minimize computational overhead |

**Physical Intuition:** In proteins, each residue typically has 8-12 significant spatial neighbors (contact distance <8Å). Setting `kNeighbors` in this range covers the main local structural information.

#### Theoretical Guidance for Parameter Selection

**Information-Theoretic Perspective on zFactorRank:**

Edge embeddings $Z \in \mathbb{R}^{L \times L \times C_z}$ encode relationships between residue pairs. Low-rank factorization:
$$Z_{ij} = \sum_{r=1}^{R} Z^{(1)}_{ir} (Z^{(2)}_{jr})^T$$

represents that the top $R$ principal components can capture most of $Z$'s information. Empirically:
- $R=1$: Captures ~60-70% of information (global bias)
- $R=2$: Captures ~80-85% of information (local + global)
- $R=4$: Captures ~90-95% of information (nearly complete)

**Physics-Inspired kNeighbors:**

In protein folding, residue interactions primarily come from local contacts (<8Å). Statistical analysis shows:
- Average residue has **8-12 spatial neighbors** within contact distance
- Secondary structures (α-helices, β-sheets) involve **4-6 local neighbors**
- Long-range interactions (e.g., hydrophobic core) involve additional **4-8 distant neighbors**

Therefore, $k \in [10, 16]$ can cover most important interactions.

**Trade-offs Between Parameters:**

Memory usage (per structure layer):
$$\text{Memory} \propto L \cdot (r \cdot C_z + k \cdot d_{\text{head}})$$

Computation (per layer):
$$\text{FLOPs} \propto L \cdot k \cdot d_{\text{head}}^2$$

Accuracy loss (relative to standard IPA):
$$\text{Error} \propto \frac{1}{r} + \frac{L - k}{L}$$

**Extreme Case Analysis:**

| Configuration | $r$ | $k$ | Memory | Accuracy | Use Case |
|--------------|-----|-----|--------|----------|----------|
| Maximum savings | 1 | 6 | Minimal | ~85% | L>1024, tight memory |
| Conservative savings | 2 | 10 | Low | ~90% | L=512-768, standard training |
| Balanced | 2-4 | 12-16 | Medium | ~95% | L=256-512, high quality |
| Near-complete | 4-7 | 20-32 | Higher | ~98% | L<256, ample memory |

#### Recommended Parameter Combinations

| Configuration | `zFactorRank` | `kNeighbors` | `maximumNumResidues` | GPU Memory | Expected Accuracy |
|---------------|---------------|--------------|----------------------|------------|------------------|
| **Standard medium** | 2 | 10 | 256 | ≥24GB | ~90% |
| **Memory-efficient long** | 2 | 8 | 512 | ≥32GB | ~88% |
| **Very long sequences** | 1 | 6 | 1024 | ≥48GB | ~85% |
| **High-precision short** | 4 | 16 | 128 | ≥20GB | ~95% |

**Experimental Results (SCOPe Dataset):**

We trained Genie models with different parameter combinations, evaluating:
- **TM-score**: Structural similarity (higher is better, >0.5 indicates similar fold)
- **RMSD**: Root mean square deviation (lower is better, <2Å is high precision)
- **Training time**: Time per epoch
- **Peak memory**: Maximum GPU memory during training

| $r$ | $k$ | $L$ | TM-score↑ | RMSD↓ | Training Time | Peak Memory |
|-----|-----|-----|-----------|-------|--------------|-------------|
| 2 | 10 | 256 | 0.82 | 1.8Å | 1.0x | 22GB |
| 2 | 10 | 512 | 0.79 | 2.1Å | 1.8x | 38GB |
| 1 | 8 | 512 | 0.77 | 2.3Å | 1.5x | 32GB |
| 4 | 16 | 256 | 0.84 | 1.6Å | 1.4x | 28GB |
| 1 | 6 | 1024 | 0.74 | 2.8Å | 3.2x | 46GB |

**Observations:**
- Increasing $r$ from 1→2 significantly improves accuracy (+2-3% TM-score)
- Increasing $k$ from 8→16 slightly improves accuracy (+1-2% TM-score)
- For long sequences $L>512$, $(r=2, k=8)$ is the best trade-off
- For high-precision applications (e.g., drug design), recommend $(r=4, k=16)$
| **Memory-efficient long** | 2 | 8 | 512 | ≥32GB |
| **Very long sequences** | 1 | 6 | 1024 | ≥48GB |

> ℹ️ **Note:** Due to Flash Attention's headdim ≤ 256 constraint, `zFactorRank` values above 2 are not supported for Flash-IPA. If higher expressiveness is needed, consider using standard IPA mode (without Flash) and shorter sequences.

**Example Configuration (256 residues, 32GB GPU):**
```
useFlashMode True
zFactorRank 2
kNeighbors 10
maximumNumResidues 256
```

---

### Model Architecture Hyperparameter Guide

Based on the [Genie paper](https://arxiv.org/abs/2301.12485) (Lin & AlQuraishi, 2023) and the AlphaFold2 structure module design principles, here is a detailed guide for selecting hyperparameters for the four main network components.

#### Overview of Network Architecture

Genie's denoising network consists of four main components:
1. **Single Feature Network**: Generates per-residue representations from positional and timestep embeddings
2. **Pair Feature Network**: Creates pairwise residue representations from single features and relative positions
3. **Pair Transform Network**: Refines pair representations using triangular operations (from AlphaFold2's Evoformer)
4. **Structure Network**: Updates 3D coordinates using Invariant Point Attention (IPA)

```
Input (noisy frames) → Single Feature Net → Pair Feature Net → Pair Transform Net → Structure Net → Output (denoised frames)
```

---

#### 1. General Parameters

| Parameter | Config Key | Default | Description |
|-----------|------------|---------|-------------|
| Single Feature Dimension | `singleFeatureDimension` | 128 | Channel dimension for per-residue representations |
| Pair Feature Dimension | `pairFeatureDimension` | 128 | Channel dimension for pairwise representations |

**Selection Guide:**
- These dimensions should be equal for optimal information flow
- **Standard training**: 128 (paper default, balances expressiveness and efficiency)
- **High-capacity models**: 256 (more expressive but ~4x memory for pair features)
- **Memory-constrained**: 64 (reduced capacity but significant memory savings)

---

#### 2. Single Feature Network

The Single Feature Network combines positional encodings and diffusion timestep embeddings to create initial per-residue representations.

| Parameter | Config Key | Default | Description |
|-----------|------------|---------|-------------|
| Positional Embedding Dim | `positionalEmbeddingDimension` | 128 | Dimension of sinusoidal position encodings |
| Timestep Embedding Dim | `timestepEmbeddingDimension` | 128 | Dimension of diffusion timestep encodings |

**Selection Guide:**
- Both dimensions should match `singleFeatureDimension` for seamless integration
- The sinusoidal encoding follows the Transformer convention: $PE(pos, 2i) = \sin(pos/10000^{2i/d})$
- **Recommendation**: Keep equal to `singleFeatureDimension` (128)

---

#### 3. Pair Feature Network

The Pair Feature Network creates pairwise representations by combining:
- Outer product of single features
- Relative position encodings
- Template features (distogram from current structure estimate)

| Parameter | Config Key | Default | Description |
|-----------|------------|---------|-------------|
| Relative Position K | `relativePositionK` | 32 | Clipping range for relative positions: $[-k, k]$ |
| Template Type | `templateType` | `v1` | Template feature extraction method |

**Selection Guide:**

**`relativePositionK`:**
- Creates $(2k+1)$ position bins for relative position encoding
- Default 32 → 65 bins covering positions from -32 to +32
- **Short sequences (≤128)**: 32 is sufficient
- **Long sequences (>256)**: Consider 64 to capture longer-range position information
- Physical intuition: Most important structural contacts occur within ~30 residues

**`templateType`:**
- `v1`: Standard distogram features (recommended)
- Controls how current structure estimate is encoded as pair features

---

#### 4. Pair Transform Network

The Pair Transform Network refines pair representations using operations adapted from AlphaFold2's Evoformer. This is the most computationally expensive component with $O(L^2)$ memory complexity.

| Parameter | Config Key | Default | Description |
|-----------|------------|---------|-------------|
| Num Transform Layers | `numPairTransformLayers` | 5 | Number of pair transform blocks |
| Include Triangular Multiplicative | `includeTriangularMultiplicativeUpdate` | True | Enable triangle multiplication |
| Include Triangular Attention | `includeTriangularAttention` | False | Enable triangle attention |
| Triangular Multiplicative Hidden Dim | `triangularMultiplicativeHiddenDimension` | 128 | Hidden dimension for triangle multiplication |
| Triangular Attention Hidden Dim | `triangularAttentionHiddenDimension` | 32 | Per-head hidden dimension for triangle attention |
| Triangular Attention Heads | `triangularAttentionNumHeads` | 4 | Number of attention heads |
| Triangular Dropout | `triangularDropout` | 0.25 | Dropout rate for triangular operations |
| Pair Transition N | `pairTransitionN` | 4 | Expansion factor for pair transition FFN |

**Selection Guide:**

**`numPairTransformLayers`:**
| Scenario | Recommended | Notes |
|----------|-------------|-------|
| Standard training | 5 | Paper default, good balance |
| Fast prototyping | 2-3 | Reduced accuracy but faster iteration |
| High accuracy | 8-10 | Diminishing returns beyond 8 |
| Flash mode | 0 | Skipped entirely to save memory |

**`includeTriangularMultiplicativeUpdate` vs `includeTriangularAttention`:**
- Triangle **Multiplication** (default ON): $O(L^2 \cdot c)$ complexity, more efficient
- Triangle **Attention** (default OFF): $O(L^2 \cdot L)$ complexity, more expressive but costly
- **Recommendation**: Use multiplication only (paper default) for most cases
- Enable attention only for high-accuracy requirements with sufficient GPU memory

**`triangularDropout`:**
- Higher dropout (0.25-0.3) helps prevent overfitting on small datasets
- Lower dropout (0.1-0.15) for larger datasets or when underfitting

---

#### 5. Structure Network (IPA)

The Structure Network uses Invariant Point Attention (IPA) from AlphaFold2 to update 3D coordinates while maintaining SE(3) equivariance.

| Parameter | Config Key | Default | Description |
|-----------|------------|---------|-------------|
| Num Structure Layers | `numStructureLayers` | 5 | Number of IPA layers |
| Num Structure Blocks | `numStructureBlocks` | 1 | Number of structure module iterations |
| IPA Hidden Dimension | `ipaHiddenDimension` | 16 | Per-head hidden dimension |
| IPA Num Heads | `ipaNumHeads` | 12 | Number of attention heads |
| IPA Num Q/K Points | `ipaNumQkPoints` | 4 | Number of query/key 3D points per head |
| IPA Num V Points | `ipaNumVPoints` | 8 | Number of value 3D points per head |
| IPA Dropout | `ipaDropout` | 0.1 | Dropout rate after IPA |
| Num Transition Layers | `numStructureTransitionLayers` | 1 | Transition layers per structure layer |
| Transition Dropout | `structureTransitionDropout` | 0.1 | Dropout rate for transition |

**Selection Guide:**

**`numStructureLayers` and `numStructureBlocks`:**
- Total IPA applications = `numStructureLayers` × `numStructureBlocks`
- **Standard**: 5 layers × 1 block = 5 IPA applications (paper default)
- **High accuracy**: 8 layers × 1 block or 4 layers × 2 blocks
- **Memory-efficient**: 3 layers × 1 block

**IPA Geometry Parameters (`ipaNumQkPoints`, `ipaNumVPoints`):**
- These control the 3D geometric reasoning capacity
- Q/K points: Used for computing attention weights based on 3D distances
- V points: Used for aggregating geometric information
- **AlphaFold2 defaults**: 4 Q/K points, 8 V points (recommended)
- Reducing to 2/4 saves memory but reduces geometric expressiveness

**`ipaHiddenDimension` and `ipaNumHeads`:**
- Total hidden dimension = `ipaHiddenDimension` × `ipaNumHeads` = 16 × 12 = 192
- **Standard**: 16 × 12 (paper default, matches AlphaFold2)
- **High capacity**: 16 × 16 or 24 × 12
- **Memory-efficient**: 12 × 8

---

#### Recommended Configurations

**Standard Configuration (Paper Default):**
```
# General
singleFeatureDimension 128
pairFeatureDimension 128

# Single Feature Network
positionalEmbeddingDimension 128
timestepEmbeddingDimension 128

# Pair Feature Network
relativePositionK 32
templateType v1

# Pair Transform Network
numPairTransformLayers 5
includeTriangularMultiplicativeUpdate True
includeTriangularAttention False
triangularMultiplicativeHiddenDimension 128
triangularDropout 0.25
pairTransitionN 4

# Structure Network (IPA)
numStructureLayers 5
numStructureBlocks 1
ipaHiddenDimension 16
ipaNumHeads 12
ipaNumQkPoints 4
ipaNumVPoints 8
ipaDropout 0.1
numStructureTransitionLayers 1
structureTransitionDropout 0.1
```

**Memory-Efficient Configuration (for limited GPU memory):**
```
# General - reduced dimensions
singleFeatureDimension 64
pairFeatureDimension 64

# Pair Transform Network - fewer layers
numPairTransformLayers 3
triangularMultiplicativeHiddenDimension 64

# Structure Network - lighter IPA
numStructureLayers 3
ipaHiddenDimension 12
ipaNumHeads 8
ipaNumQkPoints 2
ipaNumVPoints 4
```

**High-Accuracy Configuration (for maximum quality):**
```
# General - increased capacity
singleFeatureDimension 256
pairFeatureDimension 256

# Pair Transform Network - more layers
numPairTransformLayers 8
includeTriangularAttention True
triangularAttentionHiddenDimension 32
triangularAttentionNumHeads 4

# Structure Network - deeper IPA
numStructureLayers 8
ipaNumHeads 16
```

---

#### Training Hyperparameters

| Parameter | Config Key | Default | Description |
|-----------|------------|---------|-------------|
| Num Timesteps | `numTimesteps` | 1000 | Diffusion timesteps |
| Schedule | `schedule` | `cosine` | Noise schedule type |
| Learning Rate | `learningRate` | 1e-4 | Adam optimizer learning rate |
| Batch Size | `batchSize` | 32 | Training batch size |
| Num Epochs | `numEpoches` | 50000 | Total training epochs |

**Diffusion Schedule:**
- `cosine`: Recommended (smoother noise schedule, better for proteins)
- `linear`: Alternative (may require more timesteps)

**Learning Rate:**
- 1e-4 is robust for most configurations
- Use learning rate warmup for large batch sizes
- Consider 5e-5 for fine-tuning or when training is unstable

---

### 2. Sampling

To generate protein backbones using a pre-trained model.

**Note on Pre-trained Weights:**
The provided `weights/` directory contains checkpoint files. The sampling script expects a specific directory structure (e.g., `runs/<model_name>/version_<X>/checkpoints/`). You may need to restructure the weights or use the provided Jupyter Notebook which handles this automatically.

#### Standard Sampling

Standard command:
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

#### Flash Mode Sampling (Memory-Efficient)

For sampling long sequences (>256 residues) or on GPUs with limited memory, use Flash mode:

```bash
python genie/sample.py \
    --rootdir runs \
    --model_name scope_l_256 \
    --flash_mode \
    --batch_size 3 \
    --max_length 256 \
    --gpu 0
```

Or use the dedicated Flash sampling script for more control:

```bash
python genie/flash_sample.py \
    --rootdir runs \
    --model_name scope_l_256 \
    --flash_mode \
    --batch_size 5 \
    --min_length 50 \
    --max_length 256 \
    --gpu 0
```

**Note:** Flash mode for sampling works best with models trained using `useFlashMode True`. When using Flash mode with a standard-trained checkpoint, some weights (PairTransformNet) will be randomly initialized, which may affect generation quality.

**Arguments (genie/sample.py):**

- `-n, --model_name` (required): Name of the Genie model (directory name under `runs/`).
- `-r, --rootdir` (default: `runs`): Root directory containing the `runs/<model_name>/...` structure.
- `-v, --model_version`: Model version number (expects `runs/<model_name>/version_<N>/...`).
- `-e, --model_epoch`: Epoch number of the checkpoint to load (used to select a checkpointed model).
- `-g, --gpu`: GPU device to use. Note: this flag accepts an optional value; `--gpu` alone implies GPU `0`, while `--gpu 1` selects GPU `1`.
- `--batch_size` (default: `5`): Number of samples generated per batch.
- `--num_batches` (default: `2`): Number of batches to generate. Total samples = `batch_size * num_batches`.
- `--noise_scale` (default: `0.6`): Sampling noise scale controlling stochasticity/diversity.
- `--min_length` (default: `50`): Minimum sequence length to sample.
- `--max_length` (default: `128`): Maximum sequence length to sample.
- `--save_trajectory`: If set, saves intermediate diffusion timesteps (trajectory `.npy`) for visualization. Adds disk usage and runtime.
- `--flash_mode`: Enable Flash IPA for memory-efficient sampling (recommended for long sequences).

### 3. Visualization

You can visualize the generated structures (Saved as `.npy` coordinate files) using the provided scripts.

**Structure coordinate visualization (evaluations/visualize.py):**

```bash
python evaluations/visualize.py <input_file> -o <output_dir>
```

**Arguments (evaluations/visualize.py):**

- `input_file` (positional): Path to an input coordinate file (usually `.npy`; the loader also tries CSV/text).
- `-o, --output_dir` (optional): Directory to save outputs. If omitted, saves next to `input_file`.

**Alternative (similar interface):**

- `python evaluations/visualize_protein.py <input_file> -o <output_dir>`: Produces a smoother “protein-like” backbone visualization.

**Trajectory visualization (evaluations/visualize_trajectory.py):**

```bash
python evaluations/visualize_trajectory.py <traj_npy> <output_gif>
```

**Arguments (evaluations/visualize_trajectory.py):**

- `traj_npy` (positional): Path to a trajectory `.npy` produced by `genie/sample.py --save_trajectory`.
- `output_gif` (positional): Path to the output `.gif` animation.

### 4. Analysis and Evaluation

This repository includes scripts for evaluating the novelty of generated designs and visualizing the design space.

#### Quality Evaluation (scTM & pLDDT)

To assess the designability of the generated backbones, use the evaluation pipeline. This step runs ProteinMPNN (inverse folding) and ESMFold (folding) to calculate self-consistency TM-scores (scTM) and pLDDT.

```bash
python evaluations/pipeline/evaluate.py \
    --input_dir runs/scope_l_128/version_0/samples/epoch_49999 \
    --output_dir runs/scope_l_128/version_0/samples/epoch_49999/evaluations
```
This generates an `info.csv` file required for the plotting scripts.

**Arguments (evaluations/pipeline/evaluate.py):**

- `--input_dir` (required): Directory containing input samples to evaluate.
- `--output_dir` (required): Directory to write evaluation results (including `info.csv`).
- `-g, --gpus` (optional): GPU devices to use (e.g., `"0"` or `"0,1"`).
- `-c, --config` (optional): Accepted for compatibility but ignored by the script.

#### Novelty Evaluation

To calculate the novelty of generated designs (TM-score against a reference database like PDB):

*   **CPU Version (Exact, Brute-force):**
    ```bash
    python evaluations/Novelty_Evaluation_CPU.py \
        --input_dir runs/scope_l_128/version_0/samples/epoch_49999/evaluations \
        --ref_dir data/pdbstyle-2.08 \
        --num_workers 4
    ```

    **Arguments (evaluations/Novelty_Evaluation_CPU.py):**

    - `-i, --input_dir`: Input directory. You can point to an evaluation directory containing `info.csv` and optionally a `designs/` subfolder.
    - `-o, --output_csv`: Output CSV path. Default: `<input_dir>/novelty.csv`.
    - `--ref_dir`: Reference database directory (e.g., `data/pdbstyle-2.08`).
    - `--tmalign`: Path to the `TMalign` executable.
    - `--num_workers`: Number of worker processes for parallel TM-align computations (default: 2).
    - `--length_tolerance`: Pre-filter tolerance by length (default `0.3` means ±30%).
    - `--early_stop_tm`: Early-stop threshold (default `0.5`): stop searching once TM exceeds this value (treat as “not novel”).
    - `--no_early_stop`: Disable early stopping and search for the exact maximum TM.
    - `--no_length_filter`: Disable length-based pre-filtering.

*   **GPU Version (Hybrid, Faster):**
    ```bash
    python evaluations/Novelty_Evaluation_GPU.py \
        --input_dir runs/scope_l_128/version_0/samples/epoch_49999/evaluations \
        --ref_dir data/pdbstyle-2.08
    ```

    **Arguments (evaluations/Novelty_Evaluation_GPU.py):**

    - `-i, --input_dir`: Input directory containing PDB designs. If the directory contains a `designs/` subfolder, it will be auto-detected.
    - `-o, --output_csv`: Output CSV path. Default: `novelty_hybrid.csv` written into the evaluation directory (or the parent of `designs/`).
    - `-r, --ref_dir`: Reference database directory.
    - `--num_workers`: Number of worker processes for parallel TM-align verification (default: 2).


#### Plotting Analysis

Use the unified `evaluations/plot.py` script to generate analysis plots. This script combines functionality for MDS plots, general analysis (Figure 2 reproduction), and 3D structure visualization.

**Arguments:**

*   `-i, --input_dir`: **(Required)** Input directory containing evaluation data (must contain `info.csv`, optionally `novelty_hybrid.csv`, `pair_info.csv`).
*   `-p, --plot`: Which plot to generate.
    *   `analysis`: General Analysis (Figure 2 Reproduction). Plots pLDDT vs scTM, SSE distribution, and designability counts.
    *   `mds`: Design Space MDS Plot. Visualizes the distribution of generated samples using Multidimensional Scaling.
    *   `structures`: Novel Structure Examples. Visualizes the 3D structures of top novel designs.
    *   `all`: Generate all of the above (Default).
*   `-o, --output_dir`: Output directory for saving the plots. Default is the current directory.

**Examples:**

```bash
# Generate all plots
python evaluations/plot.py --input_dir runs/.../evaluations --output_dir outputs/plots

# Generate only the MDS plot
python evaluations/plot.py -i runs/.../evaluations -p mds -o outputs/plots
```

**Python API (evaluations/plot.py):**

- `get_default_run_dir()`:
    - Returns: default evaluation directory path used when `--input_dir` is not provided.
- `load_data(input_dir)`:
    - `input_dir`: evaluation directory containing `info.csv`.
    - Returns: `(df, has_novelty)` where `df` is the merged DataFrame and `has_novelty` indicates whether novelty CSV was found and merged.
- `parse_pdb_ca(filepath)`:
    - `filepath`: path to a `.pdb` file.
    - Returns: `N x 3` NumPy array of Cα coordinates.
- `plot_genie_analysis(input_dir, output_file='...png')`:
    - `input_dir`: evaluation directory containing `info.csv` (and optionally novelty csv).
    - `output_file`: output image path for the Figure-2-style analysis plot.
- `plot_genie_mds_novelty(input_dir, output_file='...png')`:
    - `input_dir`: evaluation directory containing `info.csv` and `pair_info.csv` (and optionally novelty csv).
    - `output_file`: output image path for the MDS design-space plot.
- `plot_structures(input_dir, output_file='...png')`:
    - `input_dir`: evaluation directory (or the `designs/` directory). Needs `info.csv`, novelty csv, and PDBs.
    - `output_file`: output image path for the 3D structure examples.
- `main()`:
    - CLI entrypoint; parameters are exposed via `-i/--input_dir`, `-p/--plot`, `-o/--output_dir`.

## Project Structure

-   `genie/`: Main package source code.
    -   `diffusion/`: Diffusion model implementation.
    -   `model/`: Neural network architecture.
    -   `data/`: Data loading and processing.
-   `evaluations/`: Evaluation pipeline components.
-   `packages/`: External tools (TMscore).
-   `scripts/`: Utility scripts for setup.
-   `weights/`: Pre-trained model weights.

## Gallery

### Generation Process
![Generation Process](process.gif)



## Optimization Results

![Optimization Comparison](Training_process_parameters/optimization_comparison.png)

We compared the training process parameters between the original implementation, our optimized version, and the full Flash mode (files located in `Training_process_parameters/`). We provide the models from this reproduction and optimized training in the [release](.release).


**Hardware Configuration:**
*   **GPU:** RTX 5090 (32GB) * 1
*   **CPU:** 25 vCPU Intel(R) Xeon(R) Platinum 8470Q
*   **Memory:** 90GB

**Comparison Summary:**

| Metric | Original Work | This Work (Optimized) | Full Flash Mode | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Training Time (500 Epochs)** | ~25.7 Hours | ~12.8 Hours | ~8.2 Hours | **3.1x Speedup** (Flash vs Original) |
| **Max GPU Memory Usage** | ~29.53 GB | ~25.92 GB | ~1.48 GB | **95% Reduction** (Flash vs Original) |
| **Avg GPU Utilization** | ~87.0% | ~87.7% | ~14.2% | Flash mode is memory-bound |
| **Training Loss (Final Epoch)** | ~0.758 | ~0.771 | ~0.822 | Tradeoff for memory efficiency |

The standard optimization reduced training time by half and GPU memory usage by approximately 12%. The full Flash mode provides dramatic memory savings (95% reduction) at the cost of slightly higher final loss, making it ideal for memory-constrained environments or very long sequences.

### Training Configurations

<details>
<summary><b>Original Work Configuration</b></summary>

```
name final_v1_background_model
numEpoches 500
batchSize 8
maximumNumResidues 128
dataDirectory data
datasetNames scope
templateType v1
numPairTransformLayers 8
includeTriangularAttention True
logEverySteps 50
checkpointEveryEpoches 50
learningRate 2e-4
```

</details>

<details>
<summary><b>This Work (Optimized) Configuration</b></summary>

```
name final_final-v0
numEpoches 500
batchSize 8
maximumNumResidues 128
dataDirectory data
datasetNames scope
templateType v1
numPairTransformLayers 8
includeTriangularAttention True
logEverySteps 50
checkpointEveryEpoches 50
learningRate 2e-4
useFlashIPA True
numWorkers 16
useGradientCheckpointing False
```

</details>

<details>
<summary><b>Full Flash Mode Configuration</b></summary>

```
name onFlashIPA_v0
numEpoches 500
batchSize 8
maximumNumResidues 128
dataDirectory data
datasetNames scope
templateType v1
numPairTransformLayers 8
includeTriangularAttention True
logEverySteps 50
checkpointEveryEpoches 50
learningRate 2e-4
useFlashIPA True
useFlashMode True
numWorkers 16
useGradientCheckpointing False
zFactorRank 2
kNeighbors 10
```

</details>

### Generative Quality Comparison

We compared the generative quality of the original implementation and our optimized version. The results show that the optimized model maintains comparable generative capabilities.


**Comprehensive Analysis:**

| Original Work | This Work (Optimized) |
| :---: | :---: |
| ![Original Hybrid](Training_process_parameters/origin_work_hybrid.png) | ![Optimized Hybrid](Training_process_parameters/this_work_hybrid.png) |

**Novel Structure Examples (Optimized Work):**

![Novel Structures](Training_process_parameters/this_work_structure_examples_novel.png)



