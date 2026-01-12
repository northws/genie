# Genie: De Novo Protein Design

Genie is a diffusion-based model for de novo protein design through equivariantly diffusing oriented residue clouds.

This project is a reproduction and optimization of [https://github.com/aqlaboratory/genie](https://github.com/aqlaboratory/genie).

**Read this in other languages:  [中文](README_zh.md)**

**View the demo notebook:** [genie_demo.ipynb](genie_demo.ipynb)
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

Given a protein backbone represented by C$\alpha$ coordinates $\mathbf{x}_0$, the forward process gradually adds Gaussian noise over $T$ timesteps:

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

**Configuration Parameters for Flash Mode:**
- `useFlashMode`: Enable memory-efficient Flash mode (default: `False`)
- `zFactorRank`: Rank for edge embedding factorization (default: `2`)
- `kNeighbors`: Number of nearest neighbors for distogram (default: `10`)

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
> $$\text{headdim\_eff} = \max\left(c_{\text{hidden}} + 5 \cdot n_{\text{qk\_point}} + r \cdot n_{\text{head}}, \quad c_{\text{hidden}} + 3 \cdot n_{\text{v\_point}} + r \cdot \frac{c_z}{4}\right)$$
>
> **Parameter Definitions:**
> - $c_{\text{hidden}}$: IPA hidden dimension (`ipaHiddenDimension`), hidden channels per attention head
> - $n_{\text{qk\_point}}$: Query/Key 3D points (`ipaNumQkPoints`), used for SE(3)-equivariant attention weights
> - $n_{\text{v\_point}}$: Value 3D points (`ipaNumVPoints`), used for aggregating geometric information
> - $n_{\text{head}}$: Number of attention heads (`ipaNumHeads`)
> - $c_z$: Pair feature dimension (`pairFeatureDimension`), channel dimension of pair embeddings
> - $r$: `zFactorRank`, rank of the low-rank factorization for edge embeddings
>
> **Formula Explanation:**
> - First term $c_{\text{hidden}} + 5 \cdot n_{\text{qk\_point}} + r \cdot n_{\text{head}}$: Effective Q/K dimension (scalar features + 5 point coordinate components + bias factors)
> - Second term $c_{\text{hidden}} + 3 \cdot n_{\text{v\_point}} + r \cdot c_z/4$: Effective V dimension (scalar features + 3D point coordinates + downsampled edge features)
> - The maximum of both terms determines the headdim required for Flash Attention
>
> With default IPA parameters ($c_{\text{hidden}}=16$, $n_{\text{qk\_point}}=4$, $n_{\text{v\_point}}=8$, $n_{\text{head}}=12$, $c_z=128$):
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

#### Recommended Parameter Combinations

| Configuration | `zFactorRank` | `kNeighbors` | `maximumNumResidues` | GPU Memory |
|---------------|---------------|--------------|----------------------|------------|
| **Standard medium** | 2 | 10 | 256 | ≥24GB |
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
| Single Feature Dimension | `singleFeatureDimension` | 128 | Channel dimension for per-residue representations ($c_s$) |
| Pair Feature Dimension | `pairFeatureDimension` | 128 | Channel dimension for pairwise representations ($c_p$) |

**Selection Guide:**
- These dimensions should be equal ($c_s = c_p$) for optimal information flow
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

## Citations and Acknowledgements

This project is built upon several excellent open-source projects and academic research results：
### Core Algorithm & Models
*   **Genie (Original Implementation)**:
    Lin, Y. C., & AlQuraishi, M. (2023). Generating protein backbone structures with equivariant diffusion models. *arXiv preprint arXiv:2301.12485*.
    [[Paper]](https://arxiv.org/abs/2301.12485) [[Code]](https://github.com/aqlaboratory/genie)

*   **Flash-IPA (Optimization)**:
    Flagship Pioneering. (2023). Flash-IPA: Accelerated Invariant Point Attention. GitHub.
    [[Code]](https://github.com/flagshippioneering/flash_ipa)

### Evaluation Pipeline
*   **ProteinMPNN (Sequence Design)**:
    Dauparas, J., et al. (2022). Robust deep learning–based protein sequence design using ProteinMPNN. *Science*, 378(6615), 49-56.
    [[Paper]](https://www.science.org/doi/10.1126/science.add2187) [[Code]](https://github.com/dauparas/ProteinMPNN)

*   **ESMFold / ESM-2 (Structure Prediction)**:
    Lin, Z., et al. (2023). Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*, 379(6637), 1123-1130.
    [[Paper]](https://www.science.org/doi/10.1126/science.ade2574) [[Code]](https://github.com/facebookresearch/esm)

*   **TM-score & TM-align (Structure Alignment)**:
    Zhang, Y., & Skolnick, J. (2005). TM-align: a protein structure alignment algorithm based on the TM-score. *Nucleic Acids Research*, 33(7), 2302-2309.
    [[Paper]](https://academic.oup.com/nar/article/33/7/2302/2401364) [[Code]](https://zhanggroup.org/TM-align/)

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



