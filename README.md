# Genie: De Novo Protein Design

Genie is a diffusion-based model for de novo protein design through equivariantly diffusing oriented residue clouds.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/a-lab-i/genie.git
    cd genie
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment (e.g., Conda or venv).
    ```bash
    pip install -e .
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

### 1. Sampling

To generate protein backbones using a pre-trained model.

**Note on Pre-trained Weights:**
The provided `weights/` directory contains checkpoint files. The sampling script expects a specific directory structure (e.g., `runs/<model_name>/version_<X>/checkpoints/`). You may need to restructure the weights or use the provided Jupyter Notebook which handles this automatically.

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

### 2. Training

To train a new model from scratch.

```bash
python genie/train.py \
    --config example_configuration \
    --gpus 0,1
```

Configuration files define model hyperparameters and training settings. See `genie/config.py` for details.

### 3. Visualization

You can visualize the generated structures (Saved as `.npy` coordinate files) using the provided scripts.

```bash
python visualize.py
```
(Note: You might need to modify `visualize.py` or use `visualize_protein.py` to point to your specific output file).

### 4. Analysis and Evaluation

This repository includes scripts for evaluating the novelty of generated designs and visualizing the design space.

#### Novelty Evaluation

To calculate the novelty of generated designs (TM-score against a reference database like PDB):

*   **CPU Version (Exact, Brute-force):**
    ```bash
    python Novelty_Evaluation_CPU.py \
        --input_dir runs/scope_l_128/version_0/samples/epoch_49999/evaluations \
        --ref_dir data/pdbstyle-2.08 \
        --num_workers 4
    ```

*   **GPU Version (Hybrid, Fast Screening):**
    Uses ProteinMPNN embeddings to screen candidates before running TM-align.
    ```bash
    python Novelty_Evaluation_GPU.py \
        --input_dir runs/scope_l_128/version_0/samples/epoch_49999/evaluations
    ```

#### Plotting Analysis

*   **Design Space MDS Plot:**
    Visualizes the distribution of generated samples using Multidimensional Scaling (MDS).
    ```bash
    python plot_genie_mds_novelty.py \
        --input_dir runs/.../evaluations \
        --output_file mds_plot.png
    ```

*   **General Analysis (Figure 2 Reproduction):**
    Plots pLDDT vs scTM, SSE distribution, and designability counts.
    ```bash
    python plot_genie_analysis.py \
        --input_dir runs/.../evaluations \
        --output_file analysis_plot.png
    ```

## Project Structure

-   `genie/`: Main package source code.
    -   `diffusion/`: Diffusion model implementation.
    -   `model/`: Neural network architecture.
    -   `data/`: Data loading and processing.
-   `evaluations/`: Evaluation pipeline components.
-   `packages/`: External tools (TMscore).
-   `scripts/`: Utility scripts for setup.
-   `weights/`: Pre-trained model weights.
