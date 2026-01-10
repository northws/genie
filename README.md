# Genie: De Novo Protein Design

Genie is a diffusion-based model for de novo protein design through equivariantly diffusing oriented residue clouds.

This project is a reproduction and optimization of [https://github.com/aqlaboratory/genie](https://github.com/aqlaboratory/genie).

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

To train a new model from scratch.

```bash
python genie/train.py \
    --config example_configuration \
    --gpus 0,1
```

Configuration files define model hyperparameters and training settings. See `genie/config.py` for details.

### 2. Sampling

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

### 3. Visualization

You can visualize the generated structures (Saved as `.npy` coordinate files) using the provided scripts.

```bash
python evaluations/visualize.py
```
(Note: You might need to modify `evaluations/visualize.py` or use `evaluations/visualize_protein.py` to point to your specific output file).

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

#### Novelty Evaluation

To calculate the novelty of generated designs (TM-score against a reference database like PDB):

*   **CPU Version (Exact, Brute-force):**
    ```bash
    python evaluations/Novelty_Evaluation_CPU.py \
        --input_dir runs/scope_l_128/version_0/samples/epoch_49999/evaluations \
        --ref_dir data/pdbstyle-2.08 \
        --num_workers 4
    ```

*   **GPU Version (Hybrid, Faster):**
    ```bash
    python evaluations/Novelty_Evaluation_GPU.py \
        --input_dir runs/scope_l_128/version_0/samples/epoch_49999/evaluations \
        --ref_dir data/pdbstyle-2.08
    ```


#### Plotting Analysis

*   **Design Space MDS Plot:**
    Visualizes the distribution of generated samples using Multidimensional Scaling (MDS).
    ```bash
    python evaluations/plot_genie_mds_novelty.py \
        --input_dir runs/.../evaluations \
        --output_file mds_plot.png
    ```

*   **General Analysis (Figure 2 Reproduction):**
    Plots pLDDT vs scTM, SSE distribution, and designability counts.
    ```bash
    python evaluations/plot_genie_analysis.py \
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

### Novel Structure Examples
![Novel Structures](Training_process_parameters/genie_structure_examples_novel.png)

### Design Space Analysis
![Design Space MDS](Training_process_parameters/genie_design_space_mds_hybrid.png)

### Comprehensive Analysis
![Analysis Results](Training_process_parameters/genie_analysis_figure2_repro_v2_hybrid.png)

## Optimization Results

![Optimization Comparison](Training_process_parameters/optimization_comparison.png)

We compared the training process parameters between the original implementation and our optimized version (files located in `Training_process_parameters/`).

**Hardware Configuration:**
*   **GPU:** RTX 5090 (32GB) * 1
*   **CPU:** 25 vCPU Intel(R) Xeon(R) Platinum 8470Q
*   **Memory:** 90GB

**Comparison Summary:**

| Metric | Original Work | This Work (Optimized) | Improvement |
| :--- | :--- | :--- | :--- |
| **Training Time (500 Epochs)** | ~25.7 Hours | ~12.8 Hours | **~2.0x Speedup** |
| **Max GPU Memory Usage** | ~29.53 GB | ~25.92 GB | **~12% Reduction** |
| **Training Loss (Final Epoch)** | ~0.758 | ~0.771 | Comparable |

The optimization reduced training time by half and GPU memory usage by approximately 12%. Analysis of step-wise loss (smoothed) confirms that the slight difference in final epoch loss is due to stochastic fluctuations, and both models exhibit identical convergence behavior.



