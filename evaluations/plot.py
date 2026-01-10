

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import MDS
import matplotlib.gridspec as gridspec
import os
import argparse


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_default_run_dir():
    """Get the default run directory based on script location."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, "runs", "final_final-v0", "version_3", "samples", "epoch_499", "evaluations")


def load_data(input_dir):
    """Load info.csv and optionally novelty data."""
    csv_path = os.path.join(input_dir, "info.csv")
    novelty_path = os.path.join(input_dir, "novelty_hybrid.csv")
    if not os.path.exists(novelty_path):
        novelty_path = os.path.join(input_dir, "novelty.csv")
    
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None, False
    
    # Load Novelty Data
    has_novelty = False
    if os.path.exists(novelty_path):
        print(f"Loading novelty data from {novelty_path}...")
        try:
            novelty_df = pd.read_csv(novelty_path)
            df = df.merge(novelty_df[['domain', 'max_tm_to_pdb']], on='domain', how='left')
            has_novelty = True
        except Exception as e:
            print(f"Error loading Novelty CSV: {e}")
    
    return df, has_novelty


def parse_pdb_ca(filepath):
    """Parse PDB file and extract CA (alpha carbon) coordinates."""
    coords = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
    return np.array(coords)


# ============================================================================
# MAIN PLOTTING FUNCTIONS
# ============================================================================

def plot_genie_analysis(input_dir, output_file="genie_analysis_figure2_repro_v2_hybrid.png"):
    """
    Create Figure 2 recreation with 4 subplots:
    A: Heatmap of pLDDT vs highest scTM
    B: SSE distribution for confidently designable domains
    C: Length distribution histogram
    D: Designability counts bar chart
    """
    df, has_novelty = load_data(input_dir)
    if df is None:
        return False
    
    # Validate required columns
    required_columns = ['scTM', 'pLDDT', 'pct_helix', 'pct_strand', 'seqlen']
    for col in required_columns:
        if col not in df.columns:
            print(f"Missing column: {col}")
            return False
    
    # Define classifications
    df['Designable'] = df['scTM'] > 0.5
    df['Confidently Designable'] = (df['scTM'] > 0.5) & (df['pLDDT'] > 70)
    
    if has_novelty:
        df['Novel and Confidently Designable'] = df['Confidently Designable'] & (df['max_tm_to_pdb'] < 0.5)
    else:
        df['Novel and Confidently Designable'] = df['Confidently Designable']
    
    # Set style
    sns.set_theme(style="ticks")
    plt.rcParams['font.family'] = 'sans-serif'
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # === A: Heatmap pLDDT vs scTM ===
    ax_a = axes[0, 0]
    x = df['scTM']
    y = df['pLDDT']
    h = ax_a.hist2d(x, y, bins=[25, 25], range=[[0, 1], [0, 100]], cmap="Blues", cmin=1)
    ax_a.set_xlabel("Highest scTM")
    ax_a.set_ylabel("pLDDT")
    ax_a.set_title("A. Genie: pLDDT vs Highest scTM")
    ax_a.axvline(0.5, color='green', linestyle='--')
    ax_a.axhline(70, color='red', linestyle='--')
    fig.colorbar(h[3], ax=ax_a, label='Number of Domains')
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 105)
    
    # === B: Heatmap SSE (Confidently Designable) ===
    ax_b = axes[1, 0]
    cd_df = df[df['Confidently Designable']]
    
    if not cd_df.empty:
        x_sse = cd_df['pct_helix']
        y_sse = cd_df['pct_strand']
        h2 = ax_b.hist2d(x_sse, y_sse, bins=[10, 10], range=[[0, 1], [0, 1]], cmap="Reds", cmin=1)
        ax_b.set_xlabel("Percentage of Helices")
        ax_b.set_ylabel("Percentage of Strands")
        ax_b.set_title(f"B. Genie: SSE Distribution (N={len(cd_df)})")
        fig.colorbar(h2[3], ax=ax_b, label='Density')
    else:
        ax_b.text(0.5, 0.5, "No Confidently Designable Data", ha='center')
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, 1)
    
    # === C: Histogram by Length ===
    ax_c = axes[0, 1]
    if not cd_df.empty:
        bins = np.arange(40, 140, 10)
        cd_df_copy = cd_df.copy()
        cd_df_copy['length_bin'] = pd.cut(cd_df_copy['seqlen'], bins=bins, labels=bins[:-1])
        counts_by_len = cd_df_copy['length_bin'].value_counts().sort_index()
        
        ax_c.bar(counts_by_len.index.astype(int), counts_by_len.values, width=6, color='green', align='edge', label='Genie')
        
        ax_c.set_xlabel("Sequence Length")
        ax_c.set_ylabel("Number of Designed Domains")
        ax_c.set_title("C. Confidently Designable Domains vs Length")
        ax_c.legend()
    else:
        ax_c.text(0.5, 0.5, "No Data", ha='center')
    
    # === D: Bar Chart of Counts ===
    ax_d = axes[1, 1]
    counts_dict = {
        'Designable': df['Designable'].sum(),
        'Confidently\nDesignable': df['Confidently Designable'].sum(),
        'Novel and\nConfidently\nDesignable': df['Novel and Confidently Designable'].sum()
    }
    
    total_generated = len(df)
    labels = list(counts_dict.keys())
    values = list(counts_dict.values())
    bar_colors = ['#17becf', '#9467bd', '#ff69b4']
    
    bars = ax_d.bar(labels, values, color=bar_colors)
    ax_d.set_ylabel("Number of Domains")
    ax_d.set_title(f"D. Designability Counts (Total Attempted: {total_generated})")
    
    for rect in bars:
        height = rect.get_height()
        ax_d.text(rect.get_x() + rect.get_width()/2.0, height + 1, f'{int(height)}', ha='center', va='bottom')
    
    # Finalize
    plt.suptitle("Recreation of Figure 2 Analysis for Genie Model", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    plt.savefig(output_file, dpi=150)
    print(f"Plot saved to {output_file}")
    plt.close()
    return True


def plot_genie_mds_novelty(input_dir, output_file="genie_design_space_mds_hybrid.png"):
    """
    Create MDS visualization of design space with novelty coloring.
    Main plot: 2D MDS embedding colored by novelty/scTM
    Side plots: Helix%, Strand%, Sequence length distributions
    """
    df, _ = load_data(input_dir)
    if df is None:
        return False
    
    # Load pair info
    input_pair_path = os.path.join(input_dir, "pair_info.csv")
    print("Loading pair_info.csv...")
    try:
        pairs = pd.read_csv(input_pair_path)
    except Exception as e:
        print(f"Error loading pair info: {e}")
        return False
    
    # Check for novelty
    input_novelty_path = os.path.join(input_dir, "novelty_hybrid.csv")
    if not os.path.exists(input_novelty_path):
        input_novelty_path = os.path.join(input_dir, "novelty.csv")
    
    use_novelty = False
    if os.path.exists(input_novelty_path):
        print(f"Loading novelty data from {input_novelty_path}...")
        try:
            novelty_df = pd.read_csv(input_novelty_path)
            df_merged = df.merge(novelty_df[['domain', 'max_tm_to_pdb']], on='domain', how='left')
            n_matched = df_merged['max_tm_to_pdb'].notna().sum()
            print(f"Novelty data matched for {n_matched}/{len(df)} designs.")
            if n_matched > 0:
                df = df_merged
                use_novelty = True
        except Exception as e:
            print(f"Error loading novelty csv: {e}")
    
    print(f"Loaded {len(df)} total designs.")
    
    # Filter and create distance matrix
    valid_domains = set(df['domain'])
    pairs = pairs[pairs['domain_1'].isin(valid_domains) & pairs['domain_2'].isin(valid_domains)]
    print(f"Filtered pairs: {len(pairs)} rows.")
    
    # Create TM distance matrix
    domain_list = sorted(list(valid_domains))
    domain_to_idx = {d: i for i, d in enumerate(domain_list)}
    n = len(domain_list)
    
    tm_matrix = np.zeros((n, n))
    pairs['idx1'] = pairs['domain_1'].map(domain_to_idx)
    pairs['idx2'] = pairs['domain_2'].map(domain_to_idx)
    
    idx1 = pairs['idx1'].values
    idx2 = pairs['idx2'].values
    tm_vals = pairs['tm'].values
    
    tm_matrix[idx1, idx2] = tm_vals
    tm_matrix_sym = np.maximum(tm_matrix, tm_matrix.T)
    
    distances = 1.0 - tm_matrix_sym
    np.fill_diagonal(distances, 0)
    
    # Filter out unconnected domains
    has_data_mask = np.any(tm_matrix_sym > 0, axis=1)
    valid_indices = np.where(has_data_mask)[0]
    if len(valid_indices) < len(domain_list):
        print(f"Dropping {len(domain_list) - len(valid_indices)} domains due to missing pair data.")
        domain_list = [domain_list[i] for i in valid_indices]
        distances = distances[valid_indices][:, valid_indices]
        df = df[df['domain'].isin(domain_list)]
        df = df.set_index('domain').reindex(domain_list).reset_index()
    
    # Run MDS
    print(f"Running MDS on {len(domain_list)} points...")
    embedding = MDS(n_components=2, metric=True, dissimilarity='precomputed', random_state=42, n_jobs=-1)
    X_transformed = embedding.fit_transform(distances)
    df['x_mds'] = X_transformed[:, 0]
    df['y_mds'] = X_transformed[:, 1]
    
    # Plot
    sns.set_theme(style="ticks")
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 4, wspace=0.3, hspace=0.3)
    
    ax_main = fig.add_subplot(gs[:, 1:])
    ax_helix = fig.add_subplot(gs[0, 0])
    ax_strand = fig.add_subplot(gs[1, 0])
    ax_len = fig.add_subplot(gs[2, 0])
    
    def clean_axis(ax, title):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.05, 0.9, title, transform=ax.transAxes, fontsize=12, fontweight='bold')
    
    # Main plot
    if use_novelty:
        color_col = 'max_tm_to_pdb'
        cbar_label = "Maximum TM Score to PDB (Hybrid Protocol)"
        print("Plotting with Novelty (DB Match) colors.")
    else:
        color_col = 'scTM'
        cbar_label = "Maximum TM Score (scTM) [Novelty Data Not Found]"
        print("Plotting with scTM colors (Novelty approximation).")
    
    sc = ax_main.scatter(df['x_mds'], df['y_mds'],
                         c=df[color_col], cmap='RdYlBu',
                         s=40, alpha=0.9, edgecolors='none', vmin=0.3, vmax=1.0)
    ax_main.set_xlabel("MDS Dimension 1", fontsize=14)
    ax_main.set_ylabel("MDS Dimension 2", fontsize=14)
    ax_main.set_xticks([])
    ax_main.set_yticks([])
    
    cbar = plt.colorbar(sc, ax=ax_main, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=14)
    
    # Side plots
    sc_h = ax_helix.scatter(df['x_mds'], df['y_mds'],
                            c=df['pct_helix'], cmap='Blues',
                            s=10, alpha=0.8)
    clean_axis(ax_helix, "Helix")
    ax_helix.set_ylabel("Percentage of Helices")
    cbar_h = plt.colorbar(sc_h, ax=ax_helix, location='left', pad=0.15)
    
    sc_s = ax_strand.scatter(df['x_mds'], df['y_mds'],
                             c=df['pct_strand'], cmap='Blues',
                             s=10, alpha=0.8)
    clean_axis(ax_strand, "Strand")
    ax_strand.set_ylabel("Percentage of Strands")
    cbar_s = plt.colorbar(sc_s, ax=ax_strand, location='left', pad=0.15)
    
    sc_l = ax_len.scatter(df['x_mds'], df['y_mds'],
                          c=df['seqlen'], cmap='Blues',
                          s=10, alpha=0.8)
    clean_axis(ax_len, "Length")
    ax_len.set_ylabel("Sequence Length")
    cbar_l = plt.colorbar(sc_l, ax=ax_len, location='left', pad=0.15)
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_file}")
    plt.close()
    return True


def plot_structures(input_dir, output_file="genie_structure_examples_novel.png"):
    """
    Visualize top 4 novel confidently designable structures in 3D.
    Structures are colored by N->C terminus gradient.
    """
    # Handle directory structure
    designs_dir = os.path.join(input_dir, "designs")
    if not os.path.exists(designs_dir):
        designs_dir = input_dir
        run_dir = os.path.dirname(input_dir)
    else:
        run_dir = input_dir
    
    info_path = os.path.join(run_dir, "info.csv")
    novelty_path = os.path.join(run_dir, "novelty_hybrid.csv")
    if not os.path.exists(novelty_path):
        novelty_path = os.path.join(run_dir, "novelty.csv")
    
    # Load and select top novel structures
    print("Loading data...")
    try:
        df = pd.read_csv(info_path)
        df_novel = pd.read_csv(novelty_path)
        
        merged = df.merge(df_novel, on='domain')
        
        # Filter: Confidently Designable
        confident = merged[(merged['scTM'] > 0.5) & (merged['pLDDT'] > 70)]
        
        # Top novel
        top_novel = confident.sort_values('max_tm_to_pdb', ascending=True).head(4)
        
        pdb_ids = top_novel['domain'].tolist()
        tm_scores = top_novel['max_tm_to_pdb'].tolist()
        
        print(f"Selected Top 4 Novel Domains: {pdb_ids}")
        print(f"TM Scores: {tm_scores}")
        
    except Exception as e:
        print(f"Error loading/processing data: {e}")
        pdb_ids = ['120_7', '76_6', '55_15', '68_1']
        tm_scores = [None]*4
    
    # Plot
    fig = plt.figure(figsize=(20, 5))
    
    for i, pid in enumerate(pdb_ids):
        path = os.path.join(designs_dir, f"{pid}.pdb")
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
        
        coords = parse_pdb_ca(path)
        
        ax = fig.add_subplot(1, 4, i+1, projection='3d')
        ax.set_proj_type('persp', focal_length=0.2)
        
        xs = coords[:, 0]
        ys = coords[:, 1]
        zs = coords[:, 2]
        
        # Gradient color
        N = len(coords)
        colors = plt.cm.jet(np.linspace(0, 1, N))
        
        # Plot segments
        for j in range(N-1):
            ax.plot(xs[j:j+2], ys[j:j+2], zs[j:j+2], color=colors[j], linewidth=3, alpha=0.9)
        
        ax.scatter(xs, ys, zs, c=colors, s=30, depthshade=True, edgecolor='black', linewidth=0.2)
        
        tm_text = f" (TM={tm_scores[i]:.2f})" if tm_scores[i] is not None else ""
        ax.set_title(f"Design {pid}{tm_text}", fontsize=14)
        ax.axis('off')
        
        # Auto-scaling
        max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max() / 2.0
        mid_x = (xs.max()+xs.min()) * 0.5
        mid_y = (ys.max()+ys.min()) * 0.5
        mid_z = (zs.max()+zs.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.suptitle("Top Confidently Designable & Novel Structures (Colored N->C Terminus)", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_file, dpi=150)
    print(f"Saved structures to {output_file}")
    plt.close()
    return True


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified Genie plotting module")
    parser.add_argument("-i", "--input_dir", type=str, default=get_default_run_dir(),
                        help="Input directory containing evaluation data")
    parser.add_argument("-p", "--plot", type=str, choices=['analysis', 'mds', 'structures', 'all'],
                        default='all', help="Which plot to generate")
    parser.add_argument("-o", "--output_dir", type=str, default=".",
                        help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Create output directory if needed
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Run selected plots
    plots = {
        'analysis': (plot_genie_analysis, "genie_analysis_figure2_repro_v2_hybrid.png"),
        'mds': (plot_genie_mds_novelty, "genie_design_space_mds_hybrid.png"),
        'structures': (plot_structures, "genie_structure_examples_novel.png")
    }
    
    selected = plots.keys() if args.plot == 'all' else [args.plot]
    
    for plot_name in selected:
        plot_func, output_name = plots[plot_name]
        output_path = os.path.join(args.output_dir, output_name)
        print(f"\n{'='*60}")
        print(f"Generating {plot_name} plot...")
        print(f"{'='*60}")
        
        try:
            plot_func(args.input_dir, output_path)
        except Exception as e:
            print(f"Error generating {plot_name} plot: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
