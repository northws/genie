import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS
import matplotlib.gridspec as gridspec
import os

# Configuration
input_info_path = "/root/autodl-tmp/genie/runs/final_final-v0/version_3/samples/epoch_499/evaluations/info.csv"
input_pair_path = "/root/autodl-tmp/genie/runs/final_final-v0/version_3/samples/epoch_499/evaluations/pair_info.csv"
input_novelty_path = "/root/autodl-tmp/genie/runs/final_final-v0/version_3/samples/epoch_499/evaluations/novelty_hybrid.csv"
output_file = "genie_design_space_mds_hybrid.png"

print("Loading info.csv...")
df = pd.read_csv(input_info_path)

# Filter Confidently Designable? NO. User requested "all generated samples".
# But checking if we have novelty data for all.
print(f"Loaded {len(df)} total designs.")

# Try to load novelty info
use_novelty = False
if os.path.exists(input_novelty_path):
    print(f"Loading novelty data from {input_novelty_path}...")
    try:
        novelty_df = pd.read_csv(input_novelty_path)
        # Merge LEFT (keep all designs)
        df_merged = df.merge(novelty_df[['domain', 'max_tm_to_pdb']], on='domain', how='left')
        
        # Check coverage
        n_matched = df_merged['max_tm_to_pdb'].notna().sum()
        print(f"Novelty data matched for {n_matched}/{len(df)} designs.")
        
        if n_matched > 0:
             df = df_merged
             use_novelty = True
    except Exception as e:
        print(f"Error loading novelty csv: {e}")

# If we are plotting ALL, we need to ensure pair info covers ALL.
print("Loading pair_info.csv...")
pairs = pd.read_csv(input_pair_path)

valid_domains = set(df['domain'])
# Filter pairs to only include valid domains
pairs = pairs[pairs['domain_1'].isin(valid_domains) & pairs['domain_2'].isin(valid_domains)]
print(f"Filtered pairs: {len(pairs)} rows.")

# Create Matrix
# We need to map domain names to indices
domain_list = sorted(list(valid_domains))
domain_to_idx = {d: i for i, d in enumerate(domain_list)}
n = len(domain_list)

tm_matrix = np.zeros((n, n))
# Fill matrix
pairs['idx1'] = pairs['domain_1'].map(domain_to_idx)
pairs['idx2'] = pairs['domain_2'].map(domain_to_idx)

# Use numpy indexing
idx1 = pairs['idx1'].values
idx2 = pairs['idx2'].values
tm_vals = pairs['tm'].values

tm_matrix[idx1, idx2] = tm_vals

# Symmetrize
tm_matrix_sym = np.maximum(tm_matrix, tm_matrix.T)

# Distance metric
# D = 1 - TM
distances = 1.0 - tm_matrix_sym
np.fill_diagonal(distances, 0) # Distance to self is 0

# Check for unconnected components or empty rows/cols
has_data_mask = np.any(tm_matrix_sym > 0, axis=1) 

valid_indices = np.where(has_data_mask)[0]
if len(valid_indices) < len(domain_list):
    print(f"Dropping {len(domain_list) - len(valid_indices)} domains due to missing pair data.")
    domain_list = [domain_list[i] for i in valid_indices]
    distances = distances[valid_indices][:, valid_indices]
    df = df[df['domain'].isin(domain_list)]
    # Re-sort df to match domain_list order
    df = df.set_index('domain').reindex(domain_list).reset_index()

print(f"Running MDS on {len(domain_list)} points...")
# Use metric MDS
embedding = MDS(n_components=2, metric=True, dissimilarity='precomputed', random_state=42, n_jobs=-1)
X_transformed = embedding.fit_transform(distances)
df['x_mds'] = X_transformed[:, 0]
df['y_mds'] = X_transformed[:, 1]


# === Plotting ===
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

# --- Main Plot ---
# Check what to use for color
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
                     s=40, alpha=0.9, edgecolors='none', vmin=0.3, vmax=1.0) # Adjusted range
ax_main.set_xlabel("MDS Dimension 1", fontsize=14)
ax_main.set_ylabel("MDS Dimension 2", fontsize=14)
ax_main.set_xticks([])
ax_main.set_yticks([])

# Colorbar for Main
cbar = plt.colorbar(sc, ax=ax_main, fraction=0.046, pad=0.04)
cbar.set_label(cbar_label, fontsize=14)

# --- Subplots ---
# 1. Helix
sc_h = ax_helix.scatter(df['x_mds'], df['y_mds'], 
                        c=df['pct_helix'], cmap='Blues', 
                        s=10, alpha=0.8)
clean_axis(ax_helix, "Helix")
ax_helix.set_ylabel("Percentage of Helices")
cbar_h = plt.colorbar(sc_h, ax=ax_helix, location='left', pad=0.15) 

# 2. Strand
sc_s = ax_strand.scatter(df['x_mds'], df['y_mds'], 
                         c=df['pct_strand'], cmap='Blues', 
                         s=10, alpha=0.8)
clean_axis(ax_strand, "Strand")
ax_strand.set_ylabel("Percentage of Strands")
cbar_s = plt.colorbar(sc_s, ax=ax_strand, location='left', pad=0.15)

# 3. Length
sc_l = ax_len.scatter(df['x_mds'], df['y_mds'], 
                      c=df['seqlen'], cmap='Blues', 
                      s=10, alpha=0.8)
clean_axis(ax_len, "Length")
ax_len.set_ylabel("Sequence Length")
cbar_l = plt.colorbar(sc_l, ax=ax_len, location='left', pad=0.15)

plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Saved plot to {output_file}")
