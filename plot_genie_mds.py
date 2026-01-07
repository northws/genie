import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import MDS
import matplotlib.gridspec as gridspec

# Configuration
input_info_path = "/root/autodl-tmp/genie/runs/final_final-v0/version_3/samples/epoch_499/evaluations/info.csv"
input_pair_path = "/root/autodl-tmp/genie/runs/final_final-v0/version_3/samples/epoch_499/evaluations/pair_info.csv"
output_file = "genie_design_space_mds.png"

print("Loading info.csv...")
df = pd.read_csv(input_info_path)

# Filter Confidently Designable
# Criteria from previous turn: scTM > 0.5 AND pLDDT > 70
df_conf = df[(df['scTM'] > 0.5) & (df['pLDDT'] > 70)].copy()
valid_domains = set(df_conf['domain'])
print(f"Found {len(df_conf)} confidently designable domains.")

if len(df_conf) < 10:
    print("Not enough domains to plot.")
    exit()

print("Loading pair_info.csv...")
pairs = pd.read_csv(input_pair_path)

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
# This pandas iteration might be slow, let's try to map efficiently
# Map domains to indices in the dataframe
pairs['idx1'] = pairs['domain_1'].map(domain_to_idx)
pairs['idx2'] = pairs['domain_2'].map(domain_to_idx)

# Use numpy indexing
# Note: pairs might have duplicates or missing reverse pairs, or be full.
# We will fill what we have.
idx1 = pairs['idx1'].values
idx2 = pairs['idx2'].values
tm_vals = pairs['tm'].values

tm_matrix[idx1, idx2] = tm_vals

# Symmetrize: (M + M.T) / 2
# But first, check if we have data. If A-B is missing but B-A exists, we might have zeros.
# If tm_matrix[i,j] is 0 and i!=j, it might be missing.
# Let's assume the file is reasonably complete or symmetric.
# A safe symmetrization if we have one-way data:
# If M[i,j] > 0 and M[j,i] == 0, copy.
# If both > 0, average.

# Simple average of transpose (treating 0 as valid 0 score? TM score is rarely exactly 0 unless missing/bad align).
# Ideally TM score > 0.
# Let's do max for safety against missing reverse pairs.
tm_matrix_sym = np.maximum(tm_matrix, tm_matrix.T)

# Distance metric
# D = 1 - TM
distances = 1.0 - tm_matrix_sym
np.fill_diagonal(distances, 0) # Distance to self is 0

# Check for unconnected components or empty rows/cols
# If a domain has no pairs, its row sum in TM might be 0 (except diagonal).
# But valid_domains came from info.csv and we filtered pairs.
# If pairs are missing for some valid domains, we should drop them from MDS.
has_data_mask = np.any(tm_matrix_sym > 0, axis=1) # Diagonal is 0 in tm_matrix before? No pair_info usually doesn't have self-self?
                                                  # Typically pair_info.csv might not have self-self. 
                                                  # tm_matrix was initialized 0.
                                                  # So sum > 0 means it has neighbors.

valid_indices = np.where(has_data_mask)[0]
if len(valid_indices) < len(domain_list):
    print(f"Dropping {len(domain_list) - len(valid_indices)} domains due to missing pair data.")
    domain_list = [domain_list[i] for i in valid_indices]
    distances = distances[valid_indices][:, valid_indices]
    df_conf = df_conf[df_conf['domain'].isin(domain_list)]
    # Re-sort df_conf to match domain_list order
    df_conf = df_conf.set_index('domain').reindex(domain_list).reset_index()

print(f"Running MDS on {len(domain_list)} points...")
# Use metric MDS
embedding = MDS(n_components=2, metric=True, dissimilarity='precomputed', random_state=42, n_jobs=-1)
X_transformed = embedding.fit_transform(distances)
df_conf['x_mds'] = X_transformed[:, 0]
df_conf['y_mds'] = X_transformed[:, 1]

# === Plotting ===
# Figure layout:
# Left column: 3 subplots (Helix, Strand, Length)
# Right column: 1 big subplot (Main)
# GridSpec: 3 rows, 2 columns.
# Left plots take (0,0), (1,0), (2,0).
# Main plot takes (0,1) to (2,1) -> spans rows.

fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(3, 4, wspace=0.3, hspace=0.3) # 4 columns to give main plot more width

ax_main = fig.add_subplot(gs[:, 1:]) # Spans all rows, cols 1-3
ax_helix = fig.add_subplot(gs[0, 0])
ax_strand = fig.add_subplot(gs[1, 0])
ax_len = fig.add_subplot(gs[2, 0])

# Helper to remove axes spines/ticks for cleaner look like the paper
def clean_axis(ax, title):
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_xlabel("MDS Dimension 1") # Only bottom?
    # ax.set_ylabel("MDS Dimension 2") # Only left?
    # The paper shows some axes labels on the main plot but not ticks
    ax.text(0.05, 0.9, title, transform=ax.transAxes, fontsize=12, fontweight='bold')

# --- Main Plot ---
# Color by scTM (Max TM Score)
# Cmap: 'RdYlBu'
sc = ax_main.scatter(df_conf['x_mds'], df_conf['y_mds'], 
                     c=df_conf['scTM'], cmap='RdYlBu', 
                     s=40, alpha=0.9, edgecolors='none')
ax_main.set_xlabel("MDS Dimension 1", fontsize=14)
ax_main.set_ylabel("MDS Dimension 2", fontsize=14)
ax_main.set_xticks([])
ax_main.set_yticks([])

# Colorbar for Main
cbar = plt.colorbar(sc, ax=ax_main, fraction=0.046, pad=0.04)
cbar.set_label("Maximum TM Score Relative to PDB Structures", fontsize=14)

# --- Subplots ---
# 1. Helix
# Cmap: 'Blues'
sc_h = ax_helix.scatter(df_conf['x_mds'], df_conf['y_mds'], 
                        c=df_conf['pct_helix'], cmap='Blues', 
                        s=10, alpha=0.8)
clean_axis(ax_helix, "Helix")
ax_helix.set_ylabel("Percentage of Helices")
# Add small colorbar or just rely on intensity visual? Paper has side bars.
cbar_h = plt.colorbar(sc_h, ax=ax_helix, location='left', pad=0.15) # Put on left per figure?
# Actually the figure has them on the LEFT of the box. 
# location='left' puts it outside.

# 2. Strand
sc_s = ax_strand.scatter(df_conf['x_mds'], df_conf['y_mds'], 
                         c=df_conf['pct_strand'], cmap='Blues', 
                         s=10, alpha=0.8)
clean_axis(ax_strand, "Strand")
ax_strand.set_ylabel("Percentage of Strands")
cbar_s = plt.colorbar(sc_s, ax=ax_strand, location='left', pad=0.15)

# 3. Length
sc_l = ax_len.scatter(df_conf['x_mds'], df_conf['y_mds'], 
                      c=df_conf['seqlen'], cmap='Blues', 
                      s=10, alpha=0.8)
clean_axis(ax_len, "Length")
ax_len.set_ylabel("Sequence Length")
cbar_l = plt.colorbar(sc_l, ax=ax_len, location='left', pad=0.15)


plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Saved plot to {output_file}")
