import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
sns.set_theme(style="ticks")
plt.rcParams['font.family'] = 'sans-serif'

# Load data
csv_path = "/root/autodl-tmp/genie/runs/final_final-v0/version_3/samples/epoch_499/evaluations/info.csv"
novelty_path = "/root/autodl-tmp/genie/runs/final_final-v0/version_3/samples/epoch_499/evaluations/novelty_hybrid.csv"
output_file = "genie_analysis_figure2_repro_v2_hybrid.png"

print(f"Loading data from {csv_path}...")
try:
    df = pd.read_csv(csv_path)
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit(1)

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

# Ensure columns exist
required_columns = ['scTM', 'pLDDT', 'pct_helix', 'pct_strand', 'seqlen']
for col in required_columns:
    if col not in df.columns:
        print(f"Missing column: {col}")
        exit(1)

# Definitions
df['Designable'] = df['scTM'] > 0.5
df['Confidently Designable'] = (df['scTM'] > 0.5) & (df['pLDDT'] > 70)

if has_novelty:
    # Novel Definition: Confidently Designable AND Max TM < 0.5
    df['Novel and Confidently Designable'] = df['Confidently Designable'] & (df['max_tm_to_pdb'] < 0.5)
else:
    # Fallback
    df['Novel and Confidently Designable'] = df['Confidently Designable'] 

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12)) 
# Layout:
# A (Top Left)   C (Top Right)
# B (Bottom Left) D (Bottom Right)

# === A: Heatmap pLDDT vs scTM (Genie) ===
ax_a = axes[0, 0]
x = df['scTM']
y = df['pLDDT']
# 2D Histogram
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
    # 2D Histogram
    h2 = ax_b.hist2d(x_sse, y_sse, bins=[10, 10], range=[[0, 1], [0, 1]], cmap="Reds", cmin=1)
    ax_b.set_xlabel("Percentage of Helices")
    ax_b.set_ylabel("Percentage of Strands")
    ax_b.set_title(f"B. Genie: SSE Distribution (N={len(cd_df)})")
    fig.colorbar(h2[3], ax=ax_b, label='Density')
else:
    ax_b.text(0.5, 0.5, "No Confidently Designable Data", ha='center')
ax_b.set_xlim(0, 1)
ax_b.set_ylim(0, 1)

# === C: Histogram by Length (Confidently Designable) ===
ax_c = axes[0, 1]
if not cd_df.empty:
    # Bins
    bins = np.arange(40, 140, 10) # 50, 60... 130
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
# Using colors similar to paper: Cyan, Purple, Pink
bar_colors = ['#17becf', '#9467bd', '#ff69b4'] 

bars = ax_d.bar(labels, values, color=bar_colors)
ax_d.set_ylabel("Number of Domains")
ax_d.set_title(f"D. Designability Counts (Total Attempted: {total_generated})")

# Add text labels
for rect in bars:
    height = rect.get_height()
    ax_d.text(rect.get_x() + rect.get_width()/2.0, height + 1, f'{int(height)}', ha='center', va='bottom')

# Adjust layout
plt.suptitle("Recreation of Figure 2 Analysis for Genie Model", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.92)

# Save
plt.savefig(output_file, dpi=150)
print(f"Plot saved to {output_file}")
