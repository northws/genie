import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Configuration
run_dir = "/root/autodl-tmp/genie/runs/final_final-v0/version_3/samples/epoch_499/evaluations"
designs_dir = os.path.join(run_dir, "designs")
info_path = os.path.join(run_dir, "info.csv")
novelty_path = os.path.join(run_dir, "novelty_hybrid.csv")
output_file = "genie_structure_examples_novel.png"

# Load Data
print("Loading data...")
try:
    df = pd.read_csv(info_path)
    df_novel = pd.read_csv(novelty_path)
    
    # Merge
    merged = df.merge(df_novel, on='domain')
    
    # Filter: Confidently Designable (scTM > 0.5, pLDDT > 70)
    confident = merged[(merged['scTM'] > 0.5) & (merged['pLDDT'] > 70)]
    
    # Sort by Novelty (Max TM ascending)
    top_novel = confident.sort_values('max_tm_to_pdb', ascending=True).head(4)
    
    pdb_ids = top_novel['domain'].tolist()
    tm_scores = top_novel['max_tm_to_pdb'].tolist()
    
    print(f"Selected Top 4 Novel Domains: {pdb_ids}")
    print(f"TM Scores: {tm_scores}")
    
except Exception as e:
    print(f"Error loading/processing data: {e}")
    # Fallback
    pdb_ids = ['120_7', '76_6', '55_15', '68_1']
    tm_scores = [None]*4

def parse_pdb_ca(filepath):
    coords = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
    return np.array(coords)

fig = plt.figure(figsize=(20, 5))

for i, pid in enumerate(pdb_ids):
    path = os.path.join(designs_dir, f"{pid}.pdb")
    if not os.path.exists(path):
        # Fallback to base dir check or try adding .pdb
        print(f"File not found: {path}")
        continue
        
    coords = parse_pdb_ca(path)
    
    ax = fig.add_subplot(1, 4, i+1, projection='3d')
    ax.set_proj_type('persp', focal_length=0.2)
    
    xs = coords[:, 0]
    ys = coords[:, 1]
    zs = coords[:, 2]
    
    # Gradient color based on index (Rainbow like PDB)
    # Normalized index
    N = len(coords)
    colors = plt.cm.jet(np.linspace(0, 1, N))
    
    # Plot segments
    for j in range(N-1):
        ax.plot(xs[j:j+2], ys[j:j+2], zs[j:j+2], color=colors[j], linewidth=3, alpha=0.9)
    
    # Add atoms as small spheres
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
