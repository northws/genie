import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def draw_flowchart(title, nodes, filename):
    """
    Draws a vertical flowchart defined by nodes.
    nodes: list of dicts with 'label', 'sublabel', 'width', 'height', 'color'
    """
    fig, ax = plt.subplots(figsize=(8, len(nodes) * 2.5 + 2))
    ax.set_ylim(0, len(nodes) * 2.5 + 1)
    ax.set_xlim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, len(nodes) * 2.5 + 0.5, title, ha='center', va='center', fontsize=16, fontweight='bold')

    prev_y = None
    prev_height = 0 # Initialize
    
    # Calculate positions (top to bottom)
    start_y = len(nodes) * 2.5 - 1.5
    
    box_width = 6
    x_center = 5
    
    for i, node in enumerate(nodes):
        y_center = start_y - i * 2.5
        box_height = node.get('height', 1.5)
        
        # Color mapping
        color = node.get('color', '#dddddd')
        
        # Draw arrow from previous
        if prev_y is not None:
            # Arrow
            ax.add_patch(patches.FancyArrowPatch(
                (x_center, prev_y - prev_height/2), 
                (x_center, y_center + box_height/2),
                arrowstyle='->', mutation_scale=20, color='#555555', lw=2
            ))
            
            # Label on arrow (optional)
            if 'arrow_label' in node:
                 ax.text(x_center + 0.2, (prev_y - prev_height/2 + y_center + box_height/2)/2, 
                         node['arrow_label'], va='center', fontsize=10, color='#333333')

        # Draw Box
        rect = patches.FancyBboxPatch(
            (x_center - box_width/2, y_center - box_height/2), 
            box_width, box_height,
            boxstyle="round,pad=0.1,rounding_size=0.2",
            facecolor=color, edgecolor='#555555', linewidth=1.5
        )
        ax.add_patch(rect)
        
        # Text
        main_label = node['label']
        if 'formula' in node:
            main_label +=f"\n{node['formula']}"
            
        ax.text(x_center, y_center + 0.2 if 'sublabel' in node else y_center, 
                main_label, 
                ha='center', va='center', fontsize=12, fontweight='bold', wrap=True)
        
        if 'sublabel' in node:
            ax.text(x_center, y_center - 0.3, node['sublabel'], 
                    ha='center', va='center', fontsize=10, style='italic', wrap=True)
            
        # Side notes (dimensions)
        if 'dim' in node:
            ax.text(x_center + box_width/2 + 0.2, y_center, node['dim'],
                   ha='left', va='center', fontsize=10, color='#333333')

        prev_y = y_center
        prev_height = box_height

    plt.tight_layout()
    output_path = os.path.join('/Users/zaldehyde/genie/docs/images', filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated {output_path}")

# Colors
C_INPUT = '#e0e0e0'  # Grey
C_SINGLE = '#bbdefb' # Blue
C_PAIR = '#c8e6c9'   # Green
C_TRANS = '#ffe0b2'  # Orange
C_STRUCT = '#ffcdd2' # Red
C_FLASH = '#e1bee7'  # Purple
C_MHC = '#b2dfdb'    # Teal

# Diagram 1: Genie
genie_nodes = [
    {
        'label': 'Input',
        'sublabel': 'Noisy Frames + Timesteps',
        'color': C_INPUT,
        'dim': '$[B, L, 4, 3]$'
    },
    {
        'label': 'SingleFeatureNet',
        'sublabel': 'Sequence + Timestep Embeddings',
        'color': C_SINGLE,
        'dim': '$S: [B, L, C_s]$',
        'arrow_label': '$T$'
    },
    {
        'label': 'PairFeatureNet',
        'sublabel': 'Relative Pos + Template Feats',
        'color': C_PAIR,
        'dim': '$P: [B, L, L, C_p]$',
        'arrow_label': '$S$'
    },
    {
        'label': 'PairTransformNet',
        'sublabel': 'Triangular Update & Attention x N',
        'color': C_TRANS,
        'formula': '$O(L^3)$ Compute\n$O(L^2)$ Memory',
        'dim': '$P_{updated}: [B, L, L, C_p]$'
    },
    {
        'label': 'StructureNet',
        'sublabel': 'Invariant Point Attention (IPA) x N\nStandard Backbone',
        'color': C_STRUCT,
        'formula': '$O(L^2)$ Attention',
        'dim': '$T_{final}: [B, L, 4, 3]$',
        'arrow_label': '$S, P$'
    }
]

# Diagram 2: Genie + FlashIPA
flash_nodes = [
    {
        'label': 'Input',
        'sublabel': 'Noisy Frames + Timesteps',
        'color': C_INPUT
    },
    {
        'label': 'SingleFeatureNet',
        'sublabel': 'Sequence + Timestep Embeddings',
        'color': C_SINGLE
    },
    {
        'label': 'PairFeatureNet',
        'sublabel': 'Relative Pos + Template Feats',
        'color': C_PAIR
    },
    # Note: PairTransformNet skipped
    {
        'label': 'FlashStructureNet',
        'sublabel': 'Flash IPA (Optimized) x N\n1D Bias Mode',
        'color': C_FLASH,
        'formula': '$O(L)$ Effective Memory\n$O(L)$ Factorized Attention',
        'dim': 'Output: [B, L, 4, 3]',
        'arrow_label': '$S, P$ (No PairTransform)'
    }
]

# Diagram 3: Genie + mHC
mhc_nodes = [
    {
        'label': 'Input',
        'sublabel': 'Noisy Frames + Timesteps',
        'color': C_INPUT
    },
    {
        'label': 'SingleFeatureNet',
        'sublabel': 'Sequence + Timestep Embeddings',
        'color': C_SINGLE
    },
    {
        'label': 'PairFeatureNet',
        'sublabel': 'Relative Pos + Template Feats',
        'color': C_PAIR
    },
    {
        'label': 'PairTransformNet',
        'sublabel': 'Triangular Update & Attention x N',
        'color': C_TRANS
    },
    {
        'label': 'mHCStructureNet',
        'sublabel': 'Doubly Stochastic Mixing\nExpanded Residual Stream',
        'color': C_MHC,
        'formula': 'Training Stability++\nStandard IPA $O(L^2)$',
        'dim': 'Output: [B, L, 4, 3]',
        'arrow_label': '$S, P$'
    }
]

# Diagram 4: Genie + FlashIPA + mHC
combined_nodes = [
    {
        'label': 'Input',
        'sublabel': 'Noisy Frames + Timesteps',
        'color': C_INPUT
    },
    {
        'label': 'SingleFeatureNet',
        'sublabel': 'Sequence + Timestep Embeddings',
        'color': C_SINGLE
    },
    {
        'label': 'PairFeatureNet',
        'sublabel': 'Relative Pos + Template Feats',
        'color': C_PAIR
    },
    {
        'label': 'PairTransformNet',
        'sublabel': 'Triangular Update & Attention',
        'color': C_TRANS,
        'formula': '(Optional, used in implementation)'
    },
    {
        'label': 'LinearFactorizer',
        'sublabel': 'Low-rank Factorization of Pair Features',
        'color': '#ffcc80', # Orange distinct
        'formula': '$L^2 \\to L \\times R$',
        'dim': '$Z_{fact}: [B, L, R]$',
        'arrow_label': '$P$'
    },
    {
        'label': 'mHCFlashStructureNet',
        'sublabel': 'Flash IPA + mHC Mixing',
        'color': '#e1bee7', # Reuse purple variant or mix
        'formula': 'Stability + Efficiency\n$O(L)$ Memory',
        'dim': 'Output: [B, L, 4, 3]',
        'arrow_label': '$S, Z_{fact}$'
    }
]

if __name__ == "__main__":
    if not os.path.exists('/Users/zaldehyde/genie/docs/images'):
        os.makedirs('/Users/zaldehyde/genie/docs/images')
        
    print("Generating diagrams...")
    draw_flowchart('Standard Genie Architecture', genie_nodes, 'genie_arch.png')
    draw_flowchart('Genie + FlashIPA Architecture', flash_nodes, 'genie_flash_arch.png')
    draw_flowchart('Genie + mHC Architecture', mhc_nodes, 'genie_mhc_arch.png')
    draw_flowchart('Genie + FlashIPA + mHC Architecture', combined_nodes, 'genie_flash_mhc_arch.png')
    print("Done.")
