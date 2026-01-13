import graphviz
import os

# Ensure dot is in PATH
os.environ["PATH"] += os.pathsep + '/opt/anaconda3/envs/py/bin'

def create_diagram(name, label, filename, features):
    dot = graphviz.Digraph(name, comment=label)
    dot.attr(rankdir='TB', compound='true', splines='ortho', nodesep='0.8', ranksep='0.8')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Helvetica', fontsize='11', margin='0.3,0.2')
    
    # Colors
    C_INPUT = '#E0E0E0'
    C_SINGLE = '#BBDEFB'
    C_PAIR = '#C8E6C9'
    C_TRANS = '#FFE0B2'
    C_STRUCT = '#FFCDD2'
    C_FLASH = '#E1BEE7'
    C_MHC = '#B2DFDB'
    C_LOSS = '#FF8A80'
    C_GRAD = '#FF5252' # Red for gradients

    # --- Nodes ---
    
    # Input Cluster
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='Input', color='lightgrey')
        c.node('Input', 'Noisy Protocol (T)\nTimesteps', fillcolor=C_INPUT)

    # Embedding
    dot.node('Single', 'SingleFeatureNet\n(Embeddings)', fillcolor=C_SINGLE)
    
    # Pair Gen
    dot.node('Pair', 'PairFeatureNet\n(RelPos + Templates)', fillcolor=C_PAIR)
    
    # Pair Transform (Conditional)
    if features.get('pair_transform'):
        transform_label = features.get('pair_transform_label', 'PairTransformNet\nTriangular Update\nO(L³)')
        transform_color = features.get('pair_transform_color', C_TRANS)
        dot.node('Transform', transform_label, fillcolor=transform_color)
    
    # Factorizer (Conditional - for Flash+mHC combo sometimes or just Flash)
    if features.get('factorizer'):
        dot.node('Factorizer', 'Linear Factorizer\nL² → L×R', fillcolor='#FFCC80')

    # Structure Net
    struct_label = features.get('structure_label', 'StructureNet')
    struct_color = features.get('structure_color', C_STRUCT)
    dot.node('Structure', struct_label, fillcolor=struct_color)
    
    # Output
    dot.node('Output', 'Denoised Structure\n(T_final)', fillcolor=C_INPUT)
    
    # Loss
    with dot.subgraph(name='cluster_loss') as c:
        c.attr(label='Optimization', style='dashed')
        c.node('Loss', 'Loss Function\n(MSE / Frame Loss)', fillcolor=C_LOSS, shape='ellipse')
        
        # Extra mHC Loss if requested
        if features.get('mhc_loss_term'):
             c.node('MHCLoss', 'mHC Regularization\n(Stability)', fillcolor=C_MHC, shape='ellipse')
             c.node('TotalLoss', 'Total Loss', fillcolor=C_LOSS, shape='doubleoctagon')

    # --- Edges (Forward) ---
    dot.edge('Input', 'Single', xlabel=' S')
    dot.edge('Single', 'Pair', xlabel=' S')
    
    # Path for Pair Features
    if features.get('pair_transform'):
        dot.edge('Pair', 'Transform', xlabel=' P')
        last_pair_node = 'Transform'
    elif features.get('factorizer'):
        dot.edge('Pair', 'Factorizer', xlabel=' P')
        last_pair_node = 'Factorizer'
    else:
        last_pair_node = 'Pair'

    # Connection to Structure
    dot.edge('Single', 'Structure', xlabel=' S')
    dot.edge(last_pair_node, 'Structure', xlabel=' P / Z')
    dot.edge('Input', 'Structure', xlabel=' T') # Input coords also go to StructureNet
    
    dot.edge('Structure', 'Output', xlabel=' T_pred')
    
    # To Loss
    dot.edge('Output', 'Loss', xlabel=' vs Ground Truth')

    # mHC Loss connections
    if features.get('mhc_loss_term'):
        dot.edge('Structure', 'MHCLoss', style='dashed', xlabel=' Residuals')
        dot.edge('Loss', 'TotalLoss')
        dot.edge('MHCLoss', 'TotalLoss')
        loss_node = 'TotalLoss'
    else:
        loss_node = 'Loss'

    # --- Edges (Backward / Backpropagation) ---
    # We add invisible edges or colored edges representing gradients
    # Backprop flows from Loss -> Trainable Modules
    
    dot.edge(loss_node, 'Structure', color=C_GRAD, style='dashed', xlabel=' Gradients')
    
    if features.get('pair_transform'):
        dot.edge('Structure', 'Transform', color=C_GRAD, style='dashed')
        dot.edge('Transform', 'Pair', color=C_GRAD, style='dashed')
    elif features.get('factorizer'):
        dot.edge('Structure', 'Factorizer', color=C_GRAD, style='dashed')
        dot.edge('Factorizer', 'Pair', color=C_GRAD, style='dashed')
    else:
        dot.edge('Structure', 'Pair', color=C_GRAD, style='dashed')
        
    dot.edge('Structure', 'Single', color=C_GRAD, style='dashed')
    
    # Save
    output_path = os.path.join('docs', 'images', filename)
    # render returns the path to the PDF/PNG
    result = dot.render(output_path, format='png', cleanup=True)
    print(f"Generated {result}")

if __name__ == "__main__":
    if not os.path.exists('docs/images'):
        os.makedirs('docs/images')
        
    # 1. Standard Genie
    create_diagram('Genie', 'Standard Genie', 'genie_arch', {
        'pair_transform': True,
        'structure_label': 'StructureNet\n(Standard IPA)\nO(L²)',
        'structure_color': '#FFCDD2' # Red-ish
    })
    
    # 2. Genie + FlashIPA
    create_diagram('GenieFlash', 'Genie + FlashIPA', 'genie_flash_arch', {
        'pair_transform': False, # Skipped
        'structure_label': 'FlashStructureNet\n(Flash IPA - 1D Bias)\nO(L)',
        'structure_color': '#E1BEE7' # Purple
    })
    
    # 3. Genie + mHC
    create_diagram('GenieMHC', 'Genie + mHC', 'genie_mhc_arch', {
        'pair_transform': True,
        'pair_transform_label': 'mHCPairTransformNet\n(mHC Residual Mixing)\nO(L³)',
        'structure_label': 'mHCStructureNet\n(Doubly Stochastic)\nStability++',
        'structure_color': '#B2DFDB' # Teal
    })
    
    # 4. Genie + FlashIPA + mHC (Full Architecture)
    create_diagram('GenieFlashMHC', 'Genie + FlashIPA + mHC (Full)', 'genie_flash_mhc_arch', {
        'pair_transform': True,
        'pair_transform_label': 'mHCPairTransformNet\n(mHC Mixing)',
        'pair_transform_color': '#B2DFDB',  # Teal for mHC
        'factorizer': True,
        'structure_label': 'mHCFlashStructureNet\n(Flash IPA + mHC)',
        'structure_color': '#E1BEE7',
        'mhc_loss_term': False  # Full mHC in architecture, no separate loss term needed
    })
    
    # 5. Genie + FlashIPA + mHC Loss Only (Regularization)
    create_diagram('GenieFlashMHCLoss', 'Genie + FlashIPA + mHC Loss', 'genie_flash_mhc_loss_arch', {
        'pair_transform': False,  # FlashMode skips PairTransformNet
        'structure_label': 'FlashStructureNet\n(Flash IPA - 1D Bias)\nO(L)',
        'structure_color': '#E1BEE7',
        'mhc_loss_term': True  # mHC as loss regularization only
    })
