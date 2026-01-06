import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Bio.PDB import PDBIO
from Bio.PDB.StructureBuilder import StructureBuilder
from Bio.PDB import Atom, Residue, Chain, Model, Structure
import sys
import os
import argparse
import re
from scipy import interpolate

def load_data(filepath):
    try:
        data = np.loadtxt(filepath, delimiter=',')
        return data
    except Exception:
        try:
            return np.load(filepath)
        except Exception as e:
            print(f"Failed to load data: {e}")
            return None

def create_biopython_structure(coords, structure_id="STRUCT"):
    """
    Creates a Bio.PDB Structure object from C-alpha coordinates.
    Assume all residues are Glycine (GLY).
    """
    sb = StructureBuilder()
    sb.init_structure(structure_id)
    sb.init_model(0)
    sb.init_chain('A')
    
    for i, (x, y, z) in enumerate(coords):
        # Residue ID: (' ', resseq, ' ')
        res_id = (' ', i + 1, ' ')
        # Create a generic Glycine residue
        res = Residue.Residue(res_id, 'GLY', ' ')
        
        # Atom name, coord, B-factor, occupancy, altloc, fullname, serial_number, element
        # Standard PDB C-alpha name is 'CA'
        atom = Atom.Atom(name='CA', coord=np.array([x, y, z], dtype='float32'), 
                         bfactor=0.0, occupancy=1.0, altloc=' ', 
                         fullname=' CA ', serial_number=i+1, element='C')
        
        res.add(atom)
        sb.chain.add(res)
        
    return sb.structure

def save_pdb(structure, output_path):
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_path)
    print(f"Saved PDB to {output_path}")

def plot_protein_structure(coords, output_image):
    """
    Visualizes the protein backbone as a smooth curve with N->C gradient.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('persp', focal_length=0.2)
    
    # Data points
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    n_residues = len(coords)
    
    # Spline interpolation for smooth backbone "ribbon" look
    # We need parameter t (index)
    t = np.arange(n_residues)
    
    # Interpolate if we have enough points, else just plot lines
    if n_residues > 3:
        try:
            # k=3 cubic spline, s=0 no smoothing (pass through points)
            tck, u = interpolate.splprep([x, y, z], s=0, k=3)
            # Generate more points for smoothness
            u_fine = np.linspace(0, 1, n_residues * 10)
            x_fine, y_fine, z_fine = interpolate.splev(u_fine, tck)
            
            # Color gradient: Blue (N-term) to Red (C-term)
            # We plot segments to handle changing colors
            points = np.array([x_fine, y_fine, z_fine]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # Create a color map aligned with the progress along the chain
            cmap = plt.get_cmap('jet')
            norm = plt.Normalize(0, 1)
            colors = cmap(np.linspace(0, 1, len(segments)))
            
            # We can't use Line3DCollection easily with varying colors in standard matplotlib 3d efficiently 
            # without some tweaks, but scatter implies dots. 
            # A simple loop for segments is slow but works for PDB sizes usually < 1000 res.
            # But for speed, let's use the basic scatter trace or direct plot.
            # A common trick is just scatter dense points.
            
            ax.scatter(x_fine, y_fine, z_fine, c=u_fine, cmap='jet', s=10, depthshade=True, alpha=0.5, label='Backbone Trace')
            
            # Highlight C-alpha positions (original points)
            ax.scatter(x, y, z, c='black', s=20, depthshade=True, alpha=0.8, marker='o') # CA atoms
            
        except Exception as e:
            print(f"Interpolation failed: {e}. Falling back to simple line.")
            ax.plot(x, y, z, c='blue')
    else:
        ax.plot(x, y, z, c='blue', linewidth=2)
        ax.scatter(x, y, z, c='red')

    # Start/End markers
    ax.text(x[0], y[0], z[0], "N", color='blue', fontsize=12, fontweight='bold')
    ax.text(x[-1], y[-1], z[-1], "C", color='red', fontsize=12, fontweight='bold')

    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.view_init(elev=20., azim=-35)
    ax.dist = 8 # Closer view

    plt.title('Protein Backbone (N-Blue -> C-Red)')
    # Removed legend because scatter with cmap doesn't support generic legend easily without proxy artists
    plt.savefig(output_image, dpi=300)
    print(f"Saved visualization to {output_image}")
    plt.close(fig)

def visualize_sample(input_file, output_dir=None):
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found.")
        return

    data = load_data(input_file)
    if data is None:
        return

    if data.shape[1] != 3:
        print("Error: Data must have 3 columns (x, y, z).")
        return

    # Info parsing
    filename = os.path.basename(input_file)
    match = re.match(r'(\d+)_(\d+)\.', filename)
    if match:
        print(f"Processing sample {match.group(2)} ({match.group(1)} residues)...")
    
    base_name = os.path.splitext(filename)[0]
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_base = os.path.join(output_dir, base_name)
    else:
        save_base = os.path.join(os.path.dirname(input_file), base_name)
    
    # 1. Generate PDB
    pdb_output = save_base + ".pdb"
    structure = create_biopython_structure(data, structure_id=base_name)
    save_pdb(structure, pdb_output)
    
    # 2. Visualize
    img_output = save_base + "_protein.png"
    plot_protein_structure(data, img_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize protein backbone from coordinates.")
    parser.add_argument("input_file", help="Path to input .npy file")
    parser.add_argument("--output_dir", "-o", help="Directory to save output files", default=None)
    
    args = parser.parse_args()
    
    visualize_sample(args.input_file, args.output_dir)
