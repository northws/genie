import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import argparse
import re

def load_data(filepath):
    # Try loading as text/csv since inspection showed it's csv
    try:
        data = np.loadtxt(filepath, delimiter=',')
        return data
    except Exception as e:
        # Fallback to npy if it was binary
        try:
            data = np.load(filepath)
            return data
        except Exception as e2:
            print(f"Failed to load data: {e} | {e2}")
            return None


def plot_structure(coords, output_image):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Increase perspective effect (lower focal_length = more perspective distortion/depth feeling)
    # Default is usually 1.0 (approaching 0 perspective). 0.2 is a common value for strong 3D.
    ax.set_proj_type('persp', focal_length=0.2) 
    
    xs = coords[:, 0]
    ys = coords[:, 1]
    zs = coords[:, 2]
    
    # "Atoms use dark blue dots"
    # depthshade=True gives a sense of depth (fog/fading)
    ax.scatter(xs, ys, zs, c='darkblue', s=150, depthshade=True, label='Atom')
    
    # "Bonds use light color"
    ax.plot(xs, ys, zs, c='lightblue', linewidth=3, label='Bond', alpha=0.8)
    
    # Axis with units (Angstrom)
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    
    # Set view to see structure better
    ax.view_init(elev=20., azim=-35)
    
    # Bring camera closer to enhance perspective
    ax.dist = 8

    plt.title('Structure Visualization')
    plt.legend()
    plt.savefig(output_image, dpi=300)
    print(f"Saved visualization to {output_image}")
    plt.close(fig)

def visualize_sample(input_file, output_dir=None):
    """
    Generic function to visualize a sample file.
    Args:
        input_file: Path to the .npy or .txt sample file.
        output_dir: Directory to save outputs. If None, uses input file's directory.
    """
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found.")
        return

    data = load_data(input_file)
    if data is None:
        return

    if data.shape[1] != 3:
        print("Error: Data must have 3 columns (x, y, z).")
        return

    # Derive generic info from filename if possible (x_y.npy)
    filename = os.path.basename(input_file)
    match = re.match(r'(\d+)_(\d+)\.', filename)
    if match:
        expected_atoms = int(match.group(1))
        sample_id = match.group(2)
        print(f"Processing sample {sample_id} with expected {expected_atoms} Carbon atoms...")
        if len(data) != expected_atoms:
            print(f"Warning: Filename indicates {expected_atoms} atoms but found {len(data)}.")
    
    base_name = os.path.splitext(filename)[0]
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_base = os.path.join(output_dir, base_name)
    else:
        # Default to same directory as input
        save_base = os.path.join(os.path.dirname(input_file), base_name)
    
    img_output = save_base + "_vis.png"
    
    plot_structure(data, img_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize structure coordinates.")
    parser.add_argument("input_file", help="Path to input .npy file")
    parser.add_argument("--output_dir", "-o", help="Directory to save output files", default=None)
    
    args = parser.parse_args()
    
    visualize_sample(args.input_file, args.output_dir)
