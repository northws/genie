
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import argparse
import os

def visualize_trajectory(traj_file, output_file):
    # Load trajectory: [Steps, Length, 3]
    try:
        traj = np.load(traj_file)
        # Subsample to speed up (5x relative to previous)
        traj = traj[::5]
    except Exception as e:
        print(f"Error loading {traj_file}: {e}")
        return

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('persp', focal_length=0.2)
    
    # Static setup
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title(f'Generation Process')

    # Find global bounds to keep camera steady
    all_coords = traj.reshape(-1, 3)
    min_bound = all_coords.min(axis=0)
    max_bound = all_coords.max(axis=0)
    center = (min_bound + max_bound) / 2
    span = (max_bound - min_bound).max() / 2
    
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)

    # Initial plot
    xs = traj[0, :, 0]
    ys = traj[0, :, 1]
    zs = traj[0, :, 2]
    
    line, = ax.plot(xs, ys, zs, c='lightblue', linewidth=2, alpha=0.8)
    scatter = ax.scatter(xs, ys, zs, c=np.arange(len(xs)), cmap='viridis', s=50, depthshade=True)

    def update(frame):
        coords = traj[frame]
        xs = coords[:, 0]
        ys = coords[:, 1]
        zs = coords[:, 2]
        
        # Update line
        line.set_data(xs, ys)
        line.set_3d_properties(zs)
        
        # Update scatter
        scatter._offsets3d = (xs, ys, zs)
        
        ax.set_title(f'Step {frame}/{len(traj)-1}')
        return line, scatter

    ani = animation.FuncAnimation(fig, update, frames=len(traj), interval=20, blit=False)
    
    print(f"Saving animation to {output_file}...")
    ani.save(output_file, writer='pillow', fps=50)
    print("Done.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='Path to trajectory .npy file')
    parser.add_argument('output_file', help='Path to output .gif file')
    args = parser.parse_args()
    
    visualize_trajectory(args.input_file, args.output_file)
