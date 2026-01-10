import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
origin_loss_epoch = pd.read_csv('Training_process_parameters/origin_work_epoch_loss.csv')
this_loss_epoch = pd.read_csv('Training_process_parameters/this_work_epoch_loss.csv')
origin_loss_step = pd.read_csv('Training_process_parameters/origin_work_step_loss.csv')
this_loss_step = pd.read_csv('Training_process_parameters/this_work_step_loss.csv')

origin_mem = pd.read_csv('Training_process_parameters/origin_work_train_GPUmemory.csv')
this_mem = pd.read_csv('Training_process_parameters/this_work_train_GPUmemory.csv')

# Configure style
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12})
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Training Time Comparison
origin_time = origin_mem.iloc[:, 0].max() / 3600  # Convert to hours
this_time = this_mem.iloc[:, 0].max() / 3600    # Convert to hours

times = [origin_time, this_time]
labels = ['Original Work', 'This Work\n(Optimized)']
colors = ['#1f77b4', '#2ca02c']

bars = axes[0].bar(labels, times, color=colors, alpha=0.8)
axes[0].set_ylabel('Training Time (Hours)')
axes[0].set_title('Total Training Time (500 Epochs)')
axes[0].grid(axis='y', linestyle='--', alpha=0.5)

# Add text labels on bars
for bar in bars:
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}h',
                ha='center', va='bottom')

# Calculate speedup
speedup = origin_time / this_time
axes[0].text(0.5, 0.9, f'{speedup:.1f}x Speedup', 
             transform=axes[0].transAxes, ha='center', fontweight='bold', fontsize=12)


# 2. GPU Memory Comparison
origin_max_mem = origin_mem.iloc[:, 1].max() / (1024**3) # GB
this_max_mem = this_mem.iloc[:, 1].max() / (1024**3)     # GB

mems = [origin_max_mem, this_max_mem]

bars_mem = axes[1].bar(labels, mems, color=colors, alpha=0.8)
axes[1].set_ylabel('Peak GPU Memory (GB)')
axes[1].set_title('Peak GPU Memory Usage')
axes[1].grid(axis='y', linestyle='--', alpha=0.5)

# Add text labels
for bar in bars_mem:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f} GB',
                ha='center', va='bottom')

reduction = (1 - this_max_mem / origin_max_mem) * 100
axes[1].text(0.5, 0.9, f'-{reduction:.0f}% Memory', 
             transform=axes[1].transAxes, ha='center', fontweight='bold', fontsize=12)


# 3. Training Loss Curves (Step Loss with Smoothing)
# Apply rolling average to smooth the step loss
window_size = 100
origin_smooth = origin_loss_step['skilled-firefly-1 - train_loss_step'].rolling(window=window_size).mean()
this_smooth = this_loss_step['final_final-v0 - train_loss_step'].rolling(window=window_size).mean()

# Use index as x-axis
axes[2].plot(origin_smooth, label='Original Work (Smoothed)', color='#1f77b4', linewidth=2)
axes[2].plot(this_smooth, label='This Work (Smoothed)', color='#2ca02c', linewidth=2, linestyle='--')

axes[2].set_xlabel('Training Steps (x50)')
axes[2].set_ylabel('Training Loss')
axes[2].set_title('Training Loss Convergence (Moving Avg)')
axes[2].legend()
axes[2].grid(True, linestyle='--', alpha=0.3)
axes[2].set_ylim(0.5, 1.5) # Zoom in to relevant range

plt.tight_layout()
plt.savefig('optimization_comparison.png', dpi=300, bbox_inches='tight')
print("Comparison plot saved to optimization_comparison.png")
