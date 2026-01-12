import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
origin_loss_epoch = pd.read_csv('Training_process_parameters/origin_work_epoch_loss.csv')
this_loss_epoch = pd.read_csv('Training_process_parameters/this_work_epoch_loss.csv')
flash_loss_epoch = pd.read_csv('Training_process_parameters/完整flash_epoch_loss.csv')

origin_loss_step = pd.read_csv('Training_process_parameters/origin_work_step_loss.csv')
this_loss_step = pd.read_csv('Training_process_parameters/this_work_step_loss.csv')
flash_loss_step = pd.read_csv('Training_process_parameters/完整flash_step_loss.csv')

origin_mem = pd.read_csv('Training_process_parameters/origin_work_train_GPUmemory.csv')
this_mem = pd.read_csv('Training_process_parameters/this_work_train_GPUmemory.csv')
flash_mem = pd.read_csv('Training_process_parameters/完整flash_memory.csv')

origin_gpu = pd.read_csv('Training_process_parameters/origin_work_gpu_use.csv')
this_gpu = pd.read_csv('Training_process_parameters/this_work_gpu_use.csv')
flash_gpu = pd.read_csv('Training_process_parameters/完整flash_gpu_use.csv')

# Configure style
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

labels = ['Original\nWork', 'This Work\n(Optimized)', 'Full Flash\nMode']
colors = ['#1f77b4', '#2ca02c', '#ff7f0e']

# 1. Training Time Comparison
origin_time = origin_mem.iloc[:, 0].max() / 3600  # Convert to hours
this_time = this_mem.iloc[:, 0].max() / 3600    # Convert to hours
flash_time = flash_mem.iloc[:, 0].max() / 3600  # Convert to hours

times = [origin_time, this_time, flash_time]

bars = axes[0, 0].bar(labels, times, color=colors, alpha=0.8)
axes[0, 0].set_ylabel('Training Time (Hours)')
axes[0, 0].set_title('Total Training Time (500 Epochs)')
axes[0, 0].grid(axis='y', linestyle='--', alpha=0.5)

# Add text labels on bars
for bar in bars:
    height = bar.get_height()
    axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}h',
                ha='center', va='bottom', fontsize=10)

# Calculate speedup
speedup = origin_time / flash_time
axes[0, 0].text(0.5, 0.92, f'{speedup:.1f}x Speedup (Flash vs Original)', 
             transform=axes[0, 0].transAxes, ha='center', fontweight='bold', fontsize=11)


# 2. GPU Memory Comparison
origin_max_mem = origin_mem.iloc[:, 1].max() / (1024**3) # GB
this_max_mem = this_mem.iloc[:, 1].max() / (1024**3)     # GB
flash_max_mem = flash_mem.iloc[:, 1].max() / (1024**3)   # GB

mems = [origin_max_mem, this_max_mem, flash_max_mem]

bars_mem = axes[0, 1].bar(labels, mems, color=colors, alpha=0.8)
axes[0, 1].set_ylabel('Peak GPU Memory (GB)')
axes[0, 1].set_title('Peak GPU Memory Usage')
axes[0, 1].grid(axis='y', linestyle='--', alpha=0.5)

# Add text labels
for bar in bars_mem:
    height = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f} GB',
                ha='center', va='bottom', fontsize=10)

reduction = (1 - flash_max_mem / origin_max_mem) * 100
axes[0, 1].text(0.5, 0.92, f'-{reduction:.0f}% Memory (Flash vs Original)', 
             transform=axes[0, 1].transAxes, ha='center', fontweight='bold', fontsize=11)


# 3. GPU Utilization Comparison
origin_util = origin_gpu.iloc[:, 1].mean()
this_util = this_gpu.iloc[:, 1].mean()
flash_util = flash_gpu.iloc[:, 1].mean()

utils = [origin_util, this_util, flash_util]

bars_util = axes[1, 0].bar(labels, utils, color=colors, alpha=0.8)
axes[1, 0].set_ylabel('Average GPU Utilization (%)')
axes[1, 0].set_title('Average GPU Utilization')
axes[1, 0].grid(axis='y', linestyle='--', alpha=0.5)
axes[1, 0].set_ylim(0, 100)

# Add text labels
for bar in bars_util:
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10)

axes[1, 0].text(0.5, 0.92, 'Flash mode is memory-bandwidth bound', 
             transform=axes[1, 0].transAxes, ha='center', fontweight='bold', fontsize=11, style='italic')


# 4. Training Loss Curves (Epoch Loss with Smoothing)
from scipy.interpolate import make_interp_spline

# Get epoch and loss columns
origin_epochs = origin_loss_epoch['epoch'].values
origin_loss = origin_loss_epoch.iloc[:, 1].values
this_epochs = this_loss_epoch['epoch'].values
this_loss = this_loss_epoch.iloc[:, 1].values
flash_epochs = flash_loss_epoch['epoch'].values
flash_loss = flash_loss_epoch.iloc[:, 1].values

# Create smooth curves using spline interpolation
def smooth_curve(x, y, num_points=500):
    x_smooth = np.linspace(x.min(), x.max(), num_points)
    spl = make_interp_spline(x, y, k=3)
    y_smooth = spl(x_smooth)
    return x_smooth, y_smooth

origin_x_smooth, origin_y_smooth = smooth_curve(origin_epochs, origin_loss)
this_x_smooth, this_y_smooth = smooth_curve(this_epochs, this_loss)
flash_x_smooth, flash_y_smooth = smooth_curve(flash_epochs, flash_loss)

# Plot smooth epoch loss curves
axes[1, 1].plot(origin_x_smooth, origin_y_smooth, label='Original Work', color='#1f77b4', linewidth=2)
axes[1, 1].plot(this_x_smooth, this_y_smooth, label='This Work (Optimized)', color='#2ca02c', linewidth=2)
axes[1, 1].plot(flash_x_smooth, flash_y_smooth, label='Full Flash Mode', color='#ff7f0e', linewidth=2)

axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Training Loss')
axes[1, 1].set_title('Training Loss Convergence (Epoch)')
axes[1, 1].legend(loc='upper right')
axes[1, 1].grid(True, linestyle='--', alpha=0.3)
axes[1, 1].set_ylim(0.7, 1.5) # Zoom in to relevant range

plt.tight_layout()
plt.savefig('Training_process_parameters/optimization_comparison.png', dpi=300, bbox_inches='tight')
print("Comparison plot saved to Training_process_parameters/optimization_comparison.png")
