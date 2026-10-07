import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Data
N = [2, 3, 4, 5]
# N=3 values come from the paper's Transfer_GSM8K benchmark (mean across seeds)
accuracy = [98.0, 95.0, 98.7, 98.0]
comm_tokens = [13.4, 52.2, 157.2, 246.2]
total_tokens = [1400.0, 1680.0, 3642.4, 4539.5] # estimated total tokens for N=3

os.makedirs("08-06results/plots", exist_ok=True)

# Create figure
fig, ax1 = plt.subplots(figsize=(8, 5), facecolor='white')

# Accuracy Plot (Primary Axis)
color = '#1f77b4' # Premium blue
ax1.set_xlabel('Agent Pool Size (N)', fontsize=12, fontweight='bold', labelpad=10)
ax1.set_ylabel('Accuracy (%)', color=color, fontsize=12, fontweight='bold', labelpad=10)
line1 = ax1.plot(N, accuracy, marker='o', linewidth=2.5, color=color, markersize=8, label='Accuracy (%)')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.set_ylim(90, 100)
ax1.set_xticks(N)

# Communication Tokens Plot (Secondary Axis)
ax2 = ax1.twinx()
color = '#ff7f0e' # Premium orange
ax2.set_ylabel('Avg. Comm Tokens per Sample', color=color, fontsize=12, fontweight='bold', labelpad=10)
line2 = ax2.plot(N, comm_tokens, marker='s', linewidth=2.5, color=color, markersize=8, label='Comm Tokens')
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(0, 300)

# Add Legend
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', frameon=True, facecolor='white', edgecolor='#e2e2e2')

plt.title('System Scaling & Communication Overhead vs. Agent Pool Size (N)\n(SparseComm on GSM8K, 150 test samples)', fontsize=13, fontweight='bold', pad=15)
fig.tight_layout()

# Save
plot_path = "08-06results/plots/agent_scaling_study.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Scaling study plot saved to: {plot_path}")
