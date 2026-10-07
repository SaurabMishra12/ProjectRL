import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Ensure target directories exist
os.makedirs("08-06results/plots", exist_ok=True)
artifact_dir = "/home/saurab/.gemini/antigravity-ide/brain/fd73fec3-b16f-44af-88fb-0351b4e10822"
os.makedirs(artifact_dir, exist_ok=True)

# Data
train_n = [100, 200, 300, 400]
arm_00_static_ring = [56.3, 95.3, 94.7, 93.3]
arm_01_ppo_topo    = [66.0, 92.0, 94.0, 96.7]
arm_10_sparse_ring = [63.0, 94.7, 94.7, 95.3]
arm_11_sparsecomm  = [65.0, 98.0, 92.7, 93.3]

comm_tokens_arm11  = [496.7, 124.6, 123.3, 58.9]

# Styling setup
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True, gridspec_kw={'height_ratios': [2, 1.2]})
fig.patch.set_facecolor('white')

# Colors
color_static = '#7f7f7f'      # Static Ring (Arm 00) - Neutral Gray
color_ppo    = '#2ca02c'      # PPO topology (Arm 01) - Green
color_sring  = '#9467bd'      # Sparse Ring (Arm 10) - Purple
color_scomm  = '#d62728'      # SparseComm (Arm 11) - Red/Crimson
color_comm   = '#1f77b4'      # Comm tokens - Deep Blue

# --- PANEL 1: Accuracy Curves ---
ax1.plot(train_n, arm_00_static_ring, marker='o', linewidth=2.2, markersize=7, color=color_static, label='Arm 00: Static Ring', linestyle='--')
ax1.plot(train_n, arm_01_ppo_topo,    marker='s', linewidth=2.2, markersize=7, color=color_ppo,    label='Arm 01: PPO topology', linestyle='-.')
ax1.plot(train_n, arm_10_sparse_ring, marker='^', linewidth=2.2, markersize=7, color=color_sring,  label='Arm 10: Sparse Ring', linestyle=':')
ax1.plot(train_n, arm_11_sparsecomm,  marker='D', linewidth=3.0, markersize=8, color=color_scomm,  label='Arm 11: SparseComm (Ours)', linestyle='-')

# Highlight peak for SparseComm
ax1.annotate('Peak Accuracy: 98.0%', xy=(200, 98.0), xytext=(220, 99.2),
             arrowprops=dict(facecolor=color_scomm, shrink=0.08, width=1.5, headwidth=6),
             fontsize=10, fontweight='bold', color=color_scomm,
             bbox=dict(boxstyle="round,pad=0.3", fc="#ffebee", ec=color_scomm, lw=1))

ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', labelpad=10)
ax1.set_ylim(50, 102)
ax1.set_yticks(np.arange(50, 101, 10))
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.95, edgecolor='#cccccc', fontsize=10.5)
ax1.set_title('Factorial-Arm Accuracy & Communication Cost vs. Training Scale (GSM8K)', fontsize=13.5, fontweight='bold', pad=12)

# --- PANEL 2: Communication Cost ---
ax2.plot(train_n, comm_tokens_arm11, marker='h', linewidth=2.5, markersize=8, color=color_comm, label='Comm. Tokens (SparseComm)')
ax2.fill_between(train_n, comm_tokens_arm11, color=color_comm, alpha=0.15)

# Value annotations on tokens plot
for x, y in zip(train_n, comm_tokens_arm11):
    ax2.annotate(f'{y:.1f}', xy=(x, y), xytext=(x, y + 35),
                 ha='center', fontsize=9.5, fontweight='bold', color=color_comm,
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color_comm, lw=0.8, alpha=0.9))

ax2.set_xlabel('Training Scale (N Training Examples)', fontsize=12, fontweight='bold', labelpad=8)
ax2.set_ylabel('Comm. Tokens', fontsize=12, fontweight='bold', labelpad=10)
ax2.set_xticks(train_n)
ax2.set_xticklabels(['100', '200', '300', '400'], fontsize=11)
ax2.set_ylim(0, 580)
ax2.grid(True, linestyle='--', alpha=0.6)

# Key Insight Annotation
ax2.text(0.55, 0.72, 'Efficiency keeps improving\n(496.7 → 58.9 tokens)', 
         transform=ax2.transAxes, fontsize=10, fontweight='bold', color='#0d47a1',
         bbox=dict(boxstyle="round,pad=0.4", fc="#e3f2fd", ec="#1e88e5", lw=1))

plt.tight_layout()

# Save paths
plot_path_local = "08-06results/plots/factorial_scale_ablation.png"
plot_path_artifact = os.path.join(artifact_dir, "factorial_scale_ablation.png")

plt.savefig(plot_path_local, dpi=300, bbox_inches='tight')
plt.savefig(plot_path_artifact, dpi=300, bbox_inches='tight')
plt.close()

print(f"Plot successfully saved to:\n  - {plot_path_local}\n  - {plot_path_artifact}")

# Also generate a single-panel twinx version for alternative presentation
fig, ax1 = plt.subplots(figsize=(9, 5.5))
fig.patch.set_facecolor('white')

# Primary axis: Accuracy curves
l1 = ax1.plot(train_n, arm_00_static_ring, marker='o', linewidth=2, color=color_static, label='Arm 00: Static Ring', linestyle='--')
l2 = ax1.plot(train_n, arm_01_ppo_topo,    marker='s', linewidth=2, color=color_ppo,    label='Arm 01: PPO topology', linestyle='-.')
l3 = ax1.plot(train_n, arm_10_sparse_ring, marker='^', linewidth=2, color=color_sring,  label='Arm 10: Sparse Ring', linestyle=':')
l4 = ax1.plot(train_n, arm_11_sparsecomm,  marker='D', linewidth=2.8, color=color_scomm,  label='Arm 11: SparseComm', linestyle='-')

ax1.set_xlabel('Training Examples (N)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold', color='#333333')
ax1.set_ylim(50, 102)
ax1.set_xticks(train_n)
ax1.grid(True, linestyle='--', alpha=0.5)

# Secondary axis: Comm tokens
ax2 = ax1.twinx()
l5 = ax2.plot(train_n, comm_tokens_arm11, marker='h', linewidth=2.5, color=color_comm, label='Comm. Tokens (SparseComm)', linestyle='-')
ax2.set_ylabel('Comm. Tokens (Arm 11)', fontsize=12, fontweight='bold', color=color_comm)
ax2.tick_params(axis='y', labelcolor=color_comm)
ax2.set_ylim(0, 550)

# Combine legends
lines = l1 + l2 + l3 + l4 + l5
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center right', frameon=True, facecolor='white', framealpha=0.9, edgecolor='#cccccc')

plt.title('Training-Scale Accuracy & Communication Trade-off (GSM8K)', fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()

twin_local = "08-06results/plots/factorial_scale_twinx.png"
twin_artifact = os.path.join(artifact_dir, "factorial_scale_twinx.png")
plt.savefig(twin_local, dpi=300, bbox_inches='tight')
plt.savefig(twin_artifact, dpi=300, bbox_inches='tight')
plt.close()
print(f"Twin-X plot saved to:\n  - {twin_local}\n  - {twin_artifact}")
