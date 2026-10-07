import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set publication-ready styling
# 'seaborn-v0_8-white' uses a white background without the heavy gray grids
plt.style.use('seaborn-v0_8-white')
sns.set_context("paper", font_scale=1.5)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

out_dir = r"c:\Users\msaur\OneDrive\Desktop\projectRL\results\paper_plots"
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------
# Plot 1: Pareto Frontier (Accuracy vs Token Cost)
# ---------------------------------------------------------
def plot_pareto_frontier():
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Data for GSM8K (N=292 / deepseek)
    methods = ['No-Comm MV', 'Static Routing', 'Multi-Agent Debate', 'IB-Swarm (Ours)']
    tokens = [530, 963, 12897, 2315]
    acc = [42.81, 45.00, 45.00, 46.23]
    
    colors = ['#7f8c8d', '#e67e22', '#e74c3c', '#27ae60']
    markers = ['o', 's', 'D', '*']
    sizes = [150, 150, 150, 500] # Increased star size (was 350)
    
    for i in range(len(methods)):
        ax.scatter(tokens[i], acc[i], color=colors[i], marker=markers[i], s=sizes[i], label=methods[i], zorder=5, edgecolor='black' if i==3 else 'none')
        
    ax.set_xscale('log')
    ax.set_xlabel('Tokens per Sample (Log Scale)')
    ax.set_ylabel('Accuracy (%)')
    # Removed title
    
    # Reduced legend size
    ax.legend(loc='lower right', frameon=True, shadow=False, fontsize=12, edgecolor='black')
    
    # Pareto curve line approximation
    ax.plot([530, 2315], [42.81, 46.23], 'k--', alpha=0.3, zorder=1)
    
    # Add annotation for token reduction
    ax.annotate(r'$\mathbf{6\times\ fewer\ tokens}$', 
                xy=(2315, 46.23), xytext=(2800, 45.5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6),
                fontsize=12, color='black')

    # Add light grid for readability but keep background white
    ax.grid(True, linestyle='--', alpha=0.5, color='gray')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pareto_frontier.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

# ---------------------------------------------------------
# Plot 2: Mode Distribution Collapse
# ---------------------------------------------------------
def plot_mode_distribution():
    fig, ax = plt.subplots(figsize=(8, 6))
    
    modes = ['BLOCK\n[0, 0.25)', 'SUMMARIZE\n[0.25, 0.50)', 'FILTER\n[0.50, 0.75)', 'FULL\n[0.75, 1.0]']
    gsm8k_freq = [0.0, 31.7, 68.3, 0.0]
    aqua_freq = [0.0, 50.1, 49.9, 0.0]
    
    x = np.arange(len(modes))
    width = 0.35
    
    ax.bar(x - width/2, gsm8k_freq, width, label='GSM8K', color='#3498db', edgecolor='black')
    ax.bar(x + width/2, aqua_freq, width, label='AQuA-RAT', color='#9b59b6', edgecolor='black')
    
    ax.set_ylabel('Frequency (%)')
    # Removed title
    ax.set_xticks(x)
    ax.set_xticklabels(modes)
    
    # Increase y limit for breathing room
    ax.set_ylim(0, 75)
    
    # Reduced legend size
    ax.legend(loc='upper right', frameon=True, fontsize=12, edgecolor='black')
    
    for i, v in enumerate(gsm8k_freq):
        if v > 0:
            ax.text(i - width/2, v + 1.5, f'{v}%', ha='center', fontweight='bold', fontsize=12)
    for i, v in enumerate(aqua_freq):
        if v > 0:
            ax.text(i + width/2, v + 1.5, f'{v}%', ha='center', fontweight='bold', fontsize=12)
            
    # Add light grid
    ax.grid(True, axis='y', linestyle='--', alpha=0.5, color='gray')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'mode_distribution.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

# ---------------------------------------------------------
# Plot 3: Divergence Paradox
# ---------------------------------------------------------
def plot_divergence_paradox():
    fig, ax = plt.subplots(figsize=(8, 6))
    
    stages = ['Round 0\n(Independent)', 'Round T\n(Post-Communication)']
    
    # Data
    indiv_acc = [41.74, 29.56]
    ensemble_acc = [42.49, 45.89]
    
    ax.plot(stages, indiv_acc, marker='o', markersize=10, linewidth=3, color='#e74c3c', label='Individual Agent Avg.')
    ax.plot(stages, ensemble_acc, marker='s', markersize=10, linewidth=3, color='#2ecc71', label='Ensemble (Majority Vote)')
    
    # Adjust offsets to avoid overlap with lines and y-axis. Use smaller fontsize.
    # Individual Agent numbers: place below
    ax.text(0, indiv_acc[0] - 1.5, f'{indiv_acc[0]:.1f}%', ha='center', va='top', color='#c0392b', fontweight='bold', fontsize=12)
    ax.text(1, indiv_acc[1] - 1.5, f'{indiv_acc[1]:.1f}%', ha='center', va='top', color='#c0392b', fontweight='bold', fontsize=12)
    
    # Ensemble numbers: place above
    ax.text(0, ensemble_acc[0] + 1.5, f'{ensemble_acc[0]:.1f}%', ha='center', va='bottom', color='#27ae60', fontweight='bold', fontsize=12)
    ax.text(1, ensemble_acc[1] + 1.5, f'{ensemble_acc[1]:.1f}%', ha='center', va='bottom', color='#27ae60', fontweight='bold', fontsize=12)
    
    # Add differential labels
    ax.text(0.5, (indiv_acc[0]+indiv_acc[1])/2 + 2.5, '−12.2%', ha='center', va='center', color='#c0392b', fontweight='bold', fontsize=12, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.5))
    ax.text(0.5, (ensemble_acc[0]+ensemble_acc[1])/2 + 2, '+3.4%', ha='center', va='center', color='#27ae60', fontweight='bold', fontsize=12, bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=0.5))

    ax.set_ylabel('Accuracy (%)')
    # Removed title
    
    # Add padding to limits to avoid y-axis overlap
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(25, 52)
    
    # Move legend to the empty bottom-left space to avoid any overlaps
    ax.legend(loc='lower left', frameon=True, fontsize=12, edgecolor='black')
    
    # Add light grid
    ax.grid(True, linestyle='--', alpha=0.5, color='gray')
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'divergence_paradox.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

# ---------------------------------------------------------
# Plot 4: Answer Change Penalty
# ---------------------------------------------------------
def plot_answer_change_penalty():
    fig, ax = plt.subplots(figsize=(7, 6))
    
    categories = ['Changed Answer\nDuring Comm.', 'Kept Original\nAnswer']
    acc = [31.0, 68.6]
    colors = ['#e74c3c', '#2ecc71']
    
    bars = ax.bar(categories, acc, color=colors, edgecolor='black', width=0.5)
    
    ax.set_ylabel('Accuracy (%)')
    # Removed title
    ax.set_ylim(0, 80)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                f'{height}%', ha='center', va='bottom', fontweight='bold', fontsize=14)
                
    # Add light grid
    ax.grid(True, axis='y', linestyle='--', alpha=0.5, color='gray')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'answer_change_penalty.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

if __name__ == "__main__":
    plot_pareto_frontier()
    plot_mode_distribution()
    plot_divergence_paradox()
    plot_answer_change_penalty()
    print("All updated plots generated successfully in: ", out_dir)
