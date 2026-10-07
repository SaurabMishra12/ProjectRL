import matplotlib.pyplot as plt
import numpy as np
import os

# Set style parameters for minimal, publication-quality aesthetic
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.labelsize': 11.5,
    'axes.titlesize': 13,
    'xtick.labelsize': 10.5,
    'ytick.labelsize': 10.5,
    'legend.fontsize': 10,
    'figure.titlesize': 13.5,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

# Exact data from the table
data = {
    'GSM8K': {
        'n': np.array([100, 200, 300, 400]),
        'mean': np.array([56.6, 96.2, 95.2, 94.6]),
        'sd': np.array([2.15, 1.60, 2.40, 2.65]),
        'max_change': np.array([4.0, 4.0, 5.0, 4.0]),
        'color': '#1D4ED8',      # Deep Blue
        'marker': 'o'
    },
    'AQuA-RAT': {
        'n': np.array([100, 200, 300, 400]),
        'mean': np.array([85.8, 72.2, 72.6, 62.0]),
        'sd': np.array([0.40, 2.23, 1.85, 2.45]),
        'max_change': np.array([1.0, 5.0, 4.0, 4.0]),
        'color': '#047857',      # Deep Emerald
        'marker': 's'
    },
    'HotpotQA': {
        'n': np.array([100]),
        'mean': np.array([40.8]),
        'sd': np.array([1.72]),
        'max_change': np.array([5.0]),
        'color': '#C2410C',      # Deep Terracotta / Rust Amber
        'marker': '^'
    }
}

fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=300)

# Background
ax.set_facecolor('#FFFFFF')
fig.patch.set_facecolor('#FFFFFF')

# Subtle dashed horizontal gridlines only
ax.grid(axis='y', linestyle='--', linewidth=0.55, color='#E2E8F0', alpha=0.85, zorder=0)

# Plot each dataset
for name, d in data.items():
    n_pts = d['n']
    means = d['mean']
    sds = d['sd']
    col = d['color']
    marker = d['marker']

    if len(n_pts) > 1:
        # Shaded ribbon representing ± 1 S.d.
        ax.fill_between(
            n_pts, means - sds, means + sds,
            color=col, alpha=0.10, zorder=2
        )
        # Trajectory line
        ax.plot(
            n_pts, means,
            color=col, linewidth=2.0, zorder=3,
            label=f"{name}"
        )

    # Error bars with clean caps and crisp markers
    ax.errorbar(
        n_pts, means, yerr=sds,
        fmt=marker, color=col, ecolor=col,
        elinewidth=1.2, capsize=3.5, capthick=1.2,
        markersize=6, markeredgewidth=1.2, markeredgecolor='white',
        zorder=4,
        label=f"{name}" if len(n_pts) == 1 else None
    )

# Subtle direct data callouts with soft halo background to prevent overlap
offsets = {
    'GSM8K': [
        (100, 56.6, 7, 0, 'bottom', 'center'),
        (200, 96.2, 7, 0, 'bottom', 'center'),
        (300, 95.2, 7, 0, 'bottom', 'center'),
        (400, 94.6, 7, -6, 'bottom', 'right'),
    ],
    'AQuA-RAT': [
        (100, 85.8, 7, 0, 'bottom', 'center'),
        (200, 72.2, -13, 0, 'top', 'center'),
        (300, 72.6, -13, 0, 'top', 'center'),
        (400, 62.0, -13, -6, 'top', 'right'),
    ],
    'HotpotQA': [
        (100, 40.8, 0, 10, 'center', 'left')
    ]
}

bbox_style = dict(boxstyle='round,pad=0.18', facecolor='white', edgecolor='none', alpha=0.75)

for name, points in offsets.items():
    col = data[name]['color']
    for x, y, yoff, xoff, va, ha in points:
        text = f"{y:.1f}%"
        if name == 'HotpotQA':
            text = f"HotpotQA: {y:.1f}% ± 1.7%"
        ax.annotate(
            text,
            xy=(x, y),
            xytext=(xoff, yoff),
            textcoords='offset points',
            ha=ha,
            va=va,
            fontsize=8.5,
            color=col,
            fontweight='bold' if name == 'HotpotQA' else 'normal',
            bbox=bbox_style,
            zorder=5
        )

# Direct curve end labels
ax.text(412, 94.6, 'GSM8K', color='#1D4ED8', va='center', fontweight='bold', fontsize=10)
ax.text(412, 62.0, 'AQuA-RAT', color='#047857', va='center', fontweight='bold', fontsize=10)

# Axes ranges & formatting
ax.set_xlim(75, 465)
ax.set_xticks([100, 200, 300, 400])
ax.set_xticklabels(['100', '200', '300', '400'])
ax.set_ylim(30, 105)
ax.set_yticks([30, 40, 50, 60, 70, 80, 90, 100])
ax.set_yticklabels(['30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])

# Minimalist labels and title
ax.set_xlabel('Pretraining Dataset Size ($n_{\\mathrm{train}}$)', labelpad=7)
ax.set_ylabel('Mean Accuracy (± 1 S.d.)', labelpad=7)
ax.set_title('Pretraining Accuracy & Convergence Across Sample Scales', pad=12, fontweight='bold', loc='left')

# Subtle note on Max Change
fig.text(
    0.13, 0.02,
    '* Shaded bands indicate ±1 s.d. across epochs. Max epoch-to-epoch change: GSM8K (4–5%), AQuA-RAT (1–5%), HotpotQA (5.0%).',
    fontsize=7.8, color='#64748B', style='italic'
)

# Despine: keep only bottom and left axis
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#CBD5E1')
ax.spines['bottom'].set_color('#CBD5E1')
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

# Tick parameters
ax.tick_params(axis='both', which='both', color='#94A3B8', length=3.5, width=0.8)

plt.subplots_adjust(bottom=0.16)

# Output paths
output_dir = 'results/plots'
os.makedirs(output_dir, exist_ok=True)
png_path = os.path.join(output_dir, 'scaling_stability_minimal.png')
pdf_path = os.path.join(output_dir, 'scaling_stability_minimal.pdf')

plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.savefig(pdf_path, bbox_inches='tight')
print(f"Refined figure saved to {png_path} and {pdf_path}")
