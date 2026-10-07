"""
Full Analysis Script for projectRL 08-06results
================================================
Generates all plots, tables, and insights for:
1. Transfer learning results (300 samples from 100-trained model)
2. Transfer learning results (300 samples from 400-trained model)
3. Ablation study across 4 arms at 100-200-300-400 training sizes
4. OLS factorial analysis
5. Training accuracy per epoch (all sizes and datasets)
6. Topology benchmark results (at 400 training samples)
7. Symmetry/fluctuation analysis
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = '/home/saurab/Documents/projectRLMAIN/08-06results'
PLOTS = os.path.join(BASE, 'plots')
os.makedirs(PLOTS, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
C = {
    'Transfer':  '#1F77B4', # Standard academic blue
    'FullComm':  '#FF7F0E', # Academic orange
    'NoComm':    '#2CA02C', # Academic green
    'Arm_00_StaticRing': '#4A7FB5',
    'Arm_01_PPOOnly':    '#E85A5A',
    'Arm_10_SparseRing': '#5AC68D',
    'Arm_11_Full':       '#984EA3',
    'GSM8K':    '#1F77B4',
    'AQUA':     '#FF7F0E',
    'HOTPOTQA': '#2CA02C',
    'bg': '#FFFFFF',
    'fg': '#333333',
    'grid': '#E5E5E5',
}

ARM_LABELS = {
    'Arm_00_StaticRing': 'Static Ring\n(A=0,B=0)',
    'Arm_01_PPOOnly':    'PPO Only\n(A=0,B=1)',
    'Arm_10_SparseRing': 'Sparse Ring\n(A=1,B=0)',
    'Arm_11_Full':       'Full (Our)\n(A=1,B=1)',
}
ARM_KEYS = list(ARM_LABELS.keys())

STYLE = {
    'figure.facecolor': '#FFFFFF',
    'axes.facecolor':   '#FFFFFF',
    'axes.edgecolor':   '#333333',
    'axes.labelcolor':  '#333333',
    'text.color':       '#333333',
    'xtick.color':      '#333333',
    'ytick.color':      '#333333',
    'grid.color':       '#E5E5E5',
    'grid.alpha':       0.8,
    'legend.facecolor': '#FFFFFF',
    'legend.edgecolor': '#E5E5E5',
    'font.family':      'DejaVu Sans',
}
plt.rcParams.update(STYLE)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_ablation(size):
    fname = os.path.join(BASE, f'ablation_{size}.json')
    with open(fname) as f:
        return json.load(f)

def load_final_results():
    with open(os.path.join(BASE, 'final_results.json')) as f:
        return json.load(f)

def load_seed_ckpt(method, dataset, seed):
    fname = os.path.join(BASE, f'seed_ckpt_{method}_{dataset}_seed{seed}.json')
    with open(fname) as f:
        return json.load(f)

def count_jsonl(filepath):
    correct, total, f1s, toks = 0, 0, [], []
    try:
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    d = json.loads(line)
                    total += 1
                    if d.get('is_correct', False): correct += 1
                    f1s.append(d.get('f1_score', 0))
                    toks.append(d.get('total_tokens_transmitted', 0))
                except: pass
    except: pass
    return correct, total, f1s, toks

def load_training_epochs():
    """Load epoch-by-epoch training accuracy for all sizes and datasets."""
    datasets = {'GSM8K': 'gsm8k', 'AQUA': 'aqua', 'HOTPOTQA': 'hotpotqa'}
    sizes = [100, 200, 300, 400]
    result = {}
    for ds_name, ds_key in datasets.items():
        result[ds_name] = {}
        for sz in sizes:
            sz_label = '' if sz == 100 else f'_{sz}'
            accs = []
            for ep in range(1, 6):
                fname = os.path.join(BASE, 'continuous',
                    f'Train_Pretrain_{ds_name}{sz_label}_Epoch{ep}_{ds_key}_stream.jsonl')
                c, t, _, _ = count_jsonl(fname)
                accs.append(c / t if t else 0)
            result[ds_name][sz] = accs
    return result

def load_topology():
    with open(os.path.join(BASE, 'topology_benchmark_results.json')) as f:
        topo = json.load(f)
    rows = []
    for item in topo:
        samples = item.get('sample_results', [])
        c = sum(1 for s in samples if s.get('is_correct', False))
        n = len(samples)
        rows.append({
            'method': item['method_name'],
            'dataset': item['dataset_name'],
            'accuracy': c / n if n else 0,
            'n': n,
        })
    return rows

def load_transfer_stream_split():
    """
    Load stream files and split into first 300 / last 300.

    - Transfer: uses per-seed files (Transfer_{DS}_seed{N}_{key}_stream.jsonl);
      each seed file may itself contain 600 records (first 300 = 100-trained,
      last 300 = 400-trained).  For seeds where the file has only 300 records
      the whole file is 'first_300' (100-trained evaluation only).
    - MAD / NoComm: uses the single pooled file that concatenates all 3 seeds
      back-to-back.  Each seed block is 300 samples, so:
        seed0 block  = rows   0-299  (100-trained model)
        seed1 block  = rows 300-599  (100-trained model)
        seed2 block  = rows 600-899  (100-trained model)
      … but the user confirmed the pooled file is 900 total = 3 seeds × 300
      samples, all tested on the same checkpoint.  The 'first 300' and 'last 300'
      in this context means: first seed block vs last seed block.
    """
    datasets = {'GSM8K': 'gsm8k', 'AQUA': 'aqua', 'HOTPOTQA': 'hotpotqa'}
    result = {}

    for ds_name, ds_key in datasets.items():
        result[ds_name] = {}

        # ── Transfer: per-seed files ──────────────────────────────────────────
        seeds_first_tr, seeds_last_tr = [], []
        for seed in range(3):
            fname = os.path.join(BASE, 'continuous',
                f'Transfer_{ds_name}_seed{seed}_{ds_key}_stream.jsonl')
            if not os.path.exists(fname):
                continue
            data = []
            with open(fname) as f:
                for line in f:
                    l = line.strip()
                    if l:
                        try: data.append(json.loads(l))
                        except: pass
            first = data[:300]
            last  = data[300:]
            if first:
                seeds_first_tr.append(
                    sum(1 for d in first if d.get('is_correct', False)) / len(first))
            if last:
                seeds_last_tr.append(
                    sum(1 for d in last if d.get('is_correct', False)) / len(last))
        result[ds_name]['Transfer'] = {
            'first_300': seeds_first_tr,
            'last_300':  seeds_last_tr,
        }

        # ── MAD & NoComm: pooled files (900 samples = 3×300) ─────────────────
        for method_key, method_file in [('FullComm', 'Multi-Agent Debate'),
                                        ('NoComm',   'No Communication')]:
            fname = os.path.join(BASE, 'continuous',
                f'{method_file}_{ds_key}_stream.jsonl')
            if not os.path.exists(fname):
                result[ds_name][method_key] = {'first_300': [], 'last_300': []}
                continue
            data = []
            with open(fname) as f:
                for line in f:
                    l = line.strip()
                    if l:
                        try: data.append(json.loads(l))
                        except: pass
            # Split into 300-sample blocks (one per seed)
            blocks = [data[i*300:(i+1)*300] for i in range(len(data)//300)]
            remainder = data[len(blocks)*300:]
            if remainder:
                blocks.append(remainder)
            # Per-block accuracies
            per_seed_accs = [
                sum(1 for d in blk if d.get('is_correct', False)) / len(blk)
                for blk in blocks if blk
            ]
            result[ds_name][method_key] = {
                'first_300': per_seed_accs,   # all seeds pooled for comparison
                'last_300':  [],               # pooled file has one checkpoint
            }

    return result

# ═══════════════════════════════════════════════════════════════════════════════
# PLOT FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def save(fig, name):
    path = os.path.join(PLOTS, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f'  Saved: {path}')
    return path


# ── PLOT 1: Transfer Results - 3 Datasets x 3 Methods (from 100-trained model) ─
def plot_transfer_100(stream_split):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle('Transfer Learning Results\n(First 300 Samples – Model Trained on 100 Samples)',
                 fontsize=14, fontweight='bold', y=1.02)
    datasets = ['GSM8K', 'AQUA', 'HOTPOTQA']
    methods = ['Transfer', 'FullComm', 'NoComm']
    method_labels = ['Transfer (Ours)', 'MAD (Full Comm)', 'No Comm']
    colors = [C['Transfer'], C['FullComm'], C['NoComm']]

    for ax, ds in zip(axes, datasets):
        accs = []
        errs = []
        for m in methods:
            vals = stream_split[ds][m]['first_300']
            accs.append(np.mean(vals) if vals else 0)
            errs.append(np.std(vals) if len(vals) > 1 else 0)
        bars = ax.bar(method_labels, [a * 100 for a in accs], 
                      color=colors, alpha=0.85, edgecolor='#333333', linewidth=0.8,
                      yerr=[e * 100 for e in errs], capsize=5, 
                      error_kw={'color': '#333333', 'linewidth': 1.5})
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{acc*100:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_title(f'{ds}', fontsize=12, fontweight='bold', color=C[ds])
        ax.set_ylabel('Accuracy (%)', fontsize=10)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.4)
        ax.tick_params(axis='x', rotation=15)
    fig.tight_layout()
    return save(fig, 'plot1_transfer_100train.png')


# ── PLOT 2: Transfer Results - 400-trained model (final_results) ──────────────
def plot_transfer_400(final):
    datasets = ['GSM8K', 'AQUA', 'HOTPOTQA']
    methods = ['Transfer', 'FullComm', 'NoComm']
    method_labels = ['Transfer (Ours)', 'MAD (Full Comm)', 'No Comm']
    colors = [C['Transfer'], C['FullComm'], C['NoComm']]

    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle('Transfer Learning Results\n(Last 300 Samples – Model Trained on 400 Samples)',
                 fontsize=14, fontweight='bold', y=1.02)

    for ax, ds in zip(axes, datasets):
        accs, errs = [], []
        for m in methods:
            key = f'{m}_{ds}'
            if key in final:
                accs.append(final[key]['accuracy_mean'] * 100)
                errs.append(final[key]['accuracy_std'] * 100)
            else:
                accs.append(0); errs.append(0)
        bars = ax.bar(method_labels, accs, color=colors, alpha=0.85, 
                      edgecolor='#333333', linewidth=0.8,
                      yerr=errs, capsize=5, error_kw={'color':'#333333','linewidth':1.5})
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.set_title(f'{ds}', fontsize=12, fontweight='bold', color=C[ds])
        ax.set_ylabel('Accuracy (%)', fontsize=10)
        ax.set_ylim(0, 110 if ds != 'HOTPOTQA' else 60)
        ax.grid(axis='y', alpha=0.4)
        ax.tick_params(axis='x', rotation=15)
    fig.tight_layout()
    return save(fig, 'plot2_transfer_400train.png')


# ── PLOT 3: Before vs After (100 train → 400 train) for Transfer ──────────────
def plot_transfer_before_after(stream_split, final):
    datasets = ['GSM8K', 'AQUA', 'HOTPOTQA']
    methods = ['Transfer', 'FullComm', 'NoComm']
    method_labels = ['Transfer (Ours)', 'MAD', 'No Comm']

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle('Transfer Training Effect: 100-Sample vs 400-Sample Model\n(Per Dataset, Per Method)',
                 fontsize=14, fontweight='bold', y=1.01)

    x = np.arange(len(methods))
    width = 0.35

    for ax, ds in zip(axes, datasets):
        # First 300 (100-trained)
        first = [np.mean(stream_split[ds][m]['first_300']) * 100 
                 if stream_split[ds][m]['first_300'] else 0 for m in methods]
        first_err = [np.std(stream_split[ds][m]['first_300']) * 100 
                     if len(stream_split[ds][m]['first_300']) > 1 else 0 for m in methods]
        # Last 300 (400-trained)
        last = [final.get(f'{m}_{ds}', {}).get('accuracy_mean', 0) * 100 for m in methods]
        last_err = [final.get(f'{m}_{ds}', {}).get('accuracy_std', 0) * 100 for m in methods]

        b1 = ax.bar(x - width/2, first, width, label='100-trained model',
                    color='#4A7FB5', alpha=0.8, edgecolor='#333333', lw=0.8,
                    yerr=first_err, capsize=4, error_kw={'color':'#333333','lw':1.5})
        b2 = ax.bar(x + width/2, last, width, label='400-trained model',
                    color='#E8835A', alpha=0.8, edgecolor='#333333', lw=0.8,
                    yerr=last_err, capsize=4, error_kw={'color':'#333333','lw':1.5})

        # Annotations with delta
        for i, (f, l) in enumerate(zip(first, last)):
            delta = l - f
            color = '#006600' if delta > 0 else '#CC0000'
            ax.annotate(f'Δ{delta:+.1f}%', 
                        xy=(x[i], max(f, l) + 1), ha='center', fontsize=8.5,
                        color=color, fontweight='bold')

        ax.set_title(ds, fontsize=12, fontweight='bold', color=C[ds])
        ax.set_xticks(x); ax.set_xticklabels(method_labels, fontsize=9)
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(0, 115 if ds != 'HOTPOTQA' else 60)
        ax.grid(axis='y', alpha=0.4)
        ax.legend(fontsize=8)

    fig.tight_layout()
    return save(fig, 'plot3_transfer_before_after.png')


# ── PLOT 4: Ablation - All 4 Arms at All 4 Sizes ─────────────────────────────
def plot_ablation_all_sizes():
    sizes = [100, 200, 300, 400]
    arm_keys = ARM_KEYS
    arm_labels = [ARM_LABELS[k] for k in arm_keys]
    arm_colors = [C[k] for k in arm_keys]

    # Collect data
    data = {}
    for sz in sizes:
        ab = load_ablation(sz)
        data[sz] = {}
        for ak in arm_keys:
            samples = ab[ak]
            c = sum(1 for s in samples if s.get('is_correct', False))
            n = len(samples)
            data[sz][ak] = {'acc': c/n if n else 0, 'n': n}

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle('Ablation Study: Accuracy Across All 4 Arms × Training Sizes',
                 fontsize=14, fontweight='bold')

    x = np.arange(len(sizes))
    width = 0.2
    offsets = np.linspace(-1.5, 1.5, len(arm_keys)) * width

    for i, (ak, label, color) in enumerate(zip(arm_keys, arm_labels, arm_colors)):
        accs = [data[sz][ak]['acc'] * 100 for sz in sizes]
        bars = ax.bar(x + offsets[i], accs, width*0.95, label=label.replace('\n', ' '),
                      color=color, alpha=0.85, edgecolor='#333333', lw=0.7)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    f'{acc:.1f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
        # Line connecting bars
        ax.plot(x + offsets[i], accs, 'o--', color=color, alpha=0.5, ms=4, lw=1.2)

    ax.set_xticks(x); ax.set_xticklabels([f'{sz} samples' for sz in sizes])
    ax.set_ylabel('Accuracy (%)'); ax.set_xlabel('Training Size')
    ax.set_ylim(40, 105)
    ax.grid(axis='y', alpha=0.4)
    ax.legend(loc='lower right', fontsize=9)
    fig.tight_layout()
    return save(fig, 'plot4_ablation_all_sizes.png')


# ── PLOT 5: Ablation Heatmap ──────────────────────────────────────────────────
def plot_ablation_heatmap():
    sizes = [100, 200, 300, 400]
    arm_keys = ARM_KEYS
    matrix = np.zeros((len(arm_keys), len(sizes)))
    for j, sz in enumerate(sizes):
        ab = load_ablation(sz)
        for i, ak in enumerate(arm_keys):
            samples = ab[ak]
            c = sum(1 for s in samples if s.get('is_correct', False))
            matrix[i, j] = c / len(samples) * 100 if samples else 0

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle('Ablation Accuracy Heatmap (%) – 4 Arms × 4 Training Sizes', fontsize=13)

    cmap = 'Blues'
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=50, vmax=100)
    fig.colorbar(im, ax=ax, label='Accuracy (%)')

    ax.set_xticks(range(len(sizes))); ax.set_xticklabels([f'{s}' for s in sizes])
    ax.set_yticks(range(len(arm_keys))); ax.set_yticklabels([ARM_LABELS[k].replace('\n',' ') for k in arm_keys])
    ax.set_xlabel('Training Size'); ax.set_ylabel('Ablation Arm')

    for i in range(len(arm_keys)):
        for j in range(len(sizes)):
            ax.text(j, i, f'{matrix[i, j]:.1f}%', ha='center', va='center',
                    fontsize=11, fontweight='bold',
                    color='white' if matrix[i, j] > 75 else 'black')

    fig.tight_layout()
    return save(fig, 'plot5_ablation_heatmap.png')


# ── PLOT 6: OLS Betas across Training Sizes ───────────────────────────────────
def plot_ols_betas():
    sizes = [100, 200, 300, 400]
    betas_A, betas_B, betas_AB = [], [], []
    pvals_A, pvals_B, pvals_AB = [], [], []

    for sz in sizes:
        ab = load_ablation(sz)
        ols = ab['factorial_ols_betas']
        betas_A.append(ols['beta_A_sparsity'])
        betas_B.append(ols['beta_B_ppo'])
        betas_AB.append(ols['beta_AB_interaction'])
        pvals_A.append(ols['p_values'][1])
        pvals_B.append(ols['p_values'][2])
        pvals_AB.append(ols['p_values'][3])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    fig.suptitle('OLS Factorial Analysis: Effect Sizes & P-values Across Training Sizes',
                 fontsize=14, fontweight='bold')

    # Beta plot
    x = np.arange(len(sizes))
    ax1.plot(x, betas_A, 'o-', color='#1F77B4', label='β_A (Sparsity)', lw=2, ms=7)
    ax1.plot(x, betas_B, 's-', color='#FF7F0E', label='β_B (PPO)', lw=2, ms=7)
    ax1.plot(x, betas_AB, '^-', color='#984EA3', label='β_AB (Interaction)', lw=2, ms=7)
    ax1.axhline(0, color='#333333', alpha=0.3, lw=1, ls='--')
    ax1.fill_between(x, betas_A, alpha=0.15, color='#1F77B4')
    ax1.fill_between(x, betas_B, alpha=0.15, color='#FF7F0E')
    for i, (ba, bb, bab) in enumerate(zip(betas_A, betas_B, betas_AB)):
        ax1.annotate(f'{ba:.3f}', (i, ba), textcoords='offset points', xytext=(0, 8),
                    ha='center', fontsize=8, color='#1F77B4')
        ax1.annotate(f'{bb:.3f}', (i, bb), textcoords='offset points', xytext=(0, -12),
                    ha='center', fontsize=8, color='#FF7F0E')
    ax1.set_ylabel('OLS Beta Coefficient')
    ax1.legend(); ax1.grid(alpha=0.4)

    # P-value plot (log scale)
    ax2.semilogy(x, pvals_A, 'o-', color='#1F77B4', label='p(β_A)', lw=2, ms=7)
    ax2.semilogy(x, pvals_B, 's-', color='#FF7F0E', label='p(β_B)', lw=2, ms=7)
    ax2.semilogy(x, pvals_AB, '^-', color='#984EA3', label='p(β_AB)', lw=2, ms=7)
    ax2.axhline(0.05, color='#D32F2F', lw=1.5, ls='--', label='α=0.05 threshold')
    ax2.axhline(0.1, color='#FF9800', lw=1, ls=':', label='α=0.10 threshold')
    ax2.fill_between(x, 0, 0.05, alpha=0.08, color='#D32F2F')
    for i, (pa, pb, pab) in enumerate(zip(pvals_A, pvals_B, pvals_AB)):
        ax2.annotate(f'{pa:.3f}', (i, pa), textcoords='offset points', xytext=(5, 0),
                    ha='left', fontsize=7.5, color='#1F77B4')
    ax2.set_ylabel('P-value (log scale)')
    ax2.set_xlabel('Training Size')
    ax2.set_xticks(x); ax2.set_xticklabels([str(s) for s in sizes])
    ax2.legend(fontsize=8); ax2.grid(alpha=0.4)

    fig.tight_layout()
    return save(fig, 'plot6_ols_betas.png')


# ── PLOT 7: Training Accuracy per Epoch ──────────────────────────────────────
def plot_training_epochs(epoch_data):
    datasets = ['GSM8K', 'AQUA', 'HOTPOTQA']
    sizes = [100, 200, 300, 400]
    size_colors = ['#BF7BBF', '#4C9BE8', '#6DC6A0', '#F97B6C']
    size_styles = ['o--', 's--', '^--', 'D--']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Training Accuracy per Epoch – All Datasets & Training Sizes', fontsize=14, fontweight='bold')

    for ax, ds in zip(axes, datasets):
        for sz, col, sty in zip(sizes, size_colors, size_styles):
            accs = [a * 100 for a in epoch_data[ds][sz]]
            epochs = range(1, len(accs) + 1)
            ax.plot(epochs, accs, sty, color=col, label=f'{sz} samples', lw=2, ms=7, alpha=0.9)
            # Highlight fluctuations
            for ep in range(1, len(accs)):
                delta = accs[ep] - accs[ep-1]
                if abs(delta) >= 2:
                    ax.annotate(f'{delta:+.1f}', (ep+1, accs[ep]),
                                textcoords='offset points', xytext=(3, 4),
                                fontsize=7, color=col, alpha=0.8)
        ax.set_title(ds, fontsize=12, fontweight='bold', color=C[ds])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_xticks(range(1, 6))
        ax.grid(alpha=0.4)
        ax.legend(fontsize=8, loc='best')
        ymin = min(min(epoch_data[ds][sz]) * 100 for sz in sizes) - 5
        ax.set_ylim(max(0, ymin), 105)

    fig.tight_layout()
    return save(fig, 'plot7_training_epochs.png')


# ── PLOT 8: Training Accuracy Grid (all sizes side by side per epoch) ─────────
def plot_training_epochs_grid(epoch_data):
    datasets = ['GSM8K', 'AQUA', 'HOTPOTQA']
    sizes = [100, 200, 300, 400]
    size_colors = ['#BF7BBF', '#4C9BE8', '#6DC6A0', '#F97B6C']

    fig, axes = plt.subplots(len(sizes), len(datasets), figsize=(18, 14), sharey=False)
    fig.suptitle('Training Accuracy per Epoch – Individual Size × Dataset Grid', fontsize=14, fontweight='bold')

    for col_i, ds in enumerate(datasets):
        for row_i, sz in enumerate(sizes):
            ax = axes[row_i][col_i]
            accs = [a * 100 for a in epoch_data[ds][sz]]
            epochs = list(range(1, len(accs) + 1))
            ax.plot(epochs, accs, 'o-', color=size_colors[row_i], lw=2, ms=6)
            ax.fill_between(epochs, accs, min(accs), alpha=0.2, color=size_colors[row_i])
            for ep, acc in zip(epochs, accs):
                ax.text(ep, acc + 0.3, f'{acc:.0f}', ha='center', va='bottom', fontsize=8)
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            ax.axhline(mean_acc, color='#666666', ls='--', alpha=0.5, lw=1)
            ax.set_title(f'{ds} | {sz} samples\n(μ={mean_acc:.1f}%, σ={std_acc:.1f}%)', fontsize=9)
            ax.set_xticks(epochs); ax.set_xlabel('Epoch', fontsize=8)
            ax.set_ylabel('Acc (%)', fontsize=8)
            ax.grid(alpha=0.3)
            ymin = min(accs) - 3
            ax.set_ylim(max(0, ymin), min(105, max(accs) + 5))

    fig.tight_layout()
    return save(fig, 'plot8_training_epoch_grid.png')


# ── PLOT 9: Topology Benchmark ───────────────────────────────────────────────
def plot_topology(topo_rows):
    methods = [r['method'] for r in topo_rows]
    accs = [r['accuracy'] * 100 for r in topo_rows]
    colors_topo = ['#F97B6C', '#4C9BE8', '#6DC6A0', '#FFD700', '#BF7BBF', '#FF8C94', '#87CEFA']

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle('Topology Benchmark (400-Train Model) – GSM8K\nDifferent Communication Topologies',
                 fontsize=14, fontweight='bold')

    sorted_pairs = sorted(zip(accs, methods, colors_topo[:len(methods)]), reverse=True)
    accs_s, methods_s, cols_s = zip(*sorted_pairs)

    bars = ax.barh(methods_s, accs_s, color=cols_s, alpha=0.85, edgecolor='#333333', lw=0.8)
    for bar, acc in zip(bars, accs_s):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{acc:.2f}%', va='center', fontsize=11, fontweight='bold')

    ax.axvline(accs_s[-1], color='#333333', ls='--', alpha=0.4)  # baseline
    ax.set_xlabel('Accuracy (%)')
    ax.set_xlim(85, 102)
    ax.grid(axis='x', alpha=0.4)
    fig.tight_layout()
    return save(fig, 'plot9_topology_benchmark.png')


# ── PLOT 10: Token Efficiency Analysis ────────────────────────────────────────
def plot_token_efficiency(final):
    datasets = ['GSM8K', 'AQUA', 'HOTPOTQA']
    methods = ['Transfer', 'FullComm', 'NoComm']
    method_labels = ['Transfer (Ours)', 'MAD (Full Comm)', 'No Comm']

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle('Accuracy vs Token Cost Trade-off (400-Train Model)', fontsize=14, fontweight='bold')

    for ds in datasets:
        for m, label in zip(methods, method_labels):
            key = f'{m}_{ds}'
            if key in final:
                acc = final[key]['accuracy_mean'] * 100
                tok = final[key]['token_cost_mean']
                ax.scatter(tok, acc, s=180, 
                           color=C[m], marker={'GSM8K': 'o', 'AQUA': 's', 'HOTPOTQA': '^'}[ds],
                           zorder=5, edgecolors='#333333', lw=1.0)
                ax.annotate(f'{ds[:3]}\n{label.split()[0]}', (tok, acc),
                            textcoords='offset points', xytext=(5, 5),
                            fontsize=7.5, color=C[m])

    # Legend
    patches = [mpatches.Patch(color=C[m], label=l) for m, l in zip(methods, method_labels)]
    markers = [plt.scatter([], [], marker=mk, color='grey', s=60, label=ds) 
               for ds, mk in [('GSM8K', 'o'), ('AQUA', 's'), ('HOTPOTQA', '^')]]
    ax.legend(handles=patches + markers, fontsize=9, loc='upper left')
    ax.set_xlabel('Average Token Cost per Sample')
    ax.set_ylabel('Accuracy (%)')
    ax.grid(alpha=0.4)
    fig.tight_layout()
    return save(fig, 'plot10_token_efficiency.png')


# ── PLOT 11: Fluctuation Analysis ────────────────────────────────────────────
def plot_fluctuation_analysis(epoch_data):
    datasets = ['GSM8K', 'AQUA', 'HOTPOTQA']
    sizes = [100, 200, 300, 400]

    # Compute epoch-to-epoch deltas
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Epoch-to-Epoch Accuracy Fluctuation (ΔAcc) – Symmetry Analysis', 
                 fontsize=14, fontweight='bold')
    size_colors = ['#BF7BBF', '#4C9BE8', '#6DC6A0', '#F97B6C']

    for ax, ds in zip(axes, datasets):
        for sz, col in zip(sizes, size_colors):
            accs = [a * 100 for a in epoch_data[ds][sz]]
            deltas = [accs[i] - accs[i-1] for i in range(1, len(accs))]
            epochs = range(2, len(accs) + 1)
            ax.plot(epochs, deltas, 'o-', color=col, label=f'{sz}', lw=1.8, ms=6, alpha=0.9)
            for ep, d in zip(epochs, deltas):
                if abs(d) > 0.5:
                    ax.annotate(f'{d:+.1f}', (ep, d), textcoords='offset points',
                                xytext=(0, 5 if d > 0 else -12), fontsize=7.5, color=col)
        ax.axhline(0, color='#333333', lw=1, alpha=0.4)
        ax.fill_between(range(2, 6), 0, ax.get_ylim()[0] if ax.get_ylim()[0] < 0 else 0, 
                        alpha=0.05, color='red')
        ax.fill_between(range(2, 6), 0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0,
                        alpha=0.05, color='green')
        ax.set_title(ds, color=C[ds], fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('ΔAccuracy (%)')
        ax.set_xticks(range(2, 6)); ax.grid(alpha=0.4)
        ax.legend(title='Train size', fontsize=8)

    fig.tight_layout()
    return save(fig, 'plot11_fluctuation_analysis.png')


# ── PLOT 12: Ablation Arms - Token Cost vs Accuracy ─────────────────────────
def plot_ablation_token_accuracy():
    sizes = [100, 200, 300, 400]
    arm_keys = ARM_KEYS
    markers = ['o', 's', '^', 'D']

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle('Ablation Arms: Accuracy vs Avg Token Transmitted\n(Bubble = Training Size)',
                 fontsize=14, fontweight='bold')

    for ak, mk in zip(arm_keys, markers):
        label = ARM_LABELS[ak].replace('\n', ' ')
        col = C[ak]
        for sz in sizes:
            ab = load_ablation(sz)
            samples = ab[ak]
            c = sum(1 for s in samples if s.get('is_correct', False))
            n = len(samples)
            acc = c / n * 100 if n else 0
            toks = [s.get('total_tokens_transmitted', 0) for s in samples]
            avg_tok = np.mean(toks) if toks else 0
            ax.scatter(avg_tok, acc, s=sz / 2, marker=mk, color=col, alpha=0.8,
                       edgecolors='#333333', lw=0.8, zorder=5)
            ax.annotate(f'{sz}', (avg_tok, acc), textcoords='offset points',
                        xytext=(4, 4), fontsize=7, color=col)
        # Connect sizes with line
        toks_all = []
        accs_all = []
        for sz in sizes:
            ab = load_ablation(sz)
            samples = ab[ak]
            c = sum(1 for s in samples if s.get('is_correct', False))
            n = len(samples)
            accs_all.append(c / n * 100 if n else 0)
            toks_all.append(np.mean([s.get('total_tokens_transmitted', 0) for s in samples]))
        ax.plot(toks_all, accs_all, mk + '--', color=col, alpha=0.5, lw=1.2, label=label)

    ax.set_xlabel('Average Tokens Transmitted per Sample')
    ax.set_ylabel('Accuracy (%)')
    ax.grid(alpha=0.4)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return save(fig, 'plot12_token_vs_accuracy.png')


# ── PLOT 13: Radar / Spider Chart - All Methods at 400 ───────────────────────
def plot_radar_methods(final):
    from matplotlib.patches import FancyArrowPatch
    datasets = ['GSM8K', 'AQUA', 'HOTPOTQA']
    methods_map = {
        'Transfer (Ours)': 'Transfer',
        'MAD (Full Comm)': 'FullComm',
        'No Comm': 'NoComm',
    }
    cols_radar = [C['Transfer'], C['FullComm'], C['NoComm']]

    angles = np.linspace(0, 2 * np.pi, len(datasets), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.suptitle('Method Comparison Radar\n(400-Train Model – Normalized Accuracy)', fontsize=13, fontweight='bold')

    for (m_label, m_key), col in zip(methods_map.items(), cols_radar):
        values = []
        for ds in datasets:
            key = f'{m_key}_{ds}'
            acc = final.get(key, {}).get('accuracy_mean', 0)
            # Normalize HOTPOTQA to [0,1] scale relative to best
            values.append(acc * 100)
        values += values[:1]
        ax.plot(angles, values, 'o-', color=col, lw=2, label=m_label, ms=6)
        ax.fill(angles, values, color=col, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(datasets, size=12, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(color='#CCCCCC', alpha=0.5)
    ax.set_facecolor('#FFFFFF')
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)
    fig.tight_layout()
    return save(fig, 'plot13_radar_methods.png')


# ── PLOT 14: Improvement Symmetry (Training size effect on each arm) ──────────
def plot_improvement_symmetry():
    sizes = [100, 200, 300, 400]
    arm_keys = ARM_KEYS

    data = {}
    for sz in sizes:
        ab = load_ablation(sz)
        for ak in arm_keys:
            samples = ab[ak]
            c = sum(1 for s in samples if s.get('is_correct', False))
            n = len(samples)
            if ak not in data: data[ak] = []
            data[ak].append(c / n * 100 if n else 0)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Per-Arm: Accuracy vs Training Size\n(Improvement Symmetry Analysis)', 
                 fontsize=14, fontweight='bold')
    axes = axes.flatten()

    for ax, ak in zip(axes, arm_keys):
        accs = data[ak]
        col = C[ak]
        ax.plot(sizes, accs, 'o-', color=col, lw=2.5, ms=9)
        ax.fill_between(sizes, accs, accs[0], alpha=0.2, color=col)
        for sz, acc in zip(sizes, accs):
            ax.annotate(f'{acc:.2f}%', (sz, acc), textcoords='offset points',
                        xytext=(0, 8), ha='center', fontsize=10, fontweight='bold')
        # Add improvement arrows
        for i in range(1, len(sizes)):
            delta = accs[i] - accs[i-1]
            mid_x = (sizes[i-1] + sizes[i]) / 2
            mid_y = (accs[i-1] + accs[i]) / 2
            arr_color = '#008000' if delta >= 0 else '#D00000'
            ax.annotate(f'Δ{delta:+.1f}%', (mid_x, mid_y),
                        xytext=(10, 5), textcoords='offset points',
                        fontsize=8.5, color=arr_color, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color=arr_color, lw=1.2))
        ax.set_title(ARM_LABELS[ak].replace('\n', ' '), fontsize=11, color=col, fontweight='bold')
        ax.set_xlabel('Training Samples')
        ax.set_ylabel('Accuracy (%)')
        ax.set_xticks(sizes)
        ax.set_ylim(min(accs) - 5, max(accs) + 10)
        ax.grid(alpha=0.4)

    fig.tight_layout()
    return save(fig, 'plot14_improvement_symmetry.png')


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE PRINTER
# ═══════════════════════════════════════════════════════════════════════════════

def print_all_tables(final, epoch_data, topo_rows, stream_split):
    lines = []
    sep = '=' * 80

    # ── Table 1: Transfer Results (100-trained model) ─────────────────────────
    lines.append(sep)
    lines.append('TABLE 1: TRANSFER RESULTS – 100-TRAINED MODEL (First 300 Samples per Seed)')
    lines.append(sep)
    lines.append(f'{"Dataset":<12} {"Method":<18} {"Seed 0":>8} {"Seed 1":>8} {"Seed 2":>8} {"Mean":>8} {"Std":>8}')
    lines.append('-' * 80)
    datasets = ['GSM8K', 'AQUA', 'HOTPOTQA']
    methods = ['Transfer', 'FullComm', 'NoComm']
    method_labels = ['Transfer (Ours)', 'MAD (FullComm)', 'No Comm']
    for ds in datasets:
        for m, ml in zip(methods, method_labels):
            vals = stream_split[ds][m]['first_300']
            padded = vals + [float('nan')] * (3 - len(vals))
            mean = np.nanmean(vals) if vals else 0
            std = np.nanstd(vals) if len(vals) > 1 else 0
            row = f'{ds:<12} {ml:<18}'
            for v in padded[:3]:
                row += f' {v*100:>7.2f}%' if not np.isnan(v) else '     N/A'
            row += f' {mean*100:>7.2f}% {std*100:>7.2f}%'
            lines.append(row)
    lines.append('')

    # ── Table 2: Transfer Results (400-trained model) ─────────────────────────
    lines.append(sep)
    lines.append('TABLE 2: TRANSFER RESULTS – 400-TRAINED MODEL (Last 300 Samples per Seed)')
    lines.append(sep)
    lines.append(f'{"Dataset":<12} {"Method":<18} {"Seed 0":>8} {"Seed 1":>8} {"Seed 2":>8} {"Mean":>8} {"Std":>8} {"Token":>8}')
    lines.append('-' * 80)
    for ds in datasets:
        for m, ml in zip(methods, method_labels):
            key = f'{m}_{ds}'
            if key in final:
                d = final[key]
                accs = d['all_accuracies']
                toks = d['all_token_costs']
                row = f'{ds:<12} {ml:<18}'
                for a in accs:
                    row += f' {a*100:>7.2f}%'
                row += f' {d["accuracy_mean"]*100:>7.2f}% {d["accuracy_std"]*100:>7.2f}% {d["token_cost_mean"]:>7.1f}'
                lines.append(row)
    lines.append('')

    # ── Table 3: Ablation Results ─────────────────────────────────────────────
    lines.append(sep)
    lines.append('TABLE 3: ABLATION STUDY – ACCURACY PER ARM × TRAINING SIZE')
    lines.append(sep)
    lines.append(f'{"Arm":<22} {"100":>8} {"200":>8} {"300":>8} {"400":>8} {"Improvement":>12}')
    lines.append('-' * 80)
    arm_keys = ARM_KEYS
    for ak in arm_keys:
        label = ARM_LABELS[ak].replace('\n', ' ')
        accs = []
        for sz in [100, 200, 300, 400]:
            ab = load_ablation(sz)
            samples = ab[ak]
            c = sum(1 for s in samples if s.get('is_correct', False))
            n = len(samples)
            accs.append(c / n * 100 if n else 0)
        improvement = accs[-1] - accs[0]
        row = f'{label:<22}'
        for a in accs: row += f' {a:>7.2f}%'
        row += f' {improvement:>+11.2f}%'
        lines.append(row)
    lines.append('')

    # ── Table 4: OLS Analysis ─────────────────────────────────────────────────
    lines.append(sep)
    lines.append('TABLE 4: OLS FACTORIAL ANALYSIS – β COEFFICIENTS & P-VALUES')
    lines.append(sep)
    lines.append(f'{"Train Size":<12} {"β_Intercept":>12} {"β_A(Sparsity)":>14} {"β_B(PPO)":>10} {"β_AB(Interaction)":>18} {"N":>6}')
    lines.append(f'{"":12} {"p_value":>12} {"p_A":>14} {"p_B":>10} {"p_AB":>18}')
    lines.append('-' * 80)
    for sz in [100, 200, 300, 400]:
        ab = load_ablation(sz)
        ols = ab['factorial_ols_betas']
        pv = ols['p_values']
        lines.append(f'{sz:<12} {ols["intercept"]:>12.4f} {ols["beta_A_sparsity"]:>14.4f} {ols["beta_B_ppo"]:>10.4f} {ols["beta_AB_interaction"]:>18.4f} {ols["N"]:>6}')
        sig = lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else '.' if p < 0.1 else 'ns'
        lines.append(f'{"(p-value)":<12} {pv[0]:>12.4f} {pv[1]:>14.4f}{sig(pv[1])} {pv[2]:>9.4f}{sig(pv[2])} {pv[3]:>16.4f}{sig(pv[3])}')
    lines.append('Signif codes: 0 *** 0.001 ** 0.01 * 0.05 . 0.1 ns')
    lines.append('')

    # ── Table 5: Training Accuracy per Epoch ──────────────────────────────────
    lines.append(sep)
    lines.append('TABLE 5: TRAINING ACCURACY PER EPOCH (ALL SIZES & DATASETS)')
    lines.append(sep)
    for ds in ['GSM8K', 'AQUA', 'HOTPOTQA']:
        lines.append(f'  {ds}:')
        lines.append(f'  {"Train Size":<12} {"Ep1":>7} {"Ep2":>7} {"Ep3":>7} {"Ep4":>7} {"Ep5":>7} {"Mean":>7} {"Std":>7} {"MaxFluc":>8}')
        lines.append('  ' + '-' * 72)
        for sz in [100, 200, 300, 400]:
            accs = [a * 100 for a in epoch_data[ds][sz]]
            deltas = [abs(accs[i] - accs[i-1]) for i in range(1, len(accs))]
            row = f'  {sz:<12}'
            for a in accs: row += f' {a:>6.1f}%'
            row += f' {np.mean(accs):>6.1f}% {np.std(accs):>6.2f}% {max(deltas):>7.2f}%'
            lines.append(row)
        lines.append('')

    # ── Table 6: Topology Benchmark ───────────────────────────────────────────
    lines.append(sep)
    lines.append('TABLE 6: TOPOLOGY BENCHMARK RESULTS (400-TRAIN, GSM8K)')
    lines.append(sep)
    lines.append(f'{"Rank":<6} {"Method":<25} {"Dataset":<12} {"N":>5} {"Accuracy":>10}')
    lines.append('-' * 60)
    sorted_topo = sorted(topo_rows, key=lambda x: x['accuracy'], reverse=True)
    for rank, row in enumerate(sorted_topo, 1):
        lines.append(f'{rank:<6} {row["method"]:<25} {row["dataset"]:<12} {row["n"]:>5} {row["accuracy"]*100:>9.2f}%')
    lines.append('')

    # ── Table 7: Summary – Best vs Worst per Dataset ─────────────────────────
    lines.append(sep)
    lines.append('TABLE 7: SUMMARY – BEST vs WORST METHOD PER DATASET (400-Train)')
    lines.append(sep)
    lines.append(f'{"Dataset":<12} {"Best Method":<18} {"Best Acc":>10} {"Worst Method":<18} {"Worst Acc":>10} {"Gap":>8}')
    lines.append('-' * 80)
    for ds in datasets:
        rows_ds = [(m, final.get(f'{m}_{ds}', {}).get('accuracy_mean', 0)) 
                   for m in methods]
        rows_ds = [(m, a) for m, a in rows_ds if a > 0]
        if rows_ds:
            best_m, best_a = max(rows_ds, key=lambda x: x[1])
            worst_m, worst_a = min(rows_ds, key=lambda x: x[1])
            gap = (best_a - worst_a) * 100
            bml = dict(zip(methods, method_labels))[best_m]
            wml = dict(zip(methods, method_labels))[worst_m]
            lines.append(f'{ds:<12} {bml:<18} {best_a*100:>9.2f}% {wml:<18} {worst_a*100:>9.2f}% {gap:>7.2f}%')
    lines.append('')

    output = '\n'.join(lines)
    print(output)

    # Save to file
    table_path = os.path.join(PLOTS, 'all_tables.txt')
    with open(table_path, 'w') as f:
        f.write(output)
    print(f'\nTables saved to: {table_path}')
    return output


# ═══════════════════════════════════════════════════════════════════════════════
# KEY INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════════

def print_insights(final, epoch_data, topo_rows, stream_split):
    print('\n' + '=' * 80)
    print('KEY INSIGHTS')
    print('=' * 80)

    # 1. Transfer Learning Effect
    print('\n[1] TRANSFER LEARNING EFFECT (100 → 400 samples):')
    for ds in ['GSM8K', 'AQUA', 'HOTPOTQA']:
        f300 = stream_split[ds]['Transfer']['first_300']
        l300 = final.get(f'Transfer_{ds}', {}).get('accuracy_mean', 0)
        if f300:
            delta = (l300 - np.mean(f300)) * 100
            print(f'  {ds}: {np.mean(f300)*100:.2f}% → {l300*100:.2f}% | Δ = {delta:+.2f}%')

    # 2. Token efficiency
    print('\n[2] TOKEN EFFICIENCY (Transfer vs FullComm):')
    for ds in ['GSM8K', 'AQUA', 'HOTPOTQA']:
        t_acc = final.get(f'Transfer_{ds}', {}).get('accuracy_mean', 0)
        f_acc = final.get(f'FullComm_{ds}', {}).get('accuracy_mean', 0)
        t_tok = final.get(f'Transfer_{ds}', {}).get('token_cost_mean', 0)
        f_tok = final.get(f'FullComm_{ds}', {}).get('token_cost_mean', 1)
        ratio = f_tok / t_tok if t_tok > 0 else float('inf')
        acc_delta = (t_acc - f_acc) * 100
        print(f'  {ds}: Transfer={t_acc*100:.2f}% @ {t_tok:.0f} tok | FullComm={f_acc*100:.2f}% @ {f_tok:.0f} tok | '
              f'Token savings: {ratio:.1f}x | Acc diff: {acc_delta:+.2f}%')

    # 3. Ablation best arm
    print('\n[3] BEST ABLATION ARM PER TRAINING SIZE:')
    for sz in [100, 200, 300, 400]:
        ab = load_ablation(sz)
        best_arm = ''
        best_acc = 0
        for ak in ARM_KEYS:
            samples = ab[ak]
            c = sum(1 for s in samples if s.get('is_correct', False))
            n = len(samples)
            acc = c / n if n else 0
            if acc > best_acc:
                best_acc = acc
                best_arm = ak
        print(f'  Size {sz}: Best = {ARM_LABELS[best_arm].replace(chr(10)," ")} | Acc = {best_acc*100:.2f}%')

    # 4. Training stability
    print('\n[4] TRAINING STABILITY (Std of epoch accuracy):')
    for ds in ['GSM8K', 'AQUA', 'HOTPOTQA']:
        for sz in [100, 200, 300, 400]:
            accs = [a * 100 for a in epoch_data[ds][sz]]
            deltas = [abs(accs[i]-accs[i-1]) for i in range(1, len(accs))]
            max_fluc = max(deltas)
            print(f'  {ds} | {sz}: std={np.std(accs):.2f}%, max_fluctuation={max_fluc:.2f}%')

    # 5. Topology best/worst
    if topo_rows:
        sorted_t = sorted(topo_rows, key=lambda x: x['accuracy'], reverse=True)
        print(f'\n[5] TOPOLOGY BENCHMARK: Best={sorted_t[0]["method"]} ({sorted_t[0]["accuracy"]*100:.2f}%) | '
              f'Worst={sorted_t[-1]["method"]} ({sorted_t[-1]["accuracy"]*100:.2f}%) | '
              f'Gap={( sorted_t[0]["accuracy"] - sorted_t[-1]["accuracy"])*100:.2f}%')

    # 6. OLS key findings
    print('\n[6] OLS FACTORIAL FINDINGS:')
    for sz in [100, 200, 300, 400]:
        ab = load_ablation(sz)
        ols = ab['factorial_ols_betas']
        pv = ols['p_values']
        dominant = 'β_B (PPO)' if abs(ols['beta_B_ppo']) > abs(ols['beta_A_sparsity']) else 'β_A (Sparsity)'
        print(f'  Size {sz}: Dominant effect = {dominant} | '
              f'β_A={ols["beta_A_sparsity"]:.4f}(p={pv[1]:.3f}) | '
              f'β_B={ols["beta_B_ppo"]:.4f}(p={pv[2]:.3f}) | '
              f'β_AB={ols["beta_AB_interaction"]:.4f}(p={pv[3]:.3f})')


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print('\n' + '=' * 80)
    print('FULL ANALYSIS – projectRL 08-06results')
    print('=' * 80)

    print('\n[Loading data...]')
    final = load_final_results()
    epoch_data = load_training_epochs()
    topo_rows = load_topology()
    stream_split = load_transfer_stream_split()

    print('[Generating plots...]')
    plots_generated = []
    plots_generated.append(('Plot 1', 'Transfer (100-train) Results', plot_transfer_100(stream_split)))
    plots_generated.append(('Plot 2', 'Transfer (400-train) Results', plot_transfer_400(final)))
    plots_generated.append(('Plot 3', 'Before/After Transfer', plot_transfer_before_after(stream_split, final)))
    plots_generated.append(('Plot 4', 'Ablation All Arms × Sizes', plot_ablation_all_sizes()))
    plots_generated.append(('Plot 5', 'Ablation Heatmap', plot_ablation_heatmap()))
    plots_generated.append(('Plot 6', 'OLS Betas', plot_ols_betas()))
    plots_generated.append(('Plot 7', 'Training Epochs', plot_training_epochs(epoch_data)))
    plots_generated.append(('Plot 8', 'Training Epoch Grid', plot_training_epochs_grid(epoch_data)))
    plots_generated.append(('Plot 9', 'Topology Benchmark', plot_topology(topo_rows)))
    plots_generated.append(('Plot 10', 'Token Efficiency', plot_token_efficiency(final)))
    plots_generated.append(('Plot 11', 'Fluctuation Analysis', plot_fluctuation_analysis(epoch_data)))
    plots_generated.append(('Plot 12', 'Token vs Accuracy', plot_ablation_token_accuracy()))
    plots_generated.append(('Plot 13', 'Radar Methods', plot_radar_methods(final)))
    plots_generated.append(('Plot 14', 'Improvement Symmetry', plot_improvement_symmetry()))

    print(f'\n✓ {len(plots_generated)} plots generated in: {PLOTS}')

    print('\n[Printing tables...]')
    print_all_tables(final, epoch_data, topo_rows, stream_split)

    print('\n[Key insights...]')
    print_insights(final, epoch_data, topo_rows, stream_split)

    print('\n✓ Analysis complete!')
    return plots_generated

if __name__ == '__main__':
    main()
