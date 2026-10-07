"""
Visualization: All required plots for the paper.

1. Communication graph (before vs after training)
2. Heatmap of W_ij
3. Token cost vs accuracy (scatter)
4. Error propagation comparison (bar chart)
5. Ablation comparison (bar chart)
6. Training curve (reward over episodes)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import List, Dict, Any, Optional

from evaluation import ExperimentResults, aggregate_metrics
from utils import ensure_dir

# Style configuration for publication-quality plots
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

COLORS = {
    'IB-Controller': '#2196F3',
    'Self-Consistency': '#FF9800',
    'Majority Voting': '#4CAF50',
    'Multi-Agent Debate': '#F44336',
    'Static Routing': '#9C27B0',
    'Ablation: lambda=0': '#795548',
    'Ablation: No Summarize': '#607D8B',
    'Ablation: No Filter': '#009688',
    'Ablation: Random Controller': '#E91E63',
}


def plot_communication_graph(
    W_before: np.ndarray,
    W_after: np.ndarray,
    save_path: str,
    title: str = "Communication Graph Evolution",
):
    """
    Plot communication graph before and after training.
    Edge width proportional to W_ij.
    """
    try:
        import networkx as nx
    except ImportError:
        _plot_communication_graph_simple(W_before, W_after, save_path, title)
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, W, subtitle in [(ax1, W_before, "Before Training"),
                             (ax2, W_after, "After Training")]:
        n = W.shape[0]
        G = nx.DiGraph()

        for i in range(n):
            G.add_node(i, label=f"Agent {i}")

        for i in range(n):
            for j in range(n):
                if i != j and W[i][j] > 0.05:
                    G.add_edge(i, j, weight=W[i][j])

        pos = nx.circular_layout(G)
        node_colors = ['#2196F3', '#FF9800', '#4CAF50', '#F44336', '#9C27B0'][:n]

        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_color=node_colors,
            node_size=800, alpha=0.9
        )
        nx.draw_networkx_labels(
            G, pos, ax=ax,
            labels={i: f"A{i}" for i in range(n)},
            font_color='white', font_weight='bold'
        )

        edges = G.edges(data=True)
        if edges:
            weights = [d['weight'] for _, _, d in edges]
            max_w = max(weights) if weights else 1
            widths = [3 * w / max_w for w in weights]
            alphas = [0.3 + 0.7 * w / max_w for w in weights]

            for (u, v, d), width, alpha in zip(edges, widths, alphas):
                nx.draw_networkx_edges(
                    G, pos, edgelist=[(u, v)], ax=ax,
                    width=width, alpha=alpha,
                    edge_color='#333333',
                    arrows=True, arrowsize=15,
                    connectionstyle='arc3,rad=0.1'
                )

        ax.set_title(subtitle, fontweight='bold')
        ax.axis('off')

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def _plot_communication_graph_simple(
    W_before: np.ndarray, W_after: np.ndarray,
    save_path: str, title: str
):
    """Fallback graph visualization without networkx."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, W, subtitle in [(ax1, W_before, "Before"), (ax2, W_after, "After")]:
        n = W.shape[0]
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        xs = np.cos(angles)
        ys = np.sin(angles)

        for i in range(n):
            for j in range(n):
                if i != j and W[i][j] > 0.05:
                    ax.annotate("",
                        xy=(xs[j], ys[j]), xytext=(xs[i], ys[i]),
                        arrowprops=dict(
                            arrowstyle="->",
                            lw=W[i][j] * 3,
                            alpha=0.3 + 0.7 * W[i][j],
                            color='#333'
                        )
                    )

        colors = ['#2196F3', '#FF9800', '#4CAF50', '#F44336', '#9C27B0'][:n]
        ax.scatter(xs, ys, s=800, c=colors, zorder=5)
        for i in range(n):
            ax.text(xs[i], ys[i], f'A{i}', ha='center', va='center',
                    fontweight='bold', color='white', zorder=6)
        ax.set_title(subtitle, fontweight='bold')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_weight_heatmap(
    W: np.ndarray,
    save_path: str,
    title: str = "Communication Weight Matrix W_ij",
):
    """Heatmap of the learned weight matrix."""
    fig, ax = plt.subplots(figsize=(8, 6))

    n = W.shape[0]
    labels = [f"Agent {i}" for i in range(n)]

    sns.heatmap(
        W, ax=ax,
        annot=True, fmt='.2f',
        xticklabels=labels, yticklabels=labels,
        cmap='YlOrRd', vmin=0, vmax=1,
        square=True, linewidths=0.5,
        cbar_kws={'label': 'Edge Weight'},
    )

    ax.set_xlabel('Receiver', fontweight='bold')
    ax.set_ylabel('Sender', fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=14)

    # Add bandwidth mode annotations
    for i in range(n):
        for j in range(n):
            if i != j:
                w = W[i][j]
                if w < 0.25:
                    mode = "B"
                elif w < 0.5:
                    mode = "S"
                elif w < 0.75:
                    mode = "F"
                else:
                    mode = "OK"

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_token_vs_accuracy(
    all_metrics: List[Dict[str, Any]],
    save_path: str,
    title: str = "Token Cost vs. Accuracy",
):
    """Scatter plot of token cost vs accuracy for all methods."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for m in all_metrics:
        name = m['method']
        acc = m['accuracy'] * 100
        tokens = m['token_cost']['avg_tokens_per_sample']
        color = COLORS.get(name, '#666666')

        ax.scatter(tokens, acc, s=200, c=color, zorder=5,
                   edgecolors='white', linewidth=2)
        ax.annotate(
            name, (tokens, acc),
            textcoords="offset points", xytext=(10, 10),
            fontsize=9, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.8)
        )

    ax.set_xlabel('Average Tokens per Sample', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)

    # Ideal region annotation
    ax.annotate('<-- Ideal Region\n(Low Cost, High Accuracy)',
                xy=(0.05, 0.95), xycoords='axes fraction',
                fontsize=9, fontstyle='italic', color='green', alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_error_propagation(
    all_metrics: List[Dict[str, Any]],
    save_path: str,
    title: str = "Error Propagation Rate Comparison",
):
    """Grouped bar chart of error propagation rates."""
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = [m['method'] for m in all_metrics]
    rates = [m['error_propagation_rate'] for m in all_metrics]
    colors = [COLORS.get(m, '#666666') for m in methods]

    x = np.arange(len(methods))
    bars = ax.bar(x, rates, color=colors, edgecolor='white', linewidth=1.5, width=0.6)

    # Add value labels on bars
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                f'{rate:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Error Propagation Rate', fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_ylim(0, max(rates) * 1.3 if rates else 1.0)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_ablation_comparison(
    ablation_metrics: List[Dict[str, Any]],
    main_metrics: Dict[str, Any],
    save_path: str,
    title: str = "Ablation Study Results",
):
    """Bar chart comparing ablation conditions against the full model."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    all_m = [main_metrics] + ablation_metrics
    methods = [m['method'] for m in all_m]
    accuracies = [m['accuracy'] * 100 for m in all_m]
    tokens = [m['token_cost']['avg_tokens_per_sample'] for m in all_m]
    colors = [COLORS.get(m, '#666666') for m in methods]

    x = np.arange(len(methods))

    # Accuracy comparison
    bars1 = ax1.bar(x, accuracies, color=colors, edgecolor='white', linewidth=1.5, width=0.6)
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                 f'{acc:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=35, ha='right', fontsize=8)
    ax1.set_ylabel('Accuracy (%)', fontweight='bold')
    ax1.set_title('Accuracy', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Token cost comparison
    bars2 = ax2.bar(x, tokens, color=colors, edgecolor='white', linewidth=1.5, width=0.6)
    for bar, tok in zip(bars2, tokens):
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 1,
                 f'{tok:.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=35, ha='right', fontsize=8)
    ax2.set_ylabel('Avg Tokens / Sample', fontweight='bold')
    ax2.set_title('Token Cost', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_training_curve(
    rewards: List[float],
    losses: List[float],
    save_path: str,
    title: str = "Controller Training Progress",
):
    """Plot training reward and loss over episodes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    if rewards:
        # Smooth rewards with running average
        window = min(10, len(rewards) // 3 + 1)
        if window > 1:
            smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
            ax1.plot(range(len(rewards)), rewards, alpha=0.3, color='#2196F3', label='Raw')
            ax1.plot(range(window - 1, window - 1 + len(smoothed)), smoothed,
                     color='#2196F3', linewidth=2, label='Smoothed')
        else:
            ax1.plot(range(len(rewards)), rewards, color='#2196F3', linewidth=2)
        ax1.set_xlabel('Episode', fontweight='bold')
        ax1.set_ylabel('Reward', fontweight='bold')
        ax1.set_title('Reward per Episode', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    if losses:
        ax2.plot(range(len(losses)), losses, color='#F44336', linewidth=2)
        ax2.set_xlabel('Update Step', fontweight='bold')
        ax2.set_ylabel('Loss', fontweight='bold')
        ax2.set_title('Policy Gradient Loss', fontweight='bold')
        ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_mode_distribution(
    all_metrics: List[Dict[str, Any]],
    save_path: str,
    title: str = "Bandwidth Mode Distribution",
):
    """Stacked bar chart showing mode distribution per method."""
    fig, ax = plt.subplots(figsize=(12, 6))

    methods = [m['method'] for m in all_metrics]
    mode_names = ['BLOCK', 'SUMMARIZE', 'FILTER', 'FULL']
    mode_colors = ['#F44336', '#FF9800', '#2196F3', '#4CAF50']

    x = np.arange(len(methods))
    bottom = np.zeros(len(methods))

    for mode, color in zip(mode_names, mode_colors):
        values = []
        for m in all_metrics:
            dist = m.get('mode_distribution', {})
            values.append(dist.get(mode, 0))
        values = np.array(values)
        ax.bar(x, values, bottom=bottom, color=color, label=mode,
               edgecolor='white', linewidth=0.5, width=0.6)
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Proportion', fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.legend(title='Mode')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def generate_all_plots(
    main_results: ExperimentResults,
    baseline_results: List[ExperimentResults],
    ablation_results: List[ExperimentResults],
    plots_dir: str,
    dataset_name: str = "",
):
    """Generate all required plots and save to plots_dir."""
    ensure_dir(plots_dir)
    suffix = f"_{dataset_name}" if dataset_name else ""

    # Compute metrics for all methods
    main_metrics = aggregate_metrics(main_results)
    baseline_metrics = [aggregate_metrics(r) for r in baseline_results]
    ablation_metrics = [aggregate_metrics(r) for r in ablation_results]

    all_metrics = [main_metrics] + baseline_metrics

    # 1. Communication graph (before vs after)
    n = main_results.final_W.shape[0] if main_results.final_W is not None else 3
    W_before = np.ones((n, n)) * 0.5
    np.fill_diagonal(W_before, 0)
    W_after = main_results.final_W if main_results.final_W is not None else W_before

    plot_communication_graph(
        W_before, W_after,
        os.path.join(plots_dir, f"communication_graph{suffix}.png"),
        f"Communication Graph Evolution — {dataset_name}",
    )

    # 2. Weight heatmap
    plot_weight_heatmap(
        W_after,
        os.path.join(plots_dir, f"weight_heatmap{suffix}.png"),
        f"Learned Weight Matrix — {dataset_name}",
    )

    # 3. Token cost vs accuracy
    plot_token_vs_accuracy(
        all_metrics,
        os.path.join(plots_dir, f"token_vs_accuracy{suffix}.png"),
        f"Token Cost vs. Accuracy — {dataset_name}",
    )

    # 4. Error propagation comparison
    plot_error_propagation(
        all_metrics,
        os.path.join(plots_dir, f"error_propagation{suffix}.png"),
        f"Error Propagation Rate — {dataset_name}",
    )

    # 5. Ablation comparison
    if ablation_metrics:
        plot_ablation_comparison(
            ablation_metrics, main_metrics,
            os.path.join(plots_dir, f"ablation{suffix}.png"),
            f"Ablation Study — {dataset_name}",
        )

    # 6. Training curve
    if main_results.training_rewards:
        plot_training_curve(
            main_results.training_rewards,
            main_results.training_losses,
            os.path.join(plots_dir, f"training_curve{suffix}.png"),
            f"Controller Training — {dataset_name}",
        )

    # 7. Mode distribution (bonus)
    plot_mode_distribution(
        all_metrics,
        os.path.join(plots_dir, f"mode_distribution{suffix}.png"),
        f"Bandwidth Mode Distribution — {dataset_name}",
    )

    return {
        'main_metrics': main_metrics,
        'baseline_metrics': baseline_metrics,
        'ablation_metrics': ablation_metrics,
    }
