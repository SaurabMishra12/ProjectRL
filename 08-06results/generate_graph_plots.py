import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx

BASE = '/home/saurab/Documents/projectRLMAIN/08-06results'
PLOTS = os.path.join(BASE, 'plots')
os.makedirs(PLOTS, exist_ok=True)

def generate_graph_evolution():
    print("Generating Graph Evolution Plots...")
    
    # ── Weights for each round ───────────────────────────────────────────────
    # Round 0: Independent (no edges)
    W0 = np.zeros((3, 3))
    
    # Round 1: Sparse PPO Weights
    W1 = np.array([
        [0.0, 0.126, 0.108],
        [0.412, 0.0, 0.693],
        [0.674, 0.844, 0.0]
    ])
    
    # Round 2: Focused Weights (after feedback)
    W2 = np.array([
        [0.0, 0.666, 0.465],
        [0.303, 0.0, 0.366],
        [0.623, 0.607, 0.0]
    ])
    
    weights = [W0, W1, W2]
    rounds_labels = ["Round 0\n(Independent solving)", "Round 1\n(PPO Sparse routing)", "Round 2\n(PPO Focused feedback)"]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='white')
    
    # Node positions: equilateral triangle
    pos = {
        0: (0.5, 0.8),
        1: (0.1, 0.2),
        2: (0.9, 0.2)
    }
    
    node_colors = ['#1F77B4', '#2CA02C', '#FF7F0E'] # Blue, Green, Orange
    node_labels = {
        0: "Agent 0\n(Analytical)",
        1: "Agent 1\n(Step-by-step)",
        2: "Agent 2\n(Creative)"
    }
    
    for r_idx, (W, ax, label) in enumerate(zip(weights, axes, rounds_labels)):
        ax.set_title(label, fontsize=14, fontweight='bold', pad=15)
        ax.axis('off')
        
        # Create directed graph
        G = nx.DiGraph()
        G.add_nodes_from([0, 1, 2])
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_color=node_colors, node_size=2800,
            edgecolors='#333333', linewidths=1.5, alpha=0.9
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            G, pos, labels=node_labels, ax=ax,
            font_size=10, font_weight='bold', font_color='black'
        )
        
        # Add edges and weights
        edges = []
        edge_widths = []
        edge_colors = []
        
        for i in range(3):
            for j in range(3):
                if i != j and W[i][j] > 0.02:
                    G.add_edge(i, j, weight=W[i][j])
                    edges.append((i, j))
                    # Scale width
                    edge_widths.append(W[i][j] * 5.0)
                    # Color based on weight: stronger weights are darker
                    edge_colors.append((0.2, 0.2, 0.2, min(W[i][j] + 0.1, 1.0)))
        
        # Draw directed curved edges so mutual links are visible
        if edges:
            nx.draw_networkx_edges(
                G, pos, edgelist=edges, width=edge_widths, edge_color=edge_colors,
                ax=ax, arrowsize=20, arrowstyle='-|>',
                connectionstyle='arc3,rad=0.15' # Curved edges
            )
            
            # Draw edge weights
            for (u, v) in edges:
                w_val = W[u][v]
                # Position label slightly offset from the curve
                mid_x = (pos[u][0] + pos[v][0]) / 2
                mid_y = (pos[u][1] + pos[v][1]) / 2
                # Offset based on curve
                dx = pos[v][0] - pos[u][0]
                dy = pos[v][1] - pos[u][1]
                length = np.sqrt(dx**2 + dy**2)
                nx_label = f"{w_val:.2f}"
                ax.text(
                    mid_x - 0.1*dy/length, mid_y + 0.1*dx/length,
                    nx_label, fontsize=9, fontweight='bold', color='#444444',
                    ha='center', va='center', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#CCCCCC", alpha=0.8)
                )
                
        # Draw circular border indicating connection bounds
        circle = plt.Circle((0.5, 0.5), 0.48, color='#EEEEEE', fill=False, linestyle='--', linewidth=1.0)
        ax.add_artist(circle)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0.05, 0.95)
        
    fig.tight_layout()
    plot_path = os.path.join(PLOTS, 'claim_graph_evolution.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Graph evolution plot saved to: {plot_path}")

if __name__ == '__main__':
    generate_graph_evolution()
