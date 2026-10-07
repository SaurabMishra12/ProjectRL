import nbformat

notebook_path = "SparseComm_Kaggle.ipynb"

# Read notebook
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Fixed cell code
cell_code = """# --- TOPOLOGY FAMILY BENCHMARK ---
import networkx as nx
import numpy as np
import dataclasses
import time

def get_topology_weights(topology_name: str, n_agents: int) -> np.ndarray:
    if topology_name == "Fully Connected":
        G = nx.complete_graph(n_agents)
    elif topology_name == "Ring":
        G = nx.cycle_graph(n_agents)
    elif topology_name == "Star":
        G = nx.star_graph(n_agents - 1)
    elif topology_name == "Random":
        G = nx.erdos_renyi_graph(n_agents, p=0.5, seed=42)
    elif topology_name == "Small World":
        # For N=3, k=2 is the only valid even k < 3. 
        k = min(2, n_agents - 1)
        if k % 2 != 0: k -= 1
        k = max(2, k)
        G = nx.watts_strogatz_graph(n_agents, k=k, p=0.2, seed=42) if n_agents > 3 else nx.cycle_graph(n_agents)
    elif topology_name == "Scale Free":
        G = nx.barabasi_albert_graph(n_agents, m=1, seed=42) if n_agents >= 3 else nx.complete_graph(n_agents)
    else:
        raise ValueError(f"Unknown topology {topology_name}")
    
    W = nx.to_numpy_array(G)
    np.fill_diagonal(W, 1.0)
    
    # Scale binary adjacency matrix to transmission weights (e.g. 0.6)
    W = W * 0.6 
    return W

def run_topology_benchmark():
    print("Starting Topology Family Benchmark...")
    # Load default config
    config = get_gsm8k_config()
    config.n_samples = 50  # Default eval size
    set_seed(config.seed)
    
    benchmark_results = []
    
    topologies = [
        "Fully Connected",
        "Ring",
        "Star",
        "Random",
        "Small World",
        "Scale Free"
    ]
    
    # 1. Evaluate Static Graph Topologies
    for topo in topologies:
        print(f"\\n--- Evaluating Topology: {topo} ---")
        W = get_topology_weights(topo, config.n_agents)
        Q = np.zeros((config.n_agents, config.query_dim)) # Dummy Q
        
        controller = FrozenWeightController(config, frozen_W=W, frozen_Q=Q)
        
        # NOTE: run_experiment creates the dataset and backend internally based on config
        results = run_experiment(
            config=config,
            method_name=topo,
            train=False,
            evaluate=True,
            controller=controller
        )
        benchmark_results.append(results)
        
    # 2. Evaluate Learned Topology
    print("\\n--- Evaluating Topology: Learned Topology ---")
    learned_controller = IBController(config)
    # Load model from the 08-06results folder
    model_path = r"C:\\Users\\msaur\\OneDrive\\Desktop\\projectRL\\08-06results\\gsm8k_pretrained.pt.npz"
    try:
        learned_controller.load_model(model_path)
        print(f"Successfully loaded trained controller from {model_path}")
        
        learned_results = run_experiment(
            config=config,
            method_name="Learned Topology",
            train=False,
            evaluate=True,
            controller=learned_controller
        )
        benchmark_results.append(learned_results)
    except Exception as e:
        print(f"Failed to load trained model: {e}")
        
    # 3. Print Final Benchmark Table
    metrics = [aggregate_metrics(r) for r in benchmark_results]
    print("\\n=== TOPOLOGY FAMILY BENCHMARK RESULTS ===")
    print(format_results_table(metrics, dataset_name="GSM8K Topology Benchmark"))
    
    # Save results to 08-06results folder
    save_path = r"C:\\Users\\msaur\\OneDrive\\Desktop\\projectRL\\08-06results\\topology_benchmark_results.json"
    save_json(
        [dataclasses.asdict(r) for r in benchmark_results], 
        save_path
    )
    print(f"Benchmark results saved to {save_path}.")

# Uncomment the line below to run the benchmark:
# run_topology_benchmark()
"""

# Replace the source of the last cell
nb.cells[-1].source = cell_code

# Save back to notebook
with open(notebook_path, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("Successfully fixed benchmark cell in notebook.")
