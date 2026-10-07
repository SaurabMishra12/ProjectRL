import json

with open('SparseComm_Kaggle.ipynb') as f:
    nb = json.load(f)

seen_encoder = False
seen_get_weights = False
seen_benchmark = False

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Deduplicate NumpyEncoder
        if 'class NumpyEncoder(json.JSONEncoder):' in source:
            if seen_encoder:
                # Remove this definition block
                import re
                source = re.sub(r'class NumpyEncoder\(json\.JSONEncoder\):.*?def default.*?return json\.JSONEncoder\.default\(self, obj\)\n', '', source, flags=re.DOTALL)
            else:
                seen_encoder = True
                
        # Deduplicate get_topology_weights
        if 'def get_topology_weights(topology_name: str, n_agents: int) -> np.ndarray:' in source:
            if seen_get_weights:
                import re
                source = re.sub(r'def get_topology_weights\(topology_name: str, n_agents: int\) -> np\.ndarray:.*?return W\n', '', source, flags=re.DOTALL)
            else:
                seen_get_weights = True

        # Deduplicate run_topology_benchmark
        if 'def run_topology_benchmark():' in source:
            if seen_benchmark:
                import re
                source = re.sub(r'def run_topology_benchmark\(\):.*?print\(f"Benchmark results saved to \{save_path\}\."\)\n', '', source, flags=re.DOTALL)
            else:
                seen_benchmark = True

        # Re-pack lines
        lines = source.split('\n')
        if lines and not lines[-1]:
            lines.pop()
        cell['source'] = [l + '\n' for l in lines]
        # if the last line has a newline, it's correct. Jupyter requires newlines except on the very last line sometimes, but split/join handles it.

with open('SparseComm_Kaggle.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
