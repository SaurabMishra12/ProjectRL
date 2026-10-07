import json
import re

with open('SparseComm_Kaggle.ipynb') as f:
    nb = json.load(f)

fixes = []
numpy_enc_seen = 0
topo_bench_seen = 0

for cell in nb.get('cells', []):
    if cell['cell_type'] != 'code':
        continue
    source = "".join(cell['source'])
    original = source

    # D2: Remove SECOND NumpyEncoder (line-by-line approach)
    if 'class NumpyEncoder(json.JSONEncoder):' in source:
        numpy_enc_seen += 1
        if numpy_enc_seen == 2:
            # Remove entire class definition: from 'class NumpyEncoder' to the next top-level definition
            lines = source.split('\n')
            new_lines = []
            in_class = False
            for line in lines:
                if line.startswith('class NumpyEncoder('):
                    in_class = True
                    continue
                if in_class:
                    if line and not line.startswith(' ') and not line.startswith('\t'):
                        in_class = False
                        new_lines.append(line)
                    # skip indented lines (class body)
                    continue
                new_lines.append(line)
            source = '\n'.join(new_lines)
            fixes.append("D2: Second NumpyEncoder removed")

    # D3b: Remove FIRST run_topology_benchmark
    if 'def run_topology_benchmark():' in source:
        topo_bench_seen += 1
        if topo_bench_seen == 1:
            lines = source.split('\n')
            new_lines = []
            in_func = False
            for line in lines:
                if line.startswith('def run_topology_benchmark():'):
                    in_func = True
                    continue
                if in_func:
                    if line and not line.startswith(' ') and not line.startswith('\t'):
                        in_func = False
                        new_lines.append(line)
                    continue
                new_lines.append(line)
            source = '\n'.join(new_lines)
            fixes.append("D3b: First run_topology_benchmark removed")

    if source != original:
        cell['source'] = [l + '\n' for l in source.split('\n')]
        if cell['source'] and cell['source'][-1] == '\n':
            cell['source'] = cell['source'][:-1]

with open('SparseComm_Kaggle.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

for f in fixes:
    print(f"  ✅ {f}")
