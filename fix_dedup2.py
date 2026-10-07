import json

with open('SparseComm_Kaggle.ipynb') as f:
    nb = json.load(f)

fixes = []

for cell in nb.get('cells', []):
    if cell['cell_type'] != 'code':
        continue
    source = "".join(cell['source'])
    original = source

    # Count NumpyEncoder occurrences in THIS cell
    if source.count('class NumpyEncoder(json.JSONEncoder):') == 2:
        # Remove the SECOND occurrence
        lines = source.split('\n')
        new_lines = []
        found_first = False
        in_second = False
        for line in lines:
            if 'class NumpyEncoder(json.JSONEncoder):' in line:
                if not found_first:
                    found_first = True
                    new_lines.append(line)
                else:
                    # This is the second one, start skipping
                    in_second = True
                    continue
            elif in_second:
                # Skip indented lines (class body)
                if line == '' or line.startswith(' ') or line.startswith('\t'):
                    continue
                else:
                    in_second = False
                    new_lines.append(line)
            else:
                new_lines.append(line)
        source = '\n'.join(new_lines)
        fixes.append("D2: Second NumpyEncoder removed (same cell)")

    if source != original:
        cell['source'] = [l + '\n' for l in source.split('\n')]
        if cell['source'] and cell['source'][-1] == '\n':
            cell['source'] = cell['source'][:-1]

with open('SparseComm_Kaggle.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

for f in fixes:
    print(f"  ✅ {f}")
