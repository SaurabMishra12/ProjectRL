import json

with open('SparseComm_Kaggle.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        print(f"Cell {i}: {len(cell['source'])} lines")
