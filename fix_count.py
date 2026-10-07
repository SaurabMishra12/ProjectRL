import json
import re

with open('SparseComm_Kaggle.ipynb') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Clean up the dangling DFS code
        source = re.sub(r'def count_cycles\(self, threshold: float = 0\.5\) -> int:.*?(?=\s+def reset\()', 
                        'def count_cycles(self, threshold: float = 0.5) -> int:\n        adj = (self.W > threshold).astype(int)\n        np.fill_diagonal(adj, 0)\n        return int(np.sum(adj * adj.T)) // 2\n\n', 
                        source, flags=re.DOTALL)
                        
        source = source.replace('return hotpotqa_f1_score(predicted, gold)[0] > 0.5  # TODO', 'return hotpotqa_f1_score(predicted, gold)[0] > 0.5')
        
        cell['source'] = [l + '\n' for l in source.split('\n')]
        if cell['source'] and cell['source'][-1] == '\n':
            cell['source'] = cell['source'][:-1]

with open('SparseComm_Kaggle.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
