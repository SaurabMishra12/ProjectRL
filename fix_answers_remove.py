import json
import re

with open('SparseComm_Kaggle.ipynb') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Remove the offending block entirely via regex
        pattern = r'\s*_current_consensus = 0\.0\s*if answers:\s*_normalized_answers = \[\_normalize_for_vote\(r\.answer, config\.dataset_type\) for r in answers\]\s*if _normalized_answers:\s*from collections import Counter\s*_counts = Counter\(_normalized_answers\)\s*_most_common_count = _counts\.most_common\(1\)\[0\]\[1\]\s*_current_consensus = _most_common_count / len\(_normalized_answers\)'
        source = re.sub(pattern, '', source)

        # There's also `reward_for_step = token_penalty` and `if _prev_consensus is not None:` that might be left hanging... wait, I removed those earlier or maybe they are still there?
        pattern2 = r'\s*reward_for_step = token_penalty\s*if _prev_consensus is not None:\s*reward_for_step \+= 0\.5 \* \(_current_consensus - _prev_consensus\)\s*_prev_consensus = _current_consensus'
        source = re.sub(pattern2, '', source)

        cell['source'] = [l + '\n' for l in source.split('\n')]
        if cell['source'] and cell['source'][-1] == '\n':
            cell['source'] = cell['source'][:-1]

with open('SparseComm_Kaggle.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
