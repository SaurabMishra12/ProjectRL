import json

with open('SparseComm_Kaggle.ipynb') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Inject the consensus tracking
        if 'for t in range(config.n_rounds):' in source and '_prev_consensus = None' not in source:
            source = source.replace('for t in range(config.n_rounds):', '_prev_consensus = None\n    for t in range(config.n_rounds):')
        
        target = 'reward_for_step = token_penalty # Will add delta_acc below'
        replacement = """
                _current_consensus = 0.0
                if answers:
                    _normalized_answers = [_normalize_for_vote(r.answer, config.dataset_type) for r in answers]
                    if _normalized_answers:
                        from collections import Counter
                        _counts = Counter(_normalized_answers)
                        _most_common_count = _counts.most_common(1)[0][1]
                        _current_consensus = _most_common_count / len(_normalized_answers)
                
                reward_for_step = token_penalty
                if _prev_consensus is not None:
                    reward_for_step += 0.5 * (_current_consensus - _prev_consensus)
                _prev_consensus = _current_consensus
"""
        if target in source:
            source = source.replace(target, replacement.strip('\n'))
            
        cell['source'] = [l + '\n' for l in source.split('\n')]
        if cell['source'] and cell['source'][-1] == '\n':
            cell['source'] = cell['source'][:-1]

with open('SparseComm_Kaggle.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
