import json

with open('SparseComm_Kaggle.ipynb') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Remove the previous broken block entirely
        bad_block = """
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
        source = source.replace(bad_block.strip('\n'), "")
        
        # Also remove any dangling token_penalty assignment
        source = source.replace('reward_for_step = token_penalty # Will add delta_acc below', '')
        
        # Insert the correct logic using responses_snapshots
        target = """
                token_penalty = -controller.lambda_info * \\
                    round_tokens / max(float(tokens_if_full), 1.0)
                # Fix A: intermediate reward = cost penalty only.
                # Entropy-reduction shaping removed: rewards wrong-consensus
                # collapse.
"""
        replacement = """
                token_penalty = -controller.lambda_info * \\
                    round_tokens / max(float(tokens_if_full), 1.0)
                
                _current_consensus = 0.0
                _prev_consensus = 0.0
                
                # responses_snapshots has round 0 at idx 0, round 1 at idx 1, etc.
                # all_metas[i] represents the transition from round i to round i+1.
                if i + 1 < len(responses_snapshots):
                    ans_curr = responses_snapshots[i + 1]
                    norm_curr = [_normalize_for_vote(r.answer, config.dataset_type) for r in ans_curr]
                    if norm_curr:
                        from collections import Counter
                        _current_consensus = Counter(norm_curr).most_common(1)[0][1] / len(norm_curr)
                        
                if i < len(responses_snapshots):
                    ans_prev = responses_snapshots[i]
                    norm_prev = [_normalize_for_vote(r.answer, config.dataset_type) for r in ans_prev]
                    if norm_prev:
                        from collections import Counter
                        _prev_consensus = Counter(norm_prev).most_common(1)[0][1] / len(norm_prev)
                        
                reward_for_step = token_penalty + 0.5 * (_current_consensus - _prev_consensus)
"""
        if target in source:
            source = source.replace(target, replacement)
            
        cell['source'] = [l + '\n' for l in source.split('\n')]
        if cell['source'] and cell['source'][-1] == '\n':
            cell['source'] = cell['source'][:-1]

with open('SparseComm_Kaggle.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
