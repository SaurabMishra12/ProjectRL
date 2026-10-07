import json
import re

with open('SparseComm_Kaggle.ipynb') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])

        # Fix 9: Intermediate rewards
        # We need to find the step evaluation logic.
        if 'reward_for_step = token_penalty' in source and 'for t in range(config.n_rounds):' in source:
            # Add variables for prev_acc tracking
            source = source.replace('for t in range(config.n_rounds):', '_prev_acc = None\n    for t in range(config.n_rounds):')
            # Calculate current_acc inside loop
            acc_calc_str = """
        _current_acc = 0.0
        if answers:
            _maj = weighted_majority_vote(answers, config.dataset_type, W=None)
            _current_acc = 1.0 if evaluate_answer(_maj, list(sample['answers'])[0], config.dataset_type) else 0.0
        
        reward_for_step = token_penalty
        if _prev_acc is not None:
            reward_for_step += 0.5 * (_current_acc - _prev_acc)
        _prev_acc = _current_acc
"""
            source = source.replace('reward_for_step = token_penalty', acc_calc_str.strip())
            
        # Fix 20: Unanimous Round 0 early exit
        synth_transition = """
    if len(_round0_unique) == 1 and list(_round0_unique)[0]:
        # Inject synthetic transition for controller
        if hasattr(config, 'train') and config.train and controller:
            _synth_state = np.zeros(config.n_agents * config.agent_dim)
            # Create a zero action
            _synth_action = np.zeros(config.n_agents * config.k_dim * 2 + config.n_agents * controller.query_dim + 1)
            _synth_action[-1] = 10.0 # High halt logit
            controller.store_transition(
                state=_synth_state,
                action=_synth_action,
                reward=1.0,  # Positive reward for halting early on easy questions
                value=0.0,
                log_prob=0.0,
                done=True
            )
        return SampleResult(
"""
        source = re.sub(r'if len\(_round0_unique\) == 1 and list\(_round0_unique\)\[0\]:\n\s*return SampleResult\(', synth_transition.strip() + '(\n', source)

        # Fix 22, 23: Duplicated definitions
        # This is tricky using regex per cell. If a cell ONLY contains NumpyEncoder, we can clear it.
        # But maybe they are in the same cell. Let's rely on finding and removing the second occurrence.

        cell['source'] = [line + '\n' for line in source.split('\n')]
        if cell['source'] and cell['source'][-1] == '\n':
            cell['source'] = cell['source'][:-1]

with open('SparseComm_Kaggle.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
