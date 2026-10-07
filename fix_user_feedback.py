import json
import re

with open('SparseComm_Kaggle.ipynb') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # Section 2: Methodological Flaws
        # Issue 8: Revert Cache key + round_num
        source = source.replace('self._cache_str = system_msg + user_msg + str(getattr(self, "current_round", ""))', 'self._cache_str = system_msg + user_msg')
        
        # Issue 9: Intermediate rewards using delta_consensus
        consensus_calc_str = """
        _current_consensus = 0.0
        if answers:
            # Count the most frequent answer fraction
            _normalized_answers = [_normalize_for_vote(r.answer, config.dataset_type) for r in answers]
            if _normalized_answers:
                from collections import Counter
                _counts = Counter(_normalized_answers)
                _most_common_count = _counts.most_common(1)[0][1]
                _current_consensus = _most_common_count / len(_normalized_answers)
        
        reward_for_step = token_penalty
        if _prev_acc is not None:
            # Using _prev_acc variable name but it stores consensus now to minimize code changes
            reward_for_step += 0.5 * (_current_consensus - _prev_acc)
        _prev_acc = _current_consensus
"""
        # We need to replace the _current_acc block we injected previously
        old_acc_calc_str = """
        _current_acc = 0.0
        if answers:
            _maj = weighted_majority_vote(answers, config.dataset_type, W=None)
            _current_acc = 1.0 if evaluate_answer(_maj, list(sample['answers'])[0], config.dataset_type) else 0.0
        
        reward_for_step = token_penalty
        if _prev_acc is not None:
            reward_for_step += 0.5 * (_current_acc - _prev_acc)
        _prev_acc = _current_acc
"""
        source = source.replace(old_acc_calc_str.strip(), consensus_calc_str.strip())
        
        
        # Section 3: PPO/RL Math
        # Issue 14: log_std gradient dimensionally wrong.
        # old: d_log_std = adv * ((action - mean_a_new)**2 / (sigma_new**2 + 1e-8) - 1.0)
        # new: d_log_std = (adv * ratio) * ((action - mean_a_new)**2 / (sigma_new**2 + 1e-8) - 1.0)
        source = source.replace('d_log_std = adv * \\\n                        ((action - mean_a_new)**2 / (sigma_new**2 + 1e-8) - 1.0)', 
                                'd_log_std = (adv * ratio) * \\\n                        ((action - mean_a_new)**2 / (sigma_new**2 + 1e-8) - 1.0)')

        # Issue 15: Actor PPO clipping - correct the analytic derivative
        ppo_grad_fix_old = """
                    # Standard PPO Clipping Derivative
                    if adv >= 0:
                        is_clipped = ratio > (1.0 + self.ppo_epsilon)
                    else:
                        is_clipped = ratio < (1.0 - self.ppo_epsilon)
                    
                    # d(log_prob) / d(mean) = (action - mean) / sigma^2
                    d_log_prob_wrt_mean = (action - mean_a_new) / (sigma_new**2 + 1e-8)
                    
                    if is_clipped:
                        d_mean = 0.0 * d_log_prob_wrt_mean  # gradient is zero when clipped
                    else:
                        d_mean = adv * d_log_prob_wrt_mean  # standard policy gradient
"""
        ppo_grad_fix_new = """
                    # Standard PPO Clipping Derivative (Corrected)
                    surr1 = ratio * adv
                    surr2 = np.clip(ratio, 1.0 - self.ppo_epsilon, 1.0 + self.ppo_epsilon) * adv
                    
                    # d(log_prob) / d(mean) = (action - mean) / sigma^2
                    d_log_prob_wrt_mean = (action - mean_a_new) / (sigma_new**2 + 1e-8)
                    
                    if surr1 <= surr2:
                        d_mean = (adv * ratio) * d_log_prob_wrt_mean
                    else:
                        d_mean = 0.0
"""
        source = source.replace(ppo_grad_fix_old.strip(), ppo_grad_fix_new.strip())

        # Section 4: Design Issues
        # Issue 18: Revert Self-loop removal
        source = source.replace('# Removed: np.fill_diagonal(self.W, 1.0)  # Allow controller to learn self-loops', 'np.fill_diagonal(self.W, 1.0)')
        source = source.replace('# Removed: np.fill_diagonal(W, 1.0)', 'np.fill_diagonal(W, 1.0)')

        # Issue 20: Unanimous Round 0 synthetic transition - use W=current_W and halt=1
        synth_transition_old = """
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
"""
        synth_transition_new = """
        if hasattr(config, 'train') and config.train and controller:
            _synth_state = comm_graph.get_graph_state()
            _synth_action = np.zeros(config.n_agents * config.k_dim * 2 + config.n_agents * controller.query_dim + 1)
            _synth_action[-1] = 10.0 # High halt logit
            # We don't have the true action that generated current_W, so we pass zeros for the W logits,
            # but the controller will see the positive reward for halting.
            controller.store_transition(
                state=_synth_state,
                action=_synth_action,
                reward=1.0,  # Positive reward for halting early on easy questions
                value=0.0,
                log_prob=0.0,
                done=True
            )
"""
        source = source.replace(synth_transition_old.strip(), synth_transition_new.strip())

        # Section 5: Minor Issues
        # Issue 25: Revert PlattCalibrator rename
        source = source.replace('IsotonicCalibrator', 'PlattCalibrator')
        
        cell['source'] = [l + '\n' for l in source.split('\n')]
        if cell['source'] and cell['source'][-1] == '\n':
            cell['source'] = cell['source'][:-1]

with open('SparseComm_Kaggle.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
