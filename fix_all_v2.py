"""
Final comprehensive fix script for SparseComm_Kaggle.ipynb v2.
Operates on the restored .bak baseline.
"""
import json
import re

with open('SparseComm_Kaggle.ipynb') as f:
    nb = json.load(f)

fixes_applied = []

for cell in nb.get('cells', []):
    if cell['cell_type'] != 'code':
        continue
    source = "".join(cell['source'])
    original = source

    # =========================================================================
    # C3: GAE Loop — add if is_done guard inside the for loop
    # Baseline has delta/gae inside loop (correct) but no is_done guard.
    # =========================================================================
    old_gae = """\
            mask = 0.0 if is_done else 1.0
            delta       = rewards[t] + self.gamma_discount * next_value * mask - values[t]"""
    new_gae = """\
            mask = 0.0 if is_done else 1.0
            if is_done:
                next_value = 0.0
                gae = 0.0
            delta       = rewards[t] + self.gamma_discount * next_value * mask - values[t]"""
    if old_gae in source:
        source = source.replace(old_gae, new_gae)
        fixes_applied.append("C3: GAE if-is_done guard added")

    # =========================================================================
    # C5/C6: First run_topology_benchmark fixes
    # =========================================================================
    old_load = 'dataset = load_dataset_by_name("gsm8k", config)'
    new_load = '_, dataset = load_dataset_by_name("gsm8k", n_train=0, n_test=config.n_samples, seed=config.seed)'
    if old_load in source:
        source = source.replace(old_load, new_load)
        fixes_applied.append("C6: load_dataset_by_name signature fixed")

    old_run_exp = """\
        results = run_experiment(
            config=config,
            dataset=dataset,
            backend=backend,
            controller=controller,
            method_name=topo
        )"""
    new_run_exp = """\
        results = run_experiment(
            config=config,
            method_name=topo,
            train=False,
            evaluate=True,
            controller=controller
        )"""
    if old_run_exp in source:
        source = source.replace(old_run_exp, new_run_exp)
        fixes_applied.append("C5: run_experiment kwargs fixed (static)")

    old_run_exp2 = """\
        learned_results = run_experiment(
            config=config,
            dataset=dataset,
            backend=backend,
            controller=learned_controller,
            method_name="Learned Topology"
        )"""
    new_run_exp2 = """\
        learned_results = run_experiment(
            config=config,
            method_name="Learned Topology",
            train=False,
            evaluate=True,
            controller=learned_controller
        )"""
    if old_run_exp2 in source:
        source = source.replace(old_run_exp2, new_run_exp2)
        fixes_applied.append("C5: run_experiment kwargs fixed (learned)")

    # =========================================================================
    # Hardcoded Windows paths -> os.path.join (all occurrences)
    # =========================================================================
    source = source.replace(
        r'model_path = r"C:\Users\msaur\OneDrive\Desktop\projectRL\08-06results\gsm8k_pretrained.pt.npz"',
        'model_path = os.path.join("08-06results", "gsm8k_pretrained.pt.npz")')
    source = source.replace(
        r'save_path = r"C:\Users\msaur\OneDrive\Desktop\projectRL\08-06results\topology_benchmark_results.json"',
        'save_path = os.path.join("08-06results", "topology_benchmark_results.json")')

    # =========================================================================
    # M1: _PROMPT_CACHE.clear() — between samples in the training loop
    # =========================================================================
    old_train_call = """\
            for i, sample in enumerate(pbar):
                result = run_single_sample("""
    new_train_call = """\
            for i, sample in enumerate(pbar):
                _PROMPT_CACHE.clear()
                result = run_single_sample("""
    if old_train_call in source and '_PROMPT_CACHE.clear()' not in source:
        source = source.replace(old_train_call, new_train_call)
        fixes_applied.append("M1: _PROMPT_CACHE.clear() between samples")

    # =========================================================================
    # M2: Intermediate rewards — add delta_consensus signal using responses_snapshots
    # =========================================================================
    old_reward = """\
                # Fix A: intermediate reward = cost penalty only.
                # Entropy-reduction shaping removed: rewards wrong-consensus collapse.
                reward_for_step = token_penalty"""
    new_reward = """\
                # Intermediate reward = cost penalty + consensus improvement
                _current_consensus = 0.0
                _prev_consensus = 0.0
                if i + 1 < len(responses_snapshots):
                    _curr_ans = [_normalize_for_vote(r.answer, config.dataset) for r in responses_snapshots[i + 1]]
                    if _curr_ans:
                        from collections import Counter
                        _current_consensus = Counter(_curr_ans).most_common(1)[0][1] / len(_curr_ans)
                if i < len(responses_snapshots):
                    _prev_ans = [_normalize_for_vote(r.answer, config.dataset) for r in responses_snapshots[i]]
                    if _prev_ans:
                        from collections import Counter
                        _prev_consensus = Counter(_prev_ans).most_common(1)[0][1] / len(_prev_ans)
                reward_for_step = token_penalty + 0.5 * (_current_consensus - _prev_consensus)"""
    if old_reward in source:
        source = source.replace(old_reward, new_reward)
        fixes_applied.append("M2: Intermediate rewards with delta_consensus")

    # =========================================================================
    # D1: count_cycles — replace DFS with trace(A^3)//6
    # =========================================================================
    old_cycles_start = "    def count_cycles(self, threshold: float = 0.5) -> int:"
    if old_cycles_start in source and 'def dfs(v):' in source:
        old_method_pattern = r'    def count_cycles\(self, threshold: float = 0\.5\) -> int:.*?return cycle_count'
        new_method = """    def count_cycles(self, threshold: float = 0.5) -> int:
        \"\"\"Count 3-cycles in the communication graph using trace(A^3)/6.\"\"\"
        if threshold is None:
            threshold = 0.25
        adj = (self.W >= threshold).astype(int)
        np.fill_diagonal(adj, 0)
        return int(np.trace(adj @ adj @ adj)) // 6"""
        source = re.sub(old_method_pattern, new_method, source, count=1, flags=re.DOTALL)
        fixes_applied.append("D1: count_cycles -> trace(A^3)//6")

    # =========================================================================
    # D4: Unanimous Round-0 — inject store_transition before early return
    # Build a synthetic meta dict with all required keys.
    # =========================================================================
    old_unan = """\
    if len(_round0_unique) == 1 and list(_round0_unique)[0]:
        final_answer = round0_responses[0].answer
        is_correct = evaluate_answer(final_answer, gold_answer, config.dataset)
        return SampleResult("""
    new_unan = """\
    if len(_round0_unique) == 1 and list(_round0_unique)[0]:
        final_answer = round0_responses[0].answer
        is_correct = evaluate_answer(final_answer, gold_answer, config.dataset)
        if train_controller and controller is not None:
            _synth_meta = {
                'features': np.zeros(controller.input_dim),
                'action': np.zeros(controller.action_dim),
                'log_prob': 0.0,
                'log_std': np.zeros(controller.mean_dim),
                'sigma': np.ones(controller.mean_dim),
                'value': 0.0,
                'actor_cache': None,
                'mean_action': np.zeros(controller.mean_dim),
            }
            controller.store_transition(
                _synth_meta,
                reward=1.0 if is_correct else 0.0,
                done=True
            )
        return SampleResult("""
    if old_unan in source:
        source = source.replace(old_unan, new_unan)
        fixes_applied.append("D4: Unanimous Round-0 stores synthetic transition")

    # =========================================================================
    # N1: HuggingFaceLocalBackend double-escaped newlines
    # =========================================================================
    if '\\\\n\\\\nContext from other agents:' in source:
        source = source.replace('\\\\n\\\\nContext from other agents:\\\\n{context}',
                                '\\n\\nContext from other agents:\\n{context}')
        source = source.replace('\\\\n\\\\nProvide step-by-step',
                                '\\n\\nProvide step-by-step')
        source = source.replace('{system_msg}\\\\n\\\\n{user_msg}',
                                '{system_msg}\\n\\n{user_msg}')
        fixes_applied.append("N1: HF double-escaped newlines fixed")

    # =========================================================================
    # P2/P2b: d_log_std — use (adv * ratio) instead of d_log_prob,
    # remove duplicate entropy_coef definition
    # =========================================================================
    old_ppo_logstd = """\
                    # Gradient of log_std head: entropy-regularised to prevent collapse
                    entropy_coef = 0.01
                    d_log_std    = d_log_prob * ((action - mean_a_new)**2 / (sigma_new**2 + 1e-8) - 1.0)
                    d_log_std   += entropy_coef * np.ones_like(log_std_new)  # entropy bonus"""
    new_ppo_logstd = """\
                    # Gradient of log_std: uses unclipped (adv * ratio), not clipped d_log_prob
                    d_log_std    = (adv * ratio) * ((action - mean_a_new)**2 / (sigma_new**2 + 1e-8) - 1.0)
                    d_log_std   += entropy_coef * np.ones_like(log_std_new)  # entropy bonus"""
    if old_ppo_logstd in source:
        source = source.replace(old_ppo_logstd, new_ppo_logstd)
        fixes_applied.append("P2/P2b: d_log_std uses (adv*ratio), dup entropy_coef removed")

    # =========================================================================
    # P1: PPO actor clipping — min(surr1, surr2) derivative
    # =========================================================================
    old_ppo_actor = """\
                    if adv >= 0:
                        clipped = ratio > (1.0 + self.ppo_epsilon)
                    else:
                        clipped = ratio < (1.0 - self.ppo_epsilon)
                    d_ratio = 0.0 if clipped else adv
                    d_log_prob = d_ratio * ratio
                    d_mean     = d_log_prob * (action - mean_a_new) / (sigma_new ** 2 + 1e-8)"""
    new_ppo_actor = """\
                    # PPO clipping: derivative of min(surr1, surr2)
                    d_log_prob_wrt_mean = (action - mean_a_new) / (sigma_new ** 2 + 1e-8)
                    if surr1 <= surr2:
                        d_mean = adv * d_log_prob_wrt_mean
                    else:
                        d_mean = 0.0"""
    if old_ppo_actor in source:
        source = source.replace(old_ppo_actor, new_ppo_actor)
        fixes_applied.append("P1: PPO actor uses min(surr1,surr2) derivative")

    # =========================================================================
    # NEW-4: t = 0 before the rounds loop — ONLY in run_single_sample
    # Targeted: only the one at 4-space indent (run_single_sample), not 8-space
    # =========================================================================
    old_t_run_single = '\n    for t in range(1, config.n_rounds + 1):\n'
    new_t_run_single = '\n    t = 0\n    for t in range(1, config.n_rounds + 1):\n'
    if 'def run_single_sample(' in source and old_t_run_single in source:
        source = source.replace(old_t_run_single, new_t_run_single, 1)  # only first occurrence
        fixes_applied.append("NEW-4: t=0 in run_single_sample")

    # =========================================================================
    # D6: SPRT _p_maj fallback from 0.7 to 0.9
    # =========================================================================
    if '_p_maj = 0.7  # Conservative fallback' in source:
        source = source.replace('_p_maj = 0.7  # Conservative fallback',
                                '_p_maj = 0.9  # Conservative fallback')
        fixes_applied.append("D6: SPRT _p_maj -> 0.9")

    # Write back if changed
    if source != original:
        cell['source'] = [l + '\n' for l in source.split('\n')]
        if cell['source'] and cell['source'][-1] == '\n':
            cell['source'] = cell['source'][:-1]

# =========================================================================
# D2: Remove SECOND NumpyEncoder definition
# D3a/D3b: Remove FIRST get_topology_weights and run_topology_benchmark
# Strategy: operate within-cell using regex, never delete whole cells.
# =========================================================================
numpy_encoder_count = 0
topo_weights_count = 0
topo_benchmark_count = 0

for cell in nb.get('cells', []):
    if cell['cell_type'] != 'code':
        continue
    source = "".join(cell['source'])
    original = source

    # D2: NumpyEncoder — remove SECOND occurrence
    if 'class NumpyEncoder(json.JSONEncoder):' in source:
        numpy_encoder_count += 1
        if numpy_encoder_count == 2:
            # Find and remove the class + its default method
            pattern = r'\nclass NumpyEncoder\(json\.JSONEncoder\):.*?(?=\n(?:class |def |# ---))'
            source = re.sub(pattern, '\n', source, count=1, flags=re.DOTALL)
            fixes_applied.append("D2: Second NumpyEncoder removed")

    # D3a: get_topology_weights — remove FIRST occurrence
    if 'def get_topology_weights(' in source:
        topo_weights_count += 1
        if topo_weights_count == 1:
            pattern = r'\ndef get_topology_weights\([^)]*\)[^:]*:.*?(?=\ndef )'
            source = re.sub(pattern, '\n', source, count=1, flags=re.DOTALL)
            fixes_applied.append("D3a: First get_topology_weights removed")

    # D3b: run_topology_benchmark — remove FIRST occurrence
    if 'def run_topology_benchmark():' in source:
        topo_benchmark_count += 1
        if topo_benchmark_count == 1:
            pattern = r'\ndef run_topology_benchmark\(\):.*?(?=\ndef |\n# ---)'
            source = re.sub(pattern, '\n', source, count=1, flags=re.DOTALL)
            fixes_applied.append("D3b: First run_topology_benchmark removed")

    if source != original:
        cell['source'] = [l + '\n' for l in source.split('\n')]
        if cell['source'] and cell['source'][-1] == '\n':
            cell['source'] = cell['source'][:-1]

with open('SparseComm_Kaggle.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("=== Fixes Applied ===")
for f in fixes_applied:
    print(f"  ✅ {f}")
print(f"\nTotal: {len(fixes_applied)} fixes")
