"""
Final comprehensive fix script for SparseComm_Kaggle.ipynb.
Operates on the restored .bak baseline (225KB).
Each fix is surgical and well-documented.
"""
import json
import re

with open('SparseComm_Kaggle.ipynb') as f:
    nb = json.load(f)

# Track which fixes succeeded
fixes_applied = []

for cell in nb.get('cells', []):
    if cell['cell_type'] != 'code':
        continue
    source = "".join(cell['source'])
    original = source  # for tracking changes

    # =========================================================================
    # C3: GAE Loop — add if is_done guard inside the for loop
    # The baseline has delta/gae/advantages INSIDE the loop (correct),
    # but lacks the if is_done: next_value=0.0; gae=0.0 guard.
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
    # C5/C6: First run_topology_benchmark — fix load_dataset_by_name call
    # and run_experiment kwargs.
    # The FIRST definition passes config object and has dataset/backend kwargs.
    # =========================================================================
    # Fix load_dataset_by_name("gsm8k", config) -> proper signature
    old_load = 'dataset = load_dataset_by_name("gsm8k", config)'
    new_load = '_, dataset = load_dataset_by_name("gsm8k", n_train=0, n_test=config.n_samples, seed=config.seed)'
    if old_load in source:
        source = source.replace(old_load, new_load)
        fixes_applied.append("C6: load_dataset_by_name signature fixed")

    # Fix run_experiment(config=config, dataset=dataset, backend=backend, ...)
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
        fixes_applied.append("C5: run_experiment kwargs fixed (static topo)")

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
        fixes_applied.append("C5: run_experiment kwargs fixed (learned topo)")

    # =========================================================================
    # C1/C2: Hardcoded Windows paths in topology benchmark
    # =========================================================================
    source = source.replace(
        r'model_path = r"C:\Users\msaur\OneDrive\Desktop\projectRL\08-06results\gsm8k_pretrained.pt.npz"',
        'model_path = os.path.join("08-06results", "gsm8k_pretrained.pt.npz")')
    source = source.replace(
        r'save_path = r"C:\Users\msaur\OneDrive\Desktop\projectRL\08-06results\topology_benchmark_results.json"',
        'save_path = os.path.join("08-06results", "topology_benchmark_results.json")')
    if source != original and "model_path = os.path.join" in source:
        fixes_applied.append("C1: Windows paths replaced with os.path.join")

    # =========================================================================
    # M1: _PROMPT_CACHE.clear() — in run_experiment training loop,
    # BEFORE each run_single_sample call (between samples, not within)
    # =========================================================================
    old_train_call = """\
            for i, sample in enumerate(pbar):
                result = run_single_sample("""
    new_train_call = """\
            for i, sample in enumerate(pbar):
                _PROMPT_CACHE.clear()  # Clear cache between samples to prevent cross-sample leakage
                result = run_single_sample("""
    if old_train_call in source and '_PROMPT_CACHE.clear()' not in source:
        source = source.replace(old_train_call, new_train_call)
        fixes_applied.append("M1: _PROMPT_CACHE.clear() added before each sample in training loop")

    # =========================================================================
    # M2: Intermediate rewards — add delta_consensus signal
    # The baseline has: reward_for_step = token_penalty (cost only)
    # We add: + 0.5 * (current_consensus - prev_consensus) using responses_snapshots
    # =========================================================================
    old_reward = """\
                # Fix A: intermediate reward = cost penalty only.
                # Entropy-reduction shaping removed: rewards wrong-consensus collapse.
                reward_for_step = token_penalty"""
    new_reward = """\
                # Intermediate reward = cost penalty + consensus improvement signal
                _current_consensus = 0.0
                _prev_consensus = 0.0
                # responses_snapshots[i] = round i responses; meta i spans round i -> i+1
                if i + 1 < len(responses_snapshots):
                    _curr_answers = [_normalize_for_vote(r.answer, config.dataset) for r in responses_snapshots[i + 1]]
                    if _curr_answers:
                        from collections import Counter
                        _current_consensus = Counter(_curr_answers).most_common(1)[0][1] / len(_curr_answers)
                if i < len(responses_snapshots):
                    _prev_answers = [_normalize_for_vote(r.answer, config.dataset) for r in responses_snapshots[i]]
                    if _prev_answers:
                        from collections import Counter
                        _prev_consensus = Counter(_prev_answers).most_common(1)[0][1] / len(_prev_answers)
                reward_for_step = token_penalty + 0.5 * (_current_consensus - _prev_consensus)"""
    if old_reward in source:
        source = source.replace(old_reward, new_reward)
        fixes_applied.append("M2: Intermediate rewards now include delta_consensus signal")

    # =========================================================================
    # D1: count_cycles — replace DFS with trace-based 3-cycle count
    # =========================================================================
    old_cycles_start = "    def count_cycles(self, threshold: float = 0.5) -> int:"
    if old_cycles_start in source and 'def dfs(v):' in source:
        # Replace the whole method
        old_method_pattern = r'    def count_cycles\(self, threshold: float = 0\.5\) -> int:.*?return cycle_count'
        new_method = """    def count_cycles(self, threshold: float = 0.5) -> int:
        \"\"\"Count 3-cycles in the communication graph.
        Uses trace(A^3)/6 which is O(N^3) and exact for 3-cycles.
        \"\"\"
        if threshold is None:
            threshold = 0.25
        adj = (self.W >= threshold).astype(int)
        np.fill_diagonal(adj, 0)
        return int(np.trace(adj @ adj @ adj)) // 6"""
        source = re.sub(old_method_pattern, new_method, source, flags=re.DOTALL)
        fixes_applied.append("D1: count_cycles replaced with trace(A^3)//6")

    # =========================================================================
    # D4: Unanimous Round-0 — inject store_transition before early return
    # =========================================================================
    old_unanimous = """\
    if len(_round0_unique) == 1 and list(_round0_unique)[0]:
        final_answer = round0_responses[0].answer
        is_correct = evaluate_answer(final_answer, gold_answer, config.dataset)
        return SampleResult("""
    new_unanimous = """\
    if len(_round0_unique) == 1 and list(_round0_unique)[0]:
        final_answer = round0_responses[0].answer
        is_correct = evaluate_answer(final_answer, gold_answer, config.dataset)
        # D4 fix: teach controller to halt immediately on unanimous easy questions
        if train_controller and controller is not None:
            _synth_state = comm_graph.get_graph_state()
            _synth_reward = 1.0 if is_correct else 0.0
            controller.store_transition(
                _synth_state,           # state (current graph features)
                _synth_reward,          # reward (task outcome)
                done=True               # terminal — sample is finished
            )
        return SampleResult("""
    if old_unanimous in source:
        source = source.replace(old_unanimous, new_unanimous)
        fixes_applied.append("D4: Unanimous Round-0 now stores transition before return")

    # =========================================================================
    # D5: SetTransformerEncoder RNG — not present in baseline (no RandomState)
    # The user mentioned line 2464 but the baseline doesn't have it.
    # Skip if not found.
    # =========================================================================
    if 'rng = np.random.RandomState(seed=None)' in source:
        source = source.replace('rng = np.random.RandomState(seed=None)',
                                'rng = np.random.RandomState(seed=42)')
        fixes_applied.append("D5: SetTransformer RNG seed fixed")

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
        fixes_applied.append("N1: HF backend double-escaped newlines fixed")

    # =========================================================================
    # P2/P2b: PPO log_std gradient
    # 1. Remove the second entropy_coef = 0.01 (redundant shadow)
    # 2. Fix d_log_std to use (adv * ratio) instead of d_log_prob
    # 3. Add entropy_coef to d_log_std
    # =========================================================================
    old_ppo_logstd = """\
                    # Gradient of log_std head: entropy-regularised to prevent collapse
                    entropy_coef = 0.01
                    d_log_std    = d_log_prob * ((action - mean_a_new)**2 / (sigma_new**2 + 1e-8) - 1.0)
                    d_log_std   += entropy_coef * np.ones_like(log_std_new)  # entropy bonus"""
    new_ppo_logstd = """\
                    # Gradient of log_std head: uses unclipped (adv * ratio), not d_log_prob
                    d_log_std    = (adv * ratio) * ((action - mean_a_new)**2 / (sigma_new**2 + 1e-8) - 1.0)
                    d_log_std   += entropy_coef * np.ones_like(log_std_new)  # entropy bonus"""
    if old_ppo_logstd in source:
        source = source.replace(old_ppo_logstd, new_ppo_logstd)
        fixes_applied.append("P2/P2b: d_log_std uses (adv*ratio), duplicate entropy_coef removed")

    # =========================================================================
    # P1: PPO actor clipping — replace heuristic with min(surr1,surr2) derivative
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
                        # Unclipped branch: gradient flows through ratio * adv
                        d_mean = adv * d_log_prob_wrt_mean
                    else:
                        # Clipped branch: clip(ratio) is constant w.r.t. params
                        d_mean = 0.0"""
    if old_ppo_actor in source:
        source = source.replace(old_ppo_actor, new_ppo_actor)
        fixes_applied.append("P1: PPO actor clipping uses min(surr1,surr2) derivative")

    # =========================================================================
    # NEW-4: Initialize t=0 before the rounds loop
    # =========================================================================
    old_t_loop = '    for t in range(1, config.n_rounds + 1):'
    new_t_loop = '    t = 0\n    for t in range(1, config.n_rounds + 1):'
    if old_t_loop in source and '    t = 0\n    for t in range(1, config.n_rounds + 1):' not in source:
        source = source.replace(old_t_loop, new_t_loop)
        fixes_applied.append("NEW-4: t=0 initialized before rounds loop")

    # =========================================================================
    # D2: NumpyEncoder — deduplicate (keep FIRST, remove SECOND)
    # We handle this by counting occurrences within the same source
    # =========================================================================
    # (handled at cell level below)

    # =========================================================================
    # D3a/D3b: Duplicate get_topology_weights / run_topology_benchmark
    # The FIRST definition (with bugs) and SECOND (correct) are in separate cells.
    # We keep the SECOND (which we also patched above for C5/C6).
    # We'll handle this at cell level.
    # =========================================================================

    # =========================================================================
    # D6: SPRT _p_maj fallback
    # =========================================================================
    if '_p_maj = 0.7  # Conservative fallback' in source:
        source = source.replace('_p_maj = 0.7  # Conservative fallback',
                                '_p_maj = 0.9  # Conservative fallback — prevents premature halting before calibrator is fitted')
        fixes_applied.append("D6: SPRT _p_maj raised from 0.7 to 0.9")

    # Write back
    if source != original:
        cell['source'] = [l + '\n' for l in source.split('\n')]
        if cell['source'] and cell['source'][-1] == '\n':
            cell['source'] = cell['source'][:-1]

# =========================================================================
# D2: Remove the SECOND NumpyEncoder definition (keep first, remove second)
# D3a/D3b: Remove the FIRST get_topology_weights and run_topology_benchmark
# definitions (they have the C5/C6 bugs). Keep the SECOND definitions.
# =========================================================================
# Strategy: Track which cell contains the FIRST vs SECOND definitions.
# For NumpyEncoder: keep first, remove second.
# For get_topology_weights/run_topology_benchmark: keep second, remove first.

numpy_encoder_count = 0
topo_weights_count = 0
topo_benchmark_count = 0

for cell in nb.get('cells', []):
    if cell['cell_type'] != 'code':
        continue
    source = "".join(cell['source'])

    # D2: NumpyEncoder — remove SECOND occurrence
    if 'class NumpyEncoder(json.JSONEncoder):' in source:
        numpy_encoder_count += 1
        if numpy_encoder_count == 2:
            # Remove only the NumpyEncoder class from this cell
            # Find the class and its method
            pattern = r'class NumpyEncoder\(json\.JSONEncoder\):.*?(?=\nclass |\ndef |\n[A-Z]|\Z)'
            source = re.sub(pattern, '', source, count=1, flags=re.DOTALL)
            cell['source'] = [l + '\n' for l in source.split('\n')]
            if cell['source'] and cell['source'][-1] == '\n':
                cell['source'] = cell['source'][:-1]
            fixes_applied.append("D2: Second NumpyEncoder definition removed")

    # D3a: get_topology_weights — remove FIRST occurrence
    if 'def get_topology_weights(' in source:
        topo_weights_count += 1
        if topo_weights_count == 1:
            # Remove the function
            pattern = r'def get_topology_weights\(.*?\n(?=def |\nclass |\n# ---|\Z)'
            source = re.sub(pattern, '', source, count=1, flags=re.DOTALL)
            cell['source'] = [l + '\n' for l in source.split('\n')]
            if cell['source'] and cell['source'][-1] == '\n':
                cell['source'] = cell['source'][:-1]
            fixes_applied.append("D3a: First get_topology_weights definition removed")

    # D3b: run_topology_benchmark — remove FIRST occurrence
    if 'def run_topology_benchmark():' in source:
        topo_benchmark_count += 1
        if topo_benchmark_count == 1:
            # Remove the function
            pattern = r'def run_topology_benchmark\(\):.*?\n(?=def |\nclass |\n# ---|\Z)'
            source = re.sub(pattern, '', source, count=1, flags=re.DOTALL)
            cell['source'] = [l + '\n' for l in source.split('\n')]
            if cell['source'] and cell['source'][-1] == '\n':
                cell['source'] = cell['source'][:-1]
            fixes_applied.append("D3b: First run_topology_benchmark definition removed")


with open('SparseComm_Kaggle.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("=== Fixes Applied ===")
for f in fixes_applied:
    print(f"  ✅ {f}")
print(f"\nTotal: {len(fixes_applied)} fixes")
