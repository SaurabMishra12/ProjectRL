"""
apply_v3_fixes.py
Third-pass fixes: crashes, state encoder, voting calibration, dead code, persistence.
"""
import nbformat, sys, re as _re

NB_PATH = 'C:\\Users\\msaur\\OneDrive\\Desktop\\projectRL\\SparseComm_Kaggle.ipynb'
nb = nbformat.read(NB_PATH, as_version=4)
src = nb.cells[2].source

fixes = []

def apply(label, old, new, required=True):
    global src
    if old in src:
        src = src.replace(old, new, 1)
        fixes.append(f"[OK] {label}")
    elif required:
        fixes.append(f"[MISS] {label}")
    else:
        fixes.append(f"[SKIP] {label}")

# ===========================================================================
# BUG V3-1: Final results table crashes — baseline_results has dicts, not ExperimentResults
# ===========================================================================
apply(
    "V3-1: Fix final results table — use stats dict directly",
    """all_baseline_metrics = []
for method_name, results in baseline_results.items():
    metrics = aggregate_metrics(results)
    all_baseline_metrics.append(metrics)
    print(f"\\n{method_name}:")
    print(f"  Accuracy: {metrics['accuracy_pct']}")
    print(f"  Avg Tokens: {metrics['token_cost']['avg_tokens_per_sample']:.1f}")
    print(f"  Comm Density: {metrics['communication_density']:.3f}")
    print(f"  Error Prop: {metrics['error_propagation_rate']:.3f}")

print("\\n" + format_results_table(all_baseline_metrics, dataset_name="Baselines"))""",
    """# V3-1 fix: baseline_results holds plain stats dicts from run_multi_seed_experiment
# — they have no .sample_results, so aggregate_metrics() would crash.
all_baseline_metrics = []
for method_name, stats in baseline_results.items():
    if hasattr(stats, 'sample_results'):
        # ExperimentResults object — aggregate normally
        metrics = aggregate_metrics(stats)
        all_baseline_metrics.append(metrics)
        print(f"\\n{method_name}:")
        print(f"  Accuracy: {metrics['accuracy_pct']}")
        print(f"  Avg Tokens: {metrics['token_cost']['avg_tokens_per_sample']:.1f}")
        print(f"  Comm Density: {metrics['communication_density']:.3f}")
        print(f"  Error Prop: {metrics['error_propagation_rate']:.3f}")
    else:
        # Plain stats dict from run_multi_seed_experiment
        acc_mean = stats.get('accuracy_mean', 0.0)
        acc_ci   = stats.get('accuracy_95ci', 0.0)
        tok_mean = stats.get('token_cost_mean', 0.0)
        tok_ci   = stats.get('token_cost_95ci', 0.0)
        print(f"\\n{method_name}:")
        print(f"  Accuracy: {acc_mean:.1%} ± {acc_ci:.1%} (95% CI, n={stats.get('n_seeds', '?')} seeds)")
        print(f"  Token Cost: {tok_mean:.1f} ± {tok_ci:.1f}")
        print(f"  All seed accuracies: {[f'{a:.1%}' for a in stats.get('all_accuracies', [])]}")

if all_baseline_metrics:
    print("\\n" + format_results_table(all_baseline_metrics, dataset_name="Baselines"))"""
)

# ===========================================================================
# BUG V3-2: Nvidia backend — create_llm_backend never forwards max_tokens
# ===========================================================================
apply(
    "V3-2: create_llm_backend forwards max_tokens to Nvidia backend",
    "        return _Nv(\n            model=config.api_model,\n            api_key=config.api_key,\n            base_url=config.api_base_url,\n        )",
    "        return _Nv(\n            model=config.api_model,\n            api_key=config.api_key,\n            base_url=config.api_base_url,\n            max_tokens=getattr(config, 'max_tokens', 512),  # V3-2 fix\n        )"
)

# ===========================================================================
# BUG V3-3: SemanticStateEncoder.encode_state uses raw un-normalized answers
# ===========================================================================
apply(
    "V3-3: encode_state normalizes answers before entropy calculation",
    "        answers = [str(s.final_answer).lower().strip() for s in structured_states.values()]",
    "        # V3-3 fix: normalize answers before entropy so '24'/'24.0'/'24.00' collapse to one key\n        _raw_answers = [str(s.final_answer).lower().strip() for s in structured_states.values()]\n        _dataset_type = getattr(self, '_dataset_type', 'gsm8k')\n        answers = [_normalize_for_vote(a, _dataset_type) for a in _raw_answers]"
)
# Add _dataset_type setter to SemanticStateEncoder (next to wherever it is initialized)
apply(
    "V3-3b: IBController sets encoder._dataset_type when dataset changes",
    "        dataset_type = getattr(self.config, 'dataset', 'gsm8k')\n        structured_states = {i: self.parser.parse(reasoning_texts.get(i, \"\"), answers.get(i, \"\"), dataset_type) for i in range(self.n_agents)}  # Tier4 #20",
    "        dataset_type = getattr(self.config, 'dataset', 'gsm8k')\n        # V3-3 fix: propagate dataset_type to encoder for answer normalization\n        self.semantic_encoder._dataset_type = dataset_type\n        # V3-3b: Normalize answers before building structured states (fixes state encoder entropy)\n        _norm_answers = {i: _normalize_for_vote(answers.get(i, ''), dataset_type) for i in range(self.n_agents)}\n        structured_states = {i: self.parser.parse(reasoning_texts.get(i, ''), _norm_answers.get(i, ''), dataset_type) for i in range(self.n_agents)}  # Tier4 #20"
)

# ===========================================================================
# BUG V3-3c: Unanimous early-exit check uses raw answers — same normalization gap
# ===========================================================================
apply(
    "V3-3c: Early exit unanimous check normalizes answers",
    "    _round0_unique = set(r.answer for r in round0_responses)",
    "    # V3-3c fix: normalize before uniqueness check — '24' and '24.0' are the same answer\n    _round0_unique = set(_normalize_for_vote(r.answer, config.dataset) for r in round0_responses)"
)

# ===========================================================================
# BUG V3-4: contradiction_score is semantically inverted — rename + fix formula
# Replace with answer_diversity (fraction unique) and answer_length_variance (distinct feature)
# ===========================================================================
apply(
    "V3-4: Fix inverted contradiction_score → answer_diversity + length_variance",
    "        # 3. Contradiction Score (answer disagreement ratio)\n        unique_answers = len(set(answers))\n        contradiction_score = 1.0 - (unique_answers / max(len(answers), 1))\n\n        # 4. Novelty Score (vocabulary richness across agents)\n        all_words: set = set()\n        for t in texts:\n            all_words.update(t.lower().split())\n        avg_words_per_agent = float(np.mean([len(t.split()) for t in texts])) if texts else 0\n        novelty_score = len(all_words) / max(avg_words_per_agent * len(texts), 1)\n        novelty_score = min(1.0, novelty_score / 2.0)  # Normalize to [0,1]\n\n        # 5. Consensus Strength\n        if unique_answers == 1:\n            consensus_strength = 1.0\n        else:\n            answer_counts = Counter(answers)\n            majority_count = answer_counts.most_common(1)[0][1]\n            consensus_strength = majority_count / len(answers)\n\n        features = np.array([\n            shannon_entropy,\n            jaccard_similarity,\n            contradiction_score,\n            novelty_score,\n            consensus_strength,\n        ], dtype=np.float64)",
    "        # 3. Answer Diversity (fraction of answers that are unique — V3-4 fix: was inverted)\n        unique_answers = len(set(answers))\n        # HIGH value = HIGH disagreement (correct semantics)\n        answer_diversity = unique_answers / max(len(answers), 1)\n\n        # 4. Answer Length Variance (proxy for reasoning confidence heterogeneity)\n        # Replaces novelty_score which was largely redundant with consensus_strength.\n        if texts:\n            lengths = np.array([len(t.split()) for t in texts], dtype=float)\n            mean_len = lengths.mean()\n            length_variance = float(np.std(lengths) / (mean_len + 1e-8))  # CoV, clamped to [0,1]\n            length_variance = min(1.0, length_variance)\n        else:\n            length_variance = 0.0\n\n        # 5. Consensus Strength\n        if unique_answers == 1:\n            consensus_strength = 1.0\n        else:\n            answer_counts = Counter(answers)\n            majority_count = answer_counts.most_common(1)[0][1]\n            consensus_strength = majority_count / len(answers)\n\n        features = np.array([\n            shannon_entropy,\n            jaccard_similarity,\n            answer_diversity,    # V3-4 fix: was contradiction_score (inverted)\n            length_variance,     # V3-4 new: replaces novelty_score (was redundant)\n            consensus_strength,\n        ], dtype=np.float64)"
)
# Fix the feature names in the array comment reference (contradiction_score variable name used in log)
apply(
    "V3-4b: Fix variable name reference contradiction_score in features",
    "        return np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)",
    "        return np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)  # [entropy, jaccard, diversity, len_var, consensus]"
)

# ===========================================================================
# BUG V3-5: Voting calibration asymmetry — run_full_communication and run_no_communication
# don't get calibrator, making IB-Controller's vote artificially better
# Fix: pass calibrator as optional arg to both baselines
# ===========================================================================
apply(
    "V3-5a: run_full_communication accepts optional calibrator",
    "def run_full_communication(config: Config) -> ExperimentResults:\n    \"\"\"\n    Run multi-agent debate baseline: full communication (W=1.0 everywhere).\n    \"\"\"",
    "def run_full_communication(config: Config, calibrator: Optional['PlattCalibrator'] = None) -> ExperimentResults:\n    \"\"\"\n    Run multi-agent debate baseline: full communication (W=1.0 everywhere).\n    V3-5 fix: accept calibrator so voting quality is comparable to IB-Controller.\n    \"\"\""
)
apply(
    "V3-5b: run_full_communication passes calibrator to vote",
    "        final = weighted_majority_vote(current, dataset_type=config.dataset)\n        is_correct = evaluate_answer(final, sample[\"answer\"], config.dataset)\n\n        sr = SampleResult(\n            question=sample[\"question\"],\n            gold_answer=sample[\"answer\"],\n            final_answer=final,\n            is_correct=is_correct,\n            initial_answers=initial_answers,\n            initial_correct=initial_correct,\n            final_answers={r.agent_id: r.answer for r in current},\n            total_tokens_transmitted=comm_graph.get_total_tokens_transmitted(),",
    "        final = weighted_majority_vote(current, dataset_type=config.dataset, calibrator=calibrator)  # V3-5 fix\n        is_correct = evaluate_answer(final, sample[\"answer\"], config.dataset)\n\n        sr = SampleResult(\n            question=sample[\"question\"],\n            gold_answer=sample[\"answer\"],\n            final_answer=final,\n            is_correct=is_correct,\n            initial_answers=initial_answers,\n            initial_correct=initial_correct,\n            final_answers={r.agent_id: r.answer for r in current},\n            total_tokens_transmitted=comm_graph.get_total_tokens_transmitted(),"
)
apply(
    "V3-5c: run_no_communication accepts optional calibrator",
    "def run_no_communication(config: Config) -> ExperimentResults:\n    \"\"\"\n    Run no-communication baseline: agents solve independently with zero communication.\n    \"\"\"",
    "def run_no_communication(config: Config, calibrator: Optional['PlattCalibrator'] = None) -> ExperimentResults:\n    \"\"\"\n    Run no-communication baseline: agents solve independently with zero communication.\n    V3-5 fix: accept calibrator for fair vote comparison.\n    \"\""
)
apply(
    "V3-5d: run_no_communication passes calibrator to vote",
    "        final = weighted_majority_vote(r0, dataset_type=config.dataset)\n        is_correct = evaluate_answer(final, sample[\"answer\"], config.dataset)",
    "        final = weighted_majority_vote(r0, dataset_type=config.dataset, calibrator=calibrator)  # V3-5 fix\n        is_correct = evaluate_answer(final, sample[\"answer\"], config.dataset)"
)

# ===========================================================================
# BUG V3-6: numeric_median_vote is dead code — integrate into weighted_majority_vote
# For GSM8K when ALL answers are numeric, use weighted median instead of Bayesian mode
# ===========================================================================
apply(
    "V3-6: Integrate numeric_median_vote into weighted_majority_vote as fallback",
    "    # Numerically stable argmax\n    best_key = max(log_posterior, key=log_posterior.get)\n    return canonical_map[best_key]",
    "    # Numerically stable argmax\n    best_key = max(log_posterior, key=log_posterior.get)\n\n    # V3-6: For GSM8K, if every answer is a parseable number, use L1-optimal weighted median.\n    # The Bayesian mode can get stuck when agents produce close-but-not-identical numbers\n    # (e.g. 47, 48, 49). The weighted median minimises expected L1 error.\n    if dataset_type == 'gsm8k':\n        _numeric_entries = []\n        for key, r_item in zip(normalized, responses):\n            nums = re.findall(r'-?\\d+\\.?\\d*', key)\n            if nums:\n                try:\n                    _numeric_entries.append((float(nums[-1]), confs[responses.index(r_item)], canonical_map[key]))\n                except (ValueError, IndexError):\n                    pass\n        if len(_numeric_entries) == len(responses) and len(_numeric_entries) > 0:\n            _vals = np.array([v for v, _, _ in _numeric_entries])\n            _wts  = np.array([w for _, w, _ in _numeric_entries])\n            _wts  = _wts / (_wts.sum() + 1e-12)\n            _idx  = np.argsort(_vals)\n            _cumw = np.cumsum(_wts[_idx])\n            _med_idx = int(np.searchsorted(_cumw, 0.5))\n            return _numeric_entries[_idx[min(_med_idx, len(_idx) - 1)]][2]\n\n    return canonical_map[best_key]"
)

# ===========================================================================
# BUG V3-7: Calibrator not persisted in save_model/load_model
# ===========================================================================
apply(
    "V3-7a: save_model persists calibrator parameters",
    "    def save_model(self, filepath: str):\n        \"\"\"Save the actor and critic MLP weights to a .npz file.\"\"\"\n        import os\n        os.makedirs(os.path.dirname(filepath), exist_ok=True)\n        np.savez(filepath,\n                 actor_W1=self.actor.W1, actor_b1=self.actor.b1,\n                 actor_W2=self.actor.W2, actor_b2=self.actor.b2,\n                 actor_W3=self.actor.W3, actor_b3=self.actor.b3,\n                 critic_W1=self.critic.W1, critic_b1=self.critic.b1,\n                 critic_W2=self.critic.W2, critic_b2=self.critic.b2,\n                 critic_W3=self.critic.W3, critic_b3=self.critic.b3,\n                 # Save mean_dim so load_model can validate architecture compatibility\n                 mean_dim=np.array(self.mean_dim))",
    "    def save_model(self, filepath: str):\n        \"\"\"Save the actor and critic MLP weights to a .npz file.\"\"\"\n        import os\n        os.makedirs(os.path.dirname(filepath), exist_ok=True)\n        np.savez(filepath,\n                 actor_W1=self.actor.W1, actor_b1=self.actor.b1,\n                 actor_W2=self.actor.W2, actor_b2=self.actor.b2,\n                 actor_W3=self.actor.W3, actor_b3=self.actor.b3,\n                 critic_W1=self.critic.W1, critic_b1=self.critic.b1,\n                 critic_W2=self.critic.W2, critic_b2=self.critic.b2,\n                 critic_W3=self.critic.W3, critic_b3=self.critic.b3,\n                 # Save mean_dim so load_model can validate architecture compatibility\n                 mean_dim=np.array(self.mean_dim),\n                 # V3-7 fix: persist Platt calibrator so test-time voting matches train-time\n                 calibrator_a=np.array(self.calibrator.a),\n                 calibrator_b=np.array(self.calibrator.b),\n                 calibrator_fitted=np.array(self.calibrator._fitted))"
)
apply(
    "V3-7b: load_model restores calibrator parameters",
    "        self.critic.W3 = data['critic_W3']\n        self.critic.b3 = data['critic_b3']",
    "        self.critic.W3 = data['critic_W3']\n        self.critic.b3 = data['critic_b3']\n        # V3-7 fix: restore Platt calibrator if present in checkpoint\n        if 'calibrator_a' in data:\n            self.calibrator.a = float(data['calibrator_a'])\n            self.calibrator.b = float(data['calibrator_b'])\n            self.calibrator._fitted = bool(data['calibrator_fitted'])"
)

# ===========================================================================
# BUG V3-8: Dead misleading comment block in update_policy — remove it
# ===========================================================================
apply(
    "V3-8: Remove dead misleading comment in update_policy",
    "        # Tier4 #22: Update EMA topology prior from observed W distribution\n        if self.saved_transitions:  # Was cleared above — use pre-cleared transitions indirectly\n            pass  # Prior is updated in store_transition context below\n        self.saved_transitions = []",
    "        # Tier4 #22: EMA prior updated once per epoch in run_experiment (not here).\n        self.saved_transitions = []"
)

# ===========================================================================
# BUG V3-9: run_full_communication lacks unanimous Round-0 early exit
# Add same consensus check as run_single_sample for fair token comparison
# ===========================================================================
apply(
    "V3-9: Add unanimous early exit to run_full_communication",
    "        current = r0\n        for t in range(1, config.n_rounds + 1):\n            comm_graph.set_weights(full_W)\n            texts = {r.agent_id: r.reasoning_text for r in current}",
    "        current = r0\n\n        # V3-9 fix: add same unanimous early exit as run_single_sample for fair comparison.\n        # Without this, debate baseline wastes tokens on questions where all agents already agree.\n        _debate_unique = set(_normalize_for_vote(r.answer, config.dataset) for r in r0)\n        if len(_debate_unique) == 1 and list(_debate_unique)[0]:\n            final = weighted_majority_vote(current, dataset_type=config.dataset, calibrator=calibrator)\n            is_correct = evaluate_answer(final, sample[\"answer\"], config.dataset)\n            sr = SampleResult(\n                question=sample[\"question\"],\n                gold_answer=sample[\"answer\"],\n                final_answer=final,\n                is_correct=is_correct,\n                initial_answers=initial_answers,\n                initial_correct=initial_correct,\n                final_answers={r.agent_id: r.answer for r in r0},\n                total_tokens_transmitted=0,\n                total_tokens_if_full=0,\n                compression_ratio=0.0,\n                weight_matrices=[full_W.copy()],\n                communication_density=0.0,\n                mode_distribution={mode.name: 0 for mode in SemanticBandwidthMode if mode != SemanticBandwidthMode.BLOCK},\n                n_active_edges=0,\n                generation_tokens=total_gen,\n                halt_round=0,\n            )\n            results.sample_results.append(sr)\n            log_continuous_result(\"Multi-Agent Debate\", config.dataset, sr)\n            continue\n\n        for t in range(1, config.n_rounds + 1):\n            comm_graph.set_weights(full_W)\n            texts = {r.agent_id: r.reasoning_text for r in current}"
)

# ===========================================================================
# BUG V3-10: Bayesian vote when n_options artificially floored to 2
# Add unanimous fast-path before Bayesian computation
# ===========================================================================
apply(
    "V3-10: Unanimous fast-path in weighted_majority_vote before Bayesian step",
    "    # Step 5: Bayesian multiplicative voting (Improvement II)\n    unique_answers = list(set(normalized))\n    n_options = max(len(unique_answers), 2)",
    "    # Step 5: Bayesian multiplicative voting (Improvement II)\n    unique_answers = list(set(normalized))\n    # V3-10 fix: unanimous fast-path avoids artificial n_options=2 floor for single-answer case\n    if len(unique_answers) == 1:\n        return canonical_map[unique_answers[0]]\n    n_options = max(len(unique_answers), 2)"
)

# ===========================================================================
# Write back
# ===========================================================================
nb.cells[2].source = src
nbformat.write(nb, NB_PATH)

print("\n" + "="*60)
print("V3 FIX SUMMARY")
print("="*60)
for line in fixes:
    print(line)
ok   = len([l for l in fixes if l.startswith("[OK]")])
miss = len([l for l in fixes if l.startswith("[MISS]")])
skip = len([l for l in fixes if l.startswith("[SKIP]")])
print(f"\nTotal: {len(fixes)} | OK: {ok} | Miss: {miss} | Skip: {skip}")

# Syntax check
import py_compile
with open('temp_check.py', 'w', encoding='utf-8') as f:
    f.write(src)
try:
    py_compile.compile('temp_check.py', doraise=True)
    print("\n[OK] Syntax check passed!")
except py_compile.PyCompileError as e:
    print(f"\n[FAIL] Syntax error: {e}")
    sys.exit(1)
