"""
apply_v4_fixes.py
Fourth-pass fixes: prompt wiring, reward double-counting, voting fairness,
HotpotQA EM evaluation, Phase1->Phase2 transfer, transmit final_answer, minor issues.
"""
import nbformat, sys, py_compile

NB_PATH = r'C:\Users\msaur\OneDrive\Desktop\projectRL\SparseComm_Kaggle.ipynb'

with open('temp_check.py', 'r', encoding='utf-8') as f:
    src = f.read()

fixes = []

def apply(label, old, new, required=True):
    global src
    if old in src:
        src = src.replace(old, new, 1)
        fixes.append(('OK', label))
    elif required:
        fixes.append(('MISS', label))
    else:
        fixes.append(('SKIP', label))

# ===========================================================================
# FIX 1: Wire agent's reasoning_style prompt into both backends as the system message.
# The `prompt` arg is the fully built prompt from _build_prompt.
# Backends reconstruct user_msg from scratch (good) but the reasoning-style instruction
# in `prompt` should be injected as the leading system context.
# Simplest correct fix: pass `prompt` as the user message directly (it already includes
# the question, context, style instruction, and task description).
# The backend's _get_system_prompt still provides the format/output spec.
# ===========================================================================

# Nvidia backend: replace hardcoded user_msg construction with the agent's prompt
apply(
    "Fix1a: Nvidia backend uses agent prompt as user_msg",
    '        system_msg = self._get_system_prompt(dataset_type, agent_id)\n        user_msg = f"Question: {question}"\n        if context:\n            user_msg += f"\\n\\nContext from other agents:\\n{context}"\n        if dataset_type in ["math", "truthfulqa"]:\n            user_msg += "\\n\\nProvide step-by-step reasoning, ensuring your final answer is boxed as requested."\n        else:\n            user_msg += "\\n\\nProvide step-by-step reasoning, then give your final answer on the last line as: Answer: <your answer>"\n\n        messages = [\n            {"role": "system", "content": system_msg},\n            {"role": "user", "content": user_msg},\n        ]',
    '        # Fix1: Use agent\'s pre-built prompt (includes reasoning style) as user message.\n        # system_msg provides the output format spec; prompt provides the task + style instruction.\n        system_msg = self._get_system_prompt(dataset_type, agent_id)\n        if prompt and prompt.strip():\n            # Agent._build_prompt already includes question, context, and style instruction\n            user_msg = prompt\n        else:\n            # Fallback if called without a pre-built prompt\n            user_msg = f"Question: {question}"\n            if context:\n                user_msg += f"\\n\\nContext from other agents:\\n{context}"\n        if dataset_type in ["math", "truthfulqa"]:\n            user_msg += "\\n\\nProvide step-by-step reasoning, ensuring your final answer is boxed as requested."\n        else:\n            user_msg += "\\n\\nProvide step-by-step reasoning, then give your final answer on the last line as: Answer: <your answer>"\n\n        messages = [\n            {"role": "system", "content": system_msg},\n            {"role": "user", "content": user_msg},\n        ]'
)

# Azure backend: same fix
apply(
    "Fix1b: Azure backend uses agent prompt as user_msg",
    '        system_msg = self._get_system_prompt(dataset_type, agent_id)\n        user_msg = f"Question: {question}"\n        if context:\n            user_msg += f"\\n\\nContext from other agents:\\n{context}"\n        if dataset_type in ["math", "truthfulqa"]:\n            user_msg += "\\n\\nProvide step-by-step reasoning, ensuring your final answer is boxed as requested."\n        else:\n            user_msg += "\\n\\nProvide step-by-step reasoning, then give your final answer on the last line as: Answer: <your answer>"\n\n        messages = [\n            {"role": "system", "content": system_msg},\n            {"role": "user", "content": user_msg},\n        ]',
    '        # Fix1: Use agent\'s pre-built prompt (includes reasoning style) as user message.\n        system_msg = self._get_system_prompt(dataset_type, agent_id)\n        if prompt and prompt.strip():\n            user_msg = prompt\n        else:\n            user_msg = f"Question: {question}"\n            if context:\n                user_msg += f"\\n\\nContext from other agents:\\n{context}"\n        if dataset_type in ["math", "truthfulqa"]:\n            user_msg += "\\n\\nProvide step-by-step reasoning, ensuring your final answer is boxed as requested."\n        else:\n            user_msg += "\\n\\nProvide step-by-step reasoning, then give your final answer on the last line as: Answer: <your answer>"\n\n        messages = [\n            {"role": "system", "content": system_msg},\n            {"role": "user", "content": user_msg},\n        ]'
)

# ===========================================================================
# FIX 2: Reward double-counting — token cost applied in both intermediate steps AND
# the final compute_reward call.  Fix: zero out info_cost in compute_reward when
# called for the final step, since intermediate steps already penalised per-round tokens.
# The cleanest fix: remove -lambda_info * info_cost from compute_reward entirely
# (it was meant to be the final-step total penalty) and track it only in intermediate steps.
# To preserve backward compatibility, we add a parameter `include_token_penalty`
# so the call site can explicitly opt out.
# ===========================================================================
apply(
    "Fix2a: compute_reward gains include_token_penalty flag",
    '    def compute_reward(\n        self,\n        task_correct: bool,\n        tokens_transmitted: int,\n        tokens_if_full: int,\n        compression_ratio: float,\n        n_cycles: int,\n        current_W: np.ndarray,\n        halt_round: int = 0,\n        confidence_improvement: float = 0.0,\n        answer_entropy_reduction: float = 0.0,\n        is_final_step: bool = True,\n    ) -> float:',
    '    def compute_reward(\n        self,\n        task_correct: bool,\n        tokens_transmitted: int,\n        tokens_if_full: int,\n        compression_ratio: float,\n        n_cycles: int,\n        current_W: np.ndarray,\n        halt_round: int = 0,\n        confidence_improvement: float = 0.0,\n        answer_entropy_reduction: float = 0.0,\n        is_final_step: bool = True,\n        include_token_penalty: bool = False,  # Fix2: False by default — intermediate steps handle per-round tokens\n    ) -> float:'
)
apply(
    "Fix2b: compute_reward skips info_cost when include_token_penalty=False",
    '        # Information cost: fraction of maximum possible communication\n        # Use tokens_if_full as the natural baseline (what FullComm would transmit)\n        info_cost = tokens_transmitted / max(float(tokens_if_full), 1.0)',
    '        # Fix2: Token penalty is applied per-round in run_single_sample intermediate steps.\n        # include_token_penalty=False (default) avoids double-counting the total token cost\n        # in the final compute_reward call on top of already-penalised intermediate steps.\n        info_cost = (tokens_transmitted / max(float(tokens_if_full), 1.0)) if include_token_penalty else 0.0'
)

# ===========================================================================
# FIX 3: Fair baseline voting — strip spectral W and stability agents from IB-Controller
# vote as well, so all methods use the same weighted_majority_vote call signature.
# Calibrator is kept for IB-Controller but also passed to baselines via the V3-5 fix.
# Spectral weighting (W) and stability (agents) are "communication extras" not present
# in baselines → remove them from IB-Controller final vote for apples-to-apples comparison.
# ===========================================================================
apply(
    "Fix3: IB-Controller final vote uses same kwargs as baselines (no W, no agents)",
    '    # --- Final answer: full voting suite (improvements I-VI) ---\n    final_answer = weighted_majority_vote(\n        current_responses,\n        dataset_type=config.dataset,\n        W=comm_graph.get_weight_snapshot(),\n        agents=agent_pool.agents,\n        calibrator=controller.calibrator,\n    )',
    '    # --- Final answer: calibrated vote (same aggregator as baselines for fair comparison) ---\n    # Fix3: Removed W (spectral centrality) and agents (stability) from IB-Controller vote.\n    # Spectral/stability weights are communication-protocol extras not available to baselines,\n    # so including them creates an unfair advantage in the accuracy metric.\n    # Calibrator is kept because baselines also receive it via V3-5 fix.\n    final_answer = weighted_majority_vote(\n        current_responses,\n        dataset_type=config.dataset,\n        calibrator=controller.calibrator,\n    )'
)

# ===========================================================================
# FIX 4: HotpotQA evaluation — replace F1>=0.5 with exact match after normalization.
# Standard HotpotQA benchmark uses EM (exact match after lowercasing, article removal,
# punctuation removal, and whitespace normalization).
# We keep F1 as a secondary diagnostic but the boolean return uses EM.
# ===========================================================================
apply(
    "Fix4: HotpotQA uses exact match (EM) not F1>=0.5",
    '    else:\n        # HotpotQA — use token-level F1 (standard protocol) instead of substring match\n        pred_tokens = set(re.sub(r\'[^\\w\\s]\', \'\', predicted).lower().split())\n        gold_tokens = set(re.sub(r\'[^\\w\\s]\', \'\', gold).lower().split())\n        # Remove stopwords that inflate F1\n        stopwords = {\'the\', \'a\', \'an\', \'is\', \'was\', \'are\', \'were\', \'of\', \'in\', \'to\', \'and\', \'for\', \'on\', \'at\', \'by\', \'with\'}\n        pred_tokens -= stopwords\n        gold_tokens -= stopwords\n        if not gold_tokens:\n            return predicted.strip().lower() == gold.strip().lower()\n        if not pred_tokens:\n            return False\n        tp = len(pred_tokens & gold_tokens)\n        precision = tp / len(pred_tokens) if pred_tokens else 0\n        recall    = tp / len(gold_tokens) if gold_tokens else 0\n        f1 = 2 * precision * recall / (precision + recall)\n        return f1 >= 0.5  # Standard HotpotQA threshold',
    '    else:\n        # Fix4: HotpotQA — use Exact Match (EM) per standard benchmark protocol.\n        # EM after: lowercase, remove articles/punctuation, normalize whitespace.\n        # F1>=0.5 inflates accuracy by up to ~15% vs published benchmarks.\n        import re as _re\n        def _hotpot_normalize(s: str) -> str:\n            s = s.lower()\n            s = _re.sub(r\'\\b(a|an|the)\\b\', \' \', s)  # remove articles\n            s = _re.sub(r\'[^\\w\\s]\', \'\', s)           # remove punctuation\n            return \' \'.join(s.split())                 # normalize whitespace\n        return _hotpot_normalize(predicted) == _hotpot_normalize(gold)'
)

# ===========================================================================
# FIX 5: Phase 1 pre-trained controllers are discarded; Phase 2 trains from scratch.
# Fix: store pre-trained controllers per dataset and pass to Phase 2 via experiment_fn.
# ===========================================================================
apply(
    "Fix5a: Store pre-trained controllers dict",
    'controller = None\n\nfor ds_name in datasets_to_run:\n    print(f"\\n--- Pre-Training: {ds_name.upper()} ---")',
    '# Fix5: Store pre-trained controller per dataset so Phase 2 warm-starts from Phase 1 weights.\npretrained_controllers: dict = {}\ncontroller = None\n\nfor ds_name in datasets_to_run:\n    print(f"\\n--- Pre-Training: {ds_name.upper()} ---")'
)
apply(
    "Fix5b: Save controller after Phase 1 training",
    '    run_experiment(\n        current_config,\n        method_name=f"Pretrain_{ds_name.upper()}",\n        train=True,\n        evaluate=False,\n        controller=controller,\n    )\n    print(f"[OK] {ds_name.upper()} pre-training complete")',
    '    run_experiment(\n        current_config,\n        method_name=f"Pretrain_{ds_name.upper()}",\n        train=True,\n        evaluate=False,\n        controller=controller,\n    )\n    pretrained_controllers[ds_name] = controller  # Fix5: persist for Phase 2 warm-start\n    print(f"[OK] {ds_name.upper()} pre-training complete (weights saved for Phase 2)")'
)

# Fix5c: Phase 2 warm-starts from pre-trained controller weights (copies to avoid contamination)
apply(
    "Fix5c: Phase 2 experiment_fn warm-starts from pre-trained controller",
    '    stats = run_multi_seed_experiment(\n        current_config,\n        f"Transfer_{ds_name.upper()}",\n        n_seeds=3,\n    )',
    '    # Fix5: Build experiment_fn that warm-starts each seed from the Phase 1 controller.\n    import copy as _copy\n    _ds = ds_name  # capture loop variable\n    def _make_transfer_fn(pretrained_ctlrs, _ds_name):\n        def _transfer_fn(cfg, method_name):\n            base_ctrl = pretrained_ctlrs.get(_ds_name)\n            seed_ctrl = _copy.deepcopy(base_ctrl) if base_ctrl is not None else None\n            return run_experiment(cfg, method_name=method_name, train=True, evaluate=True, controller=seed_ctrl)\n        return _transfer_fn\n    stats = run_multi_seed_experiment(\n        current_config,\n        f"Transfer_{ds_name.upper()}",\n        n_seeds=3,\n        experiment_fn=_make_transfer_fn(pretrained_controllers, _ds),\n    )'
)

# ===========================================================================
# FIX 6: transmit() — parser called without final_answer; agent's answer lost in
# SUMMARIZE/FILTER modes because struct.final_answer="" and gets filtered out.
# Fix: extract the final_answer from the message text before parsing so it's preserved.
# ===========================================================================
apply(
    "Fix6: transmit extracts final_answer from message before parsing",
    '        elif mode == SemanticBandwidthMode.FULL or w_ij > 0.99:\n            # Send the complete structured state instead of raw text\n            struct = self.parser.parse(message, dataset_type=self.dataset_type)  # Bug A4 fix\n            filtered_msg = struct.to_formatted_string()\n            transmitted_tokens = count_tokens(filtered_msg)\n            sem_sim = 1.0\n            info_ret = 1.0\n        else:\n            # TRUE SEMANTIC COMPRESSION (Structured Schema Routing) — Tier4 #20\n            struct = self.parser.parse(message, dataset_type=self.dataset_type)  # Bug A4 fix',
    '        elif mode == SemanticBandwidthMode.FULL or w_ij > 0.99:\n            # Send the complete structured state instead of raw text\n            struct = self.parser.parse(message, dataset_type=self.dataset_type)  # Bug A4 fix\n            filtered_msg = struct.to_formatted_string()\n            transmitted_tokens = count_tokens(filtered_msg)\n            sem_sim = 1.0\n            info_ret = 1.0\n        else:\n            # TRUE SEMANTIC COMPRESSION (Structured Schema Routing) — Tier4 #20\n            # Fix6: pre-extract final_answer from raw message text so it survives\n            # SUMMARIZE/FILTER mode (which clears claims/evidence/assumptions).\n            _ans_match = re.search(r\'(?:Answer:|answer:)\\s*(.+?)(?:\\n|$)\', message)\n            _pre_answer = _ans_match.group(1).strip() if _ans_match else ""\n            struct = self.parser.parse(message, dataset_type=self.dataset_type)  # Bug A4 fix\n            if not struct.final_answer and _pre_answer:\n                struct.final_answer = _pre_answer  # Restore extracted answer'
)

# ===========================================================================
# FIX 7a: count_cycles threshold inconsistency — 0.5 used in both places now
# (already aligned in the main call). Verify and fix the default in the method.
# ===========================================================================
apply(
    "Fix7a: count_cycles default threshold aligned to 0.5 (matches SUMMARIZE boundary)",
    '    def count_cycles(self, threshold: float = None) -> int:',
    '    def count_cycles(self, threshold: float = 0.5) -> int:  # Fix7: default aligned to SUMMARIZE boundary'
)

# ===========================================================================
# FIX 7b: Gradient accumulation — steps=len(train_data) means one PPO update per epoch.
# This causes high variance. Fix: cap accumulation_steps at 16 samples for stability.
# ===========================================================================
apply(
    "Fix7b: Cap gradient accumulation steps at 16 for stable PPO updates",
    '        accumulation_steps = len(train_data)',
    '        accumulation_steps = min(len(train_data), 16)  # Fix7b: cap at 16 for stable gradient variance'
)

# Write back
with open('temp_check.py', 'w', encoding='utf-8') as f:
    f.write(src)

print("\n" + "=" * 60)
print("V4 FIX SUMMARY")
print("=" * 60)
ok = miss = skip = 0
for status, label in fixes:
    print(f"[{status}] {label}")
    if status == 'OK': ok += 1
    elif status == 'MISS': miss += 1
    else: skip += 1
print(f"\nTotal: {len(fixes)} | OK: {ok} | Miss: {miss} | Skip: {skip}")

# Syntax check
try:
    py_compile.compile('temp_check.py', doraise=True)
    print("\n[OK] Syntax check passed!")
except py_compile.PyCompileError as e:
    print(f"\n[FAIL] Syntax error: {e}")
    sys.exit(1)

# Write to notebook
nb = nbformat.read(NB_PATH, as_version=4)
nb.cells[2].source = src
nbformat.write(nb, NB_PATH)
print("[OK] Notebook updated!")
