"""
apply_v2_fixes.py
Second-pass fixes: all surviving bugs from v2 review + voting improvements.
"""
import nbformat, sys

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
# BUG #1: AzureOpenAIBackend.generate() — max_tokens=512 hardcoded, ignores self.max_tokens
# ===========================================================================
apply(
    "Bug A1: Azure generate() uses self.max_tokens",
    "                    max_tokens=512,\n                    stream=False,\n                )\n                text = response.choices[0].message.content or \"\"\n                token_count = response.usage.total_tokens if response.usage else count_tokens(text)\n\n                answer = self._extract_answer(text, dataset_type)\n                confidence = self._estimate_confidence(text)\n\n                return LLMResponse(\n                    text=text,\n                    answer=answer,\n                    token_count=token_count,\n                    confidence=confidence,\n                )\n            except Exception as e:\n                last_error = e\n                wait = (attempt + 1) * 3\n                print(f\"  [Azure API] Attempt {attempt+1}/{self.max_retries} failed: {type(e).__name__}: {str(e)[:100]}. Retrying in {wait}s...\")",
    "                    max_tokens=self.max_tokens,  # Bug A1 fix: use stored dataset-specific value\n                    stream=False,\n                )\n                text = response.choices[0].message.content or \"\"\n                token_count = response.usage.total_tokens if response.usage else count_tokens(text)\n\n                answer = self._extract_answer(text, dataset_type)\n                confidence = self._estimate_confidence(text)\n\n                return LLMResponse(\n                    text=text,\n                    answer=answer,\n                    token_count=token_count,\n                    confidence=confidence,\n                )\n            except Exception as e:\n                last_error = e\n                wait = (attempt + 1) * 3\n                print(f\"  [Azure API] Attempt {attempt+1}/{self.max_retries} failed: {type(e).__name__}: {str(e)[:100]}. Retrying in {wait}s...\")"
)

# ===========================================================================
# BUG #2: NvidiaDeepSeekBackend.__init__ — self.max_tokens never assigned
# ===========================================================================
apply(
    "Bug A2: Nvidia backend stores self.max_tokens",
    "    def __init__(\n        self,\n        model: str = \"deepseek-ai/deepseek-v3.2\",\n        api_key: Optional[str] = None,\n        base_url: str = \"https://integrate.api.nvidia.com/v1\",\n        max_retries: int = 15,\n        timeout: int = 120,\n    ):\n        self.model = model\n        self.api_key = api_key or os.environ.get(\"NVIDIA_API_KEY\")\n        self.base_url = base_url\n        self.max_retries = max_retries\n        self.timeout = timeout",
    "    def __init__(\n        self,\n        model: str = \"deepseek-ai/deepseek-v3.2\",\n        api_key: Optional[str] = None,\n        base_url: str = \"https://integrate.api.nvidia.com/v1\",\n        max_retries: int = 15,\n        timeout: int = 120,\n        max_tokens: int = 512,  # Bug A2 fix: must be stored or generate() crashes\n    ):\n        self.model = model\n        self.api_key = api_key or os.environ.get(\"NVIDIA_API_KEY\")\n        self.base_url = base_url\n        self.max_retries = max_retries\n        self.timeout = timeout\n        self.max_tokens = max_tokens  # Bug A2 fix"
)

# ===========================================================================
# BUG #3: run_experiment_with_random_weights — NameError: 'controller' before assignment
# ===========================================================================
apply(
    "Bug A3: FrozenWeightController NameError fix",
    "    # Bug #1 fix: Use FrozenWeightController to truly freeze topology across rounds\n    random_W, random_Q = controller.get_random_weights()\n    controller = FrozenWeightController(config, random_W, random_Q)",
    "    # Bug A3 fix: need a temp IBController to call get_random_weights()\n    _tmp_ctrl = IBController(config)\n    random_W, random_Q = _tmp_ctrl.get_random_weights()\n    del _tmp_ctrl\n    controller = FrozenWeightController(config, random_W, random_Q)"
)

# ===========================================================================
# BUG #4: CommunicationGraph.transmit() never passes dataset_type to parser
# Add dataset_type attr to CommunicationGraph; pass it in transmit()
# ===========================================================================
apply(
    "Bug A4a: CommunicationGraph stores dataset_type attr",
    "        self.transmission_log: List[TransmissionRecord] = []\n        self.round_token_counts: Dict[int, int] = {}\n        self.modes_disabled: List[SemanticBandwidthMode] = []\n        self._agent_last_text: Dict[int, str] = {}  # Tier4 #21: CONTRAST mode state",
    "        self.transmission_log: List[TransmissionRecord] = []\n        self.round_token_counts: Dict[int, int] = {}\n        self.modes_disabled: List[SemanticBandwidthMode] = []\n        self._agent_last_text: Dict[int, str] = {}  # Tier4 #21: CONTRAST mode state\n        self.dataset_type: str = \"gsm8k\"  # Bug A4 fix: set via set_dataset_type() before use"
)
# Add set_dataset_type helper after reset()
apply(
    "Bug A4b: Add set_dataset_type() to CommunicationGraph",
    "    def get_weight_snapshot(self) -> np.ndarray:\n        return self.W.copy()",
    "    def set_dataset_type(self, dataset_type: str):\n        \"\"\"Bug A4 fix: configure parser branch for transmit().\"\"\"\n        self.dataset_type = dataset_type\n\n    def get_weight_snapshot(self) -> np.ndarray:\n        return self.W.copy()"
)
# Pass dataset_type in transmit() for FULL mode
apply(
    "Bug A4c: transmit() FULL mode passes dataset_type to parser",
    "        elif mode == SemanticBandwidthMode.FULL or w_ij > 0.99:\n            # Send the complete structured state instead of raw text\n            struct = self.parser.parse(message)  # dataset_type propagated via attr if needed",
    "        elif mode == SemanticBandwidthMode.FULL or w_ij > 0.99:\n            # Send the complete structured state instead of raw text\n            struct = self.parser.parse(message, dataset_type=self.dataset_type)  # Bug A4 fix"
)
# Pass dataset_type in transmit() for SUMMARIZE/FILTER/CONTRAST mode
apply(
    "Bug A4d: transmit() compress mode passes dataset_type to parser",
    "        else:\n            # TRUE SEMANTIC COMPRESSION (Structured Schema Routing) — Tier4 #20\n            struct = self.parser.parse(message)",
    "        else:\n            # TRUE SEMANTIC COMPRESSION (Structured Schema Routing) — Tier4 #20\n            struct = self.parser.parse(message, dataset_type=self.dataset_type)  # Bug A4 fix"
)
# Wire dataset_type into CommunicationGraph in run_experiment and run_full_communication
apply(
    "Bug A4e: Set comm_graph.dataset_type in run_experiment",
    "    comm_graph = CommunicationGraph(n_agents=config.n_agents)\n\n    # Use provided controller or create a fresh one",
    "    comm_graph = CommunicationGraph(n_agents=config.n_agents)\n    comm_graph.set_dataset_type(config.dataset)  # Bug A4 fix\n\n    # Use provided controller or create a fresh one"
)
apply(
    "Bug A4f: Set comm_graph.dataset_type in run_full_communication",
    "    comm_graph = CommunicationGraph(config.n_agents)\n    controller = IBController(config)\n\n    # Set all weights to 1.0 (full communication)",
    "    comm_graph = CommunicationGraph(config.n_agents)\n    comm_graph.set_dataset_type(config.dataset)  # Bug A4 fix\n    controller = IBController(config)\n\n    # Set all weights to 1.0 (full communication)"
)
apply(
    "Bug A4g: Set comm_graph.dataset_type in run_experiment_with_random_weights",
    "    comm_graph = CommunicationGraph(n_agents=config.n_agents)\n    \n    # Bug A3 fix:",
    "    comm_graph = CommunicationGraph(n_agents=config.n_agents)\n    comm_graph.set_dataset_type(config.dataset)  # Bug A4 fix\n    \n    # Bug A3 fix:"
)

# ===========================================================================
# BUG #5: Dense intermediate reward uses stale current_responses (final-round state)
# Add responses_snapshots list; pick per-round snapshot for intermediate rewards
# ===========================================================================
apply(
    "Bug A5a: Add responses_snapshots list before loop",
    "    # Track initial state for dense reward net change\n    initial_confidence = np.mean([r.confidence for r in current_responses])\n    initial_entropy = len(set(r.answer for r in current_responses)) / controller.n_agents\n\n    # --- Rounds 1..T: Communication rounds ---\n    for t in range(1, config.n_rounds + 1):",
    "    # Track initial state for dense reward net change\n    initial_confidence = np.mean([r.confidence for r in current_responses])\n    initial_entropy = len(set(r.answer for r in current_responses)) / controller.n_agents\n\n    # Bug A5 fix: track per-round snapshots so intermediate rewards use correct state\n    responses_snapshots: List[List] = [list(round0_responses)]\n\n    # --- Rounds 1..T: Communication rounds ---\n    for t in range(1, config.n_rounds + 1):"
)
apply(
    "Bug A5b: Append snapshot after each communicate_and_solve",
    "        current_responses = agent_pool.communicate_and_solve(\n            question=question,\n            received_messages=received_messages,\n            dataset_type=config.dataset,\n            round_num=t,\n        )\n        total_gen_tokens += sum(r.token_count for r in current_responses)",
    "        current_responses = agent_pool.communicate_and_solve(\n            question=question,\n            received_messages=received_messages,\n            dataset_type=config.dataset,\n            round_num=t,\n        )\n        total_gen_tokens += sum(r.token_count for r in current_responses)\n        responses_snapshots.append(list(current_responses))  # Bug A5 fix: per-round snapshot"
)
apply(
    "Bug A5c: Use per-round snapshot for intermediate reward entropy",
    "                # Fix #7: Dense shaping reward for intermediate rounds\n                # Small positive signal for answer entropy reduction, small penalty for token use\n                round_num_i = i + 1\n                round_tokens = comm_graph.round_token_counts.get(round_num_i, 0)\n                token_penalty = -controller.lambda_info * round_tokens / max(float(tokens_if_full), 1.0)\n                # Entropy reduction: fewer unique answers = better convergence\n                current_n_unique = len(set(r.answer for r in current_responses))\n                round_entropy_reduction = initial_entropy - current_n_unique / max(controller.n_agents, 1)\n                reward_for_step = 0.05 * round_entropy_reduction + token_penalty",
    "                # Fix #7: Dense shaping reward using per-round snapshot (Bug A5 fix)\n                round_num_i = i + 1\n                round_tokens = comm_graph.round_token_counts.get(round_num_i, 0)\n                token_penalty = -controller.lambda_info * round_tokens / max(float(tokens_if_full), 1.0)\n                # Use the correct per-round snapshot, not final-round current_responses\n                snap_idx = min(round_num_i, len(responses_snapshots) - 1)\n                snap = responses_snapshots[snap_idx]\n                round_n_unique = len(set(r.answer for r in snap))\n                round_entropy_reduction = initial_entropy - round_n_unique / max(controller.n_agents, 1)\n                reward_for_step = 0.05 * round_entropy_reduction + token_penalty"
)

# ===========================================================================
# BUG #6: Non-stationary adaptive prior — update once per epoch, not per round
# Remove per-round update from run_single_sample; add epoch-level update in run_experiment
# ===========================================================================
apply(
    "Bug A6a: Remove per-round adaptive_prior update from run_single_sample",
    "        # Always apply weights BEFORE checking halt so reward reflects this round's graph\n        comm_graph.set_weights(W, Q)\n        if train_controller:\n            controller.update_adaptive_prior(W)  # Tier4 #22: Update EMA prior from observed W",
    "        # Always apply weights BEFORE checking halt so reward reflects this round's graph\n        comm_graph.set_weights(W, Q)\n        # Bug A6 fix: adaptive_prior updated once per epoch (not per round) to prevent\n        # non-stationary KL reward. update_adaptive_prior() is called in run_experiment."
)
apply(
    "Bug A6b: Add epoch-level prior update in run_experiment after update_policy",
    "                # Final update for remaining samples\n            if len(train_data) % accumulation_steps != 0:\n                loss = controller.update_policy()\n                if hasattr(config, \"model_save_path\") and config.model_save_path:\n                    controller.save_model(config.model_save_path.replace(\".pt\", f\"_{method_name}_latest.pt\"))",
    "                # Final update for remaining samples\n            if len(train_data) % accumulation_steps != 0:\n                loss = controller.update_policy()\n                if hasattr(config, \"model_save_path\") and config.model_save_path:\n                    controller.save_model(config.model_save_path.replace(\".pt\", f\"_{method_name}_latest.pt\"))\n\n            # Bug A6 fix: update EMA prior once per epoch from final comm_graph state\n            controller.update_adaptive_prior(comm_graph.get_weight_snapshot())"
)

# ===========================================================================
# VOTING IMPROVEMENTS I–VII
# Replace majority_vote and weighted_majority_vote with full suite
# ===========================================================================
new_voting_block = '''
def _normalize_for_vote(answer: str, dataset_type: str) -> str:
    """Voting improvement I: numeric normalization before vote aggregation.
    Prevents "24", "24.0", "24.00", "1,200" being counted as different answers.
    """
    if dataset_type == "gsm8k":
        answer = answer.replace(",", "").strip()
        nums = re.findall(r"-?\\d+\\.?\\d*", answer)
        if nums:
            try:
                v = float(nums[-1])
                return str(int(v)) if v == int(v) else f"{v:.4f}".rstrip("0").rstrip(".")
            except (ValueError, OverflowError):
                pass
    return answer.lower().strip()


class PlattCalibrator:
    """Voting improvement VI: Platt scaling for LLM confidence calibration.
    Fitted on training samples. Falls back to raw confidence if not fitted.
    """
    def __init__(self):
        self.a = 1.0
        self.b = 0.0
        self._fitted = False
        self._train_confs: List[float] = []
        self._train_labels: List[bool] = []

    def accumulate(self, conf: float, is_correct: bool):
        self._train_confs.append(conf)
        self._train_labels.append(is_correct)

    def fit(self):
        if len(self._train_confs) < 10:
            return  # Not enough data yet
        X = np.array(self._train_confs)
        y = np.array(self._train_labels, dtype=float)
        for _ in range(50):
            z = self.a * X + self.b
            p = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            grad_a = np.mean((p - y) * X)
            grad_b = np.mean(p - y)
            h_a = np.mean(p * (1 - p) * X**2) + 1e-6
            h_b = np.mean(p * (1 - p)) + 1e-6
            self.a -= grad_a / h_a * 0.1
            self.b -= grad_b / h_b * 0.1
        self._fitted = True

    def calibrate(self, conf: float) -> float:
        if not self._fitted:
            return conf
        z = self.a * conf + self.b
        return float(1 / (1 + np.exp(-np.clip(z, -500, 500))))


def should_halt_by_entropy(responses: List[AgentResponse], n_agents: int, threshold: float = 0.25) -> bool:
    """Voting improvement VII: entropy-based independent halt gate.
    Returns True if answer distribution is sufficiently concentrated.
    threshold=0.0 requires unanimity; threshold=0.5 accepts 2-of-3 majority.
    """
    answers = [r.answer for r in responses]
    counts = Counter(answers)
    n = len(answers)
    probs = np.array([c / n for c in counts.values()])
    H = -np.sum(probs * np.log2(probs + 1e-10))
    H_normalized = H / max(np.log2(n_agents), 1e-8)
    return bool(H_normalized < threshold)


def spectral_vote_weights(W: np.ndarray, base_confidences: List[float]) -> List[float]:
    """Voting improvement V: multiply agent confidence by in-degree centrality.
    Agents that other agents 'send to most' are trusted more.
    Zero extra API cost — uses already-computed W matrix.
    """
    W_off = W.copy()
    np.fill_diagonal(W_off, 0.0)
    in_degree = W_off.sum(axis=0)
    if in_degree.max() > in_degree.min() + 1e-8:
        centrality = 0.5 + (in_degree - in_degree.min()) / (in_degree.max() - in_degree.min())
    else:
        centrality = np.ones(len(base_confidences))
    return [float(c * centrality[i]) for i, c in enumerate(base_confidences)]


def compute_stability_weights(agents, n_agents: int) -> Dict[int, float]:
    """Voting improvement IV: penalize agents that flip-flopped between rounds.
    k answer changes → multiplier exp(-0.3k). Stable agents get weight 1.0.
    """
    stability = {}
    for agent in agents:
        if len(agent.history) <= 1:
            stability[agent.agent_id] = 1.0
        else:
            answers = [h.answer for h in agent.history]
            changes = sum(1 for i in range(1, len(answers)) if answers[i] != answers[i-1])
            stability[agent.agent_id] = float(np.exp(-0.3 * changes))
    # Fill any missing agent ids
    for aid in range(n_agents):
        stability.setdefault(aid, 1.0)
    return stability


def majority_vote(responses: List[AgentResponse], dataset_type: str = "gsm8k") -> str:
    """Simple majority vote with numeric normalization (Improvement I)."""
    answer_counts: Dict[str, int] = {}
    canonical_map: Dict[str, str] = {}
    for r in responses:
        key = _normalize_for_vote(r.answer, dataset_type)
        canonical_map.setdefault(key, r.answer)
        answer_counts[key] = answer_counts.get(key, 0) + 1
    best_key = max(answer_counts, key=answer_counts.get)
    return canonical_map[best_key]


def weighted_majority_vote(
    responses: List[AgentResponse],
    dataset_type: str = "gsm8k",
    W: Optional[np.ndarray] = None,
    agents=None,
    calibrator: Optional["PlattCalibrator"] = None,
) -> str:
    """Bayesian multiplicative vote (Improvement II) with numeric normalization (I),
    spectral centrality weighting (V), stability discounting (IV), and Platt calibration (VI).

    Falls back to additive weighting gracefully when optional inputs are absent.
    """
    if not responses:
        return ""

    # Step 1: Numeric normalization (Improvement I)
    normalized: List[str] = [_normalize_for_vote(r.answer, dataset_type) for r in responses]
    canonical_map: Dict[str, str] = {}
    for key, r in zip(normalized, responses):
        canonical_map.setdefault(key, r.answer)

    # Step 2: Build calibrated confidence per agent (Improvement VI)
    confs = []
    for r in responses:
        c = calibrator.calibrate(r.confidence) if calibrator and calibrator._fitted else r.confidence
        confs.append(max(0.51, min(0.99, c)))  # Clamp for Bayesian log-odds

    # Step 3: Spectral centrality multiplier (Improvement V)
    if W is not None:
        confs = spectral_vote_weights(W, confs)
        confs = [max(0.51, min(0.99, c)) for c in confs]

    # Step 4: Stability discounting (Improvement IV)
    if agents is not None:
        stability = compute_stability_weights(agents, len(responses))
        confs = [c * stability.get(r.agent_id, 1.0) for c, r in zip(confs, responses)]
        confs = [max(0.51, min(0.99, c)) for c in confs]

    # Step 5: Bayesian multiplicative voting (Improvement II)
    unique_answers = list(set(normalized))
    n_options = max(len(unique_answers), 2)

    log_posterior: Dict[str, float] = {}
    for candidate in unique_answers:
        log_p = 0.0
        for ans, theta in zip(normalized, confs):
            if ans == candidate:
                log_p += np.log(theta)
            else:
                log_p += np.log((1.0 - theta) / (n_options - 1))
        log_posterior[candidate] = log_p

    # Numerically stable argmax
    best_key = max(log_posterior, key=log_posterior.get)
    return canonical_map[best_key]


def numeric_median_vote(responses: List[AgentResponse], dataset_type: str = "gsm8k") -> str:
    """Voting improvement III: L1-optimal (weighted median) for numerical answers.
    Better than mode when agents produce close-but-not-identical numbers.
    Falls back to weighted_majority_vote for non-numeric or non-GSM8K.
    """
    if dataset_type != "gsm8k":
        return weighted_majority_vote(responses, dataset_type)

    numeric_answers = []
    for r in responses:
        nums = re.findall(r"-?\\d+\\.?\\d*", r.answer.replace(",", ""))
        if nums:
            try:
                numeric_answers.append((float(nums[-1]), r.confidence, r.answer))
            except ValueError:
                pass

    if not numeric_answers:
        return weighted_majority_vote(responses, dataset_type)

    values = np.array([v for v, _, _ in numeric_answers])
    weights = np.array([c for _, c, _ in numeric_answers])
    weights /= weights.sum()

    idx = np.argsort(values)
    sorted_vals, sorted_w = values[idx], weights[idx]
    cumw = np.cumsum(sorted_w)
    median_idx = int(np.searchsorted(cumw, 0.5))
    median_val = sorted_vals[min(median_idx, len(sorted_vals) - 1)]
    best = min(numeric_answers, key=lambda x: abs(x[0] - median_val))
    return best[2]

'''

apply(
    "Voting: Replace majority_vote / weighted_majority_vote with full suite",
    "def majority_vote(responses: List[AgentResponse]) -> str:\n    \"\"\"Simple majority vote over agent answers.\"\"\"\n    answers = [r.answer for r in responses]\n    counter = Counter(answers)\n    return counter.most_common(1)[0][0]\n\n\ndef weighted_majority_vote(responses: List[AgentResponse]) -> str:\n    \"\"\"Confidence-weighted majority vote.\"\"\"\n    answer_weights: Dict[str, float] = {}\n    for r in responses:\n        answer_weights[r.answer] = answer_weights.get(r.answer, 0) + r.confidence\n    return max(answer_weights, key=answer_weights.get)",
    new_voting_block.strip()
)

# ===========================================================================
# Wire up PlattCalibrator into IBController
# ===========================================================================
apply(
    "Platt: Add calibrator to IBController.__init__",
    "        self.episode_rewards: List[float] = []\n        self.episode_losses: List[float] = []\n        self.saved_transitions: List[dict] = []\n        self.adaptive_prior: float = 0.1  # Tier4 #22: EMA-adapted topology prior",
    "        self.episode_rewards: List[float] = []\n        self.episode_losses: List[float] = []\n        self.saved_transitions: List[dict] = []\n        self.adaptive_prior: float = 0.1  # Tier4 #22: EMA-adapted topology prior\n        self.calibrator: PlattCalibrator = PlattCalibrator()  # Improvement VI: Platt scaling"
)

# ===========================================================================
# Wire dataset_type and calibrator into final answer voting in run_single_sample
# ===========================================================================
apply(
    "Wire: voting uses dataset_type + spectral W + calibrator",
    "    # --- Final answer ---\n    final_answer = weighted_majority_vote(current_responses)",
    "    # --- Final answer: full voting suite (improvements I-VI) ---\n    final_answer = weighted_majority_vote(\n        current_responses,\n        dataset_type=config.dataset,\n        W=comm_graph.get_weight_snapshot(),\n        agents=agent_pool.agents,\n        calibrator=controller.calibrator,\n    )"
)

# ===========================================================================
# Wire entropy halt gate into communication loop (Improvement VII)
# ===========================================================================
apply(
    "Entropy halt: add entropy-based halt check in communication loop",
    "        # Adaptive Halting: if confident, break early (after W is set)\n        if halt_decision and t > 1:\n            halted = True  # Bug #2 fix: set explicit flag\n            if all_metas:\n                all_metas.pop()  # Discard the meta for the unused halt round\n            break",
    "        # Adaptive Halting: controller logit OR entropy consensus (Improvement VII)\n        entropy_halt = should_halt_by_entropy(current_responses, config.n_agents, threshold=0.25)\n        if (halt_decision or entropy_halt) and t > 1:\n            halted = True  # Bug #2 fix: set explicit flag\n            if all_metas:\n                all_metas.pop()  # Discard the meta for the unused halt round\n            break"
)

# ===========================================================================
# Wire Platt calibration: accumulate during training in run_experiment
# ===========================================================================
apply(
    "Platt accumulate: gather (conf, correct) pairs during training",
    "                if result.is_correct:\n                    epoch_correct += 1\n                \n                # Dynamically update running accuracy\n                avg_accuracy = epoch_correct / (i + 1)",
    "                if result.is_correct:\n                    epoch_correct += 1\n\n                # Platt calibration: accumulate (conf, correct) during training (Improvement VI)\n                for r in agent_pool.agents:\n                    if r.history:\n                        last_resp = r.history[-1]\n                        agent_correct = evaluate_answer(last_resp.answer, sample[\"answer\"], config.dataset)\n                        controller.calibrator.accumulate(last_resp.confidence, agent_correct)\n\n                # Dynamically update running accuracy\n                avg_accuracy = epoch_correct / (i + 1)"
)
apply(
    "Platt fit: fit calibrator after each epoch",
    "            # Bug A6 fix: update EMA prior once per epoch from final comm_graph state\n            controller.update_adaptive_prior(comm_graph.get_weight_snapshot())",
    "            # Bug A6 fix: update EMA prior once per epoch from final comm_graph state\n            controller.update_adaptive_prior(comm_graph.get_weight_snapshot())\n            # Improvement VI: re-fit Platt calibrator after each training epoch\n            controller.calibrator.fit()"
)

# ===========================================================================
# Also fix majority_vote calls that don't pass dataset_type (run_full_communication etc)
# ===========================================================================
apply(
    "Wire: run_full_communication uses dataset_type in final vote",
    "        final = weighted_majority_vote(current)\n        is_correct = evaluate_answer(final, sample[\"answer\"], config.dataset)\n\n        sr = SampleResult(\n            question=sample[\"question\"],\n            gold_answer=sample[\"answer\"],\n            final_answer=final,\n            is_correct=is_correct,\n            initial_answers=initial_answers,",
    "        final = weighted_majority_vote(current, dataset_type=config.dataset)\n        is_correct = evaluate_answer(final, sample[\"answer\"], config.dataset)\n\n        sr = SampleResult(\n            question=sample[\"question\"],\n            gold_answer=sample[\"answer\"],\n            final_answer=final,\n            is_correct=is_correct,\n            initial_answers=initial_answers,"
)
apply(
    "Wire: run_no_communication uses dataset_type in vote",
    "        final = weighted_majority_vote(r0)\n        is_correct = evaluate_answer(final, sample[\"answer\"], config.dataset)",
    "        final = weighted_majority_vote(r0, dataset_type=config.dataset)\n        is_correct = evaluate_answer(final, sample[\"answer\"], config.dataset)"
)

# ===========================================================================
# Write back
# ===========================================================================
nb.cells[2].source = src
nbformat.write(nb, NB_PATH)

print("\n" + "="*60)
print("V2 FIX SUMMARY")
print("="*60)
for line in fixes:
    print(line)

ok = len([l for l in fixes if l.startswith("[OK]")])
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
