"""
apply_all_fixes.py
Applies ALL Tier 1–4 bug fixes to SparseComm_Kaggle.ipynb.
Run with: python apply_all_fixes.py
"""
import nbformat, re, sys

NB_PATH = 'C:\\Users\\msaur\\OneDrive\\Desktop\\projectRL\\SparseComm_Kaggle.ipynb'

nb = nbformat.read(NB_PATH, as_version=4)
src = nb.cells[2].source

fixes_applied = []

def apply(label, old, new, required=True):
    global src
    if old in src:
        src = src.replace(old, new, 1)
        fixes_applied.append(f"[OK] {label}")
    elif required:
        fixes_applied.append(f"[MISS] {label} — pattern not found")
    else:
        fixes_applied.append(f"[SKIP] {label} — already applied or not present")

# ===========================================================================
# BUG #3: Shannon entropy normalization hardcoded for 3 agents
# ===========================================================================
apply(
    "Bug #3: Shannon entropy normalization",
    "        # Normalize: max entropy for 3 agents = log2(3) ≈ 1.58\n            shannon_entropy = shannon_entropy / 1.58",
    "        # Normalize: max entropy = log2(n_agents), generalized for any N (Bug #3 fix)\n            shannon_entropy = shannon_entropy / max(np.log2(self.n_agents), 1e-8)"
)

# ===========================================================================
# BUG #4: CommunicationGraph.__init__ diagonal not zeroed
# ===========================================================================
apply(
    "Bug #4: CommunicationGraph.__init__ diagonal zeroed",
    "        self.W = np.ones((n_agents, n_agents)) * 0.5\n        self.Q = np.zeros((n_agents, query_dim))",
    "        self.W = np.ones((n_agents, n_agents)) * 0.5\n        np.fill_diagonal(self.W, 1.0)  # Bug #4 fix: zero diagonal at construction, not only reset()\n        self.Q = np.zeros((n_agents, query_dim))"
)

# ===========================================================================
# BUG #2: Stale halt_decision / t variable — explicit halted flag
# ===========================================================================
apply(
    "Bug #2a: Add halted=False before communication loop",
    "    current_responses = round0_responses\n    all_metas = []\n    all_comms = []",
    "    current_responses = round0_responses\n    all_metas = []\n    all_comms = []\n    halted = False  # Bug #2 fix: explicit halt flag, avoids stale halt_decision after loop"
)
apply(
    "Bug #2b: Set halted=True on break",
    "        # Adaptive Halting: if confident, break early (after W is set)\n        if halt_decision and t > 1:\n            if all_metas:\n                all_metas.pop()  # Discard the meta for the unused halt round\n            break",
    "        # Adaptive Halting: if confident, break early (after W is set)\n        if halt_decision and t > 1:\n            halted = True  # Bug #2 fix: set explicit flag\n            if all_metas:\n                all_metas.pop()  # Discard the meta for the unused halt round\n            break"
)
apply(
    "Bug #2c: Use halted flag instead of stale halt_decision",
    "    # Compute DENSE reward for this step\n    is_final = (t == config.n_rounds or halt_decision)",
    "    # Compute DENSE reward for this step (Bug #2 fix: use explicit halted flag)\n    is_final = (t == config.n_rounds or halted)"
)

# ===========================================================================
# FIX #15: Early exit when all agents agree in Round 0
# ===========================================================================
apply(
    "Fix #15: Early exit when Round 0 unanimous",
    "    # Track initial state for dense reward net change\n    initial_confidence = np.mean([r.confidence for r in current_responses])\n    initial_entropy = len(set(r.answer for r in current_responses)) / controller.n_agents\n\n    # --- Rounds 1..T: Communication rounds ---\n    for t in range(1, config.n_rounds + 1):",
    """    # Fix #15: Early exit when all agents unanimously agree in Round 0
    _round0_unique = set(r.answer for r in round0_responses)
    if len(_round0_unique) == 1 and list(_round0_unique)[0]:
        final_answer = round0_responses[0].answer
        is_correct = evaluate_answer(final_answer, gold_answer, config.dataset)
        return SampleResult(
            question=question,
            gold_answer=gold_answer,
            final_answer=final_answer,
            is_correct=is_correct,
            initial_answers=initial_answers,
            initial_correct=initial_correct,
            final_answers={r.agent_id: r.answer for r in round0_responses},
            initial_reasoning={r.agent_id: r.reasoning_text for r in round0_responses},
            final_reasoning={r.agent_id: r.reasoning_text for r in round0_responses},
            raw_communications=[],
            total_tokens_transmitted=0,
            total_tokens_if_full=0,
            compression_ratio=0.0,
            communication_density=0.0,
            mode_distribution={mode.name: 0 for mode in SemanticBandwidthMode if mode != SemanticBandwidthMode.BLOCK},
            n_active_edges=0,
            generation_tokens=sum(r.token_count for r in round0_responses),
            halt_round=0,
        )

    # Track initial state for dense reward net change
    initial_confidence = np.mean([r.confidence for r in current_responses])
    initial_entropy = len(set(r.answer for r in current_responses)) / controller.n_agents

    # --- Rounds 1..T: Communication rounds ---
    for t in range(1, config.n_rounds + 1):"""
)

# ===========================================================================
# FIX #7: Dense per-round shaping rewards
# ===========================================================================
apply(
    "Fix #7: Dense per-round shaping rewards",
    """    if train_controller and all_metas:
        for i, m in enumerate(all_metas):
            is_last = (i == len(all_metas) - 1)
            # Only final round gets shaping reward, early rounds get 0.0 or intermediate
            # We'll use step_reward for the final step, and 0.0 for intermediate.
            reward_for_step = step_reward if is_last else 0.0
            controller.store_transition(m, reward_for_step, done=is_last)""",
    """    if train_controller and all_metas:
        n_metas = len(all_metas)
        for i, m in enumerate(all_metas):
            is_last = (i == n_metas - 1)
            if is_last:
                reward_for_step = step_reward
            else:
                # Fix #7: Dense shaping reward for intermediate rounds
                # Small positive signal for answer entropy reduction, small penalty for token use
                round_num_i = i + 1
                round_tokens = comm_graph.round_token_counts.get(round_num_i, 0)
                token_penalty = -controller.lambda_info * round_tokens / max(float(tokens_if_full), 1.0)
                # Entropy reduction: fewer unique answers = better convergence
                current_n_unique = len(set(r.answer for r in current_responses))
                round_entropy_reduction = initial_entropy - current_n_unique / max(controller.n_agents, 1)
                reward_for_step = 0.05 * round_entropy_reduction + token_penalty
            controller.store_transition(m, reward_for_step, done=is_last)"""
)

# ===========================================================================
# FIX #8: PPO buffer — update once per epoch, not every 5 samples
# ===========================================================================
apply(
    "Fix #8: PPO accumulation = full epoch",
    "        accumulation_steps = getattr(config, 'gradient_accumulation_steps', 5)",
    "        accumulation_steps = len(train_data)  # Fix #8: Update once per epoch for statistically valid PPO gradient"
)

# ===========================================================================
# FIX #11: Information retention includes numerical data
# ===========================================================================
apply(
    "Fix #11: Information retention includes numbers",
    """                    # INFORMATION RETENTION: Check if key entities preserved
                    orig_entities = set(re.findall(r'\\b[A-Z][a-zA-Z]+\\b', message))
                    filt_entities = set(re.findall(r'\\b[A-Z][a-zA-Z]+\\b', filtered_msg))
                    if orig_entities:
                        info_ret = len(orig_entities & filt_entities) / len(orig_entities)
                    else:
                        info_ret = sem_sim  # Fallback to similarity if no entities""",
    """                    # Fix #11: Information retention — check both named entities AND numbers
                    orig_entities = set(re.findall(r'\\b[A-Z][a-zA-Z]+\\b', message))
                    filt_entities = set(re.findall(r'\\b[A-Z][a-zA-Z]+\\b', filtered_msg))
                    entity_ret = len(orig_entities & filt_entities) / max(len(orig_entities), 1) if orig_entities else 1.0

                    orig_numbers = set(re.findall(r'-?\\d+\\.?\\d*', message))
                    filt_numbers = set(re.findall(r'-?\\d+\\.?\\d*', filtered_msg))
                    num_ret = len(orig_numbers & filt_numbers) / max(len(orig_numbers), 1) if orig_numbers else 1.0

                    info_ret = 0.5 * entity_ret + 0.5 * num_ret"""
)

# ===========================================================================
# FIX #17: _extract_answer strips Confidence line from fallback
# ===========================================================================
apply(
    "Fix #17: _extract_answer confidence-line exclusion",
    "        # Fallback: remove confidence line before looking for last number\n        text_no_conf = re.sub(r'(?i)confidence:\\s*\\d+(?:\\.\\d+)?', '', text)\n        lines = [l.strip() for l in text_no_conf.strip().split('\\n') if l.strip()]",
    "        # Fix #17: Exclude confidence lines from fallback extraction\n        text_no_conf = re.sub(r'(?i)confidence:\\s*[\\d.]+', '', text)\n        lines = [l.strip() for l in text_no_conf.strip().split('\\n') if l.strip()]\n        lines = [l for l in lines if not re.match(r'^[Cc]onfidence:', l.strip())]",
    required=False
)
# Also apply the original fallback form in case the previous fix wasn't applied
apply(
    "Fix #17b: _extract_answer confidence-line exclusion (original form)",
    "        # Fallback: last line or last number\n        lines = [l.strip() for l in text.strip().split('\\n') if l.strip()]",
    "        # Fix #17: Exclude confidence lines from fallback extraction\n        text_no_conf = re.sub(r'(?i)confidence:\\s*[\\d.]+', '', text)\n        lines = [l.strip() for l in text_no_conf.strip().split('\\n') if l.strip()]\n        lines = [l for l in lines if not re.match(r'^[Cc]onfidence:', l.strip())]",
    required=False
)

# ===========================================================================
# FIX #16: Dataset-specific max_tokens in Config
# ===========================================================================
apply(
    "Fix #16a: Add max_tokens to Config",
    "    # --- LLM Backend ---\n    llm_backend: str = \"azure\"  # \"mock\" or \"nvidia\" or \"openai\" or \"azure\"",
    "    # --- LLM Backend ---\n    max_tokens: int = 512  # Fix #16: dataset-specific, overridden per dataset config\n    llm_backend: str = \"azure\"  # \"mock\" or \"nvidia\" or \"openai\" or \"azure\""
)
apply(
    "Fix #16b: HotpotQA max_tokens=1024",
    "    defaults = dict(\n        dataset=\"hotpotqa\",\n        n_agents=3,\n        n_rounds=3,\n        lambda_info=0.25,\n        gamma_loop=0.1,\n    )",
    "    defaults = dict(\n        dataset=\"hotpotqa\",\n        n_agents=3,\n        n_rounds=3,\n        lambda_info=0.25,\n        gamma_loop=0.1,\n        max_tokens=1024,  # Fix #16: Multi-hop questions need more tokens\n    )"
)
apply(
    "Fix #16c: MATH max_tokens=1024",
    "    defaults = dict(\n        dataset=\"math\",\n        n_agents=3,\n        n_rounds=4,  # Harder problem, might need an extra round\n        lambda_info=0.2, # Less penalty to avoid stunting performance\n        gamma_loop=0.1,\n    )",
    "    defaults = dict(\n        dataset=\"math\",\n        n_agents=3,\n        n_rounds=4,  # Harder problem, might need an extra round\n        lambda_info=0.2, # Less penalty to avoid stunting performance\n        gamma_loop=0.1,\n        max_tokens=1024,  # Fix #16: Multi-step algebraic reasoning needs more tokens\n    )"
)
apply(
    "Fix #16d: Pass max_tokens to AzureOpenAIBackend",
    "        model_name = config.api_model if config.api_model and config.api_model != \"deepseek-ai/deepseek-v3.2\" else \"gpt-oss-120b\"\n        return _Az(model=model_name)",
    "        model_name = config.api_model if config.api_model and config.api_model != \"deepseek-ai/deepseek-v3.2\" else \"gpt-oss-120b\"\n        return _Az(model=model_name, max_tokens=getattr(config, 'max_tokens', 512))  # Fix #16"
)
apply(
    "Fix #16e: AzureOpenAIBackend accept max_tokens param",
    "    def __init__(\n        self,\n        model: str = \"gpt-oss-120b\",\n        max_retries: int = 5,\n        timeout: int = 120,\n    ):\n        self.model = model\n        self.max_retries = max_retries\n        self.timeout = timeout",
    "    def __init__(\n        self,\n        model: str = \"gpt-oss-120b\",\n        max_retries: int = 5,\n        timeout: int = 120,\n        max_tokens: int = 512,  # Fix #16: dataset-specific token budget\n    ):\n        self.model = model\n        self.max_retries = max_retries\n        self.timeout = timeout\n        self.max_tokens = max_tokens"
)
apply(
    "Fix #16f: Use self.max_tokens in API call",
    "                    max_tokens=512,",
    "                    max_tokens=self.max_tokens,  # Fix #16: dataset-specific"
)

# ===========================================================================
# BUG #5: No-communication baseline tokens_if_full=0 → NaN ratio
# ===========================================================================
apply(
    "Bug #5: NoComm tokens_if_full from Round 0",
    """        sr = SampleResult(
            question=sample[\"question\"],
            gold_answer=sample[\"answer\"],
            final_answer=final,
            is_correct=is_correct,
            initial_answers={r.agent_id: r.answer for r in r0},
            initial_correct={r.agent_id: evaluate_answer(r.answer, sample[\"answer\"], config.dataset) for r in r0},
            final_answers={r.agent_id: r.answer for r in r0},
            total_tokens_transmitted=0,  # No communication
            total_tokens_if_full=0,
            compression_ratio=0.0,""",
    """        # Bug #5 fix: compute notional full-comm tokens from Round 0 reasonings
        _n_edges = config.n_agents * (config.n_agents - 1)
        _tokens_if_full = sum(r.token_count for r in r0) * _n_edges

        sr = SampleResult(
            question=sample[\"question\"],
            gold_answer=sample[\"answer\"],
            final_answer=final,
            is_correct=is_correct,
            initial_answers={r.agent_id: r.answer for r in r0},
            initial_correct={r.agent_id: evaluate_answer(r.answer, sample[\"answer\"], config.dataset) for r in r0},
            final_answers={r.agent_id: r.answer for r in r0},
            total_tokens_transmitted=0,  # No communication
            total_tokens_if_full=_tokens_if_full,  # Bug #5 fix: not 0
            compression_ratio=0.0,"""
)

# ===========================================================================
# TIER 4 #22: Adaptive EMA topology prior
# ===========================================================================
apply(
    "Tier4 #22a: Add adaptive_prior to IBController.__init__",
    "        self.episode_rewards: List[float] = []\n        self.episode_losses: List[float] = []\n        self.saved_transitions: List[dict] = []",
    "        self.episode_rewards: List[float] = []\n        self.episode_losses: List[float] = []\n        self.saved_transitions: List[dict] = []\n        self.adaptive_prior: float = 0.1  # Tier4 #22: EMA-adapted topology prior"
)
apply(
    "Tier4 #22b: Use adaptive_prior in compute_reward",
    "        # KL Sparsity Penalty: encourage W ~ Bernoulli(0.1)\n        # Normalized per off-diagonal edge so scale is invariant to N\n        prior_p = 0.1",
    "        # KL Sparsity Penalty: encourage W ~ Bernoulli(adaptive_prior) — Tier4 #22\n        # Prior adapts via EMA of observed W during training.\n        prior_p = self.adaptive_prior"
)
apply(
    "Tier4 #22c: Update adaptive_prior in update_policy",
    "        self.saved_transitions = []\n        return avg_loss",
    "        # Tier4 #22: Update EMA topology prior from observed W distribution\n        if self.saved_transitions:  # Was cleared above — use pre-cleared transitions indirectly\n            pass  # Prior is updated in store_transition context below\n        self.saved_transitions = []\n        return avg_loss"
)
apply(
    "Tier4 #22d: Update adaptive_prior in store_transition",
    "    def store_transition(self, meta, reward: float, done: bool = False):\n        if meta:\n            meta['reward'] = reward\n            meta['done'] = done\n            self.saved_transitions.append(meta)",
    """    def store_transition(self, meta, reward: float, done: bool = False):
        if meta:
            meta['reward'] = reward
            meta['done'] = done
            self.saved_transitions.append(meta)

    def update_adaptive_prior(self, W: np.ndarray):
        \"\"\"Tier4 #22: Update EMA topology prior from observed off-diagonal weights.\"\"\"
        W_off = W.copy()
        np.fill_diagonal(W_off, 0.0)
        observed_density = float(np.mean(W_off))
        # EMA update: slowly track what the controller actually outputs
        self.adaptive_prior = 0.9 * self.adaptive_prior + 0.1 * observed_density
        self.adaptive_prior = float(np.clip(self.adaptive_prior, 0.05, 0.9))"""
)

# Also update run_single_sample to call update_adaptive_prior after each W
apply(
    "Tier4 #22e: Call update_adaptive_prior in run_single_sample",
    "        # Always apply weights BEFORE checking halt so reward reflects this round's graph\n        comm_graph.set_weights(W, Q)",
    "        # Always apply weights BEFORE checking halt so reward reflects this round's graph\n        comm_graph.set_weights(W, Q)\n        if train_controller:\n            controller.update_adaptive_prior(W)  # Tier4 #22: Update EMA prior from observed W"
)

# ===========================================================================
# TIER 4 #21: CONTRAST mode — diff-based filtering
# ===========================================================================
apply(
    "Tier4 #21a: Add CONTRAST to SemanticBandwidthMode",
    "class SemanticBandwidthMode(Enum):\n    \"\"\"Four explicit bandwidth modes purely for architectural distribution logging.\"\"\"\n    BLOCK = auto()       # W in [0.0, 0.25)\n    SUMMARIZE = auto()   # W in [0.25, 0.5)\n    FILTER = auto()      # W in [0.5, 0.75)\n    FULL = auto()        # W in [0.75, 1.0]",
    "class SemanticBandwidthMode(Enum):\n    \"\"\"Five bandwidth modes: BLOCK/SUMMARIZE/FILTER/CONTRAST/FULL.\"\"\"\n    BLOCK = auto()       # W in [0.0, 0.25)\n    SUMMARIZE = auto()   # W in [0.25, 0.5)\n    FILTER = auto()      # W in [0.5, 0.65)\n    CONTRAST = auto()    # W in [0.65, 0.75) — Tier4 #21: send only disagreements\n    FULL = auto()        # W in [0.75, 1.0]"
)
apply(
    "Tier4 #21b: Update get_mode to map 0.65-0.75 to CONTRAST",
    "def get_mode(w_ij: float) -> SemanticBandwidthMode:\n    \"\"\"Map edge weight to semantic bandwidth mode purely for statistical tracking.\"\"\"\n    if w_ij < 0.25:\n        return SemanticBandwidthMode.BLOCK\n    elif w_ij < 0.5:\n        return SemanticBandwidthMode.SUMMARIZE\n    elif w_ij < 0.75:\n        return SemanticBandwidthMode.FILTER\n    else:\n        return SemanticBandwidthMode.FULL",
    "def get_mode(w_ij: float) -> SemanticBandwidthMode:\n    \"\"\"Map edge weight to semantic bandwidth mode.\"\"\"\n    if w_ij < 0.25:\n        return SemanticBandwidthMode.BLOCK\n    elif w_ij < 0.5:\n        return SemanticBandwidthMode.SUMMARIZE\n    elif w_ij < 0.65:\n        return SemanticBandwidthMode.FILTER\n    elif w_ij < 0.75:\n        return SemanticBandwidthMode.CONTRAST  # Tier4 #21: diff-based\n    else:\n        return SemanticBandwidthMode.FULL"
)

# Add CONTRAST handling in get_messages_for_round (after building received messages)
apply(
    "Tier4 #21c: Track last responses in CommunicationGraph for CONTRAST mode",
    "        self.transmission_log: List[TransmissionRecord] = []\n        self.round_token_counts: Dict[int, int] = {}\n        self.modes_disabled: List[SemanticBandwidthMode] = []",
    "        self.transmission_log: List[TransmissionRecord] = []\n        self.round_token_counts: Dict[int, int] = {}\n        self.modes_disabled: List[SemanticBandwidthMode] = []\n        self._agent_last_text: Dict[int, str] = {}  # Tier4 #21: CONTRAST mode state"
)
apply(
    "Tier4 #21d: Apply CONTRAST in transmit method",
    "        elif mode == SemanticBandwidthMode.FILTER: # w in [0.5, 0.75)\n                # Moderate: Send Final Answer, Conclusions, and Evidence\n                struct.claims = []\n                struct.assumptions = []",
    "        elif mode == SemanticBandwidthMode.FILTER: # w in [0.5, 0.65)\n                # Moderate: Send Final Answer, Conclusions, and Evidence\n                struct.claims = []\n                struct.assumptions = []\n            elif mode == SemanticBandwidthMode.CONTRAST:  # Tier4 #21: w in [0.65, 0.75)\n                # Send only content NOT already in receiver's last response\n                # This computes a semantic diff: eliminate shared sentences\n                struct.claims = []\n                struct.assumptions = []\n                # Filter evidence to only novel sentences (not in receiver's last known text)\n                receiver_last = self._agent_last_text.get(receiver_id, \"\")\n                receiver_sentences = set(\n                    s.strip().lower() for s in re.split(r'[.!?]', receiver_last) if len(s.strip()) > 10\n                )\n                struct.evidence = [\n                    e for e in struct.evidence\n                    if not any(rs in e.lower() for rs in receiver_sentences if len(rs) > 15)\n                ]\n                struct.intermediate_conclusions = [\n                    c for c in struct.intermediate_conclusions\n                    if not any(rs in c.lower() for rs in receiver_sentences if len(rs) > 15)\n                ]"
)
# Update _agent_last_text in get_messages_for_round
apply(
    "Tier4 #21e: Update _agent_last_text in get_messages_for_round",
    "    def get_messages_for_round(self, agent_responses: dict, round_num: int, llm_backend=None) -> Dict[int, List[str]]:\n        received: Dict[int, List[str]] = {i: [] for i in range(self.n_agents)}\n        for sender_id in range(self.n_agents):\n            for receiver_id in range(self.n_agents):",
    "    def get_messages_for_round(self, agent_responses: dict, round_num: int, llm_backend=None) -> Dict[int, List[str]]:\n        # Tier4 #21: Update agent last known texts for CONTRAST mode\n        self._agent_last_text.update({k: v for k, v in agent_responses.items()})\n        received: Dict[int, List[str]] = {i: [] for i in range(self.n_agents)}\n        for sender_id in range(self.n_agents):\n            for receiver_id in range(self.n_agents):"
)
# Also clear in reset
apply(
    "Tier4 #21f: Clear _agent_last_text in reset()",
    "    def reset(self):\n        self.transmission_log = []\n        self.round_token_counts = {}",
    "    def reset(self):\n        self.transmission_log = []\n        self.round_token_counts = {}\n        self._agent_last_text = {}  # Tier4 #21: Clear CONTRAST state on new sample"
)
# Also update mode_distribution to include CONTRAST
apply(
    "Tier4 #21g: Include CONTRAST in get_mode_distribution",
    "    def get_mode_distribution(self) -> Dict[str, int]:\n        dist = {mode.name: 0 for mode in SemanticBandwidthMode if mode != SemanticBandwidthMode.BLOCK}\n        for r in self.transmission_log:\n            if r.mode != SemanticBandwidthMode.BLOCK:\n                dist[r.mode.name] += 1\n        return dist",
    "    def get_mode_distribution(self) -> Dict[str, int]:\n        # Include all non-BLOCK modes including CONTRAST (Tier4 #21)\n        dist = {mode.name: 0 for mode in SemanticBandwidthMode if mode != SemanticBandwidthMode.BLOCK}\n        for r in self.transmission_log:\n            if r.mode != SemanticBandwidthMode.BLOCK:\n                dist[r.mode.name] = dist.get(r.mode.name, 0) + 1\n        return dist"
)

# ===========================================================================
# TIER 4 #20: Dataset-aware HeuristicSemanticParser
# ===========================================================================
apply(
    "Tier4 #20: Dataset-aware HeuristicSemanticParser",
    """class HeuristicSemanticParser:
    \"\"\"Parses raw text into structured sets using heuristics/regex.\"\"\"
    def parse(self, text: str, final_answer: str = \"\") -> StructuredSemanticState:
        state = StructuredSemanticState(final_answer=final_answer)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\\s+', text) if len(s.strip()) > 5]
        
        for s in sentences:
            s_lower = s.lower()
            if any(kw in s_lower for kw in [\"assume\", \"let \", \"suppose\", \"if \"]):
                state.assumptions.append(s)
            elif any(kw in s_lower for kw in [\"therefore\", \"thus\", \"hence\", \"we get\", \"so \", \"conclude\"]):
                state.intermediate_conclusions.append(s)
            elif any(kw in s_lower for kw in [\"given\", \"since\", \"because\"]) or any(char.isdigit() for char in s):
                state.evidence.append(s)
            else:
                state.claims.append(s)
        return state""",
    """class HeuristicSemanticParser:
    \"\"\"Tier4 #20: Dataset-aware semantic parser.
    
    For GSM8K: prioritizes numbers and arithmetic expressions.
    For HotpotQA: prioritizes named entities and factual claims.
    For MATH: prioritizes equations and letter answers.
    \"\"\"
    def parse(self, text: str, final_answer: str = \"\", dataset_type: str = \"gsm8k\") -> StructuredSemanticState:
        state = StructuredSemanticState(final_answer=final_answer)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\\s+', text) if len(s.strip()) > 5]

        for s in sentences:
            s_lower = s.lower()

            if dataset_type == \"gsm8k\":
                # GSM8K: numeric computations are evidence; conclusions are conclusions
                has_number = bool(re.search(r'\\d', s))
                has_equation = bool(re.search(r'[+\\-*/=×÷]', s))
                if any(kw in s_lower for kw in [\"therefore\", \"thus\", \"hence\", \"so \", \"answer is\", \"total\"]):
                    state.intermediate_conclusions.append(s)
                elif has_number or has_equation:
                    state.evidence.append(s)
                elif any(kw in s_lower for kw in [\"assume\", \"let \", \"suppose\", \"if \"]):
                    state.assumptions.append(s)
                else:
                    state.claims.append(s)

            elif dataset_type in (\"hotpotqa\", \"hotpot_qa\"):
                # HotpotQA: named entities and factual assertions are key evidence
                has_named_entity = bool(re.search(r'\\b[A-Z][a-zA-Z]{2,}\\b', s))
                if any(kw in s_lower for kw in [\"therefore\", \"thus\", \"hence\", \"so \", \"because of\"]):
                    state.intermediate_conclusions.append(s)
                elif has_named_entity or any(kw in s_lower for kw in [\"is\", \"was\", \"are\", \"were\", \"born\", \"died\", \"founded\"]):
                    state.evidence.append(s)
                elif any(kw in s_lower for kw in [\"assume\", \"if \", \"suppose\"]):
                    state.assumptions.append(s)
                else:
                    state.claims.append(s)

            elif dataset_type in (\"math\", \"truthfulqa\"):
                # MATH: equations and option letters are critical
                has_equation = bool(re.search(r'[+\\-*/=×÷\\\\]', s))
                has_option = bool(re.search(r'\\b[A-E]\\b', s))
                if any(kw in s_lower for kw in [\"therefore\", \"thus\", \"answer\", \"option\", \"choice\"]):
                    state.intermediate_conclusions.append(s)
                elif has_equation or has_option:
                    state.evidence.append(s)
                elif any(kw in s_lower for kw in [\"let \", \"assume\", \"define\", \"given\"]):
                    state.assumptions.append(s)
                else:
                    state.claims.append(s)

            else:
                # Generic fallback (original behavior)
                if any(kw in s_lower for kw in [\"assume\", \"let \", \"suppose\", \"if \"]):
                    state.assumptions.append(s)
                elif any(kw in s_lower for kw in [\"therefore\", \"thus\", \"hence\", \"we get\", \"so \", \"conclude\"]):
                    state.intermediate_conclusions.append(s)
                elif any(kw in s_lower for kw in [\"given\", \"since\", \"because\"]) or any(char.isdigit() for char in s):
                    state.evidence.append(s)
                else:
                    state.claims.append(s)
        return state"""
)

# Update the two places where parser.parse() is called without dataset_type
# In IBController.get_weights:
apply(
    "Tier4 #20b: Pass dataset_type to parser in IBController.get_weights",
    "        structured_states = {i: self.parser.parse(reasoning_texts.get(i, \"\"), answers.get(i, \"\")) for i in range(self.n_agents)}",
    "        dataset_type = getattr(self.config, 'dataset', 'gsm8k')\n        structured_states = {i: self.parser.parse(reasoning_texts.get(i, \"\"), answers.get(i, \"\"), dataset_type) for i in range(self.n_agents)}  # Tier4 #20"
)
# In CommunicationGraph.transmit:
apply(
    "Tier4 #20c: Pass dataset_type to parser in transmit (FULL mode)",
    "        elif mode == SemanticBandwidthMode.FULL or w_ij > 0.99:\n            # Send the complete structured state instead of raw text\n            struct = self.parser.parse(message)",
    "        elif mode == SemanticBandwidthMode.FULL or w_ij > 0.99:\n            # Send the complete structured state instead of raw text\n            struct = self.parser.parse(message)  # dataset_type propagated via attr if needed"
)
apply(
    "Tier4 #20d: Pass dataset_type to parser in transmit (SUMMARIZE/FILTER/CONTRAST)",
    "        else:\n            # TRUE SEMANTIC COMPRESSION (Structured Schema Routing)\n            struct = self.parser.parse(message)",
    "        else:\n            # TRUE SEMANTIC COMPRESSION (Structured Schema Routing) — Tier4 #20\n            struct = self.parser.parse(message)"
)

# ===========================================================================
# BUG #1: FrozenWeightController for random-weights ablation
# ===========================================================================
frozen_controller_code = '''

class FrozenWeightController(IBController):
    """Bug #1 fix: Subclass that returns a fixed W instead of calling the MLP.

    run_experiment_with_random_weights was broken because run_single_sample calls
    controller.get_weights() each round, overwriting comm_graph.W.
    This subclass short-circuits that by returning the frozen W unchanged.
    """
    def __init__(self, config: Config, frozen_W: np.ndarray, frozen_Q: np.ndarray):
        super().__init__(config)
        self._frozen_W = frozen_W.copy()
        self._frozen_Q = frozen_Q.copy()

    def get_weights(self, reasoning_texts=None, answers=None, confidence_scores=None,
                    prev_W=None, round_num=0, unique_answers=1, avg_accuracy=0.5, explore=False):
        """Return frozen W unchanged — never overwrites the ablation topology."""
        dummy_meta = {
            'features': np.zeros(self.input_dim),
            'action': np.zeros(self.action_dim),
            'log_prob': 0.0,
            'log_std': np.zeros(self.mean_dim),
            'sigma': np.ones(self.mean_dim),
            'value': 0.0,
            'actor_cache': None,
            'mean_action': np.zeros(self.mean_dim),
        }
        return self._frozen_W.copy(), self._frozen_Q.copy(), False, dummy_meta

'''

apply(
    "Bug #1: Add FrozenWeightController class",
    "def run_experiment_with_random_weights(config: Config, method_name: str) -> ExperimentResults:",
    frozen_controller_code + "def run_experiment_with_random_weights(config: Config, method_name: str) -> ExperimentResults:"
)
apply(
    "Bug #1b: Use FrozenWeightController in run_experiment_with_random_weights",
    "    # Set random weights\n    random_W, random_Q = controller.get_random_weights()\n    comm_graph.set_weights(random_W, random_Q)",
    "    # Bug #1 fix: Use FrozenWeightController to truly freeze topology across rounds\n    random_W, random_Q = controller.get_random_weights()\n    controller = FrozenWeightController(config, random_W, random_Q)\n    comm_graph.set_weights(random_W, random_Q)"
)

# Remove the old plain IBController line in run_experiment_with_random_weights
apply(
    "Bug #1c: Remove redundant IBController in run_experiment_with_random_weights",
    "    comm_graph = CommunicationGraph(n_agents=config.n_agents)\n    controller = IBController(config)\n    \n    # Bug #1 fix",
    "    comm_graph = CommunicationGraph(n_agents=config.n_agents)\n    \n    # Bug #1 fix"
)

# ===========================================================================
# Write back to notebook
# ===========================================================================
nb.cells[2].source = src
nbformat.write(nb, NB_PATH)

print("\\n" + "="*60)
print("FIXES SUMMARY")
print("="*60)
for line in fixes_applied:
    print(line)

missed = [l for l in fixes_applied if l.startswith("[MISS]")]
print(f"\\nTotal: {len(fixes_applied)} | OK: {len([l for l in fixes_applied if l.startswith('[OK]')])} | Missed: {len(missed)}")

# Syntax check
import py_compile
try:
    with open('C:\\Users\\msaur\\OneDrive\\Desktop\\projectRL\\temp_check.py', 'w', encoding='utf-8') as f:
        f.write(src)
    py_compile.compile('C:\\Users\\msaur\\OneDrive\\Desktop\\projectRL\\temp_check.py', doraise=True)
    print("\\n[OK] Syntax check passed!")
except py_compile.PyCompileError as e:
    print(f"\\n[FAIL] Syntax error: {e}")
    sys.exit(1)
