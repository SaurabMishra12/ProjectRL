"""
Apply all 6 mathematical fixes to temp_check.py.
Handles CRLF line endings and UTF-8 em-dash characters.
"""

import re

# Read as binary then decode so we get exact CRLF content
with open('temp_check.py', 'r', encoding='utf-8', newline='') as f:
    src = f.read()

# Normalise to LF for matching, will write back with original endings
src_lf = src.replace('\r\n', '\n')

def apply(src, old, new, tag):
    old_lf = old.replace('\r\n', '\n')
    new_lf = new.replace('\r\n', '\n')
    assert old_lf in src, f"{tag}: target not found"
    return src.replace(old_lf, new_lf)

# ============================================================
# FIX 1: Remove consistency bonus from _compute_confidence
# ============================================================
F1_OLD = '''    def _compute_confidence(self, llm_response: LLMResponse) -> float:
        """
        Compute confidence based on LLM's internal estimate and
        consistency with previous answers.
        """
        base_confidence = llm_response.confidence

        if not self.history:
            return base_confidence

        # Check consistency with previous rounds
        prev_answers = [r.answer for r in self.history[-3:]]
        if prev_answers and llm_response.answer in prev_answers:
            # Consistent with previous answers -> boost confidence
            consistency_count = prev_answers.count(llm_response.answer)
            consistency_bonus = 0.1 * consistency_count / len(prev_answers)
            return min(0.95, base_confidence + consistency_bonus)
        elif prev_answers:
            # Changed answer -> reduce confidence slightly
            return max(0.1, base_confidence - 0.05)

        return base_confidence'''

F1_NEW = '''    def _compute_confidence(self, llm_response: LLMResponse) -> float:
        """
        Fix 1: Return raw LLM confidence without modification.

        Consistency bonus removed. The isotonic regression calibrator in
        PlattCalibrator learns P(correct | raw_conf) from (conf, label) pairs.
        Adding a consistency bonus pre-calibration distorts the learned mapping.
        """
        return llm_response.confidence'''

src_lf = apply(src_lf, F1_OLD, F1_NEW, "FIX 1")
print("Fix 1 applied.")

# ============================================================
# FIX 2: Remove shaping rewards; drop confidence_improvement / answer_entropy_reduction
# ============================================================
F2_SIG_OLD = '''    def compute_reward(
        self,
        task_correct: bool,
        tokens_transmitted: int,
        tokens_if_full: int,
        compression_ratio: float,
        n_cycles: int,
        current_W: np.ndarray,
        halt_round: int = 0,
        confidence_improvement: float = 0.0,
        answer_entropy_reduction: float = 0.0,
        is_final_step: bool = True,
        include_token_penalty: bool = False,  # Fix2: False by default \u2014 intermediate steps handle per-round tokens
    ) -> float:
        # Task reward: I(Z; Y) - accuracy signal
        task_rew = (1.0 if task_correct else -0.5) if is_final_step else 0.0

        # Shaping rewards (only at final step to avoid double-counting)
        shaping_reward = 0.0
        if is_final_step:
            shaping_reward += confidence_improvement * 0.3
            shaping_reward += answer_entropy_reduction * 0.2
            shaping_reward += 0.05

        # Fix2: Token penalty is applied per-round in run_single_sample intermediate steps.
        # include_token_penalty=False (default) avoids double-counting the total token cost
        # in the final compute_reward call on top of already-penalised intermediate steps.
        info_cost = (tokens_transmitted / max(float(tokens_if_full), 1.0)) if include_token_penalty else 0.0

        # KL Sparsity Penalty: encourage W ~ Bernoulli(adaptive_prior) \u2014 Tier4 #22
        # Prior adapts via EMA of observed W during training.
        prior_p = self.adaptive_prior'''

F2_SIG_NEW = '''    def compute_reward(
        self,
        task_correct: bool,
        tokens_transmitted: int,
        tokens_if_full: int,
        compression_ratio: float,
        n_cycles: int,
        current_W: np.ndarray,
        halt_round: int = 0,
        is_final_step: bool = True,
        include_token_penalty: bool = False,
    ) -> float:
        """
        Fix 2: Pure task + cost reward. All shaping terms removed.

        R = task_rew - lambda_info*cost - lambda_kl*KL - gamma_loop*loop - halt_penalty

        Shaping rewards (+alpha*delta_entropy, +beta*delta_confidence) can be maximised
        even when the answer is wrong (agents become confidently wrong together).
        Only the gold-answer-tied task reward provides a valid learning signal.
        Intermediate steps receive zero task reward; the terminal reward is sufficient.
        """
        # +1.0 correct, -0.5 wrong (final step only); 0.0 at intermediate steps
        task_rew = (1.0 if task_correct else -0.5) if is_final_step else 0.0

        info_cost = (tokens_transmitted / max(float(tokens_if_full), 1.0)) if include_token_penalty else 0.0

        # Fix 4: use module-level SPARSITY_PRIOR constant (stationary, interpretable)
        prior_p = SPARSITY_PRIOR'''

src_lf = apply(src_lf, F2_SIG_OLD, F2_SIG_NEW, "FIX 2 signature")
print("Fix 2 (signature) applied.")

F2_RET_OLD = '''        return (
            task_rew
            + shaping_reward
            - self.lambda_info * info_cost
            - self.lambda_kl * kl_divergence
            - self.gamma_loop * loop_penalty
            - halt_penalty
        )'''
F2_RET_NEW = '''        return (
            task_rew
            - self.lambda_info * info_cost
            - self.lambda_kl * kl_divergence
            - self.gamma_loop * loop_penalty
            - halt_penalty
        )'''
src_lf = apply(src_lf, F2_RET_OLD, F2_RET_NEW, "FIX 2 return")
print("Fix 2 (return) applied.")

F2_CALL_OLD = '''    step_reward = controller.compute_reward(
        task_correct=is_correct if is_final else False,
        tokens_transmitted=tokens_transmitted,
        tokens_if_full=tokens_if_full,
        compression_ratio=compression_ratio,
        n_cycles=n_cycles,
        current_W=W,
        halt_round=t,
        is_final_step=is_final,
        include_token_penalty=is_final,
    )'''
F2_CALL_NEW = '''    step_reward = controller.compute_reward(
        task_correct=is_correct if is_final else False,
        tokens_transmitted=tokens_transmitted,
        tokens_if_full=tokens_if_full,
        compression_ratio=compression_ratio,
        n_cycles=n_cycles,
        current_W=W,
        halt_round=t,
        is_final_step=is_final,
        include_token_penalty=is_final,
    )'''
# This one is unchanged since we already removed conf/entropy params earlier; verify it exists
assert F2_CALL_NEW.replace('\r\n','\n') in src_lf, "FIX 2 call: already correct or not found - checking"
print("Fix 2 (call-site): already correct (confidence_improvement/answer_entropy_reduction already removed).")

# ============================================================
# FIX 3: SPRT with calibrated p_maj, tau=log(19)~2.944
# ============================================================
F3_OLD = '''        # V5 Priority #6: SPRT Halting Logic
        # H1: Consensus reached (p_agree = 0.9)
        # H0: Random noise (p_agree = 0.1)
        p1 = 0.9; p0 = 0.1
        ans_list = [_normalize_for_vote(r.answer, config.dataset) for r in current_responses]
        agreements = 0
        total_pairs = 0
        for i in range(len(ans_list)):
            for j in range(i+1, len(ans_list)):
                agreements += int(ans_list[i] == ans_list[j])
                total_pairs += 1

        if total_pairs > 0:
            k = agreements; n = total_pairs
            llr = k * np.log(p1/p0) + (n - k) * np.log((1-p1)/(1-p0))
            sprt_halt = llr > 2.0  # Strong evidence threshold for H1
        else:
            sprt_halt = False'''

F3_NEW = '''        # Fix 3: SPRT Halting with calibrated p_maj
        # H0: majority answer is random (p=0.5)
        # H1: majority answer is correct (p=p_maj, from calibrated confidence)
        # LLR = k*log(p_maj/0.5) + (N-k)*log((1-p_maj)/0.5)
        # Halt when LLR > tau = log((1-alpha)/alpha), alpha=0.05 -> tau~2.944
        _SPRT_TAU = 2.944    # log(19), alpha=0.05
        _SPRT_P0  = 0.5      # H0: random baseline
        ans_list = [_normalize_for_vote(r.answer, config.dataset) for r in current_responses]
        from collections import Counter as _Counter
        _ans_counts = _Counter(ans_list)
        _majority_ans, _k = _ans_counts.most_common(1)[0]
        _N = len(ans_list)
        # Calibrated p_maj: max calibrated confidence among agents voting for majority answer
        _maj_confs = [
            r.confidence for r in current_responses
            if _normalize_for_vote(r.answer, config.dataset) == _majority_ans
        ]
        if controller.calibrator._fitted and _maj_confs:
            _p_maj = float(np.clip(
                controller.calibrator.calibrate(float(np.max(_maj_confs))), 0.501, 0.999
            ))
        else:
            _p_maj = 0.7  # Conservative fallback when calibrator not yet fitted
        _llr = (
            _k * np.log(_p_maj / _SPRT_P0) +
            (_N - _k) * np.log((1.0 - _p_maj) / (1.0 - _SPRT_P0))
        )
        sprt_halt = _llr > _SPRT_TAU'''

src_lf = apply(src_lf, F3_OLD, F3_NEW, "FIX 3")
print("Fix 3 applied.")

# ============================================================
# FIX 4: Fixed KL prior SPARSITY_PRIOR = 0.1
# ============================================================
F4A_OLD = '''class IBController:
    """'''
F4A_NEW = '''# Fix 4: Fixed sparsity prior. KL divergence target is Bernoulli(0.1).
# 90% of edges should be OFF on average. Stationary and interpretable.
SPARSITY_PRIOR: float = 0.1

class IBController:
    """'''
src_lf = apply(src_lf, F4A_OLD, F4A_NEW, "FIX 4a")
print("Fix 4a applied.")

F4B_OLD = '        self.adaptive_prior: float = 0.1  # Tier4 #22: EMA-adapted topology prior\n'
F4B_NEW = '        # Fix 4: adaptive_prior removed; SPARSITY_PRIOR = 0.1 constant used instead\n'
src_lf = apply(src_lf, F4B_OLD, F4B_NEW, "FIX 4b")
print("Fix 4b applied.")

F4C_OLD = '''    def update_adaptive_prior(self, W: np.ndarray):
        """Tier4 #22: EMA update of sparsity prior from observed topology.
        Called once per training epoch (not per round) to reduce non-stationarity.
        """
        observed_density = float(np.mean(W[~np.eye(W.shape[0], dtype=bool)]))
        self.adaptive_prior = 0.9 * self.adaptive_prior + 0.1 * observed_density
        self.adaptive_prior = float(np.clip(self.adaptive_prior, 0.05, 0.9))
'''
F4C_NEW = '''    # Fix 4: update_adaptive_prior() removed.
    # KL prior is now the module-level SPARSITY_PRIOR = 0.1 constant.
    # This eliminates epoch-boundary non-stationarity in the reward signal.
'''
src_lf = apply(src_lf, F4C_OLD, F4C_NEW, "FIX 4c")
print("Fix 4c applied.")

F4D_OLD = '            controller.update_adaptive_prior(comm_graph.get_weight_snapshot())\n'
F4D_NEW = '            # Fix 4: adaptive_prior update removed (fixed SPARSITY_PRIOR=0.1 used instead)\n'
src_lf = apply(src_lf, F4D_OLD, F4D_NEW, "FIX 4d")
print("Fix 4d applied.")

# ============================================================
# FIX 5: Replace confounded 2-arm ablation with 2^2 factorial
# ============================================================
F5_OLD = '''# Ablation A: Sparse-only (fixed RING topology, no PPO)
# Bug #11 Fix: was using random weights; ring topology is the correct sparse-only baseline
print("\\n  Ablation A: Sparse-only (fixed ring topology, no PPO)...")
_ring_ctrl = IBController(ablation_config)
_ring_W, _ring_Q = _ring_ctrl.get_static_weights("ring")
del _ring_ctrl
from copy import deepcopy as _deepcopy
_ring_frozen = FrozenWeightController(ablation_config, _ring_W, _ring_Q)
_ring_comm = CommunicationGraph(n_agents=ablation_config.n_agents)
_ring_comm.set_weights(_ring_W, _ring_Q)
_ring_pool = AgentPool(ablation_config.n_agents, create_llm_backend(ablation_config), ablation_config.reasoning_styles)
_, _ring_test = load_dataset_by_name(ablation_config.dataset, n_train=0, n_test=ablation_config.n_samples, seed=ablation_config.seed)
_ring_results = ExperimentResults(method_name="Ablation_SparseOnly", dataset_name=ablation_config.dataset)
for _s in tqdm(_ring_test, desc="Ablation_SparseOnly"):
    _r = run_single_sample(
        question=_s["question"], gold_answer=_s["answer"],
        agent_pool=_ring_pool, comm_graph=_ring_comm,
        controller=_ring_frozen, config=ablation_config, train_controller=False,
    )
    _ring_results.sample_results.append(_r)
sparse_only = _ring_results
ablation_results["Ablation_SparseOnly"] = sparse_only

# Ablation B: PPO-only (no sparsity: lambda_info = 0, KL weight implicitly 0)
print("\\n  Ablation B: PPO-only (dense fully-connected, no sparsity constraints)...")
no_sparse_config = _deepcopy(ablation_config)
no_sparse_config.lambda_info = 0.0   # No token cost penalty
no_sparse_config.lambda_kl = 0.0     # Bug fix: Disable KL penalty for dense graph
no_sparse_config.gamma_loop = 0.0    # No cycle penalty

# Pretrain controller for ablation without sparsity
temp_controller = IBController(no_sparse_config)
ppo_only = run_experiment(no_sparse_config, method_name="Ablation_PPOOnly", train=True, evaluate=True, controller=temp_controller)
ablation_results["Ablation_PPOOnly"] = ppo_only'''

F5_NEW = '''# =========================================================
# Fix 5: 2^2 Factorial Ablation Design
# Factor A: Sparsity (KL penalty)   0=off, 1=on
# Factor B: PPO learning             0=off (fixed ring), 1=on
# Calibrator: always ON (post-hoc mapping, not a structural choice)
#
# | A | B | Arm name           |
# |---|---|--------------------|
# | 0 | 0 | Arm_00_StaticRing  | Fixed ring, no KL, no PPO (pure baseline)
# | 0 | 1 | Arm_01_PPOOnly     | PPO learns topology, no sparsity pressure
# | 1 | 0 | Arm_10_SparseRing  | Fixed ring with KL penalty
# | 1 | 1 | Arm_11_Full        | PPO + KL sparsity (full system)
#
# OLS after collecting: y = beta0 + betaA*A + betaB*B + betaAB*AB + eps
# =========================================================
from copy import deepcopy as _deepcopy

# ---- ARM 00: Static ring, no KL, no PPO ----
print("\\n  Arm 00: Static ring, no KL, no PPO (baseline)...")
_a00_cfg = _deepcopy(ablation_config)
_a00_cfg.lambda_kl = 0.0; _a00_cfg.lambda_info = 0.0; _a00_cfg.gamma_loop = 0.0
_a00_tmp = IBController(_a00_cfg)
_a00_W, _a00_Q = _a00_tmp.get_static_weights("ring"); del _a00_tmp
_a00_frozen = FrozenWeightController(_a00_cfg, _a00_W, _a00_Q)
_a00_comm = CommunicationGraph(n_agents=_a00_cfg.n_agents); _a00_comm.set_weights(_a00_W, _a00_Q)
_a00_pool = AgentPool(_a00_cfg.n_agents, create_llm_backend(_a00_cfg), _a00_cfg.reasoning_styles)
_, _a00_test = load_dataset_by_name(_a00_cfg.dataset, n_train=0, n_test=_a00_cfg.n_samples, seed=_a00_cfg.seed)
_a00_res = ExperimentResults(method_name="Arm_00_StaticRing", dataset_name=_a00_cfg.dataset)
for _s in tqdm(_a00_test, desc="Arm_00_StaticRing"):
    _a00_res.sample_results.append(run_single_sample(
        question=_s["question"], gold_answer=_s["answer"],
        agent_pool=_a00_pool, comm_graph=_a00_comm,
        controller=_a00_frozen, config=_a00_cfg, train_controller=False,
    ))
ablation_results["Arm_00_StaticRing"] = _a00_res
save_kaggle_checkpoint({**ablation_results}, filepath="08-06results/ablation_arm00.json")

# ---- ARM 01: PPO learns topology, KL=0 (no sparsity pressure) ----
print("\\n  Arm 01: PPO-only, no KL sparsity...")
_a01_cfg = _deepcopy(ablation_config); _a01_cfg.lambda_kl = 0.0; _a01_cfg.gamma_loop = 0.0
_a01_ctrl = IBController(_a01_cfg)
ablation_results["Arm_01_PPOOnly"] = run_experiment(
    _a01_cfg, method_name="Arm_01_PPOOnly", train=True, evaluate=True, controller=_a01_ctrl)
save_kaggle_checkpoint({**ablation_results}, filepath="08-06results/ablation_arm01.json")

# ---- ARM 10: Fixed sparse ring, KL penalty active, no PPO ----
print("\\n  Arm 10: Fixed ring with KL penalty, no PPO...")
_a10_cfg = _deepcopy(ablation_config)  # lambda_kl kept from ablation_config (A=1)
_a10_tmp = IBController(_a10_cfg)
_a10_W, _a10_Q = _a10_tmp.get_static_weights("ring"); del _a10_tmp
_a10_frozen = FrozenWeightController(_a10_cfg, _a10_W, _a10_Q)
_a10_comm = CommunicationGraph(n_agents=_a10_cfg.n_agents); _a10_comm.set_weights(_a10_W, _a10_Q)
_a10_pool = AgentPool(_a10_cfg.n_agents, create_llm_backend(_a10_cfg), _a10_cfg.reasoning_styles)
_, _a10_test = load_dataset_by_name(_a10_cfg.dataset, n_train=0, n_test=_a10_cfg.n_samples, seed=_a10_cfg.seed)
_a10_res = ExperimentResults(method_name="Arm_10_SparseRing", dataset_name=_a10_cfg.dataset)
for _s in tqdm(_a10_test, desc="Arm_10_SparseRing"):
    _a10_res.sample_results.append(run_single_sample(
        question=_s["question"], gold_answer=_s["answer"],
        agent_pool=_a10_pool, comm_graph=_a10_comm,
        controller=_a10_frozen, config=_a10_cfg, train_controller=False,
    ))
ablation_results["Arm_10_SparseRing"] = _a10_res
save_kaggle_checkpoint({**ablation_results}, filepath="08-06results/ablation_arm10.json")

# ---- ARM 11: PPO + KL sparsity (full system, reuse Phase 1 pretrained controller) ----
print("\\n  Arm 11: PPO + KL sparsity (full system)...")
_a11_ctrl = pretrained_controllers.get("gsm8k")
if _a11_ctrl is None:
    _a11_ctrl = IBController(ablation_config)
    ablation_results["Arm_11_Full"] = run_experiment(
        ablation_config, method_name="Arm_11_Full", train=True, evaluate=True, controller=_a11_ctrl)
else:
    ablation_results["Arm_11_Full"] = run_experiment(
        ablation_config, method_name="Arm_11_Full", train=False, evaluate=True, controller=_a11_ctrl)
save_kaggle_checkpoint({**ablation_results}, filepath="08-06results/ablation_arm11.json")

# ---- OLS: main effects and 2-way interaction ----
try:
    _A  = np.array([0., 0., 1., 1.])   # Sparsity
    _B  = np.array([0., 1., 0., 1.])   # PPO
    _AB = _A * _B
    _X  = np.column_stack([np.ones(4), _A, _B, _AB])
    _order = ["Arm_00_StaticRing", "Arm_01_PPOOnly", "Arm_10_SparseRing", "Arm_11_Full"]
    _y = np.array([
        float(np.mean([s.correct for s in ablation_results[arm].sample_results]))
        for arm in _order
    ])
    _beta = np.linalg.lstsq(_X, _y, rcond=None)[0]
    print("\\n  2^2 Factorial OLS (accuracy):")
    print(f"    intercept:          {_beta[0]:.4f}")
    print(f"    beta_A (sparsity):  {_beta[1]:+.4f}")
    print(f"    beta_B (PPO):       {_beta[2]:+.4f}")
    print(f"    beta_AB (interact): {_beta[3]:+.4f}")
    ablation_results["factorial_ols_betas"] = {
        "intercept": float(_beta[0]), "beta_A_sparsity": float(_beta[1]),
        "beta_B_ppo": float(_beta[2]), "beta_AB_interaction": float(_beta[3]),
    }
except Exception as _e:
    print(f"  [WARN] OLS failed: {_e}")'''

src_lf = apply(src_lf, F5_OLD, F5_NEW, "FIX 5")
print("Fix 5 applied.")

# Write back
with open('temp_check.py', 'w', encoding='utf-8', newline='') as f:
    f.write(src_lf)
print("\nAll fixes written successfully.")
