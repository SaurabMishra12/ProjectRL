"""Apply fixes 4 and 5 to temp_check.py."""

with open('temp_check.py', 'r', encoding='utf-8', newline='') as f:
    src = f.read().replace('\r\n', '\n')

def apply(src, old, new, tag):
    old = old.replace('\r\n', '\n')
    new = new.replace('\r\n', '\n')
    assert old in src, f'{tag}: target not found\n{repr(old[:100])}'
    return src.replace(old, new)

# FIX 4a: SPARSITY_PRIOR constant before IBController
F4A_OLD = 'class IBController:\n    """'
F4A_NEW = ('# Fix 4: Fixed sparsity prior. KL divergence target is Bernoulli(0.1).\n'
           '# 90% of edges OFF on average. Stationary and interpretable.\n'
           'SPARSITY_PRIOR: float = 0.1\n\n'
           'class IBController:\n    """')
src = apply(src, F4A_OLD, F4A_NEW, 'FIX 4a')
print('Fix 4a applied.')

# FIX 4b: Remove adaptive_prior init
F4B_OLD = '        self.adaptive_prior: float = 0.1  # Tier4 #22: EMA-adapted topology prior\n'
F4B_NEW = '        # Fix 4: adaptive_prior removed; SPARSITY_PRIOR = 0.1 constant used\n'
src = apply(src, F4B_OLD, F4B_NEW, 'FIX 4b')
print('Fix 4b applied.')

# FIX 4c: Any leftover prior_p = self.adaptive_prior
if 'prior_p = self.adaptive_prior' in src:
    src = src.replace(
        'prior_p = self.adaptive_prior',
        'prior_p = SPARSITY_PRIOR  # Fix 4: fixed constant'
    )
    print('Fix 4c applied.')
else:
    print('Fix 4c: already done.')

# FIX 4d: Remove update_adaptive_prior method body
F4D_START = '    def update_adaptive_prior(self, W: np.ndarray):'
assert F4D_START in src, 'FIX 4d: method not found'
idx = src.find(F4D_START)
# Find end: next method definition at same indentation
end_marker = '\n    def '
next_method = src.find(end_marker, idx + len(F4D_START))
old_method = src[idx:next_method + 1]  # include leading newline of next method
new_method_comment = '    # Fix 4: update_adaptive_prior() removed. SPARSITY_PRIOR=0.1 used.\n'
src = src[:idx] + new_method_comment + src[next_method + 1:]
print('Fix 4d applied.')

# FIX 4e: Remove call to update_adaptive_prior in run_experiment
F4E_OLD = '            controller.update_adaptive_prior(comm_graph.get_weight_snapshot())\n'
F4E_NEW = '            # Fix 4: adaptive_prior update removed (SPARSITY_PRIOR=0.1)\n'
src = apply(src, F4E_OLD, F4E_NEW, 'FIX 4e')
print('Fix 4e applied.')

# FIX 5: 2^2 factorial ablation
F5_OLD = ('# Ablation A: Sparse-only (fixed RING topology, no PPO)\n'
          '# Bug #11 Fix: was using random weights; ring topology is the correct sparse-only baseline\n'
          'print("\\n  Ablation A: Sparse-only (fixed ring topology, no PPO)...")')
assert F5_OLD in src, 'FIX 5: ablation header not found'
idx5 = src.find(F5_OLD)
# Find end of the old ablation block
end5_marker = '\nsave_kaggle_checkpoint({**live_results, **baseline_results, **ablation_results})'
idx5_end = src.find(end5_marker, idx5)
assert idx5_end >= 0, 'FIX 5: end marker not found'

old_ablation = src[idx5:idx5_end]

F5_NEW = '''# =========================================================
# Fix 5: 2^2 Factorial Ablation Design
# Factor A: Sparsity (KL penalty)   0=off, 1=on
# Factor B: PPO learning             0=off (fixed ring), 1=on
# Calibrator: always ON (post-hoc mapping)
#
# | A | B | Arm name           |
# |---|---|--------------------|
# | 0 | 0 | Arm_00_StaticRing  | Fixed ring, no KL, no PPO
# | 0 | 1 | Arm_01_PPOOnly     | PPO, no sparsity pressure
# | 1 | 0 | Arm_10_SparseRing  | Fixed ring with KL penalty
# | 1 | 1 | Arm_11_Full        | PPO + KL sparsity (full system)
#
# OLS: y = beta0 + betaA*A + betaB*B + betaAB*AB + eps
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

# ---- ARM 01: PPO learns topology, KL=0 ----
print("\\n  Arm 01: PPO-only, no KL sparsity...")
_a01_cfg = _deepcopy(ablation_config); _a01_cfg.lambda_kl = 0.0; _a01_cfg.gamma_loop = 0.0
_a01_ctrl = IBController(_a01_cfg)
ablation_results["Arm_01_PPOOnly"] = run_experiment(
    _a01_cfg, method_name="Arm_01_PPOOnly", train=True, evaluate=True, controller=_a01_ctrl)
save_kaggle_checkpoint({**ablation_results}, filepath="08-06results/ablation_arm01.json")

# ---- ARM 10: Fixed ring with KL penalty, no PPO ----
print("\\n  Arm 10: Fixed ring with KL penalty, no PPO...")
_a10_cfg = _deepcopy(ablation_config)
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

# ---- ARM 11: PPO + KL sparsity (full system; reuse Phase 1 controller) ----
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
    _A  = np.array([0., 0., 1., 1.])
    _B  = np.array([0., 1., 0., 1.])
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

src = src[:idx5] + F5_NEW + src[idx5_end:]
print('Fix 5 applied.')

with open('temp_check.py', 'w', encoding='utf-8', newline='') as f:
    f.write(src)
print('\nAll fixes written.')
