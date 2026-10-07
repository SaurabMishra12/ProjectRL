"""
Apply 4 targeted fixes:
A. Remove entropy shaping from intermediate reward; cost-only
B. Always include_token_penalty=False in final compute_reward call
C. Disable calibrator for all factorial ablation arms (remove confound)
D. Guard SPRT calibrator check against None
"""

with open('temp_check.py', 'r', encoding='utf-8', newline='') as f:
    src = f.read().replace('\r\n', '\n')

def apply(src, old, new, tag):
    old = old.replace('\r\n', '\n')
    new = new.replace('\r\n', '\n')
    assert old in src, f'{tag}: target not found\n{repr(old[:120])}'
    result = src.replace(old, new)
    assert result != src, f'{tag}: replacement produced no change'
    return result

# ============================================================
# FIX A: Remove entropy shaping from intermediate reward block.
# Remove: initial_entropy, round_entropy_reduction computations.
# Intermediate reward = token_penalty only.
# ============================================================

# A1: Remove initial_entropy computation (it's only used for round_entropy_reduction)
FA1_OLD = (
    '    # Track initial state for dense reward net change\n'
    '    initial_confidence = np.mean([r.confidence for r in current_responses])\n'
    '    initial_entropy = len(set(r.answer for r in current_responses)) / controller.n_agents\n'
)
FA1_NEW = (
    '    # Track initial confidence for calibrator accumulation\n'
    '    initial_confidence = np.mean([r.confidence for r in current_responses])\n'
    '    # Fix A: initial_entropy removed (entropy shaping dropped from intermediate reward)\n'
)
src = apply(src, FA1_OLD, FA1_NEW, 'FIX A1 (remove initial_entropy)')
print('Fix A1 applied.')

# A2: Remove round_n_unique, round_entropy_reduction, replace reward_for_step
FA2_OLD = (
    '                # Use the correct per-round snapshot, not final-round current_responses\n'
    '                snap_idx = min(round_num_i, len(responses_snapshots) - 1)\n'
    '                snap = responses_snapshots[snap_idx]\n'
    '                round_n_unique = len(set(r.answer for r in snap))\n'
    '                round_entropy_reduction = initial_entropy - round_n_unique / max(controller.n_agents, 1)\n'
    '                reward_for_step = 0.05 * round_entropy_reduction + token_penalty\n'
)
FA2_NEW = (
    '                # Fix A: intermediate reward = cost penalty only.\n'
    '                # Entropy-reduction shaping removed: rewards wrong-consensus collapse.\n'
    '                reward_for_step = token_penalty\n'
)
src = apply(src, FA2_OLD, FA2_NEW, 'FIX A2 (cost-only intermediate reward)')
print('Fix A2 applied.')

# ============================================================
# FIX B: Always include_token_penalty=False in compute_reward call.
# Per-round penalties already account for all token costs.
# ============================================================
FB_OLD = (
    '        is_final_step=is_final,\n'
    '        include_token_penalty=is_final,\n'
    '    )'
)
FB_NEW = (
    '        is_final_step=is_final,\n'
    '        include_token_penalty=False,  # Fix B: cost already applied per round; avoid double-counting\n'
    '    )'
)
src = apply(src, FB_OLD, FB_NEW, 'FIX B (include_token_penalty=False)')
print('Fix B applied.')

# ============================================================
# FIX C: Disable calibrator for all 4 factorial arms.
# This removes the calibrator as a confounding variable between
# trained (PPO arms) and untrained (static-ring arms) controllers.
# ============================================================

# Arm 00: StaticRing
FC_A00_OLD = (
    '_a00_frozen = FrozenWeightController(_a00_cfg, _a00_W, _a00_Q)\n'
    '_a00_comm = CommunicationGraph(n_agents=_a00_cfg.n_agents)'
)
FC_A00_NEW = (
    '_a00_frozen = FrozenWeightController(_a00_cfg, _a00_W, _a00_Q)\n'
    '_a00_frozen.calibrator = None  # Fix C: disable calibrator to remove confound across arms\n'
    '_a00_comm = CommunicationGraph(n_agents=_a00_cfg.n_agents)'
)
src = apply(src, FC_A00_OLD, FC_A00_NEW, 'FIX C Arm00')
print('Fix C (Arm 00) applied.')

# Arm 01: PPOOnly - disable after training via run_experiment wrapper
# We'll disable it right before usage by patching the config call.
# run_experiment returns ExperimentResults; calibrator lives on the controller.
# Disable before evaluate call - since arm 01 uses run_experiment with train+evaluate,
# we disable on the controller object after construction.
FC_A01_OLD = (
    '_a01_ctrl = IBController(_a01_cfg)\n'
    'ablation_results["Arm_01_PPOOnly"] = run_experiment(\n'
    '    _a01_cfg, method_name="Arm_01_PPOOnly", train=True, evaluate=True, controller=_a01_ctrl)'
)
FC_A01_NEW = (
    '_a01_ctrl = IBController(_a01_cfg)\n'
    '_a01_ctrl.calibrator = None  # Fix C: disable calibrator to remove confound across arms\n'
    'ablation_results["Arm_01_PPOOnly"] = run_experiment(\n'
    '    _a01_cfg, method_name="Arm_01_PPOOnly", train=True, evaluate=True, controller=_a01_ctrl)'
)
src = apply(src, FC_A01_OLD, FC_A01_NEW, 'FIX C Arm01')
print('Fix C (Arm 01) applied.')

# Arm 10: SparseRing
FC_A10_OLD = (
    '_a10_frozen = FrozenWeightController(_a10_cfg, _a10_W, _a10_Q)\n'
    '_a10_comm = CommunicationGraph(n_agents=_a10_cfg.n_agents)'
)
FC_A10_NEW = (
    '_a10_frozen = FrozenWeightController(_a10_cfg, _a10_W, _a10_Q)\n'
    '_a10_frozen.calibrator = None  # Fix C: disable calibrator to remove confound across arms\n'
    '_a10_comm = CommunicationGraph(n_agents=_a10_cfg.n_agents)'
)
src = apply(src, FC_A10_OLD, FC_A10_NEW, 'FIX C Arm10')
print('Fix C (Arm 10) applied.')

# Arm 11: Full system - disable calibrator before evaluate
FC_A11_OLD = (
    '_a11_ctrl = pretrained_controllers.get("gsm8k")\n'
    'if _a11_ctrl is None:\n'
    '    _a11_ctrl = IBController(ablation_config)\n'
    '    ablation_results["Arm_11_Full"] = run_experiment(\n'
    '        ablation_config, method_name="Arm_11_Full", train=True, evaluate=True, controller=_a11_ctrl)\n'
    'else:\n'
    '    ablation_results["Arm_11_Full"] = run_experiment(\n'
    '        ablation_config, method_name="Arm_11_Full", train=False, evaluate=True, controller=_a11_ctrl)'
)
FC_A11_NEW = (
    '_a11_ctrl = pretrained_controllers.get("gsm8k")\n'
    'if _a11_ctrl is None:\n'
    '    _a11_ctrl = IBController(ablation_config)\n'
    '    _a11_ctrl_trained = run_experiment(\n'
    '        ablation_config, method_name="Arm_11_Full", train=True, evaluate=False, controller=_a11_ctrl)\n'
    'else:\n'
    '    _a11_ctrl_trained = None\n'
    '# Fix C: disable calibrator before evaluation to remove confound\n'
    '_a11_ctrl.calibrator = None\n'
    'ablation_results["Arm_11_Full"] = run_experiment(\n'
    '    ablation_config, method_name="Arm_11_Full", train=False, evaluate=True, controller=_a11_ctrl)'
)
src = apply(src, FC_A11_OLD, FC_A11_NEW, 'FIX C Arm11')
print('Fix C (Arm 11) applied.')

# ============================================================
# FIX D: Guard SPRT calibrator check against None
# ============================================================
FD_OLD = '        if controller.calibrator._fitted and _maj_confs:'
FD_NEW = '        if controller.calibrator is not None and controller.calibrator._fitted and _maj_confs:'
src = apply(src, FD_OLD, FD_NEW, 'FIX D (SPRT None guard)')
print('Fix D applied.')

# Write back
with open('temp_check.py', 'w', encoding='utf-8', newline='') as f:
    f.write(src)
print('\nAll 4 fixes written successfully.')
