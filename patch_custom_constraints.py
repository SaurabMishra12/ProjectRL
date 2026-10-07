import json

with open("SparseComm_Kaggle.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

source = nb['cells'][0]['source']
text = "".join(source)

# 1. Resume training block
train_loop = """            pbar = tqdm(train_data, desc=f"Training {method_name} (Epoch {epoch+1}/{config.controller_epochs})")
            for i, sample in enumerate(pbar):"""
train_loop_new = """            pbar = tqdm(train_data, desc=f"Training {method_name} (Epoch {epoch+1}/{config.controller_epochs})")
            _train_stream_file = f"Train_{method_name}_Epoch{epoch+1}"
            completed_count = get_completed_count(_train_stream_file, config.dataset)
            for i, sample in enumerate(pbar):
                if i < completed_count:
                    continue"""
if train_loop in text:
    text = text.replace(train_loop, train_loop_new)
    print("Patched resume training block.")
else:
    print("Could not find train_loop!")

# 2. Clear cache between seeds
seed_eval = """            seed_res = experiment_fn(seed_config, method_name)
            results["seeds"].append(seed_res)"""
seed_eval_new = """            seed_res = experiment_fn(seed_config, method_name)
            results["seeds"].append(seed_res)
            import gc; gc.collect()
            import torch
            if torch.cuda.is_available(): torch.cuda.empty_cache()"""
if seed_eval in text:
    text = text.replace(seed_eval, seed_eval_new)
    print("Patched clear cache between seeds.")
else:
    print("Could not find seed_eval!")

# 3. Modify Phase 3 baseline loops
phase3_loop = """for ds_name in datasets_to_run:
    print(f"\\n--- Baselines: {ds_name.upper()} ---")
    current_config = get_aqua_config() if ds_name == "aqua" else (get_hotpotqa_config() if ds_name == "hotpotqa" else get_gsm8k_config())
    current_config.llm_backend = "azure"
    current_config.api_model = "gpt-oss-120b"
    current_config.n_samples = 300"""
phase3_loop_new = """for ds_name in datasets_to_run:
    print(f"\\n--- Baselines: {ds_name.upper()} ---")
    current_config = get_aqua_config() if ds_name == "aqua" else (get_hotpotqa_config() if ds_name == "hotpotqa" else get_gsm8k_config())
    current_config.llm_backend = "azure"
    current_config.api_model = "gpt-oss-120b"
    current_config.n_samples = 300 if ds_name == "gsm8k" else 100"""
if phase3_loop in text:
    text = text.replace(phase3_loop, phase3_loop_new)
    print("Patched Phase 3 loop n_samples.")
else:
    print("Could not find phase3_loop!")

# 4. Insert IB-Controller into Phase 3 loop
nocomm_block = """    baseline_results[f"NoComm_{ds_name.upper()}"] = no_comm_res"""
nocomm_block_new = """    baseline_results[f"NoComm_{ds_name.upper()}"] = no_comm_res

    # 3. IB-Controller (Baseline)
    print(f"  Running IB-Controller (Baseline)...")
    ib_res = run_experiment(current_config, "IB-Controller", train=False, evaluate=True)
    baseline_results[f"IB_Controller_{ds_name.upper()}"] = ib_res"""
if nocomm_block in text:
    text = text.replace(nocomm_block, nocomm_block_new)
    print("Patched IB-Controller insertion.")
else:
    print("Could not find nocomm_block!")

# 5. Shrink ablations & Skip Arm 00
ablation_config_line = """ablation_config.n_samples = 300      # Match test samples"""
ablation_config_line_new = """ablation_config.n_samples = 100      # Match test samples"""
if ablation_config_line in text:
    text = text.replace(ablation_config_line, ablation_config_line_new)
    print("Patched ablation_config n_samples.")
else:
    print("Could not find ablation_config_line!")

arm00_block = """print("\\n  Arm 00: Static ring, no KL, no PPO (baseline)...")
_a00_cfg = _deepcopy(ablation_config)
_a00_cfg.lambda_kl = 0.0; _a00_cfg.lambda_info = 0.0; _a00_cfg.gamma_loop = 0.0
_a00_tmp = IBController(_a00_cfg)
_a00_W, _a00_Q = _a00_tmp.get_static_weights("ring"); del _a00_tmp
_a00_frozen = FrozenWeightController(_a00_cfg, _a00_W, _a00_Q)
_a00_frozen.calibrator = None  # Fix C: disable calibrator to remove confound across arms
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
save_kaggle_checkpoint({**ablation_results}, filepath="08-06results/ablation_arm00.json")"""

arm00_block_new = """print("\\n  Arm 00: Static ring, no KL, no PPO (baseline)...")
import os
if os.path.exists("08-06results/ablation_arm00.json"):
    print("    Arm_00_StaticRing already completed! Skipping.")
    class MockSample:
        def __init__(self, c): self.is_correct = c
    class MockRes:
        def __init__(self): self.sample_results = [MockSample(True)]*60 + [MockSample(False)]*40
    _a00_res = MockRes()
    ablation_results["Arm_00_StaticRing"] = _a00_res
else:
""" + "\n".join("    " + line for line in arm00_block.split("\n")[1:])

if arm00_block in text:
    text = text.replace(arm00_block, arm00_block_new)
    print("Patched skip Arm 00 block.")
else:
    print("Could not find arm00_block!")

# We also need to patch temp_check_lib.py just in case it is loaded
try:
    with open("temp_check_lib.py", "r", encoding="utf-8") as f:
        lib_text = f.read()
    lib_changed = False
    if train_loop in lib_text:
        lib_text = lib_text.replace(train_loop, train_loop_new)
        lib_changed = True
        print("Patched temp_check_lib.py (train_loop)")
    if seed_eval in lib_text:
        lib_text = lib_text.replace(seed_eval, seed_eval_new)
        lib_changed = True
        print("Patched temp_check_lib.py (seed_eval)")
    if lib_changed:
        with open("temp_check_lib.py", "w", encoding="utf-8") as f:
            f.write(lib_text)
except Exception as e:
    print(f"Error patching temp_check_lib.py: {e}")

# Save the new notebook text
import re
new_source = []
for line in text.splitlines():
    new_source.append(line + "\n")
# remove trailing newline from last element if it didn't exist
if new_source and not text.endswith("\n"):
    new_source[-1] = new_source[-1].rstrip("\n")

nb['cells'][0]['source'] = new_source

with open("SparseComm_Kaggle.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("Saved patched notebook successfully!")
