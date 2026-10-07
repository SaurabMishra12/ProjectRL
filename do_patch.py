import re
import os

with open('temp_check_lib.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add get_completed_count after log_continuous_result
func_to_add = """
def get_completed_count(method_name: str, dataset_name: str) -> int:
    import os
    filepath = f"08-06results/continuous/{method_name}_{dataset_name}_stream.jsonl"
    if not os.path.exists(filepath):
        return 0
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0
"""

if "def get_completed_count" not in content:
    content = content.replace("def create_llm_backend", func_to_add + "\ndef create_llm_backend")

# 2. Patch run_experiment (evaluate loop)
old_run_exp = """        for i, sample in enumerate(tqdm(test_data, desc=f"Evaluating {method_name}")):\n            result = run_single_sample("""
new_run_exp = """        completed_count = get_completed_count(method_name, config.dataset)
        logger.info(f"Resuming evaluate: {completed_count} samples already completed.")
        for i, sample in enumerate(tqdm(test_data, desc=f"Evaluating {method_name}")):
            if i < completed_count:
                continue
            result = run_single_sample("""

if old_run_exp in content:
    content = content.replace(old_run_exp, new_run_exp)

# 3. Patch run_full_communication
old_run_fc = """    for sample in tqdm(test_data, desc="Debate", disable=len(test_data) < 5):\n        agent_pool.reset_all()"""
new_run_fc = """    completed_count = get_completed_count("Multi-Agent Debate", config.dataset)
    for i, sample in enumerate(tqdm(test_data, desc="Debate", disable=len(test_data) < 5)):
        if i < completed_count:
            continue
        agent_pool.reset_all()"""

if old_run_fc in content:
    content = content.replace(old_run_fc, new_run_fc)
elif "for i, sample in enumerate(tqdm(test_data, desc=\"Debate\"" not in content:
    print("Could not find run_full_communication loop to patch")

# 4. Patch run_no_communication
old_run_nc = """    for sample in tqdm(test_data, desc="NoComm"):\n        agent_pool.reset_all()"""
new_run_nc = """    completed_count = get_completed_count("No Communication", config.dataset)
    for i, sample in enumerate(tqdm(test_data, desc="NoComm")):
        if i < completed_count:
            continue
        agent_pool.reset_all()"""

if old_run_nc in content:
    content = content.replace(old_run_nc, new_run_nc)
elif "for i, sample in enumerate(tqdm(test_data, desc=\"NoComm\"" not in content:
    print("Could not find run_no_communication loop to patch")


with open('temp_check_lib.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("temp_check_lib.py patched!")

# Run patch_nb.py to update the notebook
import subprocess
result = subprocess.run(["python", "patch_nb.py"], capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print("Error:", result.stderr)
