import json

with open("SparseComm_Kaggle.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

text = "".join(nb['cells'][0]['source'])

load_logic = """        completed_count = get_completed_count(method_name, config.dataset)
        logger.info(f"Resuming evaluate: {completed_count} samples already completed.")

        if completed_count > 0:
            filepath = f"08-06results/continuous/{method_name}_{config.dataset}_stream.jsonl"
            try:
                import dataclasses
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip(): continue
                        data = json.loads(line)
                        MockS = dataclasses.make_dataclass('MockS', [('is_correct', bool), ('total_tokens_transmitted', float)])
                        results.sample_results.append(MockS(data.get('is_correct', False), data.get('total_tokens_transmitted', 0.0)))
            except Exception as e:
                logger.error(f"Failed to load previous results: {e}")

        for i, sample in enumerate(tqdm(test_data, desc=f"Evaluating {method_name}")):"""

target_logic = """        completed_count = get_completed_count(method_name, config.dataset)
        logger.info(f"Resuming evaluate: {completed_count} samples already completed.")
        for i, sample in enumerate(tqdm(test_data, desc=f"Evaluating {method_name}"))"""

if "Resuming evaluate" in text and "if completed_count > 0" not in text:
    import re
    # We use regex to replace because there might be minor whitespace differences
    text = re.sub(r'        completed_count = get_completed_count\(method_name, config\.dataset\)\n        logger\.info\(f"Resuming evaluate: \{completed_count\} samples already completed\."\)\n        for i, sample in enumerate\(tqdm\(test_data, desc=f"Evaluating \{method_name\}"\)\):', load_logic, text)
    print("Patched resume loader in notebook!")
else:
    print("Already patched or could not find target!")

new_source = []
for line in text.splitlines():
    new_source.append(line + "\n")
if new_source and not text.endswith("\n"):
    new_source[-1] = new_source[-1].rstrip("\n")

nb['cells'][0]['source'] = new_source

with open("SparseComm_Kaggle.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

try:
    with open("temp_check_lib.py", "r", encoding="utf-8") as f:
        lib_text = f.read()

    lib_text = re.sub(r'        completed_count = get_completed_count\(method_name, config\.dataset\)\n        logger\.info\(f"Resuming evaluate: \{completed_count\} samples already completed\."\)\n        for i, sample in enumerate\(tqdm\(test_data, desc=f"Evaluating \{method_name\}"\)\):', load_logic, lib_text)
    
    with open("temp_check_lib.py", "w", encoding="utf-8") as f:
        f.write(lib_text)
    print("Patched temp_check_lib.py loader!")
except:
    pass
