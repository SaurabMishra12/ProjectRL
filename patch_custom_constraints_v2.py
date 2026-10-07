import json

with open("SparseComm_Kaggle.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

text = "".join(nb['cells'][0]['source'])

seed_eval = """        results = experiment_fn(seed_config, f"{method_name}_seed{seed}")"""
seed_eval_new = """        results = experiment_fn(seed_config, f"{method_name}_seed{seed}")
        import gc; gc.collect()
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()"""

if seed_eval in text:
    text = text.replace(seed_eval, seed_eval_new)
    print("Patched seed_eval cache clear in notebook!")

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

    if seed_eval in lib_text:
        lib_text = lib_text.replace(seed_eval, seed_eval_new)
        print("Patched temp_check_lib.py cache clear!")

    with open("temp_check_lib.py", "w", encoding="utf-8") as f:
        f.write(lib_text)
except Exception as e:
    print(f"Error patching temp_check_lib.py: {e}")
