import json

with open('SparseComm_Kaggle.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

source = nb['cells'][0]['source']
new_source = []

skip_phase1 = False
in_phase2 = False

for i, line in enumerate(source):
    # 1. Fix the KeyError bug
    if "stats['token_ci']" in line:
        line = line.replace("stats['token_ci']", "stats.get('token_cost_95ci', 0.0)")
    
    # Track which phase we are in
    if "PHASE 1: DATASET-SPECIFIC PRE-TRAINING" in line:
        skip_phase1 = True
    if "PHASE 2: TRANSFER EVALUATION" in line:
        skip_phase1 = False
        in_phase2 = True
        
    # Phase 1 skips
    if skip_phase1 and "datasets_to_run =" in line and "gsm8k" in line:
        # Instead of doing Phase 1, we just load the controllers into memory!
        new_source.append('    # Phase 1 bypassed. Loading saved models directly into memory.\n')
        new_source.append('    datasets_to_run = ["gsm8k", "hotpotqa", "aqua"]\n')
        new_source.append('    for ds_name in datasets_to_run:\n')
        new_source.append('        _ds = next(d for d in DATASETS if d["name"] == ds_name)\n')
        new_source.append('        controller = IBController(_ds["embedding_dim"], max_rounds=3, query_dim=32)\n')
        new_source.append('        controller.load_model(f"08-06results/{ds_name}_pretrained.pt.npz")\n')
        new_source.append('        pretrained_controllers[ds_name] = controller\n')
        new_source.append('        print(f"[OK] {ds_name.upper()} pretrained model loaded!")\n')
        new_source.append('    datasets_to_run = []  # skip the actual training loop below\n')
        continue
        
    # Phase 2 skips
    if in_phase2 and "datasets_to_run =" in line and "gsm8k" in line:
        # Skip gsm8k in phase 2 because it is already completed!
        line = '    datasets_to_run = ["hotpotqa", "aqua"]  # GSM8K already done\n'

    new_source.append(line)

nb['cells'][0]['source'] = new_source

with open('SparseComm_Kaggle.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook patched successfully!")
