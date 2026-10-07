import json

with open('SparseComm_Kaggle.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

source = nb['cells'][0]['source']

# 1. Replace Phase 1 training with model loading
# Find where Phase 1 starts and ends
p1_start = -1
p1_end = -1
for i, line in enumerate(source):
    if 'print("\\n=== PHASE 1: PRE-TRAINING ===")' in line:
        p1_start = i
    if 'print("\\n[OK] Dataset-specific pre-training complete")' in line:
        p1_end = i
        break

if p1_start != -1 and p1_end != -1:
    new_p1 = [
        'print("\\n=== PHASE 1: PRE-TRAINING (SKIPPED - LOADING FROM DISK) ===")\n',
        'pretrained_controllers: dict = {}\n',
        'for ds_name in datasets_to_run:\n',
        '    print(f"\\n--- Bypassing Pre-Training: {ds_name.upper()} ---")\n',
        '    current_config = get_aqua_config() if ds_name == "aqua" else (get_hotpotqa_config() if ds_name == "hotpotqa" else get_gsm8k_config())\n',
        '    controller = IBController(current_config)\n',
        '    controller.load_model(f"08-06results/{ds_name}_pretrained.pt.npz")\n',
        '    pretrained_controllers[ds_name] = controller\n',
        '    print(f"[OK] {ds_name.upper()} loaded from disk successfully!")\n'
    ]
    source = source[:p1_start] + new_p1 + source[p1_end:]

# 2. Skip GSM8K in Phase 2
p2_start = -1
for i, line in enumerate(source):
    if 'print("\\n=== PHASE 2: TRANSFER EVALUATION ===")' in line:
        p2_start = i
        break

if p2_start != -1:
    # insert "datasets_to_run = ['hotpotqa', 'aqua']" right after p2_start
    source.insert(p2_start + 1, 'datasets_to_run = ["hotpotqa", "aqua"]\n')

nb['cells'][0]['source'] = source

with open('SparseComm_Kaggle.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook patched perfectly!")
