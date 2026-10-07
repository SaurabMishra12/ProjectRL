import json
with open('SparseComm_Kaggle.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

source = nb['cells'][0]['source']
new_source = []

for i, line in enumerate(source):
    new_source.append(line)
    
    # Phase 1 checkpoint
    if 'print(f"[OK] {ds_name.upper()} pre-training complete' in line:
        new_source.append('    # Checkpoint Phase 1 weights\n')
        new_source.append('    if hasattr(controller, "save_model"):\n')
        new_source.append('        controller.save_model(f"08-06results/{ds_name}_pretrained.pt")\n')
        
    # Phase 2 checkpoint
    if 'live_results[f"Transfer_{ds_name.upper()}"] = stats' in line:
        new_source.append('    # Incremental save to prevent data loss\n')
        new_source.append('    save_kaggle_checkpoint(live_results, filepath="08-06results/phase2_interim.json")\n')

nb['cells'][0]['source'] = new_source

with open('SparseComm_Kaggle.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Added checkpoints successfully.')
