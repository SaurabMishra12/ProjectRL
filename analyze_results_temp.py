import json, glob
import numpy as np
from scipy import stats

def analyze_file(filepath):
    acc = []
    comp = []
    density = []
    initial_correct = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                acc.append(1.0 if data.get('is_correct', False) else 0.0)
                comp.append(data.get('compression_ratio', 1.0))
                density.append(data.get('communication_density', 1.0))
                
                ic = list(data.get('initial_correct', {}).values())
                if ic:
                    initial_correct.append(1.0 if sum(ic) > len(ic)/2 else 0.0)
            except Exception as e:
                pass
                
    if not acc: return None
    
    return {
        'count': len(acc),
        'accuracy': np.mean(acc),
        'initial_accuracy': np.mean(initial_correct) if initial_correct else 0,
        'compression': np.mean(comp),
        'density': np.mean(density)
    }

print("\n=== PHASE 1: PRE-TRAINING (Final Epochs) ===")
for ds in ['GSM8K', 'HOTPOTQA', 'AQUA']:
    f = f'08-06results/continuous/Train_Pretrain_{ds}_Epoch5_{ds.lower()}_stream.jsonl'
    res = analyze_file(f)
    if res:
        print(f"[{ds}] Samples: {res['count']}")
        print(f"[{ds}] Initial Majority Vote Acc: {res['initial_accuracy']*100:.2f}%")
        print(f"[{ds}] Final SparseComm Acc:    {res['accuracy']*100:.2f}% (+{(res['accuracy'] - res['initial_accuracy'])*100:.2f}%)")
        print(f"[{ds}] Compression Ratio:       {res['compression']:.3f} (Lower = More Tokens Saved)")
        print(f"[{ds}] Edge Density:            {res['density']:.3f} (Lower = More Sparse Graph)")
        print("-" * 40)


print("\n=== PHASE 2: TRANSFER EVALUATION (GSM8K) ===")
seeds_acc = []
seeds_comp = []
seeds_init = []

for seed in range(3):
    f = f'08-06results/continuous/Transfer_GSM8K_seed{seed}_gsm8k_stream.jsonl'
    res = analyze_file(f)
    if res:
        print(f"[Seed {seed}] Initial Acc: {res['initial_accuracy']*100:.2f}% | Final Acc: {res['accuracy']*100:.2f}% | Compression: {res['compression']:.3f}")
        seeds_acc.append(res['accuracy'])
        seeds_init.append(res['initial_accuracy'])
        seeds_comp.append(res['compression'])

if len(seeds_acc) > 1:
    acc_mean = np.mean(seeds_acc) * 100
    acc_std = np.std(seeds_acc, ddof=1) * 100
    t_crit = stats.t.ppf(0.975, len(seeds_acc)-1)
    acc_ci = t_crit * acc_std / np.sqrt(len(seeds_acc))
    
    init_mean = np.mean(seeds_init) * 100
    comp_mean = np.mean(seeds_comp)
    
    print("\n[GSM8K FINAL AGGREGATED METRICS - 3 Seeds]")
    print(f"Baseline Majority Vote: {init_mean:.2f}%")
    print(f"SparseComm Accuracy:    {acc_mean:.2f}% +/- {acc_ci:.2f}% (95% CI)")
    print(f"Average Compression:    {comp_mean:.3f}")
