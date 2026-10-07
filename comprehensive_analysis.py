import json, glob, os
import numpy as np

def analyze_file(filepath):
    acc = []
    comp = []
    density = []
    initial_correct = []
    
    try:
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
                except Exception:
                    pass
    except Exception:
        return None
                
    if not acc: return None
    
    return {
        'count': len(acc),
        'accuracy': np.mean(acc),
        'initial_accuracy': np.mean(initial_correct) if initial_correct else 0,
        'compression': np.mean(comp),
        'density': np.mean(density)
    }

print("=== PHASE 1: PRE-TRAINING (Final Epoch 5) ===")
for ds in ['GSM8K', 'HOTPOTQA', 'AQUA']:
    f = f'08-06results/continuous/Train_Pretrain_{ds}_Epoch5_{ds.lower()}_stream.jsonl'
    res = analyze_file(f)
    if res:
        print(f"{ds} | Samples: {res['count']} | Init Acc: {res['initial_accuracy']*100:.2f}% | Final Acc: {res['accuracy']*100:.2f}% | Comp: {res['compression']:.3f} | Density: {res['density']:.3f}")

print("\n=== PHASE 2: TRANSFER EVALUATION (Seed Avg) ===")
for ds in ['GSM8K', 'HOTPOTQA', 'AQUA']:
    accs, comps, dens = [], [], []
    total_samples = 0
    for seed in range(3):
        f = f'08-06results/continuous/Transfer_{ds}_seed{seed}_{ds.lower()}_stream.jsonl'
        res = analyze_file(f)
        if res:
            accs.append(res['accuracy'])
            comps.append(res['compression'])
            dens.append(res['density'])
            total_samples += res['count']
    if accs:
        print(f"{ds} | Samples: {total_samples} | Final Acc: {np.mean(accs)*100:.2f} \u00b1 {np.std(accs, ddof=1 if len(accs)>1 else 0)*100:.2f}% | Comp: {np.mean(comps):.3f} | Density: {np.mean(dens):.3f}")

print("\n=== BASELINES ===")
for ds in ['gsm8k', 'hotpotqa', 'aqua']:
    for method in ['Multi-Agent Debate', 'No Communication', 'IB-Controller']:
        f = f'08-06results/continuous/{method}_{ds}_stream.jsonl'
        res = analyze_file(f)
        if res:
            print(f"{method} ({ds.upper()}) | Samples: {res['count']} | Final Acc: {res['accuracy']*100:.2f}% | Comp: {res['compression']:.3f}")

print("\n=== ABLATIONS ===")
ablation_file = "08-06results/ablation_arm00.json"
try:
    with open(ablation_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    detailed = data.get("detailed_samples", {})
    for method, samples in detailed.items():
        if not samples: continue
        acc = np.mean([1.0 if s.get('is_correct') else 0.0 for s in samples])
        comp = np.mean([s.get('compression_ratio', 1.0) for s in samples])
        print(f"{method} (GSM8K) | Samples: {len(samples)} | Final Acc: {acc*100:.2f}% | Comp: {comp:.3f}")
except Exception as e:
    print(f"No ablation results yet or error: {e}")
