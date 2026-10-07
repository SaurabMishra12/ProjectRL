import json
import numpy as np
import glob

def analyze_file(filepath):
    acc = []
    comp = []
    density = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                acc.append(1.0 if data.get('is_correct', False) else 0.0)
                comp.append(data.get('compression_ratio', 1.0))
            except Exception: pass
    if not acc: return None
    return {'accuracy': np.mean(acc), 'compression': np.mean(comp), 'count': len(acc)}

print("=== AQUA PROGRESS ===")
seeds_acc = []
for seed in range(3):
    f = f'08-06results/continuous/Transfer_AQUA_seed{seed}_aqua_stream.jsonl'
    try:
        res = analyze_file(f)
        if res:
            print(f"[AQuA Seed {seed}] Samples: {res['count']}/300 | Acc: {res['accuracy']*100:.2f}% | Comp: {res['compression']:.3f}")
            if res['count'] == 300:
                seeds_acc.append(res['accuracy'])
    except Exception as e:
        print(f"[AQuA Seed {seed}] Not started yet or error: {e}")

if seeds_acc:
    print(f"\nCompleted Seeds Overall AQuA Acc: {np.mean(seeds_acc)*100:.2f}%")
