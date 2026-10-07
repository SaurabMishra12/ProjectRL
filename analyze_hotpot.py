import json
import numpy as np

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

seeds_acc = []
for seed in range(3):
    res = analyze_file(f'08-06results/continuous/Transfer_HOTPOTQA_seed{seed}_hotpotqa_stream.jsonl')
    if res:
        print(f"[HotpotQA Seed {seed}] Samples: {res['count']} | Acc: {res['accuracy']*100:.2f}% | Comp: {res['compression']:.3f}")
        seeds_acc.append(res['accuracy'])

if seeds_acc:
    print(f"\nOverall HotpotQA Acc: {np.mean(seeds_acc)*100:.2f}%")
