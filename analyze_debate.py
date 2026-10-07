import json
import numpy as np

def analyze_file(filepath):
    acc = []
    comp = []
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

res = analyze_file('08-06results/continuous/Multi-Agent Debate_hotpotqa_stream.jsonl')
if res:
    print(f"[Multi-Agent Debate (HotpotQA)] Samples: {res['count']}/900 | Acc: {res['accuracy']*100:.2f}% | Comp: {res['compression']:.3f}")
