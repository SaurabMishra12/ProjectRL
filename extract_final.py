import json
import numpy as np

with open('08-06results/kaggle_latest_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print("=== FINAL ABLATION RESULTS ===")
for k, v in data.get('detailed_samples', {}).items():
    if 'Arm' in k:
        acc = np.mean([s.get('is_correct', False) for s in v])
        comm = np.mean([s.get('total_tokens_transmitted', 0) for s in v])
        print(f"{k}: {len(v)} samples | Accuracy: {acc:.2%} | Comm Tokens: {comm:.2f}")

print("\n=== FACTORIAL OLS REGRESSION ===")
for k, v in data.get('stats', {}).get('factorial_ols_betas', {}).items():
    print(f"{k}: {v:.4f}")
