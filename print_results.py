import json

for dataset in ['gsm8k', 'hotpotqa']:
    filepath = f'results/full_results_{dataset}.json'
    try:
        with open(filepath, 'r') as f:
            d = json.load(f)
    except FileNotFoundError:
        print(f"No results for {dataset}")
        continue

    print(f"\n{'='*70}")
    print(f" {dataset.upper()} RESULTS")
    print(f"{'='*70}")

    print(f"\n{'Method':<25s} {'Accuracy':>8s} {'Tokens/Sample':>14s} {'Density':>8s} {'ErrProp':>8s}")
    print("-" * 70)
    for m in d.get('metrics', []):
        print(f"{m['method']:<25s} {m['accuracy_pct']:>8s} {m['token_cost']['avg_tokens_per_sample']:>14.1f} {m['communication_density']:>8.3f} {m['error_propagation_rate']:>8.3f}")

    print(f"\nAblation Studies:")
    print(f"{'Condition':<30s} {'Accuracy':>8s} {'Tokens/Sample':>14s}")
    print("-" * 55)
    for m in d.get('ablation_metrics', []):
        print(f"{m['method']:<30s} {m['accuracy_pct']:>8s} {m['token_cost']['avg_tokens_per_sample']:>14.1f}")
