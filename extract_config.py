import json, pathlib

p = pathlib.Path(r'resultsall\kaggle_latest_results (2) (1).json')
data = json.loads(p.read_text(encoding='utf-8'))
method, samples = next(iter(data['detailed_samples'].items()))
s0 = samples[0]

exclude_long = {'question','final_answers','initial_answers','raw_communications','weight_matrices','messages_received'}
print(f"Method key: {method}")
print(f"Total samples in this file: {len(samples)}")
print("\n--- Sample[0] Fields ---")
for k, v in s0.items():
    if k in exclude_long:
        size = len(v) if hasattr(v, '__len__') else 'n/a'
        print(f"  {k}: <{type(v).__name__}, len={size}>")
    else:
        val_str = json.dumps(v, default=str)
        print(f"  {k}: {val_str[:300]}")

# Also check AQuA file
print("\n\n--- AQuA file (kaggle_latest_results (8).json) ---")
p2 = pathlib.Path(r'resultsall\kaggle_latest_results (8).json')
data2 = json.loads(p2.read_text(encoding='utf-8'))
method2, samples2 = next(iter(data2['detailed_samples'].items()))
s2 = samples2[0]
print(f"Method key: {method2}")
print(f"Total samples: {len(samples2)}")
for k, v in s2.items():
    if k in exclude_long:
        size = len(v) if hasattr(v, '__len__') else 'n/a'
        print(f"  {k}: <{type(v).__name__}, len={size}>")
    else:
        val_str = json.dumps(v, default=str)
        print(f"  {k}: {val_str[:300]}")
