import json
import glob
import os
import math

results = {}

def process_samples(samples, name):
    if not samples: return
    
    correct = 0
    total = len(samples)
    tokens = []
    
    # Check reasoning factors: answer change rate, etc.
    ans_changed = 0
    ans_changed_correct = 0
    ans_unchanged_correct = 0
    
    for s in samples:
        if 'is_correct' in s and s['is_correct']:
            correct += 1
        elif 'correct' in s and s['correct']:
            correct += 1
            
        comm_tokens = s.get('communication_tokens', 0)
        tokens.append(comm_tokens)
        
        # answer change
        initial = s.get('initial_answers', [])
        final = s.get('final_answers', [])
        is_corr = s.get('is_correct', False) or s.get('correct', False)
        
        if initial and final and initial != final:
            ans_changed += 1
            if is_corr:
                ans_changed_correct += 1
        else:
            if is_corr:
                ans_unchanged_correct += 1

    acc = correct / total if total > 0 else 0
    avg_tokens = sum(tokens)/len(tokens) if tokens else 0
    
    if name not in results:
        results[name] = []
        
    results[name].append({
        'acc': acc,
        'avg_tokens': avg_tokens,
        'total': total,
        'ans_changed_rate': ans_changed / total if total > 0 else 0,
        'ans_changed_acc': ans_changed_correct / ans_changed if ans_changed > 0 else 0,
        'ans_unchanged_acc': ans_unchanged_correct / (total - ans_changed) if (total - ans_changed) > 0 else 0
    })

def scan_files():
    files = glob.glob('resultsall/*.json') + glob.glob('08-06results/*.json') + ['Master_SparseComm_Full_Results.json']
    
    for f in files:
        try:
            with open(f, 'r') as file:
                data = json.load(file)
        except Exception as e:
            continue
            
        # Format 1: Master file with 'Experiments'
        if isinstance(data, dict) and 'Experiments' in data:
            for k, v in data['Experiments'].items():
                if 'gsm8k' in k.lower():
                    if 'no communication' in k.lower() or 'mad' in k.lower() or 'multi-agent debate' in k.lower() or 'transfer' in k.lower() or 'arm_' in k.lower():
                        samples = v.get('Detailed_Samples', [])
                        if samples:
                            process_samples(samples, k)
                            
        # Format 2: detailed_samples key
        elif isinstance(data, dict) and 'detailed_samples' in data:
            # sometimes detailed_samples is a dict with keys as exp names
            ds = data['detailed_samples']
            if isinstance(ds, dict):
                for k, v in ds.items():
                    if 'gsm8k' in k.lower() or 'arm' in k.lower() or 'mad' in k.lower() or 'comm' in k.lower():
                        if isinstance(v, list):
                            process_samples(v, f + "::" + k)
            elif isinstance(ds, list):
                # We need to guess the experiment name or check filename
                if 'gsm8k' in f.lower() or 'arm' in f.lower() or 'ablation' in f.lower():
                    process_samples(ds, f)

scan_files()

# Aggregate over seeds
aggregated = {}
for k, runs in results.items():
    if len(runs) == 0: continue
    
    group = "Other"
    if 'no_comm' in k.lower() or 'no comm' in k.lower() or 'arm_00' in k.lower():
        group = 'No Comm'
    elif 'mad' in k.lower() or 'multi-agent debate' in k.lower() or 'debate' in k.lower() or 'arm_10' in k.lower():
        group = 'MAD'
    elif 'transfer_gsm8k' in k.lower() or 'our' in k.lower() or 'arm_11' in k.lower() or 'ib-controller' in k.lower():
        group = 'Our (SparseComm)'
    
    if group not in aggregated:
        aggregated[group] = []
        
    for r in runs:
        # Only take runs with >= 100 samples to avoid tiny mock runs
        if r['total'] >= 100:
            aggregated[group].append(r)

print("==== GSM8K Results Analysis ====")
for group, runs in aggregated.items():
    if not runs: continue
    print(f"\nMethod: {group} (runs: {len(runs)})")
    
    accs = [r['acc'] for r in runs]
    toks = [r['avg_tokens'] for r in runs]
    ans_change_rates = [r['ans_changed_rate'] for r in runs]
    ans_change_accs = [r['ans_changed_acc'] for r in runs]
    ans_unchange_accs = [r['ans_unchanged_acc'] for r in runs]
    
    print(f"  Accuracy: {sum(accs)/len(accs)*100:.2f}%")
    print(f"  Avg Comm Tokens: {sum(toks)/len(toks):.1f}")
    print(f"  Answer Change Rate: {sum(ans_change_rates)/len(ans_change_rates)*100:.1f}%")
    print(f"  Acc when Ans Changed: {sum(ans_change_accs)/len(ans_change_accs)*100:.1f}%")
    print(f"  Acc when Ans Unchanged: {sum(ans_unchange_accs)/len(ans_change_accs)*100:.1f}%")

