import json
import os
import math

methods = {
    'No Comm': ['No Communication_seed0_gsm8k_stream.jsonl', 'No Communication_seed1_gsm8k_stream.jsonl', 'No Communication_seed2_gsm8k_stream.jsonl'],
    'MAD': ['Multi-Agent Debate_seed0_gsm8k_stream.jsonl', 'Multi-Agent Debate_seed1_gsm8k_stream.jsonl', 'Multi-Agent Debate_seed2_gsm8k_stream.jsonl'],
    'Our': ['Transfer_GSM8K_seed0_gsm8k_stream.jsonl', 'Transfer_GSM8K_seed1_gsm8k_stream.jsonl', 'Transfer_GSM8K_seed2_gsm8k_stream.jsonl']
}

base_dir = "08-06results/continuous/"

def process_file(filepath):
    correct = 0
    total = 0
    tokens = []
    ans_changed = 0
    ans_changed_correct = 0
    ans_unchanged_correct = 0
    
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip(): continue
            try:
                s = json.loads(line)
            except:
                continue
            
            total += 1
            is_corr = s.get('is_correct', False) or s.get('correct', False)
            if is_corr:
                correct += 1
                
            comm_tokens = s.get('total_tokens_transmitted', 0)
            tokens.append(comm_tokens)
            
            # initial_answers and final_answers might be dicts mapping agent id to answer
            initial = s.get('initial_answers', {})
            final = s.get('final_answers', {})
            
            # Simple check if consensus answer changed, but here we can just check if ANY agent changed answer
            # Let's just compare the majority vote if possible, or just exact string match of the dict
            # Alternatively, if any agent's answer changed:
            changed = False
            for k in initial:
                if k in final and initial[k] != final[k]:
                    changed = True
                    break
                    
            if changed:
                ans_changed += 1
                if is_corr:
                    ans_changed_correct += 1
            else:
                if is_corr:
                    ans_unchanged_correct += 1
                    
    return correct, total, sum(tokens), ans_changed, ans_changed_correct, ans_unchanged_correct


for method, files in methods.items():
    t_correct = 0
    t_total = 0
    t_tokens = 0
    t_changed = 0
    t_changed_correct = 0
    t_unchanged_correct = 0
    
    for fname in files:
        fpath = os.path.join(base_dir, fname)
        if not os.path.exists(fpath):
            print(f"Missing: {fpath}")
            continue
            
        c, t, toks, ach, achc, aunc = process_file(fpath)
        t_correct += c
        t_total += t
        t_tokens += toks
        t_changed += ach
        t_changed_correct += achc
        t_unchanged_correct += aunc
        
    if t_total > 0:
        acc = t_correct / t_total * 100
        avg_toks = t_tokens / t_total
        change_rate = t_changed / t_total * 100
        change_acc = t_changed_correct / t_changed * 100 if t_changed > 0 else 0
        unchange_acc = t_unchanged_correct / (t_total - t_changed) * 100 if (t_total - t_changed) > 0 else 0
        
        print(f"## {method} (Across {len(files)} seeds)")
        print(f"- Total Samples: {t_total}")
        print(f"- Accuracy: {acc:.2f}%")
        print(f"- Average Comm Tokens: {avg_toks:.1f}")
        print(f"- Answer Change Rate: {change_rate:.2f}%")
        print(f"- Accuracy when Ans Changed: {change_acc:.2f}%")
        print(f"- Accuracy when Ans Unchanged: {unchange_acc:.2f}%\n")

