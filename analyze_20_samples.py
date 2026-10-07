import json
import glob
import numpy as np

print(f"{'Dataset':<12} | {'Init Acc':<10} | {'Final Acc':<10} | {'Δ Acc':<8} | {'Avg Comp':<10} | {'Avg Halt':<8}")
print("-" * 75)

for f in sorted(glob.glob('08-06results/toy_*.json')):
    dataset = f.split('_')[-1].split('.')[0].upper()
    with open(f, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    key = list(data['detailed_samples'].keys())[0]
    samples = data['detailed_samples'][key]
    
    # We only analyze the evaluate/test samples, which is what gets saved in detailed_samples.
    
    initial_accs = []
    final_accs = []
    compressions = []
    halt_rounds = []
    
    for s in samples:
        # Initial majority accuracy
        if 'initial_answers' in s and 'gold_answer' in s:
            answers = list(s['initial_answers'].values())
            # Majority vote
            from collections import Counter
            counts = Counter(answers)
            maj_ans = counts.most_common(1)[0][0]
            initial_accs.append(1.0 if maj_ans == s['gold_answer'] else 0.0)
            
        # Final accuracy
        if 'is_correct' in s:
            final_accs.append(1.0 if s['is_correct'] else 0.0)
            
        # Avg compression
        if 'token_reduction' in s:
            compressions.append(s['token_reduction'])
            
        # Avg halt round
        if 'halt_round' in s:
            halt_rounds.append(s['halt_round'])
            
    avg_init = np.mean(initial_accs) * 100 if initial_accs else 0
    avg_final = np.mean(final_accs) * 100 if final_accs else 0
    delta_acc = avg_final - avg_init
    avg_comp = np.mean(compressions) * 100 if compressions else 0
    avg_halt = np.mean(halt_rounds) if halt_rounds else 0
    
    print(f"{dataset:<12} | {avg_init:>9.1f}% | {avg_final:>9.1f}% | {delta_acc:>7.1f}% | {avg_comp:>9.1f}% | {avg_halt:>8.2f}")
