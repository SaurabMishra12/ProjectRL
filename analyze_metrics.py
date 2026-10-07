import json
import glob
import numpy as np

def analyze_jsonl(file_path):
    print(f"\nAnalyzing {file_path}")
    diversities = []
    weight_diffs = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
            except:
                continue
            
            # Agent Diversity
            # D = #unique answers / #agents
            # wait, user formula: D = #agents / #unique answers?
            # Actually user wrote: "D= #agents / #unique answers" -> Wait, usually diversity is #unique / #agents. 
            # If all answer the same, D = N / 1 = N. If all different, D = N / N = 1.
            # "If diversity is very low: all agents answer the same thing" implies user meant #unique / #agents, or maybe unique count directly.
            # I will just compute #unique initial answers.
            if 'initial_answers' in data:
                ans = list(data['initial_answers'].values())
                n_agents = len(ans)
                unique_ans = len(set(ans))
                diversity = unique_ans / n_agents
                diversities.append(diversity)
                
            # Controller Adaptation
            if 'weight_matrices' in data and len(data['weight_matrices']) > 1:
                matrices = data['weight_matrices']
                # compute frobenius norm of difference across consecutive rounds
                diffs = []
                for i in range(1, len(matrices)):
                    W_prev = np.array(matrices[i-1])
                    W_curr = np.array(matrices[i])
                    if W_prev.shape == W_curr.shape:
                        diff = np.linalg.norm(W_curr - W_prev, ord='fro')
                        diffs.append(diff)
                if diffs:
                    weight_diffs.append(np.mean(diffs))
                    
    print(f"Avg Agent Diversity (#unique / #agents): {np.mean(diversities):.3f}")
    if weight_diffs:
        print(f"Avg Controller Adaptation (Frobenius Norm ΔW): {np.mean(weight_diffs):.6f}")
    else:
        print("No communication with multiple rounds found.")

for f in glob.glob('08-06results/continuous/Train_Toy_*_stream.jsonl'):
    analyze_jsonl(f)
