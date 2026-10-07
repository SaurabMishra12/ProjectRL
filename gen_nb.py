import nbformat as nbf

nb = nbf.v4.new_notebook()

code = '''import sys
import os
sys.path.append('.')
import temp_check_lib
from temp_check_lib import get_gsm8k_config, get_hotpotqa_config, get_aqua_config, run_experiment
import numpy as np
import logging
import pandas as pd
from IPython.display import display
import json

# Suppress excessive logging to keep progress bar clean
logging.getLogger().setLevel(logging.ERROR)

temp_check_lib.RUN_TOY_ONLY = False

experiments = [
    ('GSM8K', get_gsm8k_config, 100, 300),
    ('HotpotQA', get_hotpotqa_config, 100, 300),
    ('AQuA', get_aqua_config, 100, 300),
]

all_metrics = []
csv_path = "main_results.csv"

for name, get_cfg, n_train, n_test in experiments:
    print(f"\\n{'='*20}\\nRunning {name}...\\n{'='*20}")
    cfg = get_cfg()
    cfg.n_rounds = 3
    cfg.n_train_samples = n_train
    cfg.n_samples = n_test
    # Enforce seed
    cfg.seed = 42
    
    results = run_experiment(cfg)
    
    samples = results.sample_results
    if not samples:
        print(f"No samples generated for {name}.")
        continue
        
    initial_acc = np.mean([1.0 if r.initial_majority_correct else 0.0 for r in samples])
    final_acc = np.mean([1.0 if r.is_correct else 0.0 for r in samples])
    debate_gain = final_acc - initial_acc
    compression = np.mean([r.compression_ratio for r in samples])
    sem_sim = np.mean([r.semantic_similarity for r in samples])
    info_ret = np.mean([r.information_retention for r in samples])
    diversity = np.mean([r.agent_diversity for r in samples])
    entropy = np.mean([r.answer_entropy for r in samples])
    avg_dw = np.mean([r.average_delta_w for r in samples])
    
    metric_dict = {
        'Dataset': name,
        'Train Samples': n_train,
        'Test Samples': len(samples),
        'Initial Accuracy': initial_acc,
        'Final Accuracy': final_acc,
        'Debate Gain': debate_gain,
        'Compression': compression,
        'Semantic Similarity': sem_sim,
        'Information Retention': info_ret,
        'Agent Diversity': diversity,
        'Answer Entropy': entropy,
        'Average \\u0394W': avg_dw
    }
    all_metrics.append(metric_dict)
    
    # Save incrementally to avoid data loss
    df_current = pd.DataFrame(all_metrics)
    df_current.to_csv(csv_path, index=False)
    print(f"\\nSaved interim results for {name} to {csv_path}")
    display(df_current)

print("\\nAll runs complete! Final Results:")
df_final = pd.DataFrame(all_metrics)
display(df_final)
'''

nb.cells.append(nbf.v4.new_code_cell(code))
with open('main_run.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print('Notebook main_run.ipynb created successfully.')
