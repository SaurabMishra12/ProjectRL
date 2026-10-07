import os
import sys
import numpy as np
import logging

# Ensure we don't output too much spam
logging.getLogger().setLevel(logging.ERROR)

import temp_check
from temp_check import get_gsm8k_config, get_hotpotqa_config, get_aqua_config, run_experiment

# Override to prevent exit
temp_check.RUN_TOY_ONLY = False

def main():
    datasets = [
        ("GSM8K", get_gsm8k_config),
        ("HotpotQA", get_hotpotqa_config),
        ("AQuA", get_aqua_config)
    ]
    
    print(f"{'Dataset':<15} | {'semantic_similarity':<20} | {'information_retention':<21} | {'active_edges':<12} | {'severed_edges':<13} | {'accuracy':<10}")
    print("-" * 105)
    
    for name, get_cfg in datasets:
        cfg = get_cfg()
        cfg.n_rounds = 3
        
        try:
            results = run_experiment(cfg, n_train_samples=5, n_test_samples=15)
            
            sems = []
            infos = []
            actives = []
            severeds = []
            accs = []
            
            for r in results.sample_results:
                sems.append(r.semantic_similarity)
                infos.append(r.information_retention)
                actives.append(r.n_active_edges)
                severeds.append(6 - r.n_active_edges)
                
                # Use F1 for HotpotQA to show exact match vs partial?
                # Actually user asked for accuracy
                accs.append(1.0 if r.is_correct else 0.0)
                
            print(f"{name:<15} | {np.mean(sems):<20.4f} | {np.mean(infos):<21.4f} | {np.mean(actives):<12.2f} | {np.mean(severeds):<13.2f} | {np.mean(accs):<10.4f}")
        except Exception as e:
            print(f"{name:<15} | ERROR: {e}")

if __name__ == "__main__":
    main()
