import json
import numpy as np

filepath = "08-06results/ablation_arm00.json"
try:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    detailed_samples = data.get("detailed_samples", {})
    
    for method_name, samples in detailed_samples.items():
        if not samples:
            continue
            
        print(f"--- Analysis for {method_name} ---")
        print(f"Total Samples: {len(samples)}")
        
        correct = sum(1 for s in samples if s.get("is_correct"))
        accuracy = correct / len(samples)
        print(f"Accuracy: {accuracy:.2%} ({correct}/{len(samples)})")
        
        tokens = [s.get("total_tokens_transmitted", 0) for s in samples]
        print(f"Average Tokens Transmitted: {np.mean(tokens):.2f}")
        
        comp_ratios = [s.get("compression_ratio", 0) for s in samples if "compression_ratio" in s]
        if comp_ratios:
            print(f"Average Compression Ratio: {np.mean(comp_ratios):.2f}")
            
except Exception as e:
    print(f"Error reading {filepath}: {e}")
