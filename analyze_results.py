import json
import numpy as np

filepath = "08-06results/kaggle_latest_results.json"

try:
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print("=== PERFORMANCE ANALYSIS ===")
    detailed = data.get("detailed_samples", {})
    
    for method, samples in detailed.items():
        print(f"\nMethod/Dataset: {method}")
        total = len(samples)
        if total == 0:
            continue
            
        correct = sum(1 for s in samples if s.get("is_correct", False))
        acc = (correct / total) * 100
        
        # Initial accuracy (Round 0)
        initial_correct_vals = []
        for s in samples:
            ic = s.get("initial_correct", {})
            if ic:
                # Average initial correctness across agents
                initial_correct_vals.append(sum(1 for v in ic.values() if v) / len(ic))
        
        avg_initial_acc = (np.mean(initial_correct_vals) * 100) if initial_correct_vals else 0
        
        # Token metrics
        tokens = [s.get("total_tokens_transmitted", 0) for s in samples]
        tokens_full = [s.get("total_tokens_if_full", 1) for s in samples]
        
        avg_tokens = np.mean(tokens)
        avg_tokens_full = np.mean(tokens_full)
        compression = (1.0 - (avg_tokens / max(avg_tokens_full, 1))) * 100
        
        # Halting & Density
        density = [s.get("communication_density", 1.0) for s in samples]
        halt = [s.get("halt_round", 3) for s in samples]
        
        avg_density = np.mean(density) * 100
        avg_halt = np.mean(halt)
        
        print(f"  Samples: {total}")
        print(f"  Final Accuracy: {acc:.1f}% (Initial Avg: {avg_initial_acc:.1f}%)")
        print(f"  Improvement from Debate: {acc - avg_initial_acc:+.1f}%")
        print(f"  Communication Compression: {compression:.1f}% (Tokens sent: {avg_tokens:.1f} vs Full: {avg_tokens_full:.1f})")
        print(f"  Avg Edge Density: {avg_density:.1f}%")
        print(f"  Avg Halting Round: {avg_halt:.2f} / 3.00")
        
        # Modes
        modes_total = {"broadcast": 0, "bipartite": 0, "isolated": 0, "chain": 0, "custom": 0}
        for s in samples:
            m = s.get("mode_distribution", {})
            for k, v in m.items():
                modes_total[k] = modes_total.get(k, 0) + v
                
        total_modes = sum(modes_total.values())
        if total_modes > 0:
            print("  Top Graph Topologies (Hypergraph collapse):")
            for k, v in sorted(modes_total.items(), key=lambda x: x[1], reverse=True):
                if v > 0:
                    pct = (v / total_modes) * 100
                    print(f"    - {k.capitalize()}: {pct:.1f}%")

except Exception as e:
    print(f"Error analyzing results: {e}")
