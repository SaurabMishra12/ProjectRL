import os
import json
import numpy as np

OUTPUT_DIR = "08-06results/nemotron_gsm8k_150"

with open(os.path.join(OUTPUT_DIR, "nocomm_results.json"), "r") as f:
    nocomm = json.load(f)

with open(os.path.join(OUTPUT_DIR, "mad_results.json"), "r") as f:
    mad = json.load(f)

with open(os.path.join(OUTPUT_DIR, "our_results.json"), "r") as f:
    our = json.load(f)

print(f"Loaded samples count: NoComm={len(nocomm)}, MAD={len(mad)}, Our={len(our)}")

# Correctness arrays
c_nocomm = np.array([s["is_correct"] for s in nocomm])
c_mad = np.array([s["is_correct"] for s in mad])
c_our = np.array([s["is_correct"] for s in our])

print("\n--- ACCURACY OVERLAP ---")
all_three = np.logical_and(np.logical_and(c_nocomm, c_mad), c_our)
mad_and_our = np.logical_and(c_mad, c_our)
nocomm_and_our = np.logical_and(c_nocomm, c_our)
nocomm_and_mad = np.logical_and(c_nocomm, c_mad)

only_mad = np.logical_and(c_mad, np.logical_not(np.logical_or(c_nocomm, c_our)))
only_our = np.logical_and(c_our, np.logical_not(np.logical_or(c_nocomm, c_mad)))
only_nocomm = np.logical_and(c_nocomm, np.logical_not(np.logical_or(c_mad, c_our)))

none_correct = np.logical_not(np.logical_or(np.logical_or(c_nocomm, c_mad), c_our))
at_least_one = np.logical_or(np.logical_or(c_nocomm, c_mad), c_our)

print(f"Solved by ALL 3 methods: {np.sum(all_three)} ({np.mean(all_three):.2%})")
print(f"Solved by MAD & SparseComm: {np.sum(mad_and_our)} ({np.mean(mad_and_our):.2%})")
print(f"Solved by NoComm & SparseComm: {np.sum(nocomm_and_our)} ({np.mean(nocomm_and_our):.2%})")
print(f"Solved by NoComm & MAD: {np.sum(nocomm_and_mad)} ({np.mean(nocomm_and_mad):.2%})")
print(f"Solved ONLY by MAD: {np.sum(only_mad)}")
print(f"Solved ONLY by SparseComm: {np.sum(only_our)}")
print(f"Solved ONLY by NoComm: {np.sum(only_nocomm)}")
print(f"Solved by AT LEAST ONE method: {np.sum(at_least_one)} ({np.mean(at_least_one):.2%})")
print(f"Failed by ALL 3 methods: {np.sum(none_correct)} ({np.mean(none_correct):.2%})")

# Agent-level initial correctness analysis
print("\n--- INITIAL VS FINAL ACCURACY ---")
for name, data in [("NoComm", nocomm), ("MAD", mad), ("SparseComm", our)]:
    initial_correct_counts = []
    final_correct = [s["is_correct"] for s in data]
    for s in data:
        # s['initial_correct'] is a dict of agent_id -> bool
        init_c = s.get("initial_correct", {})
        if isinstance(init_c, dict):
            num_init_corr = sum(1 for v in init_c.values() if v)
        else:
            num_init_corr = 0
        initial_correct_counts.append(num_init_corr)
    
    avg_init_agents_correct = np.mean(initial_correct_counts)
    print(f"{name}: Avg Initial Correct Agents per sample (out of 3 or 4): {avg_init_agents_correct:.2f}")
    
    # Check flip stats (initial consensus -> final answer)
    flips_wrong_to_right = 0
    flips_right_to_wrong = 0
    for s in data:
        init_c = s.get("initial_correct", {})
        init_majority_correct = (sum(1 for v in init_c.values() if v) >= 2) if isinstance(init_c, dict) else False
        fin_c = s["is_correct"]
        if not init_majority_correct and fin_c:
            flips_wrong_to_right += 1
        elif init_majority_correct and not fin_c:
            flips_right_to_wrong += 1
    print(f"  Majority Wrong -> Final Right: {flips_wrong_to_right}")
    print(f"  Majority Right -> Final Wrong: {flips_right_to_wrong}")

print("\n--- TOKEN AND COMPRESSION STATS ---")
for name, data in [("NoComm", nocomm), ("MAD", mad), ("SparseComm", our)]:
    gen_toks = [s.get("generation_tokens", 0) for s in data]
    tx_toks = [s.get("total_tokens_transmitted", 0) for s in data]
    print(f"{name}:")
    print(f"  Total Gen Tokens: {sum(gen_toks):,}")
    print(f"  Avg Gen Tokens/Sample: {np.mean(gen_toks):.1f} ± {np.std(gen_toks):.1f}")
    print(f"  Total Transmitted Tokens: {sum(tx_toks):,}")
    print(f"  Avg Transmitted Tokens/Sample: {np.mean(tx_toks):.1f} ± {np.std(tx_toks):.1f}")
    if name == "SparseComm":
        comp_ratios = [s.get("compression_ratio", 0) for s in data]
        print(f"  Avg Compression Ratio: {np.mean(comp_ratios):.2%}")
        mode_dists = [s.get("mode_distribution", {}) for s in data]
        # Aggregate mode distribution across all samples
        agg_modes = {}
        for md in mode_dists:
            if isinstance(md, dict):
                for k, v in md.items():
                    agg_modes[k] = agg_modes.get(k, 0) + v
        print(f"  Mode Distribution aggregate: {agg_modes}")

