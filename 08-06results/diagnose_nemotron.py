import os
import json

OUTPUT_DIR = "08-06results/nemotron_gsm8k_150"

with open(os.path.join(OUTPUT_DIR, "nocomm_results.json"), "r") as f:
    nocomm = json.load(f)

with open(os.path.join(OUTPUT_DIR, "mad_results.json"), "r") as f:
    mad = json.load(f)

with open(os.path.join(OUTPUT_DIR, "our_results.json"), "r") as f:
    our = json.load(f)

print("=== DEEP DIAGNOSTIC OF NEMOTRON-3 RESULTS ===")

# Inspect 3 samples where SparseComm was ONLY one correct
only_our_idx = [i for i in range(len(our)) if our[i]["is_correct"] and not mad[i]["is_correct"] and not nocomm[i]["is_correct"]]
print(f"\nSamples where ONLY SparseComm was correct ({len(only_our_idx)} total): {only_our_idx[:5]}")
for idx in only_our_idx[:2]:
    print(f"\n--- Sample #{idx} ---")
    print(f"Question: {our[idx]['question'][:150]}...")
    print(f"Gold: {our[idx]['gold_answer']}")
    print(f"NoComm Final: {nocomm[idx]['final_answer']} (Correct: {nocomm[idx]['is_correct']})")
    print(f"MAD Final: {mad[idx]['final_answer']} (Correct: {mad[idx]['is_correct']})")
    print(f"SparseComm Final: {our[idx]['final_answer']} (Correct: {our[idx]['is_correct']})")

# Inspect 3 samples where Initial Majority was Right but SparseComm flipped to Wrong
flip_wrong_idx = []
for i in range(len(our)):
    init_c = our[i].get("initial_correct", {})
    init_maj_corr = (sum(1 for v in init_c.values() if v) >= 2) if isinstance(init_c, dict) else False
    if init_maj_corr and not our[i]["is_correct"]:
        flip_wrong_idx.append(i)

print(f"\nSamples where SparseComm Initial Majority was Right but Final Answer was Wrong ({len(flip_wrong_idx)} total): {flip_wrong_idx[:5]}")
for idx in flip_wrong_idx[:2]:
    print(f"\n--- Flip Sample #{idx} ---")
    print(f"Question: {our[idx]['question'][:150]}...")
    print(f"Gold: {our[idx]['gold_answer']}")
    print(f"Initial Correct Dict: {our[idx]['initial_correct']}")
    print(f"NoComm Final: {nocomm[idx]['final_answer']} (Correct: {nocomm[idx]['is_correct']})")
    print(f"MAD Final: {mad[idx]['final_answer']} (Correct: {mad[idx]['is_correct']})")
    print(f"SparseComm Final: {our[idx]['final_answer']} (Correct: {our[idx]['is_correct']})")
    print(f"SparseComm Mode Dist: {our[idx]['mode_distribution']}")

