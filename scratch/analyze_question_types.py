import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import json
import numpy as np

subset = json.load(open("web_eval_subset_100.json"))[:100]
w4_records = [json.loads(line) for line in open("results/W4_corrected/raw_results.jsonl")]

# Question type: comparison vs bridge
types = [s.get("type", "bridge") for s in subset]
comp_idx = [i for i, t in enumerate(types) if t == "comparison"]
bridge_idx = [i for i, t in enumerate(types) if t == "bridge"]

print("=" * 80)
print("7. QUESTION TYPE BREAKDOWN (COMPARISON vs BRIDGE REASONING)")
print("=" * 80)
print(f"Comparison questions (N={len(comp_idx)}):")
print(f"  Static EM: {np.mean([w4_records[i]['Static_Role']['is_em'] for i in comp_idx])*100:.1f}%, F1: {np.mean([w4_records[i]['Static_Role']['f1'] for i in comp_idx])*100:.2f}%")
print(f"  Learned EM: {np.mean([w4_records[i]['Learned_SparseComm']['is_em'] for i in comp_idx])*100:.1f}%, F1: {np.mean([w4_records[i]['Learned_SparseComm']['f1'] for i in comp_idx])*100:.2f}%")
print(f"  Random EM: {np.mean([w4_records[i]['Random_Routing']['is_em'] for i in comp_idx])*100:.1f}%, F1: {np.mean([w4_records[i]['Random_Routing']['f1'] for i in comp_idx])*100:.2f}%")

print(f"\nBridge questions (N={len(bridge_idx)}):")
print(f"  Static EM: {np.mean([w4_records[i]['Static_Role']['is_em'] for i in bridge_idx])*100:.1f}%, F1: {np.mean([w4_records[i]['Static_Role']['f1'] for i in bridge_idx])*100:.2f}%")
print(f"  Learned EM: {np.mean([w4_records[i]['Learned_SparseComm']['is_em'] for i in bridge_idx])*100:.1f}%, F1: {np.mean([w4_records[i]['Learned_SparseComm']['f1'] for i in bridge_idx])*100:.2f}%")
print(f"  Random EM: {np.mean([w4_records[i]['Random_Routing']['is_em'] for i in bridge_idx])*100:.1f}%, F1: {np.mean([w4_records[i]['Random_Routing']['f1'] for i in bridge_idx])*100:.2f}%")

# Question difficulty level in HotpotQA (easy, medium, hard)
levels = [s.get("level", "unknown") for s in subset]
print("\nQuestion levels:", {l: levels.count(l) for l in set(levels)})
for lvl in sorted(set(levels)):
    lvl_idx = [i for i, l in enumerate(levels) if l == lvl]
    print(f"\nLevel '{lvl}' (N={len(lvl_idx)}):")
    print(f"  Static F1:  {np.mean([w4_records[i]['Static_Role']['f1'] for i in lvl_idx])*100:.2f}%")
    print(f"  Learned F1: {np.mean([w4_records[i]['Learned_SparseComm']['f1'] for i in lvl_idx])*100:.2f}%")
    print(f"  Random F1:  {np.mean([w4_records[i]['Random_Routing']['f1'] for i in lvl_idx])*100:.2f}%")
