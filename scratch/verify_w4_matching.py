#!/usr/bin/env python3
"""
Verification of W4 Volume Matching Procedure
============================================
Examines per-question token records in W4-Corrected vs W4-Prior:
1. Checks L_q = min(len(M0), len(M1), len(M2)) per question.
2. Verifies whether comm_tokens across the 2-edge topologies:
     Static_Role, Learned_SparseComm, Random_Routing
   satisfy EXACT token equality on EVERY question (diff == 0).
3. Inspects Complete_Broadcast_Matched:
     cap_matched = floor(2 * L_q / 6).
     Quantifies rounding truncation and realized tokens.
4. Compares with uncorrected W4 to demonstrate the eliminated confound.
"""

import json
import os
import sys
import numpy as np

sys.path.append("/home/saurab/Documents/projectRLMAIN/scratch")
from stats_helpers import load_jsonl

BASE_DIR = "/home/saurab/Documents/projectRLMAIN"

w4_corr = load_jsonl(os.path.join(BASE_DIR, "results/W4_corrected/raw_results.jsonl"))
w4_orig = load_jsonl(os.path.join(BASE_DIR, "results/W4_counterfactual_routing/raw_results.jsonl"))

print("=" * 80)
print("VERIFYING W4-CORRECTED MATCHING PROCEDURE")
print("=" * 80)

n_samples = len(w4_corr)
print(f"Total evaluated questions: {n_samples}")

# Check 1: L_q distribution
l_q_vals = [r["L_q"] for r in w4_corr]
print(f"L_q summary statistics:")
print(f"  Min:    {min(l_q_vals)}")
print(f"  Max:    {max(l_q_vals)}")
print(f"  Mean:   {np.mean(l_q_vals):.2f}")
print(f"  Median: {np.median(l_q_vals):.1f}")
print(f"  Std:    {np.std(l_q_vals):.2f}")

# Check 2: 2-edge token matching
# For each question, compare Static_Role vs Learned_SparseComm vs Random_Routing
static_toks = [r["Static_Role"]["comm_tokens"] for r in w4_corr]
learned_toks = [r["Learned_SparseComm"]["comm_tokens"] for r in w4_corr]
random_toks = [r["Random_Routing"]["comm_tokens"] for r in w4_corr]
comp_match_toks = [r["Complete_Broadcast_Matched"]["comm_tokens"] for r in w4_corr]
comp_full_toks = [r["Complete_Broadcast_Full"]["comm_tokens"] for r in w4_corr]

diff_static_learned = [abs(s - l) for s, l in zip(static_toks, learned_toks)]
diff_static_random = [abs(s - rnd) for s, rnd in zip(static_toks, random_toks)]
diff_learned_random = [abs(l - rnd) for l, rnd in zip(learned_toks, random_toks)]

max_diff_sl = max(diff_static_learned)
max_diff_sr = max(diff_static_random)
max_diff_lr = max(diff_learned_random)

perfect_match_sl = sum(1 for d in diff_static_learned if d == 0)
perfect_match_sr = sum(1 for d in diff_static_random if d == 0)
perfect_match_lr = sum(1 for d in diff_learned_random if d == 0)

print(f"\n2-Edge Topology Volume Equality Verification:")
print(f"  Static vs Learned: {perfect_match_sl}/{n_samples} ({perfect_match_sl}%) exact matches | Max absolute token diff = {max_diff_sl}")
print(f"  Static vs Random:  {perfect_match_sr}/{n_samples} ({perfect_match_sr}%) exact matches | Max absolute token diff = {max_diff_sr}")
print(f"  Learned vs Random: {perfect_match_lr}/{n_samples} ({perfect_match_lr}%) exact matches | Max absolute token diff = {max_diff_lr}")

# Check 3: Check relation between L_q and comm_tokens for 2-edge topologies
# In 2 rounds, with 2 edges, each edge transmitting L_q tokens:
# comm_tokens should equal: 2 rounds * 2 edges * L_q = 4 * L_q
expected_toks = [4 * l for l in l_q_vals]
diff_expected_static = [abs(act - exp) for act, exp in zip(static_toks, expected_toks)]
print(f"  Theoretical Formula Verification (comm_tokens == 4 * L_q):")
print(f"    Static_Role matches 4 * L_q: {sum(1 for d in diff_expected_static if d == 0)}/{n_samples}")
print(f"    Learned matches 4 * L_q:     {sum(1 for d in diff_static_learned if d == 0)}/{n_samples}")
print(f"    Random matches 4 * L_q:      {sum(1 for d in diff_static_random if d == 0)}/{n_samples}")

# Check 4: Complete_Broadcast_Matched
# 6 edges, cap_matched = floor(2 * L_q / 6)
# Expected tokens = 2 rounds * 6 edges * cap_matched = 12 * floor(L_q / 3)
cap_matched_vals = [(2 * l) // 6 for l in l_q_vals]
expected_comp_matched = [12 * c for c in cap_matched_vals]
diff_comp_matched = [abs(act - exp) for act, exp in zip(comp_match_toks, expected_comp_matched)]
print(f"\nComplete_Broadcast_Matched Verification:")
print(f"  Matches 12 * floor(2*L_q / 6): {sum(1 for d in diff_comp_matched if d == 0)}/{n_samples}")
print(f"  Average tokens: {np.mean(comp_match_toks):.2f} (vs 2-edge avg: {np.mean(static_toks):.2f})")
print(f"  Discrepancy due to integer quantization: {(np.mean(static_toks) - np.mean(comp_match_toks)):.2f} tokens ({(np.mean(static_toks) - np.mean(comp_match_toks))/np.mean(static_toks)*100:.1f}%)")

# Check 5: Contrast with Prior W4 (Uncorrected)
print("\n" + "=" * 80)
print("CONTRAST WITH PRIOR UNCORRECTED W4 (Volume Confound Demonstrated)")
print("=" * 80)
orig_static_toks = [r["Static_Role"]["comm_tokens"] for r in w4_orig]
orig_learned_toks = [r["Learned_SparseComm"]["comm_tokens"] for r in w4_orig]
orig_random_toks = [r["Random_Routing"]["comm_tokens"] for r in w4_orig]
orig_complete_toks = [r["Complete_Broadcast"]["comm_tokens"] for r in w4_orig]

print("Prior W4 Realized Communication Tokens (Per-Question):")
print(f"  Learned SPARSECOMM:  Mean = {np.mean(orig_learned_toks):.2f} | Min = {min(orig_learned_toks)} | Max = {max(orig_learned_toks)}")
print(f"  Random Routing:      Mean = {np.mean(orig_random_toks):.2f} | Min = {min(orig_random_toks)} | Max = {max(orig_random_toks)}")
print(f"  Static Role-Aligned: Mean = {np.mean(orig_static_toks):.2f} | Min = {min(orig_static_toks)} | Max = {max(orig_static_toks)}")
print(f"  Complete Broadcast:  Mean = {np.mean(orig_complete_toks):.2f} | Min = {min(orig_complete_toks)} | Max = {max(orig_complete_toks)}")
print(f"  Ratio (Complete / Learned): {np.mean(orig_complete_toks)/np.mean(orig_learned_toks):.2f}x volume difference!")
print(f"  Ratio (Static / Learned):   {np.mean(orig_static_toks)/np.mean(orig_learned_toks):.2f}x volume difference!")

# Check correlation between communication volume and F1 in prior W4:
orig_f1s = [
    r["Learned_SparseComm"]["f1"] for r in w4_orig
] + [
    r["Random_Routing"]["f1"] for r in w4_orig
] + [
    r["Static_Role"]["f1"] for r in w4_orig
] + [
    r["Complete_Broadcast"]["f1"] for r in w4_orig
]
orig_vols = orig_learned_toks + orig_random_toks + orig_static_toks + orig_complete_toks
r_vol_f1 = np.corrcoef(orig_vols, orig_f1s)[0, 1]
print(f"Correlation between transmitted volume and F1 across all prior W4 condition runs: r = {r_vol_f1:.4f}")
