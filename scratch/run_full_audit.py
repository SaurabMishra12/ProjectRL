#!/usr/bin/env python3
"""
Full Execution Audit Script
===========================
Executes calculations for:
1. W1, W2, W3 Paired Confidence Intervals
2. W2 Statistical Tests Specification
3. W3 Clean vs Noisy Reproducibility & Retrieval Analysis
4. W4 Matching Procedure Verification on Per-Question Records
"""

import json
import os
import sys
import numpy as np
from scipy import stats

sys.path.append("/home/saurab/Documents/projectRLMAIN/scratch")
from stats_helpers import load_jsonl, compute_comparison, paired_t_ci, bootstrap_paired_ci, paired_perm_test

BASE_DIR = "/home/saurab/Documents/projectRLMAIN"

print("=" * 80)
print("1. CALCULATING PAIRED CONFIDENCE INTERVALS FOR W1")
print("=" * 80)

w1_records = load_jsonl(os.path.join(BASE_DIR, "results/W1_web_cost_accuracy/raw_results.jsonl"))

# W1 Methods: NoComm, MAD, SPARSECOMM
# Metrics: f1, is_em, comm_tokens, latency_sec
w1_data = {
    "NoComm": {
        "f1": [r["NoComm"]["f1"] for r in w1_records],
        "em": [float(r["NoComm"]["is_em"]) for r in w1_records],
        "comm_tokens": [r["NoComm"]["comm_tokens"] for r in w1_records],
        "gen_tokens": [r["NoComm"]["gen_tokens"] for r in w1_records],
        "latency": [r["NoComm"]["latency_sec"] for r in w1_records],
    },
    "MAD": {
        "f1": [r["MAD"]["f1"] for r in w1_records],
        "em": [float(r["MAD"]["is_em"]) for r in w1_records],
        "comm_tokens": [r["MAD"]["comm_tokens"] for r in w1_records],
        "gen_tokens": [r["MAD"]["gen_tokens"] for r in w1_records],
        "latency": [r["MAD"]["latency_sec"] for r in w1_records],
    },
    "SPARSECOMM": {
        "f1": [r["SPARSECOMM"]["f1"] for r in w1_records],
        "em": [float(r["SPARSECOMM"]["is_em"]) for r in w1_records],
        "comm_tokens": [r["SPARSECOMM"]["comm_tokens"] for r in w1_records],
        "gen_tokens": [r["SPARSECOMM"]["gen_tokens"] for r in w1_records],
        "latency": [r["SPARSECOMM"]["latency_sec"] for r in w1_records],
    }
}

w1_results = {}
for comp_name, m_a, m_b in [
    ("SPARSECOMM_vs_MAD", "SPARSECOMM", "MAD"),
    ("SPARSECOMM_vs_NoComm", "SPARSECOMM", "NoComm"),
    ("MAD_vs_NoComm", "MAD", "NoComm"),
]:
    w1_results[comp_name] = {}
    for metric in ["f1", "em", "comm_tokens", "gen_tokens", "latency"]:
        res = compute_comparison(w1_data[m_a][metric], w1_data[m_b][metric], metric)
        w1_results[comp_name][metric] = res
        print(f"[{comp_name}] {metric:12s}: diff={res['mean_diff']:+.4f} | 95% t-CI: [{res['t_ci_95'][0]:.4f}, {res['t_ci_95'][1]:.4f}] | 95% Boot: [{res['boot_pct_ci_95'][0]:.4f}, {res['boot_pct_ci_95'][1]:.4f}] | p_t={res['p_val_t_two_sided']:.4f} | p_perm={res['p_val_perm']:.4f}")

print("\n" + "=" * 80)
print("2. CALCULATING PAIRED CONFIDENCE INTERVALS FOR W2")
print("=" * 80)

w2_records = load_jsonl(os.path.join(BASE_DIR, "results/W2_context_rationing/raw_results.jsonl"))

# W2 Methods: Full_Transmission, Semantic_Rationing, Random_Truncation
w2_data = {
    "Full_Transmission": {
        "f1": [r["Full_Transmission"]["f1"] for r in w2_records],
        "em": [float(r["Full_Transmission"]["is_em"]) for r in w2_records],
        "comm_tokens": [r["Full_Transmission"]["comm_tokens"] for r in w2_records],
        "latency": [r["Full_Transmission"]["latency_sec"] for r in w2_records],
    },
    "Semantic_Rationing": {
        "f1": [r["Semantic_Rationing"]["f1"] for r in w2_records],
        "em": [float(r["Semantic_Rationing"]["is_em"]) for r in w2_records],
        "comm_tokens": [r["Semantic_Rationing"]["comm_tokens"] for r in w2_records],
        "latency": [r["Semantic_Rationing"]["latency_sec"] for r in w2_records],
    },
    "Random_Truncation": {
        "f1": [r["Random_Truncation"]["f1"] for r in w2_records],
        "em": [float(r["Random_Truncation"]["is_em"]) for r in w2_records],
        "comm_tokens": [r["Random_Truncation"]["comm_tokens"] for r in w2_records],
        "latency": [r["Random_Truncation"]["latency_sec"] for r in w2_records],
    }
}

w2_results = {}
for comp_name, m_a, m_b in [
    ("Semantic_vs_Random", "Semantic_Rationing", "Random_Truncation"),
    ("Semantic_vs_Full", "Semantic_Rationing", "Full_Transmission"),
    ("Random_vs_Full", "Random_Truncation", "Full_Transmission"),
]:
    w2_results[comp_name] = {}
    for metric in ["f1", "em", "comm_tokens", "latency"]:
        res = compute_comparison(w2_data[m_a][metric], w2_data[m_b][metric], metric)
        w2_results[comp_name][metric] = res
        print(f"[{comp_name}] {metric:12s}: diff={res['mean_diff']:+.4f} | 95% t-CI: [{res['t_ci_95'][0]:.4f}, {res['t_ci_95'][1]:.4f}] | 95% Boot: [{res['boot_pct_ci_95'][0]:.4f}, {res['boot_pct_ci_95'][1]:.4f}] | p_t={res['p_val_t_two_sided']:.4f} | p_perm={res['p_val_perm']:.4f}")

print("\n" + "=" * 80)
print("SPECIFYING THE STATISTICAL TEST FOR W2 (Semantic vs Random Truncation)")
print("=" * 80)

f1_sem = np.array(w2_data["Semantic_Rationing"]["f1"])
f1_rnd = np.array(w2_data["Random_Truncation"]["f1"])
diff_f1_w2 = f1_sem - f1_rnd

em_sem = np.array(w2_data["Semantic_Rationing"]["em"])
em_rnd = np.array(w2_data["Random_Truncation"]["em"])
diff_em_w2 = em_sem - em_rnd

# 1. Paired Student's t-test
t_stat, p_t_2s = stats.ttest_1samp(diff_f1_w2, 0.0)
p_t_1s = p_t_2s / 2.0 if t_stat > 0 else 1.0 - p_t_2s / 2.0

# 2. Wilcoxon signed-rank test
# Non-zero differences
diff_nz = diff_f1_w2[diff_f1_w2 != 0]
n_nz = len(diff_nz)
n_zero = len(diff_f1_w2) - n_nz
w_stat, p_wilc_2s = stats.wilcoxon(diff_f1_w2, zero_method="wilcox", alternative="two-sided")
w_stat_1s, p_wilc_1s = stats.wilcoxon(diff_f1_w2, zero_method="wilcox", alternative="greater")

# 3. Paired permutation test (100,000 resamples)
rng = np.random.RandomState(42)
flips_100k = rng.choice([-1.0, 1.0], size=(100000, len(diff_f1_w2)))
perm_means_100k = np.mean(diff_f1_w2 * flips_100k, axis=1)
p_perm_2s = float(np.mean(np.abs(perm_means_100k) >= abs(np.mean(diff_f1_w2))))
p_perm_1s = float(np.mean(perm_means_100k >= np.mean(diff_f1_w2)))

# 4. Sign test
pos_diff = np.sum(diff_f1_w2 > 0)
neg_diff = np.sum(diff_f1_w2 < 0)
# Binomial test on n_nz
p_sign_2s = stats.binomtest(pos_diff, n_nz, p=0.5, alternative="two-sided").pvalue
p_sign_1s = stats.binomtest(pos_diff, n_nz, p=0.5, alternative="greater").pvalue

# 5. McNemar's test for Exact Match
# Contingency table:
# b: sem=1, rnd=0
# c: sem=0, rnd=1
b = int(np.sum((em_sem == 1) & (em_rnd == 0)))
c = int(np.sum((em_sem == 0) & (em_rnd == 1)))
a = int(np.sum((em_sem == 1) & (em_rnd == 1)))
d = int(np.sum((em_sem == 0) & (em_rnd == 0)))

# McNemar without continuity correction
mcnemar_stat = (abs(b - c))**2 / (b + c) if (b + c) > 0 else 0
p_mcnemar_chi2 = 1.0 - stats.chi2.cdf(mcnemar_stat, df=1) if (b + c) > 0 else 1.0
p_mcnemar_exact = stats.binomtest(b, b + c, p=0.5, alternative="two-sided").pvalue if (b + c) > 0 else 1.0

print(f"Summary of differences for W2 F1 (Semantic - Random):")
print(f"  Sample size N = {len(diff_f1_w2)}")
print(f"  Zero differences: {n_zero} / 100 questions")
print(f"  Positive differences (Semantic > Random): {pos_diff} / 100 questions")
print(f"  Negative differences (Semantic < Random): {neg_diff} / 100 questions")
print(f"  Mean diff: {np.mean(diff_f1_w2):+.4f} (+{np.mean(diff_f1_w2)*100:.2f} percentage points)")
print(f"  Std diff:  {np.std(diff_f1_w2, ddof=1):.4f}")
print(f"  Standard Error: {stats.sem(diff_f1_w2):.4f}")
print(f"Tests on F1:")
print(f"  1. Paired Student's t-test:        t = {t_stat:.4f}, df = 99, p (two-sided) = {p_t_2s:.4f}, p (one-sided) = {p_t_1s:.4f}")
print(f"  2. Wilcoxon signed-rank test:      W = {w_stat:.1f}, N_eff = {n_nz}, p (two-sided) = {p_wilc_2s:.4f}, p (one-sided) = {p_wilc_1s:.4f}")
print(f"  3. Paired permutation (100k):      p (two-sided) = {p_perm_2s:.4f}, p (one-sided) = {p_perm_1s:.4f}")
print(f"  4. Sign test (Binomial):           pos={pos_diff}, neg={neg_diff}, p (two-sided) = {p_sign_2s:.4f}, p (one-sided) = {p_sign_1s:.4f}")
print(f"Tests on Exact Match (b={b}, c={c}):")
print(f"  5. McNemar's exact test:           p = {p_mcnemar_exact:.4f}")

print("\n" + "=" * 80)
print("3. CALCULATING PAIRED CONFIDENCE INTERVALS FOR W3 (Corrected & Original)")
print("=" * 80)

w3_corr_records = load_jsonl(os.path.join(BASE_DIR, "results/W3_corrected/raw_results.jsonl"))
w3_orig_records = load_jsonl(os.path.join(BASE_DIR, "results/W3_adaptive_stopping/raw_results.jsonl"))

def process_w3(records, label="W3"):
    w3_d = {
        "Clean_Fixed": {
            "f1": [r["Clean_Fixed"]["f1"] for r in records],
            "em": [float(r["Clean_Fixed"]["is_em"]) for r in records],
            "comm_tokens": [r["Clean_Fixed"]["comm_tokens"] for r in records],
            "rounds": [r["Clean_Fixed"]["rounds"] for r in records],
            "latency": [r["Clean_Fixed"]["latency_sec"] for r in records],
        },
        "Clean_Adaptive": {
            "f1": [r["Clean_Adaptive"]["f1"] for r in records],
            "em": [float(r["Clean_Adaptive"]["is_em"]) for r in records],
            "comm_tokens": [r["Clean_Adaptive"]["comm_tokens"] for r in records],
            "rounds": [r["Clean_Adaptive"]["rounds"] for r in records],
            "latency": [r["Clean_Adaptive"]["latency_sec"] for r in records],
        },
        "Noisy_Fixed": {
            "f1": [r["Noisy_Fixed"]["f1"] for r in records],
            "em": [float(r["Noisy_Fixed"]["is_em"]) for r in records],
            "comm_tokens": [r["Noisy_Fixed"]["comm_tokens"] for r in records],
            "rounds": [r["Noisy_Fixed"]["rounds"] for r in records],
            "latency": [r["Noisy_Fixed"]["latency_sec"] for r in records],
        },
        "Noisy_Adaptive": {
            "f1": [r["Noisy_Adaptive"]["f1"] for r in records],
            "em": [float(r["Noisy_Adaptive"]["is_em"]) for r in records],
            "comm_tokens": [r["Noisy_Adaptive"]["comm_tokens"] for r in records],
            "rounds": [r["Noisy_Adaptive"]["rounds"] for r in records],
            "latency": [r["Noisy_Adaptive"]["latency_sec"] for r in records],
        }
    }
    
    comparisons = [
        ("Clean_Adaptive_vs_Fixed", "Clean_Adaptive", "Clean_Fixed"),
        ("Noisy_Adaptive_vs_Fixed", "Noisy_Adaptive", "Noisy_Fixed"),
        ("Noisy_vs_Clean_Fixed", "Noisy_Fixed", "Clean_Fixed"),
        ("Noisy_vs_Clean_Adaptive", "Noisy_Adaptive", "Clean_Adaptive"),
    ]
    
    res_dict = {}
    print(f"\n--- {label} ---")
    for comp_name, m_a, m_b in comparisons:
        res_dict[comp_name] = {}
        for metric in ["f1", "em", "comm_tokens", "rounds", "latency"]:
            res = compute_comparison(w3_d[m_a][metric], w3_d[m_b][metric], metric)
            res_dict[comp_name][metric] = res
            print(f"[{comp_name}] {metric:12s}: diff={res['mean_diff']:+.4f} | 95% t-CI: [{res['t_ci_95'][0]:.4f}, {res['t_ci_95'][1]:.4f}] | 95% Boot: [{res['boot_pct_ci_95'][0]:.4f}, {res['boot_pct_ci_95'][1]:.4f}] | p_t={res['p_val_t_two_sided']:.4f} | p_perm={res['p_val_perm']:.4f}")
    
    # Interaction: DID on F1
    # (Clean_Adaptive - Clean_Fixed) - (Noisy_Adaptive - Noisy_Fixed)
    diff_clean_f1 = np.array(w3_d["Clean_Adaptive"]["f1"]) - np.array(w3_d["Clean_Fixed"]["f1"])
    diff_noisy_f1 = np.array(w3_d["Noisy_Adaptive"]["f1"]) - np.array(w3_d["Noisy_Fixed"]["f1"])
    did_f1 = diff_clean_f1 - diff_noisy_f1
    res_did = compute_comparison(diff_clean_f1, diff_noisy_f1, "DID_f1")
    print(f"[Interaction_DID_F1] diff={res_did['mean_diff']:+.4f} | 95% t-CI: [{res_did['t_ci_95'][0]:.4f}, {res_did['t_ci_95'][1]:.4f}] | p_t={res_did['p_val_t_two_sided']:.4f} | p_perm={res_did['p_val_perm']:.4f}")
    res_dict["Interaction_DID_F1"] = res_did
    return res_dict

w3_corr_results = process_w3(w3_corr_records, label="W3-Corrected")
w3_orig_results = process_w3(w3_orig_records, label="W3-Original (Defective Halt)")

with open(os.path.join(BASE_DIR, "scratch/w1_w2_w3_stats.json"), "w") as f:
    json.dump({
        "W1": w1_results,
        "W2": w2_results,
        "W3_Corrected": w3_corr_results,
        "W3_Original": w3_orig_results,
    }, f, indent=2)

print("\nSaved scratch/w1_w2_w3_stats.json successfully.")
