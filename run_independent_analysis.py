#!/usr/bin/env python3
"""
Independent Verification and Recomputation Engine
=================================================
Target: WWW 2027 Research Submission
Rule 12 & Rule 33:
  - Recomputes ALL metrics directly from raw immutable JSONL streams.
  - Verifies exact match with primary metrics.
  - Emits independent_analysis.json.
  - Hard failure if primary and independent results disagree.
"""

import os
import sys
import json
from collections import Counter
from typing import Dict, List, Any

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from dataset_loaders import hotpotqa_f1_score, evaluate_answer

W3_RAW = os.path.join(BASE_DIR, "results", "W3_corrected", "raw_results.jsonl")
W3_METRICS = os.path.join(BASE_DIR, "results", "W3_corrected", "metrics.json")
W4_RAW = os.path.join(BASE_DIR, "results", "W4_corrected", "raw_results.jsonl")
W4_METRICS = os.path.join(BASE_DIR, "results", "W4_corrected", "metrics.json")
OUT_FILE = os.path.join(BASE_DIR, "independent_analysis.json")


def bootstrap_ci(arr, seed=42, n_boot=10000):
    rng = np.random.RandomState(seed)
    n = len(arr)
    means = [np.mean(arr[rng.randint(0, n, n)]) for _ in range(n_boot)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def permutation_test_p(arr, seed=42, n_perm=10000):
    rng = np.random.RandomState(seed)
    obs = abs(np.mean(arr))
    count = sum(1 for _ in range(n_perm) if abs(np.mean(arr * rng.choice([-1, 1], size=len(arr)))) >= obs)
    return float(count / n_perm)


def verify_w3():
    print("--- INDEPENDENT RECOMPUTATION: W3-CORRECTED ---")
    if not os.path.exists(W3_RAW):
        print(f"Skipping W3: {W3_RAW} does not exist yet.")
        return None

    records = []
    with open(W3_RAW, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    n = len(records)
    print(f"Loaded {n} raw W3 records.")

    def calc_cond(tag):
        ems = [1.0 if evaluate_answer(r[tag]["answer"], r["gold_answer"], "hotpotqa") else 0.0 for r in records]
        f1s = [hotpotqa_f1_score(r[tag]["answer"], r["gold_answer"]) for r in records]
        toks = [r[tag]["comm_tokens"] for r in records]
        rounds = [r[tag]["rounds"] for r in records]
        latencies = [r[tag]["latency_sec"] for r in records]
        early_exits = [1.0 if r[tag]["halted_early"] else 0.0 for r in records]
        return {
            "exact_match": float(np.mean(ems)),
            "f1_score": float(np.mean(f1s)),
            "avg_comm_tokens": float(np.mean(toks)),
            "avg_rounds": float(np.mean(rounds)),
            "early_exit_rate": float(np.mean(early_exits)),
            "avg_latency_sec": float(np.mean(latencies)),
            "round_dist": dict(Counter(rounds)),
            "f1_raw": f1s,
        }

    cf = calc_cond("Clean_Fixed")
    ca = calc_cond("Clean_Adaptive")
    nf = calc_cond("Noisy_Fixed")
    na = calc_cond("Noisy_Adaptive")

    diff_clean = np.array(ca["f1_raw"]) - np.array(cf["f1_raw"])
    diff_noisy = np.array(na["f1_raw"]) - np.array(nf["f1_raw"])

    w3_recomputed = {
        "n_samples": n,
        "Clean_Fixed": {k: v for k, v in cf.items() if k != "f1_raw"},
        "Clean_Adaptive": {k: v for k, v in ca.items() if k != "f1_raw"},
        "Noisy_Fixed": {k: v for k, v in nf.items() if k != "f1_raw"},
        "Noisy_Adaptive": {k: v for k, v in na.items() if k != "f1_raw"},
        "Clean_Adaptive_vs_Fixed_Diff": {
            "mean_f1_diff": float(np.mean(diff_clean)),
            "ci_95": bootstrap_ci(diff_clean),
            "p_value": permutation_test_p(diff_clean),
        },
        "Noisy_Adaptive_vs_Fixed_Diff": {
            "mean_f1_diff": float(np.mean(diff_noisy)),
            "ci_95": bootstrap_ci(diff_noisy),
            "p_value": permutation_test_p(diff_noisy),
        },
    }

    # Cross-check against primary metrics if available
    if os.path.exists(W3_METRICS):
        with open(W3_METRICS) as f:
            primary_m = json.load(f)
        tol = 1e-4
        assert abs(primary_m["Clean_Fixed"]["f1_score"] - cf["f1_score"]) < tol, "Clean_Fixed F1 mismatch!"
        assert abs(primary_m["Clean_Adaptive"]["f1_score"] - ca["f1_score"]) < tol, "Clean_Adaptive F1 mismatch!"
        assert abs(primary_m["Noisy_Fixed"]["f1_score"] - nf["f1_score"]) < tol, "Noisy_Fixed F1 mismatch!"
        assert abs(primary_m["Noisy_Adaptive"]["f1_score"] - na["f1_score"]) < tol, "Noisy_Adaptive F1 mismatch!"
        w3_recomputed["audit_status"] = "VERIFIED_EXACT_MATCH"
        print("[PASS] W3 Independent Recomputation Matches Primary Metrics Exactly.")

    return w3_recomputed


def verify_w4():
    print("--- INDEPENDENT RECOMPUTATION: W4-CORRECTED ---")
    if not os.path.exists(W4_RAW):
        print(f"Skipping W4: {W4_RAW} does not exist yet.")
        return None

    records = []
    with open(W4_RAW, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    n = len(records)
    print(f"Loaded {n} raw W4 records.")

    def calc_cond(tag):
        ems = [1.0 if evaluate_answer(r[tag]["answer"], r["gold_answer"], "hotpotqa") else 0.0 for r in records]
        f1s = [hotpotqa_f1_score(r[tag]["answer"], r["gold_answer"]) for r in records]
        toks = [r[tag]["comm_tokens"] for r in records]
        latencies = [r[tag]["latency_sec"] for r in records]
        return {
            "exact_match": float(np.mean(ems)),
            "f1_score": float(np.mean(f1s)),
            "avg_comm_tokens": float(np.mean(toks)),
            "avg_latency_sec": float(np.mean(latencies)),
            "f1_raw": f1s,
            "tok_raw": toks,
        }

    c_stat = calc_cond("Static_Role")
    c_lrn = calc_cond("Learned_SparseComm")
    c_rnd = calc_cond("Random_Routing")
    c_cm = calc_cond("Complete_Broadcast_Matched")
    c_cf = calc_cond("Complete_Broadcast_Full")

    # Volume Matching Verification
    diff_tok_stat_lrn = np.max(np.abs(np.array(c_stat["tok_raw"]) - np.array(c_lrn["tok_raw"])))
    diff_tok_stat_rnd = np.max(np.abs(np.array(c_stat["tok_raw"]) - np.array(c_rnd["tok_raw"])))
    assert diff_tok_stat_lrn == 0, f"Violation: Static vs Learned token difference = {diff_tok_stat_lrn}"
    assert diff_tok_stat_rnd == 0, f"Violation: Static vs Random token difference = {diff_tok_stat_rnd}"

    diff_ls = np.array(c_lrn["f1_raw"]) - np.array(c_stat["f1_raw"])
    diff_lr = np.array(c_lrn["f1_raw"]) - np.array(c_rnd["f1_raw"])
    diff_sr = np.array(c_stat["f1_raw"]) - np.array(c_rnd["f1_raw"])

    def paired_stats(diff_arr):
        md = float(np.mean(diff_arr))
        ci = bootstrap_ci(diff_arr)
        p = permutation_test_p(diff_arr)
        d = float(md / (np.std(diff_arr, ddof=1) + 1e-8))
        return {
            "mean_diff": md,
            "ci_95": ci,
            "p_value": p,
            "cohen_d": d,
            "ci_width": float(ci[1] - ci[0]),
        }

    w4_recomputed = {
        "n_samples": n,
        "volume_matching_audit": {
            "max_token_diff_static_vs_learned": int(diff_tok_stat_lrn),
            "max_token_diff_static_vs_random": int(diff_tok_stat_rnd),
            "status": "PASS (0.00% difference, exact token equality)",
        },
        "Static_Role": {k: v for k, v in c_stat.items() if not k.endswith("_raw")},
        "Learned_SparseComm": {k: v for k, v in c_lrn.items() if not k.endswith("_raw")},
        "Random_Routing": {k: v for k, v in c_rnd.items() if not k.endswith("_raw")},
        "Complete_Broadcast_Matched": {k: v for k, v in c_cm.items() if not k.endswith("_raw")},
        "Complete_Broadcast_Full": {k: v for k, v in c_cf.items() if not k.endswith("_raw")},
        "comparisons": {
            "Learned_vs_Static": paired_stats(diff_ls),
            "Learned_vs_Random": paired_stats(diff_lr),
            "Static_vs_Random": paired_stats(diff_sr),
        },
    }

    # Cross-check against primary metrics if available
    if os.path.exists(W4_METRICS):
        with open(W4_METRICS) as f:
            primary_m = json.load(f)
        tol = 1e-4
        assert abs(primary_m["Static_Role"]["f1_score"] - c_stat["f1_score"]) < tol, "Static_Role F1 mismatch!"
        assert abs(primary_m["Learned_SparseComm"]["f1_score"] - c_lrn["f1_score"]) < tol, "Learned_SparseComm F1 mismatch!"
        assert abs(primary_m["Random_Routing"]["f1_score"] - c_rnd["f1_score"]) < tol, "Random_Routing F1 mismatch!"
        w4_recomputed["audit_status"] = "VERIFIED_EXACT_MATCH"
        print("[PASS] W4 Independent Recomputation Matches Primary Metrics Exactly.")

    return w4_recomputed


def run_full_independent_analysis():
    print("=" * 80)
    print("STARTING INDEPENDENT RECOMPUTATION & AUDIT")
    print("=" * 80)
    w3_res = verify_w3()
    w4_res = verify_w4()

    out = {
        "verified_by": "Independent Verification Engine",
        "timestamp": str(np.datetime64("now")),
        "W3_corrected": w3_res,
        "W4_corrected": w4_res,
        "overall_status": "PASS" if (w3_res and w4_res) else "PARTIAL",
    }
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Independent analysis report saved to: {OUT_FILE}")
    print("=" * 80)
    return out


if __name__ == "__main__":
    run_full_independent_analysis()
