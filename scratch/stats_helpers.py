#!/usr/bin/env python3
"""
Comprehensive Statistical & Integrity Audit Script
===================================================
Calculates paired confidence intervals for W1, W2, W3;
Specifies statistical tests for W2;
Examines reproducibility of Clean/Noisy W3 difference;
Verifies W4 volume matching procedure on per-question token records.
"""

import json
import os
import sys
import numpy as np
from scipy import stats

BASE_DIR = "/home/saurab/Documents/projectRLMAIN"

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def bootstrap_paired_ci(diff_arr, n_boot=10000, seed=42):
    rng = np.random.RandomState(seed)
    diff_arr = np.asarray(diff_arr, dtype=float)
    n = len(diff_arr)
    boot_means = np.array([
        np.mean(diff_arr[rng.randint(0, n, size=n)]) for _ in range(n_boot)
    ])
    pct_lower = float(np.percentile(boot_means, 2.5))
    pct_upper = float(np.percentile(boot_means, 97.5))
    
    # BCa bootstrap
    # 1. Bias correction z0
    mean_orig = np.mean(diff_arr)
    prop_less = np.mean(boot_means < mean_orig)
    if prop_less == 0:
        z0 = -3.0
    elif prop_less == 1:
        z0 = 3.0
    else:
        z0 = stats.norm.ppf(prop_less)
    
    # 2. Acceleration a (jackknife)
    jack_means = np.zeros(n)
    for i in range(n):
        jack_means[i] = (np.sum(diff_arr) - diff_arr[i]) / (n - 1)
    mean_jack = np.mean(jack_means)
    num = np.sum((mean_jack - jack_means)**3)
    denom = 6.0 * (np.sum((mean_jack - jack_means)**2)**1.5)
    a = num / denom if denom != 0 else 0.0
    
    # BCa percentiles
    z_alpha_l = stats.norm.ppf(0.025)
    z_alpha_u = stats.norm.ppf(0.975)
    
    alpha_l = stats.norm.cdf(z0 + (z0 + z_alpha_l) / (1 - a * (z0 + z_alpha_l)))
    alpha_u = stats.norm.cdf(z0 + (z0 + z_alpha_u) / (1 - a * (z0 + z_alpha_u)))
    
    # clip to [0, 1]
    alpha_l = np.clip(alpha_l, 0.0001, 0.9999)
    alpha_u = np.clip(alpha_u, 0.0001, 0.9999)
    
    bca_lower = float(np.percentile(boot_means, alpha_l * 100))
    bca_upper = float(np.percentile(boot_means, alpha_u * 100))
    
    return {
        "percentile_95": [pct_lower, pct_upper],
        "bca_95": [bca_lower, bca_upper]
    }

def paired_t_ci(diff_arr, alpha=0.05):
    diff_arr = np.asarray(diff_arr, dtype=float)
    n = len(diff_arr)
    mean_d = np.mean(diff_arr)
    se_d = stats.sem(diff_arr)
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    ci_low = float(mean_d - t_crit * se_d)
    ci_high = float(mean_d + t_crit * se_d)
    t_stat, p_val = stats.ttest_1samp(diff_arr, 0.0)
    return {
        "mean_diff": float(mean_d),
        "std_diff": float(np.std(diff_arr, ddof=1)),
        "se_diff": float(se_d),
        "t_ci_95": [ci_low, ci_high],
        "t_stat": float(t_stat),
        "p_val_two_sided": float(p_val),
        "p_val_one_sided": float(p_val / 2.0 if t_stat > 0 else 1.0 - p_val / 2.0)
    }

def paired_perm_test(diff_arr, n_perm=10000, seed=42):
    rng = np.random.RandomState(seed)
    diff_arr = np.asarray(diff_arr, dtype=float)
    obs = abs(np.mean(diff_arr))
    flips = rng.choice([-1.0, 1.0], size=(n_perm, len(diff_arr)))
    perm_means = np.abs(np.mean(diff_arr * flips, axis=1))
    p_val = float(np.mean(perm_means >= obs))
    return p_val

def compute_comparison(arr_a, arr_b, metric_name="metric"):
    diff = np.asarray(arr_a, dtype=float) - np.asarray(arr_b, dtype=float)
    t_res = paired_t_ci(diff)
    b_res = bootstrap_paired_ci(diff)
    p_perm = paired_perm_test(diff)
    cohen_d = float(t_res["mean_diff"] / (t_res["std_diff"] + 1e-12))
    return {
        "mean_a": float(np.mean(arr_a)),
        "mean_b": float(np.mean(arr_b)),
        "mean_diff": t_res["mean_diff"],
        "t_ci_95": t_res["t_ci_95"],
        "boot_pct_ci_95": b_res["percentile_95"],
        "boot_bca_ci_95": b_res["bca_95"],
        "t_stat": t_res["t_stat"],
        "p_val_t_two_sided": t_res["p_val_two_sided"],
        "p_val_t_one_sided": t_res["p_val_one_sided"],
        "p_val_perm": p_perm,
        "cohen_d": cohen_d
    }

print("Loaded statistical functions.")
