#!/usr/bin/env python3
"""
Experiment 3: Per-Cell Realized Volume from 2×2 Factorial Grid
================================================================
Extracts token volume distributions from the archived factorial arm data and
produces paper-ready tables and plots.

2×2 Design:
                   Compression OFF        Compression ON
  Topology Static  Arm_00 (StaticRing)    Arm_10 (SparseRing)
  Topology Learned Arm_01 (PPOOnly)       Arm_11 (Full SparseComm)

Data sources:
  - 08-06results/ablation_{200,300,400}_arm{00,01,10,11}.json
  - 08-06results/continuous/Arm_{01,11}_*_stream.jsonl

Usage:
    python extract_factorial_volume.py [--verify] [--no-plots]
"""

import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ABLATION_DIR = os.path.join(BASE_DIR, "08-06results")
CONTINUOUS_DIR = os.path.join(BASE_DIR, "08-06results", "continuous")
RQ3_DIR = os.path.join(BASE_DIR, "results", "rq3")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "factorial_volume")

ARM_LABELS = {
    "00": "Arm_00_StaticRing",
    "01": "Arm_01_PPOOnly",
    "10": "Arm_10_SparseRing",
    "11": "Arm_11_Full",
}

ARM_DESCRIPTIONS = {
    "00": "Static Ring + No Compression",
    "01": "Learned Topology + No Compression",
    "10": "Static Ring + Learned Compression",
    "11": "Learned Topology + Learned Compression (Full SparseComm)",
}

N_TRAIN_VALUES = [200, 300, 400]


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_ablation_arm(n_train: int, arm_code: str) -> List[dict]:
    """Load ablation arm data from JSON files."""
    filepath = os.path.join(ABLATION_DIR, f"ablation_{n_train}_arm{arm_code}.json")
    if not os.path.exists(filepath):
        print(f"  [WARN] Missing: {filepath}")
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "sample_results" in data:
        return data["sample_results"]
    return []


def load_continuous_arm(arm_code: str, dataset: str = "gsm8k") -> List[dict]:
    """Load arm data from JSONL stream files."""
    arm_label = ARM_LABELS.get(arm_code, f"Arm_{arm_code}")
    filepath = os.path.join(CONTINUOUS_DIR, f"{arm_label}_{dataset}_stream.jsonl")
    if not os.path.exists(filepath):
        return []
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def extract_volume_metrics(records: List[dict]) -> dict:
    """Extract token volume metrics from a list of sample records."""
    if not records:
        return {
            "n_samples": 0,
            "tokens_tx": [], "tokens_full": [], "reduction_ratio": [],
            "compression_ratio": [], "density": [],
            "mode_dist": defaultdict(list),
            "accuracy": 0.0,
        }

    tokens_tx = []
    tokens_full = []
    reduction_ratios = []
    compression_ratios = []
    densities = []
    mode_counts = defaultdict(list)
    correct_count = 0

    for r in records:
        tx = r.get("total_tokens_transmitted", 0)
        full = r.get("total_tokens_if_full", 0)

        tokens_tx.append(tx)
        tokens_full.append(full)

        if full > 0:
            reduction_ratios.append(1.0 - tx / full)
        else:
            reduction_ratios.append(0.0)

        compression_ratios.append(r.get("compression_ratio", 0.0))
        densities.append(r.get("communication_density", 0.0))

        md = r.get("mode_distribution", {})
        for mode_name in ["BLOCK", "PARTIAL", "FULL"]:
            mode_counts[mode_name].append(md.get(mode_name, 0))

        if r.get("is_correct", False):
            correct_count += 1

    return {
        "n_samples": len(records),
        "tokens_tx": np.array(tokens_tx, dtype=float),
        "tokens_full": np.array(tokens_full, dtype=float),
        "reduction_ratio": np.array(reduction_ratios, dtype=float),
        "compression_ratio": np.array(compression_ratios, dtype=float),
        "density": np.array(densities, dtype=float),
        "mode_dist": dict(mode_counts),
        "accuracy": correct_count / max(len(records), 1),
        "n_correct": correct_count,
    }


# ---------------------------------------------------------------------------
# Statistical Tests
# ---------------------------------------------------------------------------
def wilcoxon_signed_rank_test(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Wilcoxon signed-rank test for paired samples (pure NumPy).
    Returns (test_statistic, approximate_p_value).
    """
    diff = x - y
    diff = diff[diff != 0]  # Remove zeros
    n = len(diff)
    if n == 0:
        return 0.0, 1.0

    abs_diff = np.abs(diff)
    ranks = np.argsort(np.argsort(abs_diff)) + 1.0  # Simple ranking

    # Handle ties by averaging ranks
    unique_vals = np.unique(abs_diff)
    for val in unique_vals:
        mask = abs_diff == val
        if np.sum(mask) > 1:
            mean_rank = np.mean(ranks[mask])
            ranks[mask] = mean_rank

    w_plus = np.sum(ranks[diff > 0])
    w_minus = np.sum(ranks[diff < 0])
    T = min(w_plus, w_minus)

    # Normal approximation for p-value (n > 20)
    if n >= 20:
        mean_T = n * (n + 1) / 4
        std_T = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        z = (T - mean_T) / std_T
        p_value = 2.0 * _norm_cdf(z)  # Two-sided
    else:
        # For small samples, use a rough approximation
        mean_T = n * (n + 1) / 4
        std_T = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        z = (T - mean_T) / max(std_T, 1e-10)
        p_value = 2.0 * _norm_cdf(z)

    return float(T), float(np.clip(p_value, 0, 1))


def _norm_cdf(x):
    """Standard normal CDF."""
    return 0.5 * (1 + _erf(x / np.sqrt(2)))


def _erf(x):
    """Error function approximation."""
    sign = np.sign(x)
    x = abs(x)
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y


def mann_whitney_u(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Mann-Whitney U test (two-sided, normal approximation)."""
    n1, n2 = len(x), len(y)
    combined = np.concatenate([x, y])
    ranks = np.argsort(np.argsort(combined)) + 1.0
    R1 = np.sum(ranks[:n1])
    U1 = R1 - n1 * (n1 + 1) / 2

    mean_U = n1 * n2 / 2
    std_U = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (U1 - mean_U) / max(std_U, 1e-10)
    p_value = 2.0 * _norm_cdf(-abs(z))

    return float(U1), float(np.clip(p_value, 0, 1))


# ---------------------------------------------------------------------------
# Question-Level Matching
# ---------------------------------------------------------------------------
def load_factorial_csv(n_train: int) -> Dict[str, Dict[str, int]]:
    """Load the factorial CSV and return {question_id: {arm: is_correct}}."""
    import csv
    filepath = os.path.join(RQ3_DIR, f"table7_sensitivity_{n_train}.csv")
    if not os.path.exists(filepath):
        return {}
    result = defaultdict(dict)
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = row["question_id"]
            arm = row["arm"]
            correct = int(row["is_correct"])
            result[qid][arm] = correct
    return dict(result)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_summary_table(all_metrics: Dict[str, Dict[str, dict]]):
    """Print a formatted summary table."""
    print()
    print("┌" + "─" * 118 + "┐")
    print("│" + " Per-Cell Realized Volume Summary ".center(118) + "│")
    print("├" + "─" * 118 + "┤")
    print(
        f"│ {'n_train':>7} │ {'Arm':>28} │ {'n':>4} │ {'Acc':>6} │ "
        f"{'Tokens TX':>13} │ {'Tokens Full':>13} │ {'Reduction':>11} │ "
        f"{'Density':>9} │ {'PARTIAL':>8} │ {'FULL':>6} │"
    )
    print("├" + "─" * 118 + "┤")

    for n_train in sorted(all_metrics.keys()):
        arms = all_metrics[n_train]
        for arm_code in ["00", "01", "10", "11"]:
            m = arms.get(arm_code)
            if m is None or m["n_samples"] == 0:
                continue

            desc = ARM_DESCRIPTIONS[arm_code]
            tx_mean = np.mean(m["tokens_tx"]) if len(m["tokens_tx"]) > 0 else 0
            tx_std = np.std(m["tokens_tx"]) if len(m["tokens_tx"]) > 0 else 0
            full_mean = np.mean(m["tokens_full"]) if len(m["tokens_full"]) > 0 else 0
            red_mean = np.mean(m["reduction_ratio"]) * 100 if len(m["reduction_ratio"]) > 0 else 0
            red_std = np.std(m["reduction_ratio"]) * 100 if len(m["reduction_ratio"]) > 0 else 0
            dens_mean = np.mean(m["density"]) if len(m["density"]) > 0 else 0
            acc = m["accuracy"] * 100

            partial_total = sum(m["mode_dist"].get("PARTIAL", []))
            full_total = sum(m["mode_dist"].get("FULL", []))

            print(
                f"│ {n_train:>7} │ {desc:>28} │ {m['n_samples']:>4} │ {acc:>5.1f}% │ "
                f"{tx_mean:>7.0f}±{tx_std:>4.0f} │ {full_mean:>7.0f}      │ "
                f"{red_mean:>5.1f}%±{red_std:>4.1f} │ "
                f"{dens_mean:>8.3f} │ {partial_total:>8} │ {full_total:>6} │"
            )

        if n_train != max(all_metrics.keys()):
            print("├" + "─" * 118 + "┤")

    print("└" + "─" * 118 + "┘")


def generate_markdown_report(all_metrics: Dict[str, Dict[str, dict]],
                              stat_results: Dict[str, list]) -> str:
    """Generate a markdown report with tables and stats."""
    lines = [
        "# Per-Cell Realized Token Volume — 2×2 Factorial Grid",
        "",
        "## Summary Table",
        "",
        "| n_train | Arm | Description | n | Accuracy | Mean Tokens TX | Mean Tokens Full | "
        "Token Reduction | Mean Density |",
        "|---------|-----|-------------|---|----------|----------------|------------------|"
        "----------------|--------------|",
    ]

    for n_train in sorted(all_metrics.keys()):
        arms = all_metrics[n_train]
        for arm_code in ["00", "01", "10", "11"]:
            m = arms.get(arm_code)
            if m is None or m["n_samples"] == 0:
                continue
            desc = ARM_DESCRIPTIONS[arm_code]
            tx_mean = np.mean(m["tokens_tx"]) if len(m["tokens_tx"]) > 0 else 0
            tx_std = np.std(m["tokens_tx"]) if len(m["tokens_tx"]) > 0 else 0
            full_mean = np.mean(m["tokens_full"]) if len(m["tokens_full"]) > 0 else 0
            red_mean = np.mean(m["reduction_ratio"]) * 100 if len(m["reduction_ratio"]) > 0 else 0
            red_std = np.std(m["reduction_ratio"]) * 100 if len(m["reduction_ratio"]) > 0 else 0
            dens_mean = np.mean(m["density"]) if len(m["density"]) > 0 else 0
            acc = m["accuracy"] * 100

            lines.append(
                f"| {n_train} | {arm_code} | {desc} | {m['n_samples']} | "
                f"{acc:.1f}% | {tx_mean:.0f} ± {tx_std:.0f} | {full_mean:.0f} | "
                f"{red_mean:.1f}% ± {red_std:.1f}% | {dens_mean:.3f} |"
            )

    # Statistical comparisons
    lines.extend([
        "",
        "## Volume Comparisons (Mann-Whitney U, two-sided)",
        "",
        "| n_train | Comparison | U statistic | p-value | Interpretation |",
        "|---------|------------|-------------|---------|----------------|",
    ])

    for n_train, tests in sorted(stat_results.items()):
        for test in tests:
            interp = "Significant (p<0.05)" if test["p_value"] < 0.05 else "Not significant"
            lines.append(
                f"| {n_train} | {test['comparison']} | {test['U']:.1f} | "
                f"{test['p_value']:.4f} | {interp} |"
            )

    # Mode distribution breakdown
    lines.extend([
        "",
        "## Mode Distribution (Total Counts)",
        "",
        "| n_train | Arm | BLOCK | PARTIAL | FULL |",
        "|---------|-----|-------|---------|------|",
    ])

    for n_train in sorted(all_metrics.keys()):
        arms = all_metrics[n_train]
        for arm_code in ["00", "01", "10", "11"]:
            m = arms.get(arm_code)
            if m is None or m["n_samples"] == 0:
                continue
            block = sum(m["mode_dist"].get("BLOCK", []))
            partial = sum(m["mode_dist"].get("PARTIAL", []))
            full = sum(m["mode_dist"].get("FULL", []))
            lines.append(f"| {n_train} | {ARM_DESCRIPTIONS[arm_code]} | {block} | {partial} | {full} |")

    return "\n".join(lines)


def generate_latex_table(all_metrics: Dict[str, Dict[str, dict]]) -> str:
    """Generate a LaTeX table for the paper."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Realized token volume across 2$\times$2 factorial arms (GSM8K). "
        r"Token reduction = $1 - V_{\text{tx}}/V_{\text{full}}$.}",
        r"\label{tab:factorial_volume}",
        r"\small",
        r"\begin{tabular}{ccrrrrr}",
        r"\toprule",
        r"$n_{\text{train}}$ & Arm & $n$ & Acc (\%) & $\bar{V}_{\text{tx}}$ & "
        r"$\bar{V}_{\text{full}}$ & Reduction (\%) \\",
        r"\midrule",
    ]

    for n_train in sorted(all_metrics.keys()):
        arms = all_metrics[n_train]
        first = True
        for arm_code in ["00", "01", "10", "11"]:
            m = arms.get(arm_code)
            if m is None or m["n_samples"] == 0:
                continue
            nt = str(n_train) if first else ""
            tx_mean = np.mean(m["tokens_tx"]) if len(m["tokens_tx"]) > 0 else 0
            full_mean = np.mean(m["tokens_full"]) if len(m["tokens_full"]) > 0 else 0
            red_mean = np.mean(m["reduction_ratio"]) * 100 if len(m["reduction_ratio"]) > 0 else 0
            acc = m["accuracy"] * 100

            lines.append(
                f"  {nt} & {ARM_DESCRIPTIONS[arm_code]} & {m['n_samples']} & "
                f"{acc:.1f} & {tx_mean:.0f} & {full_mean:.0f} & {red_mean:.1f} \\\\"
            )
            first = False
        lines.append(r"\midrule")

    # Remove last \midrule
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"

    lines.extend([
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def make_plots(all_metrics: Dict[str, Dict[str, dict]], output_dir: str):
    """Generate distribution plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not available, skipping plots")
        return

    os.makedirs(output_dir, exist_ok=True)
    colors = {"00": "#3498DB", "01": "#E74C3C", "10": "#2ECC71", "11": "#9B59B6"}

    for n_train in sorted(all_metrics.keys()):
        arms = all_metrics[n_train]

        # Box + violin plot for token volume
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot 1: Token transmitted distribution
        ax = axes[0]
        data_tx = []
        labels = []
        for arm_code in ["00", "01", "10", "11"]:
            m = arms.get(arm_code)
            if m and m["n_samples"] > 0:
                data_tx.append(m["tokens_tx"])
                labels.append(f"Arm {arm_code}")

        if data_tx:
            bp = ax.boxplot(data_tx, tick_labels=labels, patch_artist=True, showfliers=True,
                           flierprops=dict(marker=".", markersize=3, alpha=0.4))
            for patch, arm_code in zip(bp["boxes"], ["00", "01", "10", "11"]):
                patch.set_facecolor(colors[arm_code])
                patch.set_alpha(0.6)
            ax.set_ylabel("Tokens Transmitted", fontsize=11)
            ax.set_title(f"Token Volume (n_train={n_train})", fontsize=12, fontweight="bold")
            ax.tick_params(axis="x", rotation=15)

        # Plot 2: Token reduction ratio
        ax = axes[1]
        data_red = []
        labels_red = []
        for arm_code in ["00", "01", "10", "11"]:
            m = arms.get(arm_code)
            if m and m["n_samples"] > 0:
                data_red.append(m["reduction_ratio"] * 100)
                labels_red.append(f"Arm {arm_code}")

        if data_red:
            bp = ax.boxplot(data_red, tick_labels=labels_red, patch_artist=True, showfliers=True,
                           flierprops=dict(marker=".", markersize=3, alpha=0.4))
            for patch, arm_code in zip(bp["boxes"], ["00", "01", "10", "11"]):
                patch.set_facecolor(colors[arm_code])
                patch.set_alpha(0.6)
            ax.set_ylabel("Token Reduction (%)", fontsize=11)
            ax.set_title(f"Token Reduction (n_train={n_train})", fontsize=12, fontweight="bold")
            ax.tick_params(axis="x", rotation=15)

        # Plot 3: Communication density
        ax = axes[2]
        data_dens = []
        labels_dens = []
        for arm_code in ["00", "01", "10", "11"]:
            m = arms.get(arm_code)
            if m and m["n_samples"] > 0:
                data_dens.append(m["density"])
                labels_dens.append(f"Arm {arm_code}")

        if data_dens:
            bp = ax.boxplot(data_dens, tick_labels=labels_dens, patch_artist=True, showfliers=True,
                           flierprops=dict(marker=".", markersize=3, alpha=0.4))
            for patch, arm_code in zip(bp["boxes"], ["00", "01", "10", "11"]):
                patch.set_facecolor(colors[arm_code])
                patch.set_alpha(0.6)
            ax.set_ylabel("Communication Density", fontsize=11)
            ax.set_title(f"Density (n_train={n_train})", fontsize=12, fontweight="bold")
            ax.tick_params(axis="x", rotation=15)

        plt.tight_layout()
        filepath = os.path.join(output_dir, f"factorial_volume_n{n_train}.png")
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {filepath}")

    # Combined overview: Mean token reduction by arm across n_train
    fig, ax = plt.subplots(figsize=(10, 6))

    x_positions = np.arange(len(N_TRAIN_VALUES))
    bar_width = 0.18

    for i, arm_code in enumerate(["00", "01", "10", "11"]):
        means = []
        stds = []
        for n_train in N_TRAIN_VALUES:
            m = all_metrics.get(n_train, {}).get(arm_code)
            if m and m["n_samples"] > 0:
                means.append(np.mean(m["reduction_ratio"]) * 100)
                stds.append(np.std(m["reduction_ratio"]) * 100 / np.sqrt(m["n_samples"]))
            else:
                means.append(0)
                stds.append(0)

        ax.bar(x_positions + i * bar_width, means, bar_width,
               yerr=stds, capsize=3,
               label=ARM_DESCRIPTIONS[arm_code],
               color=colors[arm_code], alpha=0.8,
               edgecolor="white", linewidth=0.5)

    ax.set_xlabel("n_train", fontsize=12)
    ax.set_ylabel("Mean Token Reduction (%)", fontsize=12)
    ax.set_title("Token Reduction Across Factorial Arms", fontsize=14, fontweight="bold")
    ax.set_xticks(x_positions + 1.5 * bar_width)
    ax.set_xticklabels([str(n) for n in N_TRAIN_VALUES])
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    filepath = os.path.join(output_dir, "factorial_volume_overview.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filepath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_analysis(do_plots: bool = True, do_verify: bool = False):
    """Run the full factorial volume analysis."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_metrics = {}
    stat_results = {}

    for n_train in N_TRAIN_VALUES:
        print(f"\n{'=' * 70}")
        print(f"  n_train = {n_train}")
        print(f"{'=' * 70}")

        arm_metrics = {}
        for arm_code in ["00", "01", "10", "11"]:
            records = load_ablation_arm(n_train, arm_code)

            # Also try JSONL stream if JSON is empty
            if not records:
                records = load_continuous_arm(arm_code)

            metrics = extract_volume_metrics(records)
            arm_metrics[arm_code] = metrics

            desc = ARM_DESCRIPTIONS[arm_code]
            n = metrics["n_samples"]
            if n > 0:
                tx_mean = np.mean(metrics["tokens_tx"])
                red_mean = np.mean(metrics["reduction_ratio"]) * 100
                dens_mean = np.mean(metrics["density"])
                print(f"  Arm {arm_code} ({desc}):")
                print(f"    n={n}, acc={metrics['accuracy']*100:.1f}%, "
                      f"mean_tokens_tx={tx_mean:.0f}, "
                      f"reduction={red_mean:.1f}%, density={dens_mean:.3f}")
            else:
                print(f"  Arm {arm_code} ({desc}): NO DATA")

        all_metrics[n_train] = arm_metrics

        # Statistical comparisons between arms sharing questions
        tests = []

        # Compression effect: Arm_00 vs Arm_10 (holding topology=static)
        m00 = arm_metrics.get("00", {})
        m10 = arm_metrics.get("10", {})
        if m00.get("n_samples", 0) > 0 and m10.get("n_samples", 0) > 0:
            n_min = min(len(m00["tokens_tx"]), len(m10["tokens_tx"]))
            if n_min > 0:
                U, p = mann_whitney_u(m00["tokens_tx"][:n_min], m10["tokens_tx"][:n_min])
                tests.append({"comparison": "Arm_00 vs Arm_10 (compression effect, static topology)",
                             "U": U, "p_value": p})
                print(f"    Compression effect (static): U={U:.1f}, p={p:.4f}")

        # Compression effect: Arm_01 vs Arm_11 (holding topology=learned)
        m01 = arm_metrics.get("01", {})
        m11 = arm_metrics.get("11", {})
        if m01.get("n_samples", 0) > 0 and m11.get("n_samples", 0) > 0:
            n_min = min(len(m01["tokens_tx"]), len(m11["tokens_tx"]))
            if n_min > 0:
                U, p = mann_whitney_u(m01["tokens_tx"][:n_min], m11["tokens_tx"][:n_min])
                tests.append({"comparison": "Arm_01 vs Arm_11 (compression effect, learned topology)",
                             "U": U, "p_value": p})
                print(f"    Compression effect (learned): U={U:.1f}, p={p:.4f}")

        # Topology effect: Arm_00 vs Arm_01 (holding compression=off)
        if m00.get("n_samples", 0) > 0 and m01.get("n_samples", 0) > 0:
            n_min = min(len(m00["tokens_tx"]), len(m01["tokens_tx"]))
            if n_min > 0:
                U, p = mann_whitney_u(m00["tokens_tx"][:n_min], m01["tokens_tx"][:n_min])
                tests.append({"comparison": "Arm_00 vs Arm_01 (topology effect, no compression)",
                             "U": U, "p_value": p})
                print(f"    Topology effect (no compress): U={U:.1f}, p={p:.4f}")

        # Topology effect: Arm_10 vs Arm_11 (holding compression=on)
        if m10.get("n_samples", 0) > 0 and m11.get("n_samples", 0) > 0:
            n_min = min(len(m10["tokens_tx"]), len(m11["tokens_tx"]))
            if n_min > 0:
                U, p = mann_whitney_u(m10["tokens_tx"][:n_min], m11["tokens_tx"][:n_min])
                tests.append({"comparison": "Arm_10 vs Arm_11 (topology effect, with compression)",
                             "U": U, "p_value": p})
                print(f"    Topology effect (w/ compress): U={U:.1f}, p={p:.4f}")

        stat_results[n_train] = tests

    # Print summary table
    print_summary_table(all_metrics)

    # Verification
    if do_verify:
        print(f"\n{'=' * 70}")
        print("  VERIFICATION")
        print(f"{'=' * 70}")

        # Check Arm_00 has near-zero reduction (static ring, no learned compression)
        for n_train in N_TRAIN_VALUES:
            m00 = all_metrics.get(n_train, {}).get("00", {})
            if m00.get("n_samples", 0) > 0:
                red = np.mean(m00["reduction_ratio"]) * 100
                # Static ring with fixed weights should still have some reduction
                # because communication.py applies stochastic sentence dropout
                # based on the weight values (0.6 for ring)
                print(f"  n_train={n_train}, Arm_00 token reduction: {red:.1f}%")
                if red < 60:
                    print(f"    ✓ Expected: moderate reduction from ring weights (w=0.6)")
                else:
                    print(f"    ⚠ Higher than expected for static ring")

        # Cross-check with factorial CSV accuracy
        for n_train in N_TRAIN_VALUES:
            csv_data = load_factorial_csv(n_train)
            if csv_data:
                for arm_label in ARM_LABELS.values():
                    csv_correct = sum(1 for qid, arms in csv_data.items() if arms.get(arm_label, 0) == 1)
                    csv_total = sum(1 for qid, arms in csv_data.items() if arm_label in arms)
                    if csv_total > 0:
                        csv_acc = csv_correct / csv_total * 100
                        # Find matching arm_code
                        arm_code = [k for k, v in ARM_LABELS.items() if v == arm_label][0]
                        json_acc = all_metrics.get(n_train, {}).get(arm_code, {}).get("accuracy", 0) * 100
                        match = "✓" if abs(csv_acc - json_acc) < 1.0 else "⚠"
                        print(f"  n_train={n_train}, {arm_label}: CSV={csv_acc:.1f}%, JSON={json_acc:.1f}% {match}")

    # Generate outputs
    print(f"\n{'=' * 70}")
    print("  Generating outputs")
    print(f"{'=' * 70}")

    md_report = generate_markdown_report(all_metrics, stat_results)
    md_path = os.path.join(OUTPUT_DIR, "factorial_volume_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)
    print(f"  Saved markdown: {md_path}")

    latex_table = generate_latex_table(all_metrics)
    latex_path = os.path.join(OUTPUT_DIR, "factorial_volume_results.tex")
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_table)
    print(f"  Saved LaTeX: {latex_path}")

    if do_plots:
        make_plots(all_metrics, OUTPUT_DIR)

    # Print markdown report
    print(f"\n{md_report}")

    return all_metrics, stat_results


def main():
    do_verify = "--verify" in sys.argv
    do_plots = "--no-plots" not in sys.argv

    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " Experiment 3: Per-Cell Realized Volume from 2×2 Factorial Grid ".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    run_analysis(do_plots=do_plots, do_verify=do_verify)


if __name__ == "__main__":
    main()
