#!/usr/bin/env python3
"""
Experiment 2: Paired Question-Clustered Bootstrap & Non-Inferiority Test
=========================================================================
Re-analyzes archived .jsonl streams to produce:
  1. Paired question-clustered bootstrap 95% BCa CIs for accuracy differences
  2. TOST non-inferiority p-values at δ ∈ {3, 5, 10} percentage points
  3. Summary tables in markdown and LaTeX
  4. Bootstrap distribution histograms (PNG)

Usage:
    python bootstrap_analysis.py [--verify] [--no-plots]

Data sources:
    08-06results/continuous/*.jsonl
"""

import json
import os
import sys
import hashlib
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 2026
N_BOOTSTRAP = 10_000
DELTA_MARGINS = [0.03, 0.05, 0.10]  # Non-inferiority margins (3, 5, 10 pp)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONTINUOUS_DIR = os.path.join(BASE_DIR, "08-06results", "continuous")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "bootstrap")


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def question_hash(question_text: str) -> str:
    """Deterministic hash of question text for matching across methods."""
    normalized = " ".join(question_text.lower().split())
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()[:12]


def load_jsonl_stream(filepath: str, start_row: int = 0, end_row: Optional[int] = None) -> List[dict]:
    """Load a .jsonl file, optionally slicing rows."""
    if not os.path.exists(filepath):
        print(f"  [WARN] File not found: {filepath}")
        return []
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start_row:
                continue
            if end_row is not None and i >= end_row:
                break
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def build_question_index(records: List[dict]) -> Dict[str, dict]:
    """Index records by question hash. Later occurrences overwrite earlier ones."""
    index = {}
    for r in records:
        q = r.get("question", "")
        if q:
            qh = question_hash(q)
            index[qh] = r
    return index


def load_method_data(dataset: str) -> Dict[str, Dict[str, dict]]:
    """Load all three methods for a dataset, returning {method_name: {q_hash: record}}."""
    methods = {}

    if dataset == "gsm8k":
        # MAD: seeds stored sequentially in one file, 300 per seed (but file has 1161 = ~387×3)
        mad_records = load_jsonl_stream(os.path.join(CONTINUOUS_DIR, "Multi-Agent Debate_gsm8k_stream.jsonl"))
        methods["MAD"] = build_question_index(mad_records)

        # NoComm
        nocomm_records = load_jsonl_stream(os.path.join(CONTINUOUS_DIR, "No Communication_gsm8k_stream.jsonl"))
        methods["NoComm"] = build_question_index(nocomm_records)

        # SparseComm: from Transfer_GSM8K_seed{0,1,2}, rows 300-599 (400-sample checkpoint)
        sparse_records = []
        for seed in range(3):
            fp = os.path.join(CONTINUOUS_DIR, f"Transfer_GSM8K_seed{seed}_gsm8k_stream.jsonl")
            seed_records = load_jsonl_stream(fp, start_row=300, end_row=600)
            sparse_records.extend(seed_records)
        methods["SparseComm"] = build_question_index(sparse_records)

    elif dataset == "aqua":
        mad_records = load_jsonl_stream(os.path.join(CONTINUOUS_DIR, "Multi-Agent Debate_aqua_stream.jsonl"))
        methods["MAD"] = build_question_index(mad_records)

        nocomm_records = load_jsonl_stream(os.path.join(CONTINUOUS_DIR, "No Communication_aqua_stream.jsonl"))
        methods["NoComm"] = build_question_index(nocomm_records)

        sparse_records = []
        for seed in range(3):
            fp = os.path.join(CONTINUOUS_DIR, f"Transfer_AQUA_seed{seed}_aqua_stream.jsonl")
            seed_records = load_jsonl_stream(fp)
            sparse_records.extend(seed_records)
        methods["SparseComm"] = build_question_index(sparse_records)

    elif dataset == "hotpotqa":
        mad_records = load_jsonl_stream(os.path.join(CONTINUOUS_DIR, "Multi-Agent Debate_hotpotqa_stream.jsonl"))
        methods["MAD"] = build_question_index(mad_records)

        nocomm_records = load_jsonl_stream(os.path.join(CONTINUOUS_DIR, "No Communication_hotpotqa_stream.jsonl"))
        methods["NoComm"] = build_question_index(nocomm_records)

        sparse_records = []
        for seed in range(3):
            fp = os.path.join(CONTINUOUS_DIR, f"Transfer_HOTPOTQA_seed{seed}_hotpotqa_stream.jsonl")
            seed_records = load_jsonl_stream(fp)
            sparse_records.extend(seed_records)
        methods["SparseComm"] = build_question_index(sparse_records)

    return methods


def get_paired_deltas(methods: Dict[str, Dict[str, dict]], method_a: str, method_b: str) -> Tuple[np.ndarray, int, int, int]:
    """
    Compute per-question accuracy deltas: correct_A(q) - correct_B(q).
    Returns (deltas_array, n_paired, n_only_a, n_only_b).
    """
    idx_a = methods.get(method_a, {})
    idx_b = methods.get(method_b, {})

    common_qs = set(idx_a.keys()) & set(idx_b.keys())
    only_a = len(set(idx_a.keys()) - common_qs)
    only_b = len(set(idx_b.keys()) - common_qs)

    deltas = []
    for qh in sorted(common_qs):
        correct_a = 1.0 if idx_a[qh].get("is_correct", False) else 0.0
        correct_b = 1.0 if idx_b[qh].get("is_correct", False) else 0.0
        deltas.append(correct_a - correct_b)

    return np.array(deltas), len(common_qs), only_a, only_b


# ---------------------------------------------------------------------------
# Bootstrap Engine
# ---------------------------------------------------------------------------
def bca_ci(data: np.ndarray, stat_fn, n_bootstrap: int = 10_000, alpha: float = 0.05,
           rng: np.random.Generator = None) -> Tuple[float, float, float, np.ndarray]:
    """
    BCa (bias-corrected and accelerated) bootstrap confidence interval.

    Returns: (lower, upper, point_estimate, bootstrap_distribution)
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    n = len(data)
    theta_hat = stat_fn(data)

    # Bootstrap distribution
    boot_stats = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        boot_stats[b] = stat_fn(data[indices])

    # Bias correction: z0
    z0 = _norm_ppf(np.mean(boot_stats < theta_hat))

    # Acceleration: a (jackknife)
    jackknife_stats = np.empty(n)
    for i in range(n):
        jack_sample = np.delete(data, i)
        jackknife_stats[i] = stat_fn(jack_sample)
    jack_mean = np.mean(jackknife_stats)
    num = np.sum((jack_mean - jackknife_stats) ** 3)
    den = 6.0 * (np.sum((jack_mean - jackknife_stats) ** 2) ** 1.5)
    a = num / den if abs(den) > 1e-12 else 0.0

    # Adjusted percentiles
    z_alpha = _norm_ppf(alpha / 2)
    z_1alpha = _norm_ppf(1 - alpha / 2)

    a1 = _norm_cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha)))
    a2 = _norm_cdf(z0 + (z0 + z_1alpha) / (1 - a * (z0 + z_1alpha)))

    lower = np.percentile(boot_stats, 100 * a1)
    upper = np.percentile(boot_stats, 100 * a2)

    return lower, upper, theta_hat, boot_stats


def _norm_ppf(p):
    """Inverse normal CDF (probit) using rational approximation."""
    p = np.clip(p, 1e-10, 1 - 1e-10)
    # Abramowitz & Stegun approximation 26.2.23
    if p < 0.5:
        t = np.sqrt(-2.0 * np.log(p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        return -(t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t))
    else:
        t = np.sqrt(-2.0 * np.log(1 - p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        return t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)


def _norm_cdf(x):
    """Standard normal CDF."""
    return 0.5 * (1 + _erf(x / np.sqrt(2)))


def _erf(x):
    """Error function approximation (Abramowitz & Stegun 7.1.26)."""
    sign = np.sign(x)
    x = abs(x)
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y


def tost_bootstrap_pvalue(boot_deltas: np.ndarray, delta: float) -> float:
    """
    TOST non-inferiority p-value via bootstrap.
    H0: true_delta <= -delta (SparseComm is >delta worse)
    H1: true_delta > -delta  (non-inferior)

    p-value = proportion of bootstrap samples where mean_delta <= -delta
    """
    return float(np.mean(boot_deltas <= -delta))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def format_ci(lower, upper, point, n_paired):
    """Format a CI for display."""
    return f"{point * 100:+.2f}% [{lower * 100:+.2f}%, {upper * 100:+.2f}%] (n={n_paired})"


def make_plots(results: dict, output_dir: str):
    """Generate bootstrap distribution histograms."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not available, skipping plots")
        return

    os.makedirs(output_dir, exist_ok=True)

    for dataset, comparisons in results.items():
        fig, axes = plt.subplots(1, len(comparisons), figsize=(6 * len(comparisons), 5))
        if len(comparisons) == 1:
            axes = [axes]

        for ax, (comp_name, comp_data) in zip(axes, comparisons.items()):
            boot_dist = comp_data["boot_dist"]
            point = comp_data["point"]
            lower = comp_data["lower"]
            upper = comp_data["upper"]

            ax.hist(boot_dist * 100, bins=80, density=True, alpha=0.7, color="#4A90D9", edgecolor="white", linewidth=0.3)
            ax.axvline(point * 100, color="#E74C3C", linewidth=2, label=f"Point: {point * 100:+.2f}%")
            ax.axvline(lower * 100, color="#2ECC71", linewidth=1.5, linestyle="--", label=f"95% BCa CI")
            ax.axvline(upper * 100, color="#2ECC71", linewidth=1.5, linestyle="--")
            ax.axvline(0, color="gray", linewidth=1, linestyle=":")

            # Non-inferiority margins
            for delta in DELTA_MARGINS:
                ax.axvline(-delta * 100, color="#F39C12", linewidth=1, linestyle=":",
                           alpha=0.6)

            ax.set_xlabel("Accuracy Difference (pp)", fontsize=11)
            ax.set_ylabel("Density", fontsize=11)
            ax.set_title(f"{dataset.upper()}: {comp_name}", fontsize=12, fontweight="bold")
            ax.legend(fontsize=9)

        plt.tight_layout()
        filepath = os.path.join(output_dir, f"bootstrap_{dataset}.png")
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {filepath}")


def generate_markdown_table(all_results: dict) -> str:
    """Generate a markdown summary table."""
    lines = [
        "# Paired Question-Clustered Bootstrap Analysis",
        "",
        "## Accuracy Difference CIs (95% BCa, B=10,000)",
        "",
        "| Dataset | Comparison | n_paired | Point Δ | 95% BCa CI | Interpretation |",
        "|---------|-----------|----------|---------|------------|---------------|",
    ]

    for dataset, comparisons in all_results.items():
        for comp_name, cd in comparisons.items():
            interp = "✓ Non-inferior" if cd["lower"] > -0.05 else "Inconclusive"
            lines.append(
                f"| {dataset.upper()} | {comp_name} | {cd['n_paired']} | "
                f"{cd['point'] * 100:+.2f}% | [{cd['lower'] * 100:+.2f}%, {cd['upper'] * 100:+.2f}%] | "
                f"{interp} |"
            )

    lines.extend([
        "",
        "## TOST Non-Inferiority p-values",
        "",
        "| Dataset | Comparison | δ=3pp | δ=5pp | δ=10pp |",
        "|---------|-----------|-------|-------|--------|",
    ])

    for dataset, comparisons in all_results.items():
        for comp_name, cd in comparisons.items():
            tost = cd.get("tost", {})
            lines.append(
                f"| {dataset.upper()} | {comp_name} | "
                f"p={tost.get(0.03, float('nan')):.4f} | "
                f"p={tost.get(0.05, float('nan')):.4f} | "
                f"p={tost.get(0.10, float('nan')):.4f} |"
            )

    lines.extend([
        "",
        "## Method-Level Accuracy (from matched questions)",
        "",
        "| Dataset | Method | Accuracy | n |",
        "|---------|--------|----------|---|",
    ])

    for dataset, comparisons in all_results.items():
        for comp_name, cd in comparisons.items():
            # Only print method accuaries once per dataset
            if "acc_a" in cd:
                lines.append(f"| {dataset.upper()} | {cd['method_a']} | {cd['acc_a'] * 100:.2f}% | {cd['n_paired']} |")
                lines.append(f"| {dataset.upper()} | {cd['method_b']} | {cd['acc_b'] * 100:.2f}% | {cd['n_paired']} |")
                break  # Only first comparison per dataset for method accuracies

    return "\n".join(lines)


def generate_latex_table(all_results: dict) -> str:
    """Generate LaTeX table for the paper."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Paired question-clustered bootstrap analysis (B=10{,}000). "
        r"$\Delta$ = SparseComm $-$ MAD accuracy difference. "
        r"TOST tests H$_0$: $\Delta \leq -\delta$.}",
        r"\label{tab:bootstrap}",
        r"\small",
        r"\begin{tabular}{llrrcc}",
        r"\toprule",
        r"Dataset & Comparison & $n$ & Point $\Delta$ & 95\% BCa CI & $p_{\text{TOST}}$ ($\delta\!=\!5$) \\",
        r"\midrule",
    ]

    for dataset, comparisons in all_results.items():
        first = True
        for comp_name, cd in comparisons.items():
            ds_label = dataset.upper() if first else ""
            tost_p = cd.get("tost", {}).get(0.05, float("nan"))
            p_str = f"{tost_p:.4f}" if not np.isnan(tost_p) else "---"
            lines.append(
                f"  {ds_label} & {comp_name} & {cd['n_paired']} & "
                f"{cd['point'] * 100:+.2f}\\% & "
                f"[{cd['lower'] * 100:+.2f}\\%, {cd['upper'] * 100:+.2f}\\%] & "
                f"{p_str} \\\\"
            )
            first = False

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def verify_table3_accuracies(methods: Dict[str, Dict[str, dict]], dataset: str):
    """Cross-check loaded data against known Table 3 numbers from the audit report."""
    print(f"\n  [VERIFY] Table 3 cross-check for {dataset.upper()}:")

    expected = {
        "gsm8k": {"MAD": [96.00, 96.33, 96.33], "SparseComm": [94.67, 94.33, 96.00], "NoComm": [86.33, 87.00, 87.67]},
    }

    for method_name, idx in methods.items():
        n_total = len(idx)
        n_correct = sum(1 for r in idx.values() if r.get("is_correct", False))
        acc = n_correct / max(n_total, 1)
        print(f"    {method_name}: {n_correct}/{n_total} = {acc * 100:.2f}%")

        if dataset in expected and method_name in expected[dataset]:
            exp_mean = np.mean(expected[dataset][method_name])
            if abs(acc * 100 - exp_mean) < 2.0:
                print(f"      ✓ Consistent with Table 3 mean ({exp_mean:.2f}%)")
            else:
                print(f"      ⚠ Differs from Table 3 mean ({exp_mean:.2f}%) — check seed/row selection")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_analysis(datasets=None, do_plots=True, do_verify=False):
    """Run the full bootstrap analysis."""
    if datasets is None:
        datasets = ["gsm8k", "aqua", "hotpotqa"]

    rng = np.random.default_rng(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_results = {}

    for dataset in datasets:
        print(f"\n{'=' * 70}")
        print(f"  Dataset: {dataset.upper()}")
        print(f"{'=' * 70}")

        methods = load_method_data(dataset)

        for m_name, idx in methods.items():
            n_total = len(idx)
            n_correct = sum(1 for r in idx.values() if r.get("is_correct", False))
            print(f"  {m_name}: {n_total} records, {n_correct} correct ({n_correct/max(n_total,1)*100:.1f}%)")

        if do_verify and dataset == "gsm8k":
            verify_table3_accuracies(methods, dataset)

        comparisons = {}

        # SparseComm vs MAD
        if "SparseComm" in methods and "MAD" in methods:
            deltas, n_paired, n_only_a, n_only_b = get_paired_deltas(methods, "SparseComm", "MAD")
            print(f"\n  SparseComm vs MAD: {n_paired} paired questions "
                  f"({n_only_a} SparseComm-only, {n_only_b} MAD-only)")

            if n_paired > 0:
                lower, upper, point, boot_dist = bca_ci(deltas, np.mean, N_BOOTSTRAP, rng=rng)
                tost_pvals = {}
                for delta in DELTA_MARGINS:
                    tost_pvals[delta] = tost_bootstrap_pvalue(boot_dist, delta)

                # Method accuracies on matched questions
                sparse_idx = methods["SparseComm"]
                mad_idx = methods["MAD"]
                common = set(sparse_idx.keys()) & set(mad_idx.keys())
                acc_sparse = np.mean([1.0 if sparse_idx[q].get("is_correct") else 0.0 for q in common])
                acc_mad = np.mean([1.0 if mad_idx[q].get("is_correct") else 0.0 for q in common])

                comparisons["SparseComm − MAD"] = {
                    "point": point, "lower": lower, "upper": upper,
                    "boot_dist": boot_dist, "n_paired": n_paired,
                    "tost": tost_pvals,
                    "method_a": "SparseComm", "method_b": "MAD",
                    "acc_a": acc_sparse, "acc_b": acc_mad,
                }
                print(f"    Point Δ: {point*100:+.2f}%, 95% BCa CI: [{lower*100:+.2f}%, {upper*100:+.2f}%]")
                for delta, pval in tost_pvals.items():
                    status = "REJECT H0 → non-inferior" if pval < 0.05 else "FAIL to reject"
                    print(f"    TOST (δ={delta*100:.0f}pp): p={pval:.4f} → {status}")

        # SparseComm vs NoComm
        if "SparseComm" in methods and "NoComm" in methods:
            deltas, n_paired, n_only_a, n_only_b = get_paired_deltas(methods, "SparseComm", "NoComm")
            print(f"\n  SparseComm vs NoComm: {n_paired} paired questions")

            if n_paired > 0:
                lower, upper, point, boot_dist = bca_ci(deltas, np.mean, N_BOOTSTRAP, rng=rng)
                tost_pvals = {}
                for delta in DELTA_MARGINS:
                    tost_pvals[delta] = tost_bootstrap_pvalue(boot_dist, delta)

                comparisons["SparseComm − NoComm"] = {
                    "point": point, "lower": lower, "upper": upper,
                    "boot_dist": boot_dist, "n_paired": n_paired,
                    "tost": tost_pvals,
                }
                print(f"    Point Δ: {point*100:+.2f}%, 95% BCa CI: [{lower*100:+.2f}%, {upper*100:+.2f}%]")
                for delta, pval in tost_pvals.items():
                    status = "REJECT H0 → non-inferior" if pval < 0.05 else "FAIL to reject"
                    print(f"    TOST (δ={delta*100:.0f}pp): p={pval:.4f} → {status}")

        all_results[dataset] = comparisons

    # Generate outputs
    print(f"\n{'=' * 70}")
    print("  Generating outputs")
    print(f"{'=' * 70}")

    md_table = generate_markdown_table(all_results)
    md_path = os.path.join(OUTPUT_DIR, "bootstrap_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_table)
    print(f"  Saved markdown: {md_path}")

    latex_table = generate_latex_table(all_results)
    latex_path = os.path.join(OUTPUT_DIR, "bootstrap_results.tex")
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_table)
    print(f"  Saved LaTeX: {latex_path}")

    if do_plots:
        # Strip boot_dist for plots (keep a copy)
        make_plots(all_results, OUTPUT_DIR)

    # Print final summary table
    print(f"\n{md_table}")

    return all_results


def main():
    do_verify = "--verify" in sys.argv
    do_plots = "--no-plots" not in sys.argv

    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " Experiment 2: Paired Question-Clustered Bootstrap Analysis ".center(78) + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    run_analysis(do_plots=do_plots, do_verify=do_verify)


if __name__ == "__main__":
    main()
