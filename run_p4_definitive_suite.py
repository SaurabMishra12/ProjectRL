#!/usr/bin/env python3
"""
Definitive Protocol P4 Suite Runner
===================================
Executes the comprehensive experimental redesign of Protocol P4:
  1. P4-B: Exhaustive 15-Support Evaluation (All C(6,2)=15 directed 2-edge supports)
  2. P4-C: Exact Bandwidth Allocation (25/25 at V=50, 50/50 at V=100)
  3. P4-A: Multi-Seed Learned Controller Robustness (5 controller checkpoints)
  4. P4-D: Content-Selection Control (Semantic vs. Random Word Retention)

Usage:
  .venv_linux/bin/python run_p4_definitive_suite.py --mock --samples 5
  .venv_linux/bin/python run_p4_definitive_suite.py --mode exhaustive_supports --samples 150
  .venv_linux/bin/python run_p4_definitive_suite.py --mode multi_seed --samples 150
  .venv_linux/bin/python run_p4_definitive_suite.py --mode content_retention --samples 150
  .venv_linux/bin/python run_p4_definitive_suite.py --mode all --samples 150
"""

import os
import sys
import json
import math
import time
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from tqdm import tqdm

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from config import get_gsm8k_config, Config
from utils import set_seed, count_tokens, NumpyEncoder
from dataset_loaders import load_dataset_by_name
from evaluation import ExperimentResults, SampleResult
from agents import AgentPool
from controller import IBController, FrozenWeightController
from llm_backend import AzureOpenAIBackend, MockLLMBackend
from pipeline import run_single_sample

from run_honest_fixed_budget import make_json_safe, wilson_ci
from p4_definitive_framework import (
    ALL_15_SUPPORTS,
    edges_to_adjacency,
    ExactBandwidthCommunicationGraph,
    CONTROLLER_CHECKPOINTS,
)

# Output directories
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "p4_definitive_suite")
STREAM_DIR = os.path.join(OUTPUT_DIR, "stream")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STREAM_DIR, exist_ok=True)


# =====================================================================
# CELL EXECUTION ENGINE
# =====================================================================

def execute_cell(
    cell_name: str,
    controller,
    comm_graph: ExactBandwidthCommunicationGraph,
    test_data: List[dict],
    config: Config,
    agent_pool: AgentPool,
    v_total: int,
) -> List[SampleResult]:
    """Execute a single cell with streaming and resume support."""
    stream_file = os.path.join(STREAM_DIR, f"{cell_name}_stream.jsonl")
    set_seed(config.seed)

    start_idx = 0
    completed_results = []
    if os.path.exists(stream_file):
        with open(stream_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        start_idx = len(lines)
        if start_idx >= len(test_data):
            print(f"  [DONE] {cell_name}: already completed {start_idx} samples")
            for line in lines[:len(test_data)]:
                s = json.loads(line)
                sr = SampleResult(
                    question=s.get("question", ""),
                    gold_answer=s.get("gold_answer", ""),
                    final_answer=s.get("final_answer", ""),
                    is_correct=s.get("is_correct", False),
                    total_tokens_transmitted=s.get("total_tokens_transmitted", 0),
                    total_tokens_if_full=s.get("total_tokens_if_full", 0),
                    compression_ratio=s.get("compression_ratio", 0.0),
                    communication_density=s.get("communication_density", 0.0),
                    n_active_edges=s.get("n_active_edges", 0),
                    generation_tokens=s.get("generation_tokens", 0),
                    halt_round=s.get("halt_round", 3),
                )
                completed_results.append(sr)
            return completed_results
        elif start_idx > 0:
            print(f"  [RESUME] {cell_name}: resuming from sample {start_idx}/{len(test_data)}")
            for line in lines:
                s = json.loads(line)
                sr = SampleResult(
                    question=s.get("question", ""),
                    gold_answer=s.get("gold_answer", ""),
                    final_answer=s.get("final_answer", ""),
                    is_correct=s.get("is_correct", False),
                    total_tokens_transmitted=s.get("total_tokens_transmitted", 0),
                    total_tokens_if_full=s.get("total_tokens_if_full", 0),
                    compression_ratio=s.get("compression_ratio", 0.0),
                    communication_density=s.get("communication_density", 0.0),
                    n_active_edges=s.get("n_active_edges", 0),
                    generation_tokens=s.get("generation_tokens", 0),
                    halt_round=s.get("halt_round", 3),
                )
                completed_results.append(sr)

    sample_results = list(completed_results)

    for i, sample in enumerate(tqdm(test_data, desc=f"  {cell_name}")):
        if i < start_idx:
            continue

        agent_pool.reset_all()
        comm_graph.reset()

        result = run_single_sample(
            question=sample["question"],
            gold_answer=sample["answer"],
            agent_pool=agent_pool,
            comm_graph=comm_graph,
            controller=controller,
            config=config,
            train_controller=False,
            avg_accuracy=0.5,
        )

        sample_results.append(result)

        try:
            with open(stream_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(make_json_safe(asdict(result))) + "\n")
        except Exception as e:
            print(f"  Stream write error: {e}")

    return sample_results


# =====================================================================
# EXPERIMENT 1 & 3: EXHAUSTIVE 15 SUPPORTS WITH EXACT PER-EDGE BANDWIDTH
# =====================================================================

def run_exhaustive_15_supports(
    config: Config,
    test_data: List[dict],
    agent_pool: AgentPool,
    budgets: List[int] = [50, 100],
    support_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("  EXPERIMENT: EXHAUSTIVE 15-SUPPORT EVALUATION (PER-EDGE BANDWIDTH MATCHED)")
    print("=" * 80)

    if support_keys is None:
        support_keys = list(ALL_15_SUPPORTS.keys())

    results_by_cell = {}
    support_summary = []

    for s_key in support_keys:
        s_info = ALL_15_SUPPORTS[s_key]
        edges = s_info["edges"]
        W = edges_to_adjacency(edges, n_agents=config.n_agents)
        Q = np.zeros((config.n_agents, 16))
        ctrl = FrozenWeightController(config, frozen_W=W, frozen_Q=Q)

        for v_total in budgets:
            cell_name = f"{s_key}_V{v_total}"
            b_edge = v_total // len(edges)
            print(f"\nEvaluating {s_info['name']} at V={v_total} (Ceiling: {b_edge}/edge)...")

            comm_graph = ExactBandwidthCommunicationGraph(
                n_agents=config.n_agents,
                v_total_budget=v_total,
                content_mode="semantic",
                equal_per_edge=True,
                seed=config.seed,
            )
            comm_graph.set_dataset_type(config.dataset)
            comm_graph.set_weights(W)

            results = execute_cell(
                cell_name=cell_name,
                controller=ctrl,
                comm_graph=comm_graph,
                test_data=test_data,
                config=config,
                agent_pool=agent_pool,
                v_total=v_total,
            )
            results_by_cell[cell_name] = results

            n = len(results)
            correct = sum(1 for r in results if r.is_correct)
            acc_pt, acc_lo, acc_hi = wilson_ci(correct, n)
            vtxs = np.array([r.total_tokens_transmitted for r in results])
            mean_vtx = float(np.mean(vtxs))

            print(f"  → {s_key} @ V{v_total}: Acc = {acc_pt:.2f}% [{acc_lo:.1f}, {acc_hi:.1f}], Mean V_tx = {mean_vtx:.1f}")

            support_summary.append({
                "support_key": s_key,
                "support_id": s_info["id"],
                "name": s_info["name"],
                "iso_class": s_info["iso_class"],
                "edges": [list(e) for e in edges],
                "budget": v_total,
                "per_edge_budget": b_edge,
                "n_samples": n,
                "n_correct": correct,
                "accuracy": acc_pt,
                "ci_lo": acc_lo,
                "ci_hi": acc_hi,
                "mean_vtx": mean_vtx,
            })

    return {
        "support_summary": support_summary,
        "results_by_cell": results_by_cell,
    }


# =====================================================================
# EXPERIMENT 2: MULTI-SEED CONTROLLER ROBUSTNESS
# =====================================================================

def run_multi_seed_evaluation(
    config: Config,
    test_data: List[dict],
    agent_pool: AgentPool,
    budgets: List[int] = [50, 100],
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("  EXPERIMENT: MULTI-SEED LEARNED POLICY EVALUATION")
    print("=" * 80)

    seed_results = []
    results_by_cell = {}

    for ckpt_name, ckpt_path in CONTROLLER_CHECKPOINTS.items():
        if not os.path.exists(ckpt_path):
            print(f"  [WARN] Checkpoint not found: {ckpt_path}, skipping.")
            continue

        ctrl = IBController(config)
        ctrl.load_model(ckpt_path)

        for v_total in budgets:
            cell_name = f"{ckpt_name}_V{v_total}"
            print(f"\nEvaluating {ckpt_name} at V={v_total}...")

            comm_graph = ExactBandwidthCommunicationGraph(
                n_agents=config.n_agents,
                v_total_budget=v_total,
                content_mode="semantic",
                equal_per_edge=True,
                seed=config.seed,
            )
            comm_graph.set_dataset_type(config.dataset)

            results = execute_cell(
                cell_name=cell_name,
                controller=ctrl,
                comm_graph=comm_graph,
                test_data=test_data,
                config=config,
                agent_pool=agent_pool,
                v_total=v_total,
            )
            results_by_cell[cell_name] = results

            n = len(results)
            correct = sum(1 for r in results if r.is_correct)
            acc_pt, acc_lo, acc_hi = wilson_ci(correct, n)
            vtxs = np.array([r.total_tokens_transmitted for r in results])

            print(f"  → {ckpt_name} @ V{v_total}: Acc = {acc_pt:.2f}%, Mean V_tx = {np.mean(vtxs):.1f}")

            seed_results.append({
                "checkpoint": ckpt_name,
                "budget": v_total,
                "n_samples": n,
                "n_correct": correct,
                "accuracy": acc_pt,
                "ci_lo": acc_lo,
                "ci_hi": acc_hi,
                "mean_vtx": float(np.mean(vtxs)),
            })

    return {
        "seed_results": seed_results,
        "results_by_cell": results_by_cell,
    }


# =====================================================================
# EXPERIMENT 4: CONTENT-SELECTION CONTROL (SEMANTIC VS RANDOM)
# =====================================================================

def run_content_retention_control(
    config: Config,
    test_data: List[dict],
    agent_pool: AgentPool,
    budgets: List[int] = [50, 100],
) -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("  EXPERIMENT: CONTENT RETENTION CONTROL (SEMANTIC VS RANDOM RETENTION)")
    print("=" * 80)

    # Compare S01 (Learned Out-Star 0) and S15 (Out-Star 2) under semantic vs random
    test_supports = ["S01_OutStar_0", "S15_OutStar_2"]
    modes = ["semantic", "random"]
    retention_results = []
    results_by_cell = {}

    for s_key in test_supports:
        s_info = ALL_15_SUPPORTS[s_key]
        edges = s_info["edges"]
        W = edges_to_adjacency(edges, n_agents=config.n_agents)
        Q = np.zeros((config.n_agents, 16))
        ctrl = FrozenWeightController(config, frozen_W=W, frozen_Q=Q)

        for mode in modes:
            for v_total in budgets:
                cell_name = f"Content_{s_key}_{mode}_V{v_total}"
                print(f"\nEvaluating {s_key} with mode='{mode}' at V={v_total}...")

                comm_graph = ExactBandwidthCommunicationGraph(
                    n_agents=config.n_agents,
                    v_total_budget=v_total,
                    content_mode=mode,
                    equal_per_edge=True,
                    seed=config.seed,
                )
                comm_graph.set_dataset_type(config.dataset)
                comm_graph.set_weights(W)

                results = execute_cell(
                    cell_name=cell_name,
                    controller=ctrl,
                    comm_graph=comm_graph,
                    test_data=test_data,
                    config=config,
                    agent_pool=agent_pool,
                    v_total=v_total,
                )
                results_by_cell[cell_name] = results

                n = len(results)
                correct = sum(1 for r in results if r.is_correct)
                acc_pt, acc_lo, acc_hi = wilson_ci(correct, n)
                vtxs = np.array([r.total_tokens_transmitted for r in results])

                print(f"  → {cell_name}: Acc = {acc_pt:.2f}%, Mean V_tx = {np.mean(vtxs):.1f}")

                retention_results.append({
                    "support_key": s_key,
                    "mode": mode,
                    "budget": v_total,
                    "accuracy": acc_pt,
                    "ci_lo": acc_lo,
                    "ci_hi": acc_hi,
                    "mean_vtx": float(np.mean(vtxs)),
                })

    return {
        "retention_results": retention_results,
        "results_by_cell": results_by_cell,
    }


# =====================================================================
# SYNTHESIS & REPORT GENERATION
# =====================================================================

def synthesize_and_generate_report(
    exhaustive_data: Optional[Dict[str, Any]],
    multi_seed_data: Optional[Dict[str, Any]],
    retention_data: Optional[Dict[str, Any]],
    budgets: List[int] = [50, 100],
) -> str:
    md_lines = [
        "# Definitive Protocol P4 Evaluation Suite Report",
        "",
        "**System / Protocol**: Protocol P4 Redesign (Exact Bandwidth & Support Space Enumeration)",
        f"**Date / Timestamp**: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "",
        "---",
        "",
    ]

    # Panel 1: Exhaustive 15 Supports
    if exhaustive_data and "support_summary" in exhaustive_data:
        md_lines.extend([
            "## Panel A: Exhaustive 15-Support Enumeration (Exact Per-Edge Bandwidth)",
            "",
            "Every 2-edge directed support on N=3 agents is evaluated under exact equal-bandwidth allocation:",
            "- At V=50: exactly 25 tokens/edge ceiling (25/25)",
            "- At V=100: exactly 50 tokens/edge ceiling (50/50)",
            "",
            "| ID | Support Key | Graph Isomorphism Class | Edges | V=50 Acc [%] | V=100 Acc [%] |",
            "|---:|---|---|---|---:|---:|",
        ])

        summary_list = exhaustive_data["support_summary"]
        all_keys = list(ALL_15_SUPPORTS.keys())

        v50_accs = []
        v100_accs = []
        learned_v50 = None
        learned_v100 = None

        for k in all_keys:
            s_info = ALL_15_SUPPORTS[k]
            edges_str = " + ".join(f"{i}→{j}" for (i, j) in s_info["edges"])
            row_50 = next((r for r in summary_list if r["support_key"] == k and r["budget"] == 50), None)
            row_100 = next((r for r in summary_list if r["support_key"] == k and r["budget"] == 100), None)

            acc50_str = f"{row_50['accuracy']:.2f}%" if row_50 else "—"
            acc100_str = f"{row_100['accuracy']:.2f}%" if row_100 else "—"

            if row_50:
                v50_accs.append(row_50["accuracy"])
                if k == "S01_OutStar_0":
                    learned_v50 = row_50["accuracy"]
            if row_100:
                v100_accs.append(row_100["accuracy"])
                if k == "S01_OutStar_0":
                    learned_v100 = row_100["accuracy"]

            md_lines.append(
                f"| {s_info['id']} | `{k}` | {s_info['iso_class']} | {edges_str} | {acc50_str} | {acc100_str} |"
            )

        md_lines.append("")
        md_lines.append("### Distribution Statistics across all 15 Supports")
        md_lines.append("")

        for b, accs, l_acc in [(50, v50_accs, learned_v50), (100, v100_accs, learned_v100)]:
            if accs:
                mean_a = np.mean(accs)
                std_a = np.std(accs)
                min_a = np.min(accs)
                max_a = np.max(accs)
                md_lines.append(f"**V = {b} tokens**:")
                md_lines.append(f"- Mean across all supports: **{mean_a:.2f}% ± {std_a:.2f}%** [Min: {min_a:.2f}%, Max: {max_a:.2f}%]")
                if l_acc is not None:
                    rank = sum(1 for a in accs if a <= l_acc)
                    percentile = (rank / len(accs)) * 100.0
                    md_lines.append(f"- Learned Support (`S01_OutStar_0`): **{l_acc:.2f}%**")
                    md_lines.append(f"- Empirical Percentile: **{percentile:.1f}th percentile** ({rank}/{len(accs)} supports)")
                md_lines.append("")

    # Panel 2: Multi-Seed Robustness
    if multi_seed_data and "seed_results" in multi_seed_data:
        md_lines.extend([
            "## Panel B: Learned Policy Multi-Seed / Checkpoint Robustness",
            "",
            "| Checkpoint / Seed | V=50 Acc [%] | V=100 Acc [%] |",
            "|---|---:|---:|",
        ])
        s_list = multi_seed_data["seed_results"]
        ckpts = sorted(list(set(r["checkpoint"] for r in s_list)))
        v50_s = []
        v100_s = []
        for c in ckpts:
            r50 = next((r for r in s_list if r["checkpoint"] == c and r["budget"] == 50), None)
            r100 = next((r for r in s_list if r["checkpoint"] == c and r["budget"] == 100), None)
            a50 = f"{r50['accuracy']:.2f}%" if r50 else "—"
            a100 = f"{r100['accuracy']:.2f}%" if r100 else "—"
            if r50: v50_s.append(r50["accuracy"])
            if r100: v100_s.append(r100["accuracy"])
            md_lines.append(f"| `{c}` | {a50} | {a100} |")

        if v50_s and v100_s:
            md_lines.append(f"| **Mean ± SD** | **{np.mean(v50_s):.2f}% ± {np.std(v50_s):.2f}%** | **{np.mean(v100_s):.2f}% ± {np.std(v100_s):.2f}%** |")
        md_lines.append("")

    # Panel 3: Content-Selection Control
    if retention_data and "retention_results" in retention_data:
        md_lines.extend([
            "## Panel C: Content Selection Control (Semantic vs. Random Retention)",
            "",
            "Addresses whether the communication effect stems from topology vs. truncation artifacts:",
            "",
            "| Support | Retention Mode | V=50 Acc [%] | V=100 Acc [%] | Delta (Semantic - Random) |",
            "|---|---|---:|---:|---:|",
        ])
        r_list = retention_data["retention_results"]
        for s_key in ["S01_OutStar_0", "S15_OutStar_2"]:
            s_name = ALL_15_SUPPORTS[s_key]["name"]
            sem_50 = next((r["accuracy"] for r in r_list if r["support_key"] == s_key and r["mode"] == "semantic" and r["budget"] == 50), None)
            sem_100 = next((r["accuracy"] for r in r_list if r["support_key"] == s_key and r["mode"] == "semantic" and r["budget"] == 100), None)
            ran_50 = next((r["accuracy"] for r in r_list if r["support_key"] == s_key and r["mode"] == "random" and r["budget"] == 50), None)
            ran_100 = next((r["accuracy"] for r in r_list if r["support_key"] == s_key and r["mode"] == "random" and r["budget"] == 100), None)

            sem_50_str = f"{sem_50:.2f}%" if sem_50 is not None else "—"
            sem_100_str = f"{sem_100:.2f}%" if sem_100 is not None else "—"
            ran_50_str = f"{ran_50:.2f}%" if ran_50 is not None else "—"
            ran_100_str = f"{ran_100:.2f}%" if ran_100 is not None else "—"

            delta_parts = []
            if sem_50 is not None and ran_50 is not None:
                delta_parts.append(f"+{sem_50 - ran_50:.2f} pts (V50)")
            if sem_100 is not None and ran_100 is not None:
                delta_parts.append(f"+{sem_100 - ran_100:.2f} pts (V100)")
            delta_str = ", ".join(delta_parts) if delta_parts else "—"

            md_lines.append(f"| {s_name} | Semantic | {sem_50_str} | {sem_100_str} | — |")
            md_lines.append(f"| {s_name} | Random Word Retention | {ran_50_str} | {ran_100_str} | {delta_str} |")

        md_lines.append("")

    report_text = "\n".join(md_lines)
    report_path = os.path.join(OUTPUT_DIR, "p4_definitive_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n[REPORT SAVED] {report_path}")
    return report_text


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Definitive Protocol P4 Suite Runner")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "exhaustive_supports", "multi_seed", "content_retention", "report_only"],
                        help="Which experiment panel to run")
    parser.add_argument("--samples", type=int, default=150, help="Number of questions (default 150)")
    parser.add_argument("--budgets", type=str, default="50,100", help="Comma-separated budgets")
    parser.add_argument("--mock", action="store_true", help="Use Mock LLM backend for offline verification")
    parser.add_argument("--supports", type=str, default="all",
                        help="Comma-separated support keys (or 'all', or 'unrun')")
    args = parser.parse_args()

    budgets = [int(b.strip()) for b in args.budgets.split(",")]

    config = get_gsm8k_config()
    config.n_samples = args.samples
    config.seed = 42

    if args.mock:
        config.llm_backend = "mock"
        llm_backend = MockLLMBackend()
    else:
        config.llm_backend = "azure"
        config.api_model = "gpt-oss-120b"
        llm_backend = AzureOpenAIBackend(model=config.api_model)

    agent_pool = AgentPool(config.n_agents, llm_backend, config.reasoning_styles)

    print("=" * 80)
    print("  DEFINITIVE PROTOCOL P4 EXPERIMENTAL SUITE")
    print(f"  Mode: {args.mode}, Samples: {args.samples}, Backend: {config.llm_backend}")
    print(f"  Budgets: {budgets}")
    print("=" * 80)

    # Load dataset
    _, test_data = load_dataset_by_name("gsm8k", n_train=0, n_test=args.samples, seed=config.seed)
    print(f"Loaded {len(test_data)} GSM8K test questions (seed={config.seed})")

    exhaustive_data = None
    multi_seed_data = None
    retention_data = None

    # Determine support keys
    if args.supports == "all":
        target_supports = list(ALL_15_SUPPORTS.keys())
    elif args.supports == "unrun":
        # Unrun supports: 3, 4, 5, 6, 7, 10, 11, 12, 13
        target_supports = [k for k in ALL_15_SUPPORTS.keys() if k not in [
            "S01_OutStar_0", "S02_Dyad_01", "S08_Dyad_02", "S09_Relay_021", "S14_Dyad_12", "S15_OutStar_2"
        ]]
    else:
        target_supports = [s.strip() for s in args.supports.split(",")]

    if args.mode in ["all", "exhaustive_supports", "report_only"]:
        exhaustive_data = run_exhaustive_15_supports(
            config=config,
            test_data=test_data,
            agent_pool=agent_pool,
            budgets=budgets,
            support_keys=target_supports,
        )

    if args.mode in ["all", "multi_seed", "report_only"]:
        multi_seed_data = run_multi_seed_evaluation(
            config=config,
            test_data=test_data,
            agent_pool=agent_pool,
            budgets=budgets,
        )

    if args.mode in ["all", "content_retention"]:
        retention_data = run_content_retention_control(
            config=config,
            test_data=test_data,
            agent_pool=agent_pool,
            budgets=budgets,
        )

    # Synthesize master report
    synthesize_and_generate_report(
        exhaustive_data=exhaustive_data,
        multi_seed_data=multi_seed_data,
        retention_data=retention_data,
        budgets=budgets,
    )

    print("\n" + "=" * 80)
    print("  SUITE EXECUTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
