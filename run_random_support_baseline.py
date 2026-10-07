#!/usr/bin/env python3
"""
Random Support Baseline: Concentration vs Source Selection
==========================================================
Picks K=5 random 2-edge adjacency matrices (|T|=2), runs each on the
same 150 GSM8K questions at V̄ ∈ {50, 100}, and compares accuracy to
the learned support and Ring.

HYPOTHESIS TEST:
  If Random-2 ≈ Learned → concentration (any 2 edges suffice)
  If Random-2 ≈ Ring    → source selection (the learned pair matters)

Uses the exact same infrastructure as run_honest_fixed_budget.py:
  - TotalSampleBudgetCommunicationGraph (budget-capped transmission)
  - FrozenWeightController (fixed adjacency, no MLP)
  - Same 150 questions (seed=42), same Azure backend

Usage:
    .venv_linux/bin/python run_random_support_baseline.py
"""

import os
import sys
import json
import math
import time
import itertools
import dataclasses
from dataclasses import asdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
from tqdm import tqdm

# Add current directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from config import get_gsm8k_config, Config
from utils import set_seed, count_tokens, split_sentences, NumpyEncoder
from dataset_loaders import load_dataset_by_name
from evaluation import ExperimentResults, SampleResult
from agents import AgentPool
from controller import IBController, FrozenWeightController
from communication import (
    CommunicationGraph,
    SemanticBandwidthMode,
    TransmissionRecord,
)
from pipeline import run_single_sample, create_llm_backend
from llm_backend import AzureOpenAIBackend, LLMResponse

# Reuse TotalSampleBudgetCommunicationGraph from the honest experiment
from run_honest_fixed_budget import (
    TotalSampleBudgetCommunicationGraph,
    truncate_to_token_budget,
    make_json_safe,
    wilson_ci,
)


# =====================================================================
# OUTPUT DIRECTORIES
# =====================================================================
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "random_support_baseline")
STREAM_DIR = os.path.join(OUTPUT_DIR, "stream")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STREAM_DIR, exist_ok=True)


# =====================================================================
# RANDOM 2-EDGE ADJACENCY GENERATION
# =====================================================================
def enumerate_all_2edge_subsets(n_agents: int = 3) -> List[List[Tuple[int, int]]]:
    """Enumerate all C(n*(n-1), 2) = C(6,2) = 15 pairs of directed edges."""
    all_edges = [(i, j) for i in range(n_agents) for j in range(n_agents) if i != j]
    return [list(pair) for pair in itertools.combinations(all_edges, 2)]


def edges_to_adjacency(edges: List[Tuple[int, int]], n_agents: int = 3) -> np.ndarray:
    """Convert a list of (sender, receiver) edges to a binarized adjacency matrix."""
    W = np.zeros((n_agents, n_agents), dtype=np.float64)
    for (i, j) in edges:
        W[i][j] = 1.0
    return W


def sample_random_draws(
    n_agents: int = 3,
    k_draws: int = 5,
    exclude_edges: Optional[List[Tuple[int, int]]] = None,
    seed: int = 99,
) -> List[List[Tuple[int, int]]]:
    """Sample K random 2-edge subsets, excluding the learned pair."""
    all_subsets = enumerate_all_2edge_subsets(n_agents)

    # Exclude the learned pair
    if exclude_edges is not None:
        exclude_set = frozenset(tuple(e) for e in exclude_edges)
        all_subsets = [s for s in all_subsets if frozenset(tuple(e) for e in s) != exclude_set]

    rng = np.random.RandomState(seed)
    indices = rng.choice(len(all_subsets), size=min(k_draws, len(all_subsets)), replace=False)
    return [all_subsets[i] for i in indices]


# =====================================================================
# SINGLE CELL RUNNER (adapted from run_honest_fixed_budget.run_cell)
# =====================================================================
def run_cell(
    config: Config,
    draw_name: str,
    v_total: int,
    test_data: List[dict],
    controller: FrozenWeightController,
    cell_prefix: str = "",
) -> List[SampleResult]:
    """Run a single fixed-budget cell with a frozen adjacency. Returns list of SampleResult."""
    cell_name = f"{cell_prefix}{draw_name}_V{v_total}"
    set_seed(config.seed)

    llm_backend = AzureOpenAIBackend(model=config.api_model)
    agent_pool = AgentPool(config.n_agents, llm_backend, config.reasoning_styles)
    comm_graph = TotalSampleBudgetCommunicationGraph(
        n_agents=config.n_agents, v_total_budget=v_total
    )
    comm_graph.set_dataset_type(config.dataset)

    stream_file = os.path.join(STREAM_DIR, f"{cell_name}_stream.jsonl")

    # Resume from checkpoint
    start_idx = 0
    completed_results = []
    if os.path.exists(stream_file):
        with open(stream_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        start_idx = len(lines)
        if start_idx > 0 and start_idx < len(test_data):
            print(f"  [RESUME] {cell_name}: resuming from sample {start_idx}/{len(test_data)}")
            for line in lines:
                s = json.loads(line)
                sr = SampleResult(
                    question=s.get("question", ""),
                    gold_answer=s.get("gold_answer", ""),
                    final_answer=s.get("final_answer", ""),
                    is_correct=s.get("is_correct", False),
                    exact_match=s.get("exact_match", 0.0),
                    f1_score=s.get("f1_score", 0.0),
                    total_tokens_transmitted=s.get("total_tokens_transmitted", 0),
                    total_tokens_if_full=s.get("total_tokens_if_full", 0),
                    compression_ratio=s.get("compression_ratio", 0.0),
                    communication_density=s.get("communication_density", 0.0),
                    n_active_edges=s.get("n_active_edges", 0),
                    generation_tokens=s.get("generation_tokens", 0),
                    halt_round=s.get("halt_round", 3),
                )
                completed_results.append(sr)
        elif start_idx >= len(test_data):
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

        # HARD INVARIANT: V_tx must never exceed budget
        assert result.total_tokens_transmitted <= v_total + 2, (
            f"INVARIANT VIOLATION: V_tx={result.total_tokens_transmitted} > V̄={v_total} "
            f"on question {i}"
        )

        sample_results.append(result)

        try:
            with open(stream_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(make_json_safe(asdict(result))) + "\n")
        except Exception as e:
            print(f"  Stream write error: {e}")

    return sample_results


# =====================================================================
# ANALYSIS
# =====================================================================
def analyze_and_report(
    all_results: Dict[str, List[SampleResult]],
    draws: List[List[Tuple[int, int]]],
    budgets: List[int],
    n_questions: int,
    learned_accs: Dict[int, float],
    ring_accs: Dict[int, float],
    nocomm_acc: float,
):
    """Compute and save analysis comparing random support to learned and ring."""

    print("\n" + "=" * 90)
    print(f"  RANDOM SUPPORT BASELINE RESULTS (K={len(draws)} draws, n={n_questions})")
    print("=" * 90)

    summary = {"draws": [], "comparison": {}}

    for v_total in budgets:
        print(f"\n--- V̄ = {v_total} ---")
        draw_accs = []

        for k, edges in enumerate(draws):
            draw_name = f"RandomDraw{k+1}"
            cell_key = f"{draw_name}_V{v_total}"
            results = all_results[cell_key]
            n = len(results)
            correct = sum(1 for r in results if r.is_correct)
            acc_pt, acc_lo, acc_hi = wilson_ci(correct, n)
            draw_accs.append(acc_pt)

            vtxs = np.array([r.total_tokens_transmitted for r in results])
            comm_mask = vtxs > 0
            n_comm = int(np.sum(comm_mask))
            comm_vtx = float(np.mean(vtxs[comm_mask])) if n_comm > 0 else 0.0
            comm_vol = comm_vtx / v_total if v_total > 0 else 0.0

            edge_str = " + ".join(f"{i}→{j}" for (i, j) in edges)
            print(f"  Draw {k+1} [{edge_str}]: {acc_pt:.2f}% "
                  f"[{acc_lo:.1f}, {acc_hi:.1f}] "
                  f"n_comm={n_comm}, Ṽ/V̄|comm={comm_vol:.3f}")

            summary["draws"].append({
                "draw": k + 1,
                "edges": [list(e) for e in edges],
                "budget": v_total,
                "accuracy": acc_pt,
                "ci_lo": acc_lo,
                "ci_hi": acc_hi,
                "n_correct": correct,
                "n_total": n,
                "n_comm": n_comm,
                "comm_vol_ratio": comm_vol,
            })

        mean_acc = np.mean(draw_accs)
        std_acc = np.std(draw_accs)
        min_acc = np.min(draw_accs)
        max_acc = np.max(draw_accs)

        learned_acc = learned_accs[v_total]
        ring_acc = ring_accs[v_total]

        print(f"\n  Random-2 Mean: {mean_acc:.2f}% ± {std_acc:.2f}% "
              f"[{min_acc:.2f}%, {max_acc:.2f}%]")
        print(f"  Learned Support: {learned_acc:.2f}%")
        print(f"  Ring:            {ring_acc:.2f}%")
        print(f"  NoComm:          {nocomm_acc:.2f}%")

        # Distance metrics
        dist_to_learned = abs(mean_acc - learned_acc)
        dist_to_ring = abs(mean_acc - ring_acc)
        print(f"  |Random-2 − Learned|: {dist_to_learned:.2f} pts")
        print(f"  |Random-2 − Ring|:    {dist_to_ring:.2f} pts")

        if dist_to_learned < dist_to_ring:
            verdict = "CONCENTRATION (closer to Learned)"
        elif dist_to_ring < dist_to_learned:
            verdict = "SOURCE SELECTION (closer to Ring)"
        else:
            verdict = "INCONCLUSIVE (equidistant)"
        print(f"  → {verdict}")

        summary["comparison"][f"V{v_total}"] = {
            "random_mean": mean_acc,
            "random_std": std_acc,
            "random_min": min_acc,
            "random_max": max_acc,
            "learned": learned_acc,
            "ring": ring_acc,
            "nocomm": nocomm_acc,
            "dist_to_learned": dist_to_learned,
            "dist_to_ring": dist_to_ring,
            "verdict": verdict,
        }

    # Save summary JSON
    json_path = os.path.join(OUTPUT_DIR, "random_support_results.json")
    full_json = {}
    for ck, results in all_results.items():
        full_json[ck] = [make_json_safe(asdict(r)) for r in results]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_json, f, indent=2)
    print(f"\nSaved full results: {json_path}")

    summary_json_path = os.path.join(OUTPUT_DIR, "random_support_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON: {summary_json_path}")

    # Generate Markdown summary
    md_lines = [
        "# Random Support Baseline: Concentration vs Source Selection",
        "",
        "**Experiment**: Pick K=5 random pairs of directed edges (|T|=2),",
        "run each on the same 150 GSM8K questions at V̄ ∈ {50, 100}.",
        "",
        "**Question**: Is the learned support's advantage from *concentration*",
        "(any 2 edges suffice) or *source selection* (the specific learned pair matters)?",
        "",
        "## Reference Baselines",
        "",
        f"| Condition | V̄=50 | V̄=100 |",
        f"|---|---:|---:|",
        f"| NoComm | {nocomm_acc:.2f}% | {nocomm_acc:.2f}% |",
        f"| Ring (|T|=6) | {ring_accs[50]:.2f}% | {ring_accs[100]:.2f}% |",
        f"| Learned (|T|=2, 0→1+0→2) | {learned_accs[50]:.2f}% | {learned_accs[100]:.2f}% |",
        "",
        "## Random-2 Results",
        "",
        "| Draw | Edges | V̄=50 Acc | V̄=100 Acc |",
        "|---:|---|---:|---:|",
    ]

    # Build per-draw rows
    for k, edges in enumerate(draws):
        edge_str = " + ".join(f"{i}→{j}" for (i, j) in edges)
        acc50 = next((d["accuracy"] for d in summary["draws"]
                      if d["draw"] == k + 1 and d["budget"] == 50), None)
        acc100 = next((d["accuracy"] for d in summary["draws"]
                       if d["draw"] == k + 1 and d["budget"] == 100), None)
        md_lines.append(
            f"| {k+1} | {edge_str} | "
            f"{acc50:.2f}% | {acc100:.2f}% |"
            if acc50 is not None and acc100 is not None else
            f"| {k+1} | {edge_str} | — | — |"
        )

    md_lines.extend(["", "## Summary Statistics", ""])

    for v_total in budgets:
        comp = summary["comparison"][f"V{v_total}"]
        md_lines.extend([
            f"### V̄ = {v_total}",
            "",
            f"- **Random-2 Mean**: {comp['random_mean']:.2f}% ± {comp['random_std']:.2f}%"
            f" [{comp['random_min']:.2f}%, {comp['random_max']:.2f}%]",
            f"- **Learned Support**: {comp['learned']:.2f}%",
            f"- **Ring**: {comp['ring']:.2f}%",
            f"- **|Random-2 − Learned|**: {comp['dist_to_learned']:.2f} pts",
            f"- **|Random-2 − Ring|**: {comp['dist_to_ring']:.2f} pts",
            f"- **Verdict**: {comp['verdict']}",
            "",
        ])

    md_path = os.path.join(OUTPUT_DIR, "random_support_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"Saved Markdown summary: {md_path}")

    return summary


# =====================================================================
# MAIN
# =====================================================================
def main():
    print("=" * 80)
    print("  RANDOM SUPPORT BASELINE: CONCENTRATION vs SOURCE SELECTION")
    print("  Model: Azure OpenAI gpt-oss-120b (LIVE API)")
    print("  K=5 random 2-edge draws × V̄ ∈ {50, 100} × n=150 questions")
    print("=" * 80)

    # Setup config (identical to honest experiment)
    config = get_gsm8k_config()
    config.llm_backend = "azure"
    config.api_model = "gpt-oss-120b"
    config.seed = 42
    config.n_samples = 150

    budgets = [50, 100]
    n_agents = config.n_agents  # 3

    # The learned pair: Agent 0 broadcasts to Agents 1 and 2
    LEARNED_EDGES = [(0, 1), (0, 2)]

    # Reference accuracies from honest experiment
    learned_accs = {50: 95.33, 100: 92.67}
    ring_accs = {50: 78.67, 100: 75.33}
    nocomm_acc = 78.67

    # Generate K=5 random draws (excluding learned pair)
    K = 5
    draws = sample_random_draws(
        n_agents=n_agents,
        k_draws=K,
        exclude_edges=LEARNED_EDGES,
        seed=99,  # Separate seed for draw selection (not question selection)
    )

    print(f"\nGenerated {len(draws)} random 2-edge draws:")
    for k, edges in enumerate(draws):
        edge_str = " + ".join(f"{i}→{j}" for (i, j) in edges)
        W = edges_to_adjacency(edges, n_agents)
        print(f"  Draw {k+1}: [{edge_str}]")
        print(f"    W = {W.tolist()}")

    # Load the same 150 questions
    _, full_data = load_dataset_by_name("gsm8k", n_train=0, n_test=150, seed=config.seed)
    print(f"\nLoaded {len(full_data)} test questions (same seed={config.seed})")

    # Run all cells
    all_results = {}
    total_cells = len(draws) * len(budgets)
    cell_num = 0

    for k, edges in enumerate(draws):
        draw_name = f"RandomDraw{k+1}"
        W = edges_to_adjacency(edges, n_agents)
        Q = np.zeros((n_agents, 16))

        for v_total in budgets:
            cell_num += 1
            cell_key = f"{draw_name}_V{v_total}"
            edge_str = " + ".join(f"{i}→{j}" for (i, j) in edges)
            print(f"\n{'=' * 60}")
            print(f"  Cell {cell_num}/{total_cells}: {draw_name} [{edge_str}] @ V̄={v_total}")
            print(f"{'=' * 60}")

            ctrl = FrozenWeightController(config, frozen_W=W, frozen_Q=Q)
            results = run_cell(config, draw_name, v_total, full_data, ctrl)
            all_results[cell_key] = results

            # Quick summary
            correct = sum(1 for r in results if r.is_correct)
            acc = correct / len(results) * 100
            vtxs = [r.total_tokens_transmitted for r in results]
            print(f"  → Acc: {acc:.2f}% ({correct}/{len(results)}), "
                  f"mean V_tx: {np.mean(vtxs):.1f}")

    # Analyze and report
    summary = analyze_and_report(
        all_results, draws, budgets, 150,
        learned_accs, ring_accs, nocomm_acc,
    )

    print("\n" + "=" * 80)
    print("  EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
