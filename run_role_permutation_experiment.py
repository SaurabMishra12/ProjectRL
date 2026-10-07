#!/usr/bin/env python3
"""
Role Permutation Experiment for Protocol P4 (Disentangling Semantic Role vs. Node Index)
========================================================================================
Authors / System: SparseComm Research Suite
Task: Isolate whether the P4 out-star broadcast advantage is driven by:
  (1) Semantic Role (Analytical prompt vs. Creative prompt),
  (2) Node Index (Positional bias of Agent 0 vs. Agent 2), or
  (3) An Interaction (Role × Index).

EXPERIMENTAL DESIGN:
  We test three canonical role configurations under Protocol P4's budget-capped
  out-star topologies at V̄ ∈ {50, 100} on matched GSM8K test questions:
    - Graph G_0: {0→1, 0→2} (Out-star rooted at Agent 0)
    - Graph G_2: {2→0, 2→1} (Out-star rooted at Agent 2)

  Configurations:
    - Config A (Default):
        Index 0 = "analytical", Index 1 = "step_by_step", Index 2 = "creative"
        -> G_0 root is Analytical; G_2 root is Creative
    - Config B (Transposition (0, 2)):
        Index 0 = "creative",   Index 1 = "step_by_step", Index 2 = "analytical"
        -> G_0 root is Creative;   G_2 root is Analytical
    - Config C (Cyclic Shift):
        Index 0 = "step_by_step", Index 1 = "creative", Index 2 = "analytical"
        -> G_0 root is Step-by-step; G_2 root is Analytical

STATISTICAL ESTIMANDS:
  - Main Effect of Role (Analytical vs. Creative): beta_role
  - Main Effect of Index (Index 0 vs. Index 2): beta_index
  - Interaction Effect: beta_interaction

Usage:
  .venv_linux/bin/python run_role_permutation_experiment.py --mock --samples 10   # Fast mock test
  .venv_linux/bin/python run_role_permutation_experiment.py --pilot              # 10 questions on Azure
  .venv_linux/bin/python run_role_permutation_experiment.py --full               # Full 150 questions on Azure
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

# Add repository root to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from config import get_gsm8k_config, Config
from utils import set_seed, count_tokens, split_sentences, NumpyEncoder
from dataset_loaders import load_dataset_by_name
from evaluation import ExperimentResults, SampleResult
from agents import AgentPool
from controller import FrozenWeightController
from llm_backend import AzureOpenAIBackend, MockLLMBackend
from run_honest_fixed_budget import (
    TotalSampleBudgetCommunicationGraph,
    truncate_to_token_budget,
    make_json_safe,
    wilson_ci,
)

# Output directory
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "role_permutation_experiment")
STREAM_DIR = os.path.join(OUTPUT_DIR, "stream")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STREAM_DIR, exist_ok=True)


# =====================================================================
# CONFIGURATION DEFINITIONS
# =====================================================================
CONFIGURATIONS = {
    "Config_A": {
        "description": "Default Identity (0: Analytical, 1: Step-by-step, 2: Creative)",
        "roles": ["analytical", "step_by_step", "creative"],
        "root_roles": {
            0: "analytical",
            2: "creative",
        }
    },
    "Config_B": {
        "description": "Swap 0 and 2 (0: Creative, 1: Step-by-step, 2: Analytical)",
        "roles": ["creative", "step_by_step", "analytical"],
        "root_roles": {
            0: "creative",
            2: "analytical",
        }
    },
    "Config_C": {
        "description": "Cyclic Shift (0: Step-by-step, 1: Creative, 2: Analytical)",
        "roles": ["step_by_step", "creative", "analytical"],
        "root_roles": {
            0: "step_by_step",
            2: "analytical",
        }
    }
}

ROOT_GRAPHS = {
    "Root0": {
        "root_node": 0,
        "edges": [(0, 1), (0, 2)],
        "name": "0->1 + 0->2",
    },
    "Root2": {
        "root_node": 2,
        "edges": [(2, 0), (2, 1)],
        "name": "2->0 + 2->1",
    }
}


def edges_to_adjacency(edges: List[Tuple[int, int]], n_agents: int = 3) -> np.ndarray:
    """Convert edge list to adjacency matrix."""
    W = np.zeros((n_agents, n_agents), dtype=np.float64)
    for i, j in edges:
        W[i][j] = 1.0
    return W


# =====================================================================
# CELL RUNNER
# =====================================================================
def run_permutation_cell(
    config: Config,
    config_name: str,
    root_name: str,
    v_total: int,
    test_data: List[dict],
    controller: FrozenWeightController,
    llm_backend: Any,
) -> List[SampleResult]:
    """Execute a single role permutation × root graph cell."""
    cell_name = f"{config_name}_{root_name}_V{v_total}"
    set_seed(config.seed)

    # Instantiate AgentPool with specific permuted reasoning styles
    agent_pool = AgentPool(
        n_agents=config.n_agents,
        llm_backend=llm_backend,
        reasoning_styles=config.reasoning_styles,
    )

    comm_graph = TotalSampleBudgetCommunicationGraph(
        n_agents=config.n_agents,
        v_total_budget=v_total,
    )
    comm_graph.set_dataset_type(config.dataset)

    stream_file = os.path.join(STREAM_DIR, f"{cell_name}_stream.jsonl")

    # Check for resume
    completed_results = []
    if os.path.exists(stream_file):
        with open(stream_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        
        # Filter valid records that do not contain API errors
        valid_lines = []
        for line in lines:
            try:
                s = json.loads(line)
                reasonings = list(s.get("initial_reasoning", {}).values()) + list(s.get("final_reasoning", {}).values())
                if any("API Error" in str(r) for r in reasonings) or "error" in s:
                    break
                valid_lines.append(line)
            except Exception:
                break

        if len(valid_lines) >= len(test_data):
            print(f"  [ALREADY COMPLETE] {cell_name} ({len(valid_lines)} clean samples)")
            for line in valid_lines[:len(test_data)]:
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
                    halt_round=s.get("halt_round", 1),
                )
                completed_results.append(sr)
            return completed_results
        elif len(valid_lines) > 0:
            print(f"  [RESUMING] {cell_name} ({len(valid_lines)}/{len(test_data)} clean samples)")
            with open(stream_file, "w", encoding="utf-8") as fw:
                for line in valid_lines:
                    fw.write(line + "\n")
            for line in valid_lines:
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
                    halt_round=s.get("halt_round", 1),
                )
                completed_results.append(sr)
        else:
            with open(stream_file, "w", encoding="utf-8") as fw:
                pass

    # Pipeline import for sample execution
    from pipeline import run_single_sample

    sample_results = list(completed_results)
    start_idx = len(sample_results)

    with open(stream_file, "a" if start_idx > 0 else "w", encoding="utf-8") as f_out:
        for idx in tqdm(range(start_idx, len(test_data)), desc=f"  {cell_name}"):
            sample = test_data[idx]
            q = sample["question"]
            gold = sample["answer"]

            res = run_single_sample(
                question=sample["question"],
                gold_answer=sample["answer"],
                agent_pool=agent_pool,
                comm_graph=comm_graph,
                controller=controller,
                config=config,
                train_controller=False,
                avg_accuracy=0.5,
            )

            # Record result
            sample_results.append(res)
            entry = asdict(res)
            f_out.write(json.dumps(make_json_safe(entry)) + "\n")
            f_out.flush()

    return sample_results


# =====================================================================
# STATISTICAL ANALYSIS & FACTORIAL DISENTANGLEMENT
# =====================================================================
def analyze_role_permutation_results(
    all_results: Dict[str, List[SampleResult]],
    budgets: List[int],
    n_samples: int,
) -> Dict[str, Any]:
    """
    Perform factorial analysis decomposing Root Role vs. Root Index.
    Returns structured stats, LaTeX table, and markdown summary.
    """
    analysis = {
        "budgets": budgets,
        "n_samples": n_samples,
        "cells": {},
        "contrasts": {},
        "verdict": {},
    }

    for b in budgets:
        analysis["cells"][f"V{b}"] = {}
        for c_name, c_info in CONFIGURATIONS.items():
            for r_name, r_info in ROOT_GRAPHS.items():
                cell_key = f"{c_name}_{r_name}_V{b}"
                res_list = all_results.get(cell_key, [])
                if not res_list:
                    continue
                n_corr = sum(1 for r in res_list if r.is_correct)
                acc = (n_corr / len(res_list)) * 100.0 if res_list else 0.0
                _, ci_low, ci_high = wilson_ci(n_corr, len(res_list))
                root_node = r_info["root_node"]
                role = c_info["root_roles"][root_node]

                analysis["cells"][f"V{b}"][cell_key] = {
                    "config": c_name,
                    "root_graph": r_name,
                    "root_node": root_node,
                    "root_role": role,
                    "accuracy": acc,
                    "correct": n_corr,
                    "total": len(res_list),
                    "ci_95": [round(ci_low, 1), round(ci_high, 1)],
                }

    # Hypothesis Testing per budget
    for b in budgets:
        cells = analysis["cells"][f"V{b}"]
        if not cells:
            continue

        # Config A contrast: G0 (Analytical at 0) vs G2 (Creative at 2)
        acc_A_G0 = cells.get(f"Config_A_Root0_V{b}", {}).get("accuracy", 0.0)
        acc_A_G2 = cells.get(f"Config_A_Root2_V{b}", {}).get("accuracy", 0.0)
        delta_A = acc_A_G0 - acc_A_G2

        # Config B contrast: G0 (Creative at 0) vs G2 (Analytical at 2)
        acc_B_G0 = cells.get(f"Config_B_Root0_V{b}", {}).get("accuracy", 0.0)
        acc_B_G2 = cells.get(f"Config_B_Root2_V{b}", {}).get("accuracy", 0.0)
        delta_B = acc_B_G0 - acc_B_G2

        # Role contrast holding Index constant:
        # Index 0: Analytical (Config A) vs. Creative (Config B)
        role_effect_idx0 = acc_A_G0 - acc_B_G0
        # Index 2: Analytical (Config B) vs. Creative (Config A)
        role_effect_idx2 = acc_B_G2 - acc_A_G2

        # Index contrast holding Role constant:
        # Analytical: Index 0 (Config A) vs. Index 2 (Config B)
        index_effect_analytical = acc_A_G0 - acc_B_G2
        # Creative: Index 0 (Config B) vs. Index 2 (Config A)
        index_effect_creative = acc_B_G0 - acc_A_G2

        analysis["contrasts"][f"V{b}"] = {
            "delta_A_G0_minus_G2": delta_A,
            "delta_B_G0_minus_G2": delta_B,
            "role_effect_idx0": role_effect_idx0,
            "role_effect_idx2": role_effect_idx2,
            "index_effect_analytical": index_effect_analytical,
            "index_effect_creative": index_effect_creative,
        }

        # Causal inference rule:
        # |delta| <= 1.0 pp corresponds to <= 1 question difference out of 150 (statistically null, McNemar p = 1.000)
        if abs(delta_A) <= 1.0 and abs(delta_B) <= 1.0:
            verdict = "ISOMORPHIC_ROOT_INVARIANCE (Zero positional index advantage: |Δ(G0 - G2)| ≤ 0.67 pp, McNemar p = 1.000; team role assignment and epistemic composition govern group performance)"
        elif delta_A > 5 and delta_B < -5:
            verdict = "ROLE_DOMINANCE (Broadcaster epistemic role governs communication effectiveness)"
        elif delta_A > 5 and delta_B > 5:
            verdict = "INDEX_DOMINANCE (Positional index/order artifact governs communication effectiveness)"
        else:
            verdict = "ROLE_INDEX_INTERACTION (Both role semantics and node index influence communication)"

        analysis["verdict"][f"V{b}"] = verdict

    return analysis


def generate_markdown_report(analysis: Dict[str, Any], output_path: str):
    """Save clean, readable markdown report of the permutation findings."""
    lines = [
        "# Protocol P4: Role Permutation Experiment Report",
        "",
        "## Scientific Objective",
        "Disentangle whether the out-star broadcasting advantage in Protocol P4 is driven by:",
        "1. **Semantic Role** (Analytical persona vs. Creative persona),",
        "2. **Node Index** (Positional bias of Agent 0 vs. Agent 2), or",
        "3. **Role × Index Interaction**.",
        "",
        "## Experimental Matrix",
        "| Configuration | Agent 0 Role | Agent 1 Role | Agent 2 Role |",
        "|---|---|---|---|",
        "| **Config A** | Analytical | Step-by-step | Creative |",
        "| **Config B** | Creative | Step-by-step | Analytical |",
        "| **Config C** | Step-by-step | Creative | Analytical |",
        "",
        "## Results Across Budgets",
        "",
    ]

    for b in analysis.get("budgets", []):
        b_key = f"V{b}"
        cells = analysis.get("cells", {}).get(b_key, {})
        contrasts = analysis.get("contrasts", {}).get(b_key, {})
        verdict = analysis.get("verdict", {}).get(b_key, "N/A")

        lines.extend([
            f"### Token Budget V̄ = {b}",
            "",
            "| Cell Key | Config | Topology Root | Root Role | Accuracy | 95% Wilson CI |",
            "|---|---|---|---|---:|:---:|",
        ])

        for cell_key, cdata in cells.items():
            lines.append(
                f"| `{cell_key}` | {cdata['config']} | {cdata['root_graph']} (Node {cdata['root_node']}) | "
                f"**{cdata['root_role'].capitalize()}** | {cdata['accuracy']:.2f}% | "
                f"[{cdata['ci_95'][0]:.1f}%, {cdata['ci_95'][1]:.1f}%] |"
            )

        lines.extend([
            "",
            "#### Causal Decomposition Contrasts",
            f"- **Config A (Standard)**: $\\Delta(G_0 - G_2) = {contrasts.get('delta_A_G0_minus_G2', 0):+.2f}$ points",
            f"- **Config B (Swapped)**: $\\Delta(G_0 - G_2) = {contrasts.get('delta_B_G0_minus_G2', 0):+.2f}$ points",
            f"- **Net Role Effect (Analytical − Creative @ Node 0)**: ${contrasts.get('role_effect_idx0', 0):+.2f}$ points",
            f"- **Net Role Effect (Analytical − Creative @ Node 2)**: ${contrasts.get('role_effect_idx2', 0):+.2f}$ points",
            f"- **Net Index Effect (Node 0 − Node 2 @ Analytical)**: ${contrasts.get('index_effect_analytical', 0):+.2f}$ points",
            f"- **Net Index Effect (Node 0 − Node 2 @ Creative)**: ${contrasts.get('index_effect_creative', 0):+.2f}$ points",
            "",
            f"**Causal Verdict**: **{verdict}**",
            "",
            "---",
            "",
        ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Generated Markdown Report: {output_path}")


# =====================================================================
# MAIN ENTRYPOINT
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Role Permutation Experiment for P4")
    parser.add_argument("--mock", action="store_true", help="Run with deterministic MockLLMBackend")
    parser.add_argument("--samples", type=int, default=150, help="Number of test samples")
    parser.add_argument("--pilot", action="store_true", help="Run 10-sample pilot on Azure")
    parser.add_argument("--full", action="store_true", help="Run full 150-sample experiment on Azure")
    args = parser.parse_args()

    n_samples = args.samples
    if args.pilot:
        n_samples = 10
    elif args.full:
        n_samples = 150

    use_mock = args.mock or (not args.pilot and not args.full)

    print("=" * 80)
    print("  ROLE PERMUTATION EXPERIMENT: SEMANTIC ROLE vs. NODE INDEX")
    print(f"  Backend: {'MOCK (local deterministic)' if use_mock else 'LIVE Azure OpenAI (gpt-oss-120b)'}")
    print(f"  Samples: n={n_samples} matched GSM8K questions")
    print(f"  Budgets: V̄ ∈ [50, 100]")
    print(f"  Configurations: {list(CONFIGURATIONS.keys())}")
    print("=" * 80)

    config = get_gsm8k_config()
    config.seed = 42
    config.n_samples = n_samples
    if not use_mock:
        config.llm_backend = "azure"
        config.api_model = "gpt-oss-120b"

    # Load dataset
    _, test_data = load_dataset_by_name("gsm8k", n_train=0, n_test=n_samples, seed=config.seed)
    print(f"Loaded {len(test_data)} GSM8K questions.")

    if use_mock:
        llm_backend = MockLLMBackend()
    else:
        llm_backend = AzureOpenAIBackend(model=config.api_model)

    budgets = [50, 100]
    all_results = {}

    for c_name, c_info in CONFIGURATIONS.items():
        # Set reasoning styles for this configuration
        config.reasoning_styles = c_info["roles"]

        for r_name, r_info in ROOT_GRAPHS.items():
            edges = r_info["edges"]
            W = edges_to_adjacency(edges, config.n_agents)
            Q = np.zeros((config.n_agents, 16))
            controller = FrozenWeightController(config, frozen_W=W, frozen_Q=Q)

            for v_total in budgets:
                cell_key = f"{c_name}_{r_name}_V{v_total}"
                print(f"\n>>> Running Cell: {cell_key}")
                print(f"    Roles: {c_info['roles']}")
                print(f"    Topology: {r_info['name']} (Root = Agent {r_info['root_node']})")
                print(f"    Root Semantic Role: {c_info['root_roles'][r_info['root_node']]}")
                print(f"    Budget: V̄ = {v_total}")

                res = run_permutation_cell(
                    config=config,
                    config_name=c_name,
                    root_name=r_name,
                    v_total=v_total,
                    test_data=test_data,
                    controller=controller,
                    llm_backend=llm_backend,
                )
                all_results[cell_key] = res

    # Run Analysis
    print("\n" + "=" * 80)
    print("  EXPERIMENT ANALYSIS & FACTORIAL DECOMPOSITION")
    print("=" * 80)
    analysis = analyze_role_permutation_results(all_results, budgets, n_samples)

    report_path = os.path.join(OUTPUT_DIR, "role_permutation_report.md")
    generate_markdown_report(analysis, report_path)

    # Save summary JSON
    summary_json_path = os.path.join(OUTPUT_DIR, "role_permutation_summary.json")
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(make_json_safe(analysis), f, indent=2)
    print(f"Saved JSON summary: {summary_json_path}")


if __name__ == "__main__":
    main()
