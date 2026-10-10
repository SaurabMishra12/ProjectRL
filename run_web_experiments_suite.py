#!/usr/bin/env python3
"""
Master Web Experiments Suite Orchestrator (WWW 2027)
===================================================
Executes the minimum scientifically necessary set of Web experiments in order:
  1. Experiment W1: Web Cost-Accuracy Frontier (NoComm vs. MAD vs. SPARSECOMM)
  2. Experiment W2: Web Context Rationing (Full Transmission vs. Semantic Rationing vs. Random Truncation)
  3. Experiment W3: Adaptive Stopping under Web Noise (Fixed Depth vs. Adaptive SPRT across Clean/Noisy Evidence)
  4. Experiment W4: Counterfactual Routing Isolation (Static Role vs. Learned SPARSECOMM vs. Random vs. Complete)
     [Highest Scientific Priority - tests pure routing effect over frozen information pool]

All experiments are persistent, fail-closed, resumable, and write immutable raw logs.
"""

import os
import sys
import json
import time
import argparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from run_w1_web_cost_accuracy import run_w1
from run_w2_context_rationing import run_w2
from run_w3_adaptive_stopping import run_w3
from run_w4_counterfactual_routing import run_w4

SUITE_REPORT_FILE = os.path.join(BASE_DIR, "results", "master_web_suite_summary.json")


def run_suite(n_samples: int = 100, seed: int = 42, phases: str = "all"):
    print("=" * 80)
    print("STARTING WWW 2027 WEB EXPERIMENTAL SUITE")
    print(f"Target: {n_samples} questions from Open-Book HotpotQA | Seed: {seed}")
    print(f"Phases requested: {phases}")
    print("=" * 80)
    suite_start = time.time()

    phase_list = [p.strip().upper() for p in phases.split(",")] if phases != "all" else ["W1", "W2", "W3", "W4"]
    suite_results = {}

    # -------------------------------------------------------------
    # Phase 1: W1 Web Cost-Accuracy Frontier
    # -------------------------------------------------------------
    if "W1" in phase_list:
        print("\n" + "#" * 80)
        print("LAUNCHING PHASE 1: EXPERIMENT W1 (WEB COST-ACCURACY FRONTIER)")
        print("#" * 80)
        w1_metrics = run_w1(n_samples=n_samples, seed=seed)
        suite_results["W1"] = w1_metrics

    # -------------------------------------------------------------
    # Phase 2: W2 Web Context Rationing
    # -------------------------------------------------------------
    if "W2" in phase_list:
        print("\n" + "#" * 80)
        print("LAUNCHING PHASE 2: EXPERIMENT W2 (WEB CONTEXT RATIONING)")
        print("#" * 80)
        w2_metrics = run_w2(n_samples=n_samples, seed=seed, budget_per_edge=50)
        suite_results["W2"] = w2_metrics

    # -------------------------------------------------------------
    # Phase 3: W3 Adaptive Stopping under Noise
    # -------------------------------------------------------------
    if "W3" in phase_list:
        print("\n" + "#" * 80)
        print("LAUNCHING PHASE 3: EXPERIMENT W3 (ADAPTIVE STOPPING UNDER NOISE)")
        print("#" * 80)
        w3_metrics = run_w3(n_samples=n_samples, seed=seed)
        suite_results["W3"] = w3_metrics

    # -------------------------------------------------------------
    # Phase 4: W4 Counterfactual Routing Isolation (HIGHEST PRIORITY)
    # -------------------------------------------------------------
    if "W4" in phase_list:
        print("\n" + "#" * 80)
        print("LAUNCHING PHASE 4: EXPERIMENT W4 (COUNTERFACTUAL ROUTING ISOLATION)")
        print("#" * 80)
        w4_metrics, w4_analysis = run_w4(n_samples=n_samples, seed=seed, round_budget=100, n_rounds=2)
        suite_results["W4"] = {"metrics": w4_metrics, "analysis": w4_analysis}

    suite_total_time = round(time.time() - suite_start, 2)
    suite_summary = {
        "suite_name": "WWW 2027 Web Multi-Agent Communication Suite",
        "dataset": "Open-Book HotpotQA (fullwiki)",
        "n_samples": n_samples,
        "seed": seed,
        "phases_executed": phase_list,
        "elapsed_seconds": suite_total_time,
        "results": suite_results,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "ALL_COMPLETED",
    }

    with open(SUITE_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(suite_summary, f, indent=2)

    print("\n" + "=" * 80)
    print("MASTER EXPERIMENT SUITE COMPLETED SUCCESSFULLY!")
    print(f"Total Suite Runtime: {suite_total_time:.2f} seconds")
    print(f"Summary Report saved to: {SUITE_REPORT_FILE}")
    print("=" * 80)

    return suite_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phases", type=str, default="all", help="Comma-separated: W1,W2,W3,W4 or 'all'")
    args = parser.parse_args()
    run_suite(n_samples=args.samples, seed=args.seed, phases=args.phases)
