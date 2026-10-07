#!/usr/bin/env python3
"""
Master Experiment P5: Component-Isolated, Budget-Matched Communication Control
=============================================================================
A 2x2 factorial experiment to isolate routing policy vs. adaptive stopping
under strictly matched inter-agent communication volume (V_bar = 50 tokens).

Cells:
  - Cell A: Static S1 (0->1, 0->2) + Fixed Depth (3 communication rounds)
  - Cell B: Static S1 (0->1, 0->2) + Adaptive Stopping (SPRT S > log 19 or z_r > 0)
  - Cell C: Learned Top-2 (from controller W^(r)) + Fixed Depth (3 rounds)
  - Cell D: Learned Top-2 (from controller W^(r)) + Adaptive Stopping

Reuses:
  - 100 matched GSM8K questions from P4 (PPO_Seed1_400ep_V50_stream.jsonl)
  - Pretrained 400-example PPO controller (08-06results/gsm8k_pretrained_400.pt.npz)
  - Round-0 initial agent responses and reasoning
  - Platt-calibrated majority aggregator
  - Model: gpt-oss-120b via AzureOpenAIBackend, temperature = 0.0

Usage:
  .venv_linux/bin/python run_p5_master_experiment.py --test-run 2
  .venv_linux/bin/python run_p5_master_experiment.py --samples 100
"""

import os
import sys
import json
import math
import time
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter

import numpy as np
from tqdm import tqdm

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from config import get_gsm8k_config, Config
from utils import set_seed, count_tokens, split_sentences, NumpyEncoder
from pipeline import weighted_majority_vote, _normalize_for_vote, evaluate_answer
from agents import AgentPool, AgentResponse
from controller import IBController
from llm_backend import AzureOpenAIBackend
from run_honest_fixed_budget import make_json_safe, wilson_ci
from bootstrap_analysis import bca_ci

# Output directories
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "p5_communication_control")
STREAM_DIR = os.path.join(OUTPUT_DIR, "stream")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STREAM_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Sentence-Preserving Transmission Selector (Common across all 4 cells)
# ---------------------------------------------------------------------------
def select_sentences_for_edge(text: str, budget: int) -> Tuple[str, int]:
    """
    Selects whole sentences from reasoning text until budget is reached.
    If the first sentence exceeds budget, falls back to word-level truncation
    so that no active communication edge is starved of signal.
    """
    if not text or budget <= 0:
        return "", 0
    if count_tokens(text) <= budget:
        return text, count_tokens(text)

    sentences = split_sentences(text)
    curr_sentences = []
    for s in sentences:
        candidate = " ".join(curr_sentences + [s])
        if count_tokens(candidate) <= budget:
            curr_sentences.append(s)
        else:
            break

    if curr_sentences:
        cand_text = " ".join(curr_sentences)
        return cand_text, count_tokens(cand_text)

    # Word-level fallback if first sentence is longer than budget
    words = text.split()
    low, high, best = 0, len(words), 0
    while low <= high:
        mid = (low + high) // 2
        cand = " ".join(words[:mid])
        if count_tokens(cand) <= budget:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    cand_text = " ".join(words[:best])
    return cand_text, count_tokens(cand_text)


# ---------------------------------------------------------------------------
# Stopping Rule Evaluator (Common across Cells B & D)
# ---------------------------------------------------------------------------
def evaluate_stopping_rule(
    controller: IBController,
    current_responses: List[AgentResponse],
    prev_W: np.ndarray,
    round_num: int,
    config: Config,
) -> Tuple[bool, float, float]:
    """
    Evaluates whether communication should halt:
      1. Controller learned logit z_r > 0
      2. Sequential likelihood ratio S > log 19 (tau = 2.944)
    """
    agent_texts = {r.agent_id: r.reasoning_text for r in current_responses}
    agent_answers = {r.agent_id: r.answer for r in current_responses}
    confs = [r.confidence for r in current_responses]
    unique_answers = len(set(agent_answers.values()))

    # Controller halt logit
    _, _, halt_decision, meta = controller.get_weights(
        reasoning_texts=agent_texts,
        answers=agent_answers,
        confidence_scores=confs,
        prev_W=prev_W,
        round_num=round_num + 1,
        unique_answers=unique_answers,
        avg_accuracy=0.5,
        explore=False,
    )
    halt_logit = float(meta["mean_action"][-1])

    # Sequential binomial log-likelihood ratio S > log 19
    _SPRT_TAU = 2.9444389791664403  # log(19), alpha=0.05
    _SPRT_P0 = 0.5
    ans_list = [_normalize_for_vote(r.answer, config.dataset) for r in current_responses]
    counts = Counter(ans_list)
    majority_ans, k = counts.most_common(1)[0]
    N = len(ans_list)
    maj_confs = [
        r.confidence for r in current_responses
        if _normalize_for_vote(r.answer, config.dataset) == majority_ans
    ]
    if controller.calibrator is not None and controller.calibrator._fitted and maj_confs:
        p_maj = float(np.clip(
            controller.calibrator.calibrate(float(np.max(maj_confs))), _SPRT_P0 + 0.001, 0.999
        ))
    else:
        p_maj = 0.9
    llr = float(k * np.log(p_maj / _SPRT_P0) + (N - k) * np.log((1.0 - p_maj) / (1.0 - _SPRT_P0)))
    sprt_halt = bool(llr > _SPRT_TAU)

    should_halt = bool(halt_decision or sprt_halt)
    return should_halt, halt_logit, llr


# ---------------------------------------------------------------------------
# Single Question Simulation Engine
# ---------------------------------------------------------------------------
ALL_DIRECTED_EDGES = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
STATIC_S1_EDGES = [(0, 1), (0, 2)]

# Episode budget allocation across 3 rounds (Total = 50 tokens)
ROUND_BUDGETS = {
    1: 16,  # 8 tokens for edge 1, 8 tokens for edge 2
    2: 17,  # 8 tokens for edge 1, 9 tokens for edge 2
    3: 17,  # 8 tokens for edge 1, 9 tokens for edge 2
}

def get_edge_budgets_for_round(round_num: int) -> List[int]:
    total_b = ROUND_BUDGETS[round_num]
    b1 = total_b // 2
    b2 = total_b - b1
    return [b1, b2]


def run_cell_for_question(
    routing_mode: str,   # "static" or "learned"
    stopping_mode: str,  # "fixed" or "adaptive"
    question: str,
    gold_answer: str,
    round0_responses: List[AgentResponse],
    controller: IBController,
    agent_pool: AgentPool,
    config: Config,
    response_cache: Dict[Tuple[int, int, str], AgentResponse],
) -> Dict[str, Any]:
    """
    Executes one question under the specified routing and stopping conditions.
    """
    current_responses = list(round0_responses)
    prev_W = np.zeros((config.n_agents, config.n_agents))
    total_tokens_transmitted = 0
    per_round_tokens = []
    active_edges_per_round = []
    halt_round = config.n_rounds
    halted_early = False

    for r in range(1, config.n_rounds + 1):
        # 1. Determine active edges for this round
        if routing_mode == "static":
            active_edges = list(STATIC_S1_EDGES)
            curr_W = np.zeros((config.n_agents, config.n_agents))
            for u, v in active_edges:
                curr_W[u][v] = 1.0
        elif routing_mode == "learned":
            agent_texts = {resp.agent_id: resp.reasoning_text for resp in current_responses}
            agent_answers = {resp.agent_id: resp.answer for resp in current_responses}
            confs = [resp.confidence for resp in current_responses]
            unique_answers = len(set(agent_answers.values()))

            W, _, _, _ = controller.get_weights(
                reasoning_texts=agent_texts,
                answers=agent_answers,
                confidence_scores=confs,
                prev_W=prev_W,
                round_num=r,
                unique_answers=unique_answers,
                avg_accuracy=0.5,
                explore=False,
            )
            # Rank the 6 directed edges by intensity descending and select top 2
            ranked = sorted(ALL_DIRECTED_EDGES, key=lambda e: W[e[0]][e[1]], reverse=True)
            active_edges = ranked[:2]
            curr_W = W
        else:
            raise ValueError(f"Unknown routing_mode: {routing_mode}")

        active_edges_per_round.append([[int(u), int(v)] for u, v in active_edges])

        # 2. Select sentences under exact round budget
        edge_budgets = get_edge_budgets_for_round(r)
        round_transmitted = 0
        received_messages: Dict[int, List[str]] = {i: [] for i in range(config.n_agents)}

        # Sender reasoning from current_responses (round r-1)
        resp_by_id = {resp.agent_id: resp for resp in current_responses}

        for (u, v), b_edge in zip(active_edges, edge_budgets):
            sender_text = resp_by_id[u].reasoning_text
            filtered_msg, tok = select_sentences_for_edge(sender_text, b_edge)
            round_transmitted += tok
            if filtered_msg.strip():
                received_messages[v].append(f"Agent {u}: {filtered_msg}")

        total_tokens_transmitted += round_transmitted
        per_round_tokens.append(round_transmitted)

        # 3. Agents re-solve with transmitted messages
        next_responses = []
        for agent in agent_pool.agents:
            aid = agent.agent_id
            msgs = received_messages.get(aid, [])
            if not msgs or not any(m.strip() for m in msgs):
                # Agent receives no new messages: retain previous state without API call
                next_responses.append(resp_by_id[aid])
            else:
                # Check response cache
                cache_key = (aid, r, "\n".join(sorted(msgs)))
                if cache_key in response_cache:
                    next_responses.append(response_cache[cache_key])
                else:
                    new_resp = agent.solve(
                        question=question,
                        context_messages=msgs,
                        dataset_type=config.dataset,
                        round_num=r,
                    )
                    response_cache[cache_key] = new_resp
                    next_responses.append(new_resp)

        current_responses = next_responses
        prev_W = curr_W

        # 4. Stopping evaluation
        if stopping_mode == "adaptive":
            should_halt, h_logit, s_llr = evaluate_stopping_rule(
                controller=controller,
                current_responses=current_responses,
                prev_W=prev_W,
                round_num=r,
                config=config,
            )
            if should_halt:
                halt_round = r
                halted_early = True
                break
        elif stopping_mode == "fixed":
            # Force all rounds to execute
            pass
        else:
            raise ValueError(f"Unknown stopping_mode: {stopping_mode}")

    # 5. Final Consensus Aggregation
    final_answer = weighted_majority_vote(
        current_responses,
        dataset_type=config.dataset,
        calibrator=controller.calibrator,
    )
    is_correct = evaluate_answer(final_answer, gold_answer, config.dataset)

    return {
        "final_answer": final_answer,
        "is_correct": bool(is_correct),
        "halt_round": halt_round,
        "halted_early": halted_early,
        "total_tokens_transmitted": total_tokens_transmitted,
        "per_round_tokens": per_round_tokens,
        "active_edges_per_round": active_edges_per_round,
        "final_answers": {r.agent_id: r.answer for r in current_responses},
        "final_reasoning": {r.agent_id: r.reasoning_text for r in current_responses},
    }


# ---------------------------------------------------------------------------
# Main Orchestration Loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="P5 Master Experiment")
    parser.add_argument("--samples", type=int, default=100, help="Number of questions to evaluate")
    parser.add_argument("--test-run", type=int, default=0, help="Run on small subset of questions for testing")
    args = parser.parse_args()

    n_samples = args.test_run if args.test_run > 0 else args.samples
    print("=" * 80)
    print(f"MASTER EXPERIMENT P5: COMPONENT-ISOLATED COMMUNICATION CONTROL")
    print(f"Target: {n_samples} GSM8K questions | Budget V_bar = 50 tokens")
    print("=" * 80)

    # 1. Setup Config & Seed
    config = get_gsm8k_config()
    config.n_rounds = 3
    set_seed(config.seed)

    # 2. Setup LLM Backend & Agent Pool
    backend = AzureOpenAIBackend(temperature=0.0)
    agent_pool = AgentPool(n_agents=config.n_agents, llm_backend=backend)

    # 3. Load Pretrained Controller
    ckpt_path = os.path.join(BASE_DIR, "08-06results", "gsm8k_pretrained_400.pt.npz")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    controller = IBController(config)
    controller.load_model(ckpt_path)
    print(f"Controller loaded from: {ckpt_path}")
    print(f"Calibrator fitted: {controller.calibrator._fitted}")

    # 4. Load P4 Stream Questions & Round-0 Initial States
    source_p4_path = os.path.join(
        BASE_DIR, "results", "p4_definitive_suite", "stream", "PPO_Seed1_400ep_V50_stream.jsonl"
    )
    if not os.path.exists(source_p4_path):
        raise FileNotFoundError(f"P4 source file not found: {source_p4_path}")

    dataset = []
    with open(source_p4_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= n_samples:
                break
            sample = json.loads(line.strip())
            dataset.append(sample)
    print(f"Loaded {len(dataset)} samples from P4 reference stream.")

    # 5. Define the 4 Factorial Cells
    CELL_DEFS = [
        ("Cell_A_Static_Fixed",    "static",  "fixed",    "Cell A (Static S1 + Fixed Depth)"),
        ("Cell_B_Static_Adaptive", "static",  "adaptive", "Cell B (Static S1 + Adaptive Stopping)"),
        ("Cell_C_Learned_Fixed",   "learned", "fixed",    "Cell C (Learned Top-2 + Fixed Depth)"),
        ("Cell_D_Learned_Adaptive","learned", "adaptive", "Cell D (Learned Top-2 + Adaptive Stopping)"),
    ]

    cell_results: Dict[str, List[Dict[str, Any]]] = {c[0]: [] for c in CELL_DEFS}

    # Per-question response cache across cells to share deterministic LLM outputs
    # Key: (sample_idx, agent_id, round_num, context_str)
    global_response_cache: Dict[Tuple[int, int, int, str], AgentResponse] = {}

    for cell_key, routing_mode, stopping_mode, cell_desc in CELL_DEFS:
        print("\n" + "-" * 70)
        print(f"Executing: {cell_desc}")
        print("-" * 70)

        stream_file = os.path.join(STREAM_DIR, f"{cell_key}_stream.jsonl")
        completed = []
        if os.path.exists(stream_file):
            with open(stream_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        completed.append(json.loads(line.strip()))
            print(f"  [RESUME] Found {len(completed)} existing records in {cell_key}_stream.jsonl")

        start_idx = len(completed)
        cell_results[cell_key] = list(completed)

        for i in tqdm(range(start_idx, len(dataset)), desc=cell_key):
            s = dataset[i]
            q = s["question"]
            gold = s["gold_answer"]

            # Construct round-0 responses from cached data
            r0_responses = [
                AgentResponse(
                    agent_id=int(aid),
                    answer=s["initial_answers"][aid],
                    confidence=backend._estimate_confidence(s["initial_reasoning"][aid]),
                    reasoning_text=s["initial_reasoning"][aid],
                    token_count=count_tokens(s["initial_reasoning"][aid]),
                    round_num=0,
                )
                for aid in ["0", "1", "2"]
            ]

            # Build a local question-scoped cache adapter
            class CacheAdapter:
                def __init__(self, sample_idx: int, g_cache: dict):
                    self.sample_idx = sample_idx
                    self.g_cache = g_cache

                def __contains__(self, key):
                    aid, r, ctx = key
                    return (self.sample_idx, aid, r, ctx) in self.g_cache

                def __getitem__(self, key):
                    aid, r, ctx = key
                    return self.g_cache[(self.sample_idx, aid, r, ctx)]

                def __setitem__(self, key, val):
                    aid, r, ctx = key
                    self.g_cache[(self.sample_idx, aid, r, ctx)] = val

            q_cache = CacheAdapter(i, global_response_cache)

            # Reset agent pool history for this question
            agent_pool.reset_all()
            for r0 in r0_responses:
                agent_pool.agents[r0.agent_id].history.append(r0)

            res = run_cell_for_question(
                routing_mode=routing_mode,
                stopping_mode=stopping_mode,
                question=q,
                gold_answer=gold,
                round0_responses=r0_responses,
                controller=controller,
                agent_pool=agent_pool,
                config=config,
                response_cache=q_cache,
            )

            record = {
                "sample_idx": i,
                "question": q,
                "gold_answer": gold,
                "final_answer": res["final_answer"],
                "is_correct": res["is_correct"],
                "halt_round": res["halt_round"],
                "halted_early": res["halted_early"],
                "total_tokens_transmitted": res["total_tokens_transmitted"],
                "per_round_tokens": res["per_round_tokens"],
                "active_edges_per_round": res["active_edges_per_round"],
                "final_answers": res["final_answers"],
                "initial_answers": s["initial_answers"],
            }
            cell_results[cell_key].append(record)

            with open(stream_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(make_json_safe(record)) + "\n")

    # 6. Statistical Analysis & Reporting
    print("\n" + "=" * 80)
    print("EXPERIMENT P5: EMPIRICAL RESULTS SUMMARY")
    print("=" * 80)

    summary_rows = []
    cell_acc_arrays = {}

    for cell_key, _, _, cell_desc in CELL_DEFS:
        records = cell_results[cell_key][:n_samples]
        n = len(records)
        correct_flags = np.array([1 if r["is_correct"] else 0 for r in records], dtype=int)
        cell_acc_arrays[cell_key] = correct_flags
        acc_pt, acc_lo, acc_hi = wilson_ci(int(np.sum(correct_flags)), n)

        tokens = np.array([r["total_tokens_transmitted"] for r in records], dtype=float)
        mean_tok = float(np.mean(tokens))
        std_tok = float(np.std(tokens))

        rounds = np.array([r["halt_round"] for r in records], dtype=float)
        mean_rnd = float(np.mean(rounds))
        std_rnd = float(np.std(rounds))

        summary_rows.append({
            "cell": cell_key,
            "desc": cell_desc,
            "n": n,
            "accuracy": acc_pt,
            "ci_lo": acc_lo,
            "ci_hi": acc_hi,
            "mean_tokens": mean_tok,
            "std_tokens": std_tok,
            "mean_rounds": mean_rnd,
            "std_rounds": std_rnd,
        })
        print(f"{cell_desc:45s} | Acc: {acc_pt:5.2f}% [{acc_lo:4.1f}, {acc_hi:4.1f}] | "
              f"Tokens: {mean_tok:4.1f}±{std_tok:3.1f} | Rounds: {mean_rnd:3.2f}")

    # 7. Paired Contrasts with BCa Bootstrap CIs
    print("\n" + "=" * 80)
    print("PAIRED COMPONENT CONTRASTS (N = 100 Matched Questions)")
    print("=" * 80)

    comparisons = [
        ("Δ_routing (C vs A)",        "Cell_C_Learned_Fixed",    "Cell_A_Static_Fixed",
         "Learned Routing isolated from Stopping (Fixed Depth held constant)"),
        ("Δ_stopping (B vs A)",       "Cell_B_Static_Adaptive", "Cell_A_Static_Fixed",
         "Adaptive Stopping isolated from Routing (under Static S1 Support)"),
        ("Δ_stopping|learned (D vs C)","Cell_D_Learned_Adaptive","Cell_C_Learned_Fixed",
         "Adaptive Stopping effect under Learned Top-2 Routing"),
        ("Δ_joint (D vs A)",          "Cell_D_Learned_Adaptive","Cell_A_Static_Fixed",
         "Joint Adaptive Policy vs. Static Baseline"),
    ]

    paired_summary = []
    rng = np.random.default_rng(config.seed)

    for comp_name, key1, key2, interp in comparisons:
        arr1 = cell_acc_arrays[key1]
        arr2 = cell_acc_arrays[key2]
        delta_arr = (arr1 - arr2) * 100.0  # percentage points
        point_delta = float(np.mean(delta_arr))

        # 10,000 BCa Bootstrap
        ci_lo, ci_hi, _, _ = bca_ci(
            delta_arr,
            stat_fn=np.mean,
            n_bootstrap=10_000,
            alpha=0.05,
            rng=rng,
        )

        # McNemar exact contingency
        n01 = int(np.sum((arr1 == 0) & (arr2 == 1)))
        n10 = int(np.sum((arr1 == 1) & (arr2 == 0)))
        discordant = n01 + n10
        # Exact two-sided binomial test p-value for McNemar
        from scipy import stats
        mcnemar_p = float(stats.binomtest(min(n01, n10), discordant, 0.5, alternative="two-sided").pvalue) if discordant > 0 else 1.0

        paired_summary.append({
            "comparison": comp_name,
            "cell_tested": key1,
            "cell_baseline": key2,
            "point_delta": point_delta,
            "bca_ci_lo": float(ci_lo),
            "bca_ci_hi": float(ci_hi),
            "n01": n01,
            "n10": n10,
            "mcnemar_p": mcnemar_p,
            "interpretation": interp,
        })

        print(f"{comp_name:30s} | Δ = {point_delta:+5.2f} pp | "
              f"95% BCa CI: [{ci_lo:+5.2f}, {ci_hi:+5.2f}] | "
              f"Discordant: (n01={n01}, n10={n10}) | p_McNemar = {mcnemar_p:.4f}")

    # 8. Save Comprehensive JSON & Markdown Report
    final_output = {
        "metadata": {
            "experiment": "P5_Component_Isolated_Communication_Control",
            "n_samples": n_samples,
            "budget_v_bar": 50,
            "model": "gpt-oss-120b",
            "controller": ckpt_path,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "cell_summaries": summary_rows,
        "paired_comparisons": paired_summary,
    }

    results_json_path = os.path.join(OUTPUT_DIR, "p5_master_results.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nSaved master results to: {results_json_path}")

    # Generate Markdown Report
    report_md_path = os.path.join(OUTPUT_DIR, "p5_master_report.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("# Protocol P5: Component-Isolated, Budget-Matched Communication Control\n\n")
        f.write("## 1. Executive Summary\n\n")
        f.write("Protocol P5 executes a definitive $2 \\times 2$ factorial diagnostic designed to isolate\n")
        f.write("the empirical contribution of **learned routing** from **adaptive stopping** under strict\n")
        f.write("inter-agent communication volume matching ($\\bar{V} = 50$ tokens) across 100 matched GSM8K test questions.\n\n")
        f.write("- **Factor R (Routing)**: Static $S_1$ Out-Star ($0 \\to 1, 0 \\to 2$) vs. Learned Top-2 Directed Edges (from controller $W^{(r)}$).\n")
        f.write("- **Factor H (Stopping)**: Fixed Depth (forced 3 communication rounds) vs. Adaptive Stopping ($S > \\log 19$ or $z_r > 0$).\n")
        f.write("- **Common Mechanism**: Identical sentence-preserving transmission selector applied identically across all four cells.\n")
        f.write("- **Volume Control**: Equal allocation across active communication rounds ($B_{\\text{round}} = [16, 17, 17]$ tokens).\n\n")

        f.write("## 2. 2x2 Factorial Empirical Table\n\n")
        f.write("| Cell | Routing Policy | Stopping Policy | Accuracy (%) | 95% Wilson CI | Realized Volume (tok) | Avg. Rounds |\n")
        f.write("| :--- | :--- | :--- | :---: | :---: | :---: | :---: |\n")
        for row in summary_rows:
            f.write(f"| **{row['cell']}** | {row['desc'].split('(')[1].split('+')[0].strip()} | "
                    f"{row['desc'].split('+')[1].split(')')[0].strip()} | "
                    f"**{row['accuracy']:.2f}%** | [{row['ci_lo']:.1f}, {row['ci_hi']:.1f}] | "
                    f"{row['mean_tokens']:.1f} ± {row['std_tokens']:.1f} | {row['mean_rounds']:.2f} |\n")
        f.write("\n")

        f.write("## 3. Paired Question-Level Contrasts (N = 100 Matched Questions)\n\n")
        f.write("| Comparison | Hypothesis Tested | Point $\\Delta$ (pp) | 95% BCa Bootstrap CI | McNemar $p$-value | Scientific Verdict |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :--- |\n")
        for p in paired_summary:
            sig = "Statistically Significant" if p["mcnemar_p"] < 0.05 else "Non-Significant"
            f.write(f"| **{p['comparison']}** | {p['interpretation']} | "
                    f"**{p['point_delta']:+.2f}** | [{p['bca_ci_lo']:+.2f}, {p['bca_ci_hi']:+.2f}] | "
                    f"{p['mcnemar_p']:.4f} | {sig} |\n")
        f.write("\n")

    print(f"Generated report at: {report_md_path}")


if __name__ == "__main__":
    main()
