#!/usr/bin/env python3
"""
Experiment W2: Web Context Rationing (WWW 2027)
==============================================
Tests whether semantic sentence-preserving rationing preserves multi-hop Web QA
accuracy better than arbitrary token reduction at matched communication volume:
  - Dataset: Open-Book HotpotQA (100 validated questions from web_eval_subset_100.json)
  - Conditions (over fixed communication graph & frozen round-0 evidence):
      1. Full_Transmission: Unrestricted message pass-through
      2. Semantic_Rationing: Whole sentence-preserving extraction up to budget B=50 tokens/edge
      3. Random_Truncation: Arbitrary word truncation to identical budget B=50 tokens/edge
  - Metrics: F1, Exact Match, Transmitted tokens, Degradation from Full Transmission
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, List, Any

import numpy as np
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from config import get_hotpotqa_config
from utils import set_seed, count_tokens
from llm_backend import AzureOpenAIBackend, LLMResponse
from controller import IBController
from dataset_loaders import hotpotqa_f1_score, evaluate_answer
from web_suite import (
    LocalWikipediaRetriever,
    build_agent_prompt,
    invoke_agent_with_cache,
    select_sentences_for_edge,
    select_random_truncated_edge,
    aggregate_responses,
    _GLOBAL_LLM_CACHE,
)

EXP_DIR = os.path.join(BASE_DIR, "results", "W2_context_rationing")
os.makedirs(EXP_DIR, exist_ok=True)
STREAM_FILE = os.path.join(EXP_DIR, "stream.jsonl")
METRICS_FILE = os.path.join(EXP_DIR, "metrics.json")
MANIFEST_FILE = os.path.join(EXP_DIR, "manifest.json")


def run_w2(n_samples: int = 100, seed: int = 42, budget_per_edge: int = 50):
    print("=" * 80)
    print(f"EXPERIMENT W2: WEB CONTEXT RATIONING")
    print(f"Conditions: Full Transmission vs. Semantic Rationing vs. Random Truncation")
    print(f"Budget per edge: {budget_per_edge} tokens | Dataset: Open-Book HotpotQA ({n_samples} samples)")
    print("=" * 80)
    start_time = time.time()

    import subprocess
    git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=BASE_DIR).decode().strip()
    manifest = {
        "experiment": "W2_context_rationing",
        "dataset": "Open-Book HotpotQA (fullwiki)",
        "dataset_examples": n_samples,
        "model": "gpt-oss-120b",
        "provider": "Azure OpenAI",
        "temperature": 0.0,
        "n_agents": 3,
        "rounds": 2,
        "budget_per_edge": budget_per_edge,
        "seed": seed,
        "git_commit": git_hash,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "running",
    }
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    subset_path = os.path.join(BASE_DIR, "web_eval_subset_100.json")
    with open(subset_path, "r", encoding="utf-8") as f:
        samples = json.load(f)[:n_samples]

    config = get_hotpotqa_config(seed=seed)
    set_seed(seed)
    backend = AzureOpenAIBackend(temperature=0.0)

    ckpt_path = os.path.join(BASE_DIR, "08-06results", "hotpotqa_pretrained_400.pt.npz")
    controller = IBController(config)
    controller.load_model(ckpt_path)

    completed = {}
    if os.path.exists(STREAM_FILE):
        with open(STREAM_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        completed[rec["id"]] = rec
                    except Exception:
                        pass
        print(f"Found {len(completed)} questions already completed in stream log.")

    call_tracker = {"live_calls": 0, "cached_hits": 0, "successful_calls": 0, "failed_attempts": 0, "total_tokens": 0}
    # Fixed pipeline topology: Agent 0 -> Agent 1, Agent 1 -> Agent 2
    active_edges = [(0, 1), (1, 2)]

    for sample in tqdm(samples, desc="W2 Evaluation"):
        qid = sample["id"]
        if qid in completed:
            continue

        question = sample["question"]
        gold_answer = sample["answer"]
        articles = sample["articles"]

        # Local BM25 Retrieval
        retriever = LocalWikipediaRetriever(articles)
        retrieved_docs = retriever.retrieve(question, top_k=3)

        # -------------------------------------------------------------
        # Step 1: Frozen Round 0 Initial Solve
        # -------------------------------------------------------------
        round0_responses: List[LLMResponse] = []
        for aid in range(config.n_agents):
            sys_p, usr_p = build_agent_prompt(
                agent_id=aid,
                question=question,
                evidence_passages=retrieved_docs,
                peer_messages=None,
                round_num=0,
            )
            resp = invoke_agent_with_cache(backend, aid, sys_p, usr_p, round_num=0, call_tracker=call_tracker)
            round0_responses.append(resp)

        def execute_rationing_condition(mode: str) -> Dict[str, Any]:
            t_cond_start = time.perf_counter()
            current_resps = list(round0_responses)
            total_comm = 0
            for r in range(1, 3):  # 2 communication rounds
                received = {i: [] for i in range(config.n_agents)}
                for u, v in active_edges:
                    sender_text = current_resps[u].text
                    if mode == "full":
                        msg = sender_text
                        tok = count_tokens(msg)
                    elif mode == "semantic":
                        msg, tok = select_sentences_for_edge(sender_text, budget=budget_per_edge)
                    elif mode == "random":
                        msg, tok = select_random_truncated_edge(sender_text, budget=budget_per_edge, seed=seed + r * 10)
                    else:
                        raise ValueError(f"Unknown mode: {mode}")

                    total_comm += tok
                    if msg.strip():
                        received[v].append(f"Agent {u} [{mode.upper()}]: {msg}")

                next_resps = []
                for aid in range(config.n_agents):
                    msgs = received[aid]
                    if not msgs or not any(m.strip() for m in msgs):
                        next_resps.append(current_resps[aid])
                    else:
                        sys_p, usr_p = build_agent_prompt(
                            agent_id=aid,
                            question=question,
                            evidence_passages=retrieved_docs,
                            peer_messages=msgs,
                            round_num=r,
                        )
                        # Append mode tag to distinguish in cache
                        cache_user_p = f"[{mode.upper()}_RATIONING]\n{usr_p}"
                        new_resp = invoke_agent_with_cache(backend, aid, sys_p, cache_user_p, round_num=r, call_tracker=call_tracker)
                        next_resps.append(new_resp)
                current_resps = next_resps

            final_ans = aggregate_responses(current_resps, calibrator=controller.calibrator)
            is_em = bool(evaluate_answer(final_ans, gold_answer, "hotpotqa"))
            f1 = float(hotpotqa_f1_score(final_ans, gold_answer))
            cond_latency = time.perf_counter() - t_cond_start
            return {
                "answer": final_ans,
                "is_em": is_em,
                "f1": f1,
                "comm_tokens": total_comm,
                "latency_sec": round(cond_latency, 3),
            }

        full_res = execute_rationing_condition("full")
        semantic_res = execute_rationing_condition("semantic")
        random_res = execute_rationing_condition("random")

        record = {
            "id": qid,
            "question": question,
            "gold_answer": gold_answer,
            "Full_Transmission": full_res,
            "Semantic_Rationing": semantic_res,
            "Random_Truncation": random_res,
        }
        completed[qid] = record
        with open(STREAM_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    _GLOBAL_LLM_CACHE.flush()

    records = list(completed.values())
    metrics = {
        "n_samples": len(records),
        "Full_Transmission": {
            "exact_match": float(np.mean([r["Full_Transmission"]["is_em"] for r in records])),
            "f1_score": float(np.mean([r["Full_Transmission"]["f1"] for r in records])),
            "avg_comm_tokens": float(np.mean([r["Full_Transmission"]["comm_tokens"] for r in records])),
            "avg_latency_seconds": float(np.mean([r["Full_Transmission"]["latency_sec"] for r in records])),
        },
        "Semantic_Rationing": {
            "exact_match": float(np.mean([r["Semantic_Rationing"]["is_em"] for r in records])),
            "f1_score": float(np.mean([r["Semantic_Rationing"]["f1"] for r in records])),
            "avg_comm_tokens": float(np.mean([r["Semantic_Rationing"]["comm_tokens"] for r in records])),
            "avg_latency_seconds": float(np.mean([r["Semantic_Rationing"]["latency_sec"] for r in records])),
            "delta_f1_vs_full": float(
                np.mean([r["Semantic_Rationing"]["f1"] for r in records]) - np.mean([r["Full_Transmission"]["f1"] for r in records])
            ),
        },
        "Random_Truncation": {
            "exact_match": float(np.mean([r["Random_Truncation"]["is_em"] for r in records])),
            "f1_score": float(np.mean([r["Random_Truncation"]["f1"] for r in records])),
            "avg_comm_tokens": float(np.mean([r["Random_Truncation"]["comm_tokens"] for r in records])),
            "avg_latency_seconds": float(np.mean([r["Random_Truncation"]["latency_sec"] for r in records])),
            "delta_f1_vs_full": float(
                np.mean([r["Random_Truncation"]["f1"] for r in records]) - np.mean([r["Full_Transmission"]["f1"] for r in records])
            ),
        },
        "semantic_advantage_vs_random_f1": float(
            np.mean([r["Semantic_Rationing"]["f1"] for r in records]) - np.mean([r["Random_Truncation"]["f1"] for r in records])
        ),
        "elapsed_seconds": round(time.time() - start_time, 2),
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    manifest["status"] = "completed"
    manifest["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest["metrics"] = metrics
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 80)
    print("W2 EXPERIMENT COMPLETE")
    print(json.dumps(metrics, indent=2))
    print("=" * 80)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=int, default=50)
    args = parser.parse_args()
    run_w2(n_samples=args.samples, seed=args.seed, budget_per_edge=args.budget)
