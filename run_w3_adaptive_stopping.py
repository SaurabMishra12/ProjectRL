#!/usr/bin/env python3
"""
Experiment W3: Adaptive Stopping under Web Evidence Noise (WWW 2027)
===================================================================
Evaluates SPRT sequential stopping vs. fixed depth under varying web evidence noise:
  - Dataset: Open-Book HotpotQA (100 validated questions from web_eval_subset_100.json)
  - 2x2 Factorial Design:
      Factor 1: Stopping Policy (Fixed Depth T=3 vs. Adaptive SPRT Halting)
      Factor 2: Evidence Noise (Clean Top-2 BM25 Docs vs. Full 10 Distractor-Injected Docs)
  - Metrics: F1, Exact Match, Realized communication rounds, Transmitted tokens, Early exit rate
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
    evaluate_stopping_rule,
    aggregate_responses,
    _GLOBAL_LLM_CACHE,
)

EXP_DIR = os.path.join(BASE_DIR, "results", "W3_adaptive_stopping")
os.makedirs(EXP_DIR, exist_ok=True)
STREAM_FILE = os.path.join(EXP_DIR, "stream.jsonl")
METRICS_FILE = os.path.join(EXP_DIR, "metrics.json")
MANIFEST_FILE = os.path.join(EXP_DIR, "manifest.json")


def run_w3(n_samples: int = 100, seed: int = 42):
    print("=" * 80)
    print(f"EXPERIMENT W3: ADAPTIVE STOPPING UNDER WEB EVIDENCE NOISE")
    print(f"2x2 Factorial: [Fixed Depth vs. Adaptive SPRT] x [Clean Evidence vs. Noisy Distractors]")
    print(f"Dataset: Open-Book HotpotQA ({n_samples} samples)")
    print("=" * 80)
    start_time = time.time()

    import subprocess
    git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=BASE_DIR).decode().strip()
    manifest = {
        "experiment": "W3_adaptive_stopping",
        "dataset": "Open-Book HotpotQA (fullwiki)",
        "dataset_examples": n_samples,
        "model": "gpt-oss-120b",
        "provider": "Azure OpenAI",
        "temperature": 0.0,
        "n_agents": 3,
        "max_rounds": 3,
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
    active_edges = [(0, 1), (1, 2)]

    for sample in tqdm(samples, desc="W3 Evaluation"):
        qid = sample["id"]
        if qid in completed:
            continue

        question = sample["question"]
        gold_answer = sample["answer"]
        articles = sample["articles"]

        # 1. Clean Evidence: top-2 BM25 articles
        retriever = LocalWikipediaRetriever(articles)
        clean_docs = retriever.retrieve(question, top_k=2)

        # 2. Noisy Distractor Evidence: all 10 candidate articles
        noisy_docs = articles

        def run_condition(docs: List[Dict[str, Any]], stopping_mode: str, noise_tag: str) -> Dict[str, Any]:
            t_cond_start = time.perf_counter()
            # Round 0 solve
            r0_resps = []
            for aid in range(config.n_agents):
                sys_p, usr_p = build_agent_prompt(
                    agent_id=aid,
                    question=question,
                    evidence_passages=docs,
                    peer_messages=None,
                    round_num=0,
                )
                cache_tag = f"[{noise_tag.upper()}]\n{usr_p}"
                resp = invoke_agent_with_cache(backend, aid, sys_p, cache_tag, round_num=0, call_tracker=call_tracker)
                r0_resps.append(resp)

            current_resps = list(r0_resps)
            prev_W = np.zeros((config.n_agents, config.n_agents))
            total_comm = 0
            halt_round = config.n_rounds
            halted_early = False

            for r in range(1, config.n_rounds + 1):
                if stopping_mode == "adaptive":
                    should_halt, _, _ = evaluate_stopping_rule(
                        controller, current_resps, prev_W, round_num=r - 1, config=config
                    )
                    if should_halt and r > 1:
                        halt_round = r - 1
                        halted_early = True
                        break

                received = {i: [] for i in range(config.n_agents)}
                for u, v in active_edges:
                    msg, tok = select_sentences_for_edge(current_resps[u].text, budget=50)
                    total_comm += tok
                    if msg.strip():
                        received[v].append(f"Agent {u}: {msg}")

                next_resps = []
                for aid in range(config.n_agents):
                    msgs = received[aid]
                    if not msgs or not any(m.strip() for m in msgs):
                        next_resps.append(current_resps[aid])
                    else:
                        sys_p, usr_p = build_agent_prompt(
                            agent_id=aid,
                            question=question,
                            evidence_passages=docs,
                            peer_messages=msgs,
                            round_num=r,
                        )
                        cache_tag = f"[{noise_tag.upper()}|{stopping_mode.upper()}]\n{usr_p}"
                        new_resp = invoke_agent_with_cache(backend, aid, sys_p, cache_tag, round_num=r, call_tracker=call_tracker)
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
                "rounds": halt_round,
                "halted_early": halted_early,
                "latency_sec": round(cond_latency, 3),
            }

        clean_fixed = run_condition(clean_docs, "fixed", "clean")
        clean_adaptive = run_condition(clean_docs, "adaptive", "clean")
        noisy_fixed = run_condition(noisy_docs, "fixed", "noisy")
        noisy_adaptive = run_condition(noisy_docs, "adaptive", "noisy")

        record = {
            "id": qid,
            "question": question,
            "gold_answer": gold_answer,
            "Clean_Fixed": clean_fixed,
            "Clean_Adaptive": clean_adaptive,
            "Noisy_Fixed": noisy_fixed,
            "Noisy_Adaptive": noisy_adaptive,
        }
        completed[qid] = record
        with open(STREAM_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    _GLOBAL_LLM_CACHE.flush()

    records = list(completed.values())
    metrics = {
        "n_samples": len(records),
        "Clean_Fixed": {
            "exact_match": float(np.mean([r["Clean_Fixed"]["is_em"] for r in records])),
            "f1_score": float(np.mean([r["Clean_Fixed"]["f1"] for r in records])),
            "avg_comm_tokens": float(np.mean([r["Clean_Fixed"]["comm_tokens"] for r in records])),
            "avg_rounds": float(np.mean([r["Clean_Fixed"]["rounds"] for r in records])),
            "avg_latency_seconds": float(np.mean([r["Clean_Fixed"]["latency_sec"] for r in records])),
        },
        "Clean_Adaptive": {
            "exact_match": float(np.mean([r["Clean_Adaptive"]["is_em"] for r in records])),
            "f1_score": float(np.mean([r["Clean_Adaptive"]["f1"] for r in records])),
            "avg_comm_tokens": float(np.mean([r["Clean_Adaptive"]["comm_tokens"] for r in records])),
            "avg_rounds": float(np.mean([r["Clean_Adaptive"]["rounds"] for r in records])),
            "avg_latency_seconds": float(np.mean([r["Clean_Adaptive"]["latency_sec"] for r in records])),
            "early_exit_rate": float(np.mean([r["Clean_Adaptive"]["halted_early"] for r in records])),
        },
        "Noisy_Fixed": {
            "exact_match": float(np.mean([r["Noisy_Fixed"]["is_em"] for r in records])),
            "f1_score": float(np.mean([r["Noisy_Fixed"]["f1"] for r in records])),
            "avg_comm_tokens": float(np.mean([r["Noisy_Fixed"]["comm_tokens"] for r in records])),
            "avg_rounds": float(np.mean([r["Noisy_Fixed"]["rounds"] for r in records])),
            "avg_latency_seconds": float(np.mean([r["Noisy_Fixed"]["latency_sec"] for r in records])),
        },
        "Noisy_Adaptive": {
            "exact_match": float(np.mean([r["Noisy_Adaptive"]["is_em"] for r in records])),
            "f1_score": float(np.mean([r["Noisy_Adaptive"]["f1"] for r in records])),
            "avg_comm_tokens": float(np.mean([r["Noisy_Adaptive"]["comm_tokens"] for r in records])),
            "avg_rounds": float(np.mean([r["Noisy_Adaptive"]["rounds"] for r in records])),
            "avg_latency_seconds": float(np.mean([r["Noisy_Adaptive"]["latency_sec"] for r in records])),
            "early_exit_rate": float(np.mean([r["Noisy_Adaptive"]["halted_early"] for r in records])),
        },
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
    print("W3 EXPERIMENT COMPLETE")
    print(json.dumps(metrics, indent=2))
    print("=" * 80)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_w3(n_samples=args.samples, seed=args.seed)
