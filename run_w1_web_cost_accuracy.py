#!/usr/bin/env python3
"""
Experiment W1: Web Cost-Accuracy Frontier (WWW 2027)
===================================================
Evaluates multi-agent communication efficiency in genuine Web Information Seeking:
  - Dataset: Open-Book HotpotQA (100 validated questions from web_eval_subset_100.json)
  - Conditions:
      1. NoComm: Round 0 independent solving + Platt-calibrated majority vote (0 comm tokens)
      2. MAD (Multi-Agent Debate): Fully connected broadcast across all 3 rounds (T=3)
      3. SPARSECOMM: Learned PPO controller routing + sentence-preserving semantic filtering + SPRT adaptive stopping
  - Checkpointing: Streaming JSONL records, fully resumable
  - Metrics: F1, Exact Match, Inter-agent tokens transmitted, Generation tokens, Realized rounds
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

EXP_DIR = os.path.join(BASE_DIR, "results", "W1_web_cost_accuracy")
os.makedirs(EXP_DIR, exist_ok=True)
STREAM_FILE = os.path.join(EXP_DIR, "stream.jsonl")
METRICS_FILE = os.path.join(EXP_DIR, "metrics.json")
MANIFEST_FILE = os.path.join(EXP_DIR, "manifest.json")


def run_w1(n_samples: int = 100, seed: int = 42):
    print("=" * 80)
    print(f"EXPERIMENT W1: WEB COST-ACCURACY FRONTIER")
    print(f"Conditions: NoComm vs. MAD (Fully Connected) vs. SPARSECOMM")
    print(f"Dataset: Open-Book HotpotQA (Target: {n_samples} questions)")
    print("=" * 80)
    start_time = time.time()

    # Provenance manifest
    import subprocess
    git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=BASE_DIR).decode().strip()
    manifest = {
        "experiment": "W1_web_cost_accuracy",
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

    # 1. Load validated subset
    subset_path = os.path.join(BASE_DIR, "web_eval_subset_100.json")
    with open(subset_path, "r", encoding="utf-8") as f:
        samples = json.load(f)[:n_samples]

    config = get_hotpotqa_config(seed=seed)
    set_seed(seed)
    backend = AzureOpenAIBackend(temperature=0.0)

    # Load pretrained controller
    ckpt_path = os.path.join(BASE_DIR, "08-06results", "hotpotqa_pretrained_400.pt.npz")
    controller = IBController(config)
    controller.load_model(ckpt_path)

    # Check already completed questions
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

    for sample in tqdm(samples, desc="W1 Evaluation"):
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
        # 1. Round 0 Initial Solve (Shared across conditions)
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

        r0_gen_tokens = sum(r.token_count for r in round0_responses)

        # -------------------------------------------------------------
        # Condition A: NoComm (0 communication tokens)
        # -------------------------------------------------------------
        t_nc_start = time.perf_counter()
        nocomm_ans = aggregate_responses(round0_responses, calibrator=controller.calibrator)
        nocomm_em = bool(evaluate_answer(nocomm_ans, gold_answer, "hotpotqa"))
        nocomm_f1 = float(hotpotqa_f1_score(nocomm_ans, gold_answer))
        nocomm_latency = time.perf_counter() - t_nc_start + sum(resp.latency_ms for resp in round0_responses) / 1000.0

        # -------------------------------------------------------------
        # Condition B: MAD (Multi-Agent Debate / Fully Connected)
        # -------------------------------------------------------------
        t_mad_start = time.perf_counter()
        mad_responses = list(round0_responses)
        mad_comm_tokens = 0
        mad_gen_tokens = r0_gen_tokens
        all_edges = [(u, v) for u in range(config.n_agents) for v in range(config.n_agents) if u != v]

        for r in range(1, config.n_rounds + 1):
            received = {i: [] for i in range(config.n_agents)}
            for u, v in all_edges:
                msg = mad_responses[u].text
                tok = count_tokens(msg)
                mad_comm_tokens += tok
                received[v].append(f"Agent {u}: {msg}")

            next_resps = []
            for aid in range(config.n_agents):
                sys_p, usr_p = build_agent_prompt(
                    agent_id=aid,
                    question=question,
                    evidence_passages=retrieved_docs,
                    peer_messages=received[aid],
                    round_num=r,
                )
                new_resp = invoke_agent_with_cache(backend, aid, sys_p, usr_p, round_num=r, call_tracker=call_tracker)
                next_resps.append(new_resp)
                mad_gen_tokens += new_resp.token_count
            mad_responses = next_resps

        mad_ans = aggregate_responses(mad_responses, calibrator=controller.calibrator)
        mad_em = bool(evaluate_answer(mad_ans, gold_answer, "hotpotqa"))
        mad_f1 = float(hotpotqa_f1_score(mad_ans, gold_answer))
        mad_latency = time.perf_counter() - t_mad_start

        # -------------------------------------------------------------
        # Condition C: SPARSECOMM (Dynamic Routing + Rationing + Halting)
        # -------------------------------------------------------------
        t_sc_start = time.perf_counter()
        sc_responses = list(round0_responses)
        sc_comm_tokens = 0
        sc_gen_tokens = r0_gen_tokens
        sc_prev_W = np.zeros((config.n_agents, config.n_agents))
        sc_halt_round = config.n_rounds
        sc_halted_early = False

        for r in range(1, config.n_rounds + 1):
            # Adaptive stopping check
            should_halt, _, _ = evaluate_stopping_rule(
                controller, sc_responses, sc_prev_W, round_num=r - 1, config=config
            )
            if should_halt and r > 1:
                sc_halt_round = r - 1
                sc_halted_early = True
                break

            agent_texts = {i: resp.text for i, resp in enumerate(sc_responses)}
            agent_answers = {i: resp.answer for i, resp in enumerate(sc_responses)}
            confs = [resp.confidence for resp in sc_responses]
            unique_ans = len(set(agent_answers.values()))

            W, _, _, _ = controller.get_weights(
                reasoning_texts=agent_texts,
                answers=agent_answers,
                confidence_scores=confs,
                prev_W=sc_prev_W,
                round_num=r,
                unique_answers=unique_ans,
                avg_accuracy=0.5,
                explore=False,
            )

            ranked_edges = sorted(all_edges, key=lambda e: W[e[0]][e[1]], reverse=True)
            active_edges = ranked_edges[:2]

            received = {i: [] for i in range(config.n_agents)}
            for u, v in active_edges:
                filtered_msg, tok = select_sentences_for_edge(sc_responses[u].text, budget=50)
                sc_comm_tokens += tok
                if filtered_msg.strip():
                    received[v].append(f"Agent {u}: {filtered_msg}")

            next_resps = []
            for aid in range(config.n_agents):
                msgs = received[aid]
                if not msgs or not any(m.strip() for m in msgs):
                    next_resps.append(sc_responses[aid])
                else:
                    sys_p, usr_p = build_agent_prompt(
                        agent_id=aid,
                        question=question,
                        evidence_passages=retrieved_docs,
                        peer_messages=msgs,
                        round_num=r,
                    )
                    new_resp = invoke_agent_with_cache(backend, aid, sys_p, usr_p, round_num=r, call_tracker=call_tracker)
                    next_resps.append(new_resp)
                    sc_gen_tokens += new_resp.token_count

            sc_responses = next_resps
            sc_prev_W = W

        sc_ans = aggregate_responses(sc_responses, calibrator=controller.calibrator)
        sc_em = bool(evaluate_answer(sc_ans, gold_answer, "hotpotqa"))
        sc_f1 = float(hotpotqa_f1_score(sc_ans, gold_answer))
        sc_latency = time.perf_counter() - t_sc_start

        record = {
            "id": qid,
            "question": question,
            "gold_answer": gold_answer,
            "NoComm": {
                "answer": nocomm_ans,
                "is_em": nocomm_em,
                "f1": nocomm_f1,
                "comm_tokens": 0,
                "gen_tokens": r0_gen_tokens,
                "rounds": 0,
                "latency_sec": round(nocomm_latency, 3),
            },
            "MAD": {
                "answer": mad_ans,
                "is_em": mad_em,
                "f1": mad_f1,
                "comm_tokens": mad_comm_tokens,
                "gen_tokens": mad_gen_tokens,
                "rounds": config.n_rounds,
                "latency_sec": round(mad_latency, 3),
            },
            "SPARSECOMM": {
                "answer": sc_ans,
                "is_em": sc_em,
                "f1": sc_f1,
                "comm_tokens": sc_comm_tokens,
                "gen_tokens": sc_gen_tokens,
                "rounds": sc_halt_round,
                "halted_early": sc_halted_early,
                "latency_sec": round(sc_latency, 3),
            },
        }
        completed[qid] = record
        with open(STREAM_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    _GLOBAL_LLM_CACHE.flush()

    # Aggregate Metrics
    records = list(completed.values())
    metrics = {
        "n_samples": len(records),
        "NoComm": {
            "exact_match": float(np.mean([r["NoComm"]["is_em"] for r in records])),
            "f1_score": float(np.mean([r["NoComm"]["f1"] for r in records])),
            "avg_comm_tokens": 0.0,
            "avg_gen_tokens": float(np.mean([r["NoComm"]["gen_tokens"] for r in records])),
            "avg_rounds": 0.0,
            "avg_latency_seconds": float(np.mean([r["NoComm"]["latency_sec"] for r in records])),
        },
        "MAD": {
            "exact_match": float(np.mean([r["MAD"]["is_em"] for r in records])),
            "f1_score": float(np.mean([r["MAD"]["f1"] for r in records])),
            "avg_comm_tokens": float(np.mean([r["MAD"]["comm_tokens"] for r in records])),
            "avg_gen_tokens": float(np.mean([r["MAD"]["gen_tokens"] for r in records])),
            "avg_rounds": float(np.mean([r["MAD"]["rounds"] for r in records])),
            "avg_latency_seconds": float(np.mean([r["MAD"]["latency_sec"] for r in records])),
        },
        "SPARSECOMM": {
            "exact_match": float(np.mean([r["SPARSECOMM"]["is_em"] for r in records])),
            "f1_score": float(np.mean([r["SPARSECOMM"]["f1"] for r in records])),
            "avg_comm_tokens": float(np.mean([r["SPARSECOMM"]["comm_tokens"] for r in records])),
            "avg_gen_tokens": float(np.mean([r["SPARSECOMM"]["gen_tokens"] for r in records])),
            "avg_rounds": float(np.mean([r["SPARSECOMM"]["rounds"] for r in records])),
            "avg_latency_seconds": float(np.mean([r["SPARSECOMM"]["latency_sec"] for r in records])),
            "token_reduction_vs_mad": float(
                1.0 - (np.mean([r["SPARSECOMM"]["comm_tokens"] for r in records]) / max(np.mean([r["MAD"]["comm_tokens"] for r in records]), 1.0))
            ),
            "latency_reduction_vs_mad": float(
                1.0 - (np.mean([r["SPARSECOMM"]["latency_sec"] for r in records]) / max(np.mean([r["MAD"]["latency_sec"] for r in records]), 1e-4))
            ),
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
    print("W1 EXPERIMENT COMPLETE")
    print(json.dumps(metrics, indent=2))
    print("=" * 80)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_w1(n_samples=args.samples, seed=args.seed)
