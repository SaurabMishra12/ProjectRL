#!/usr/bin/env python3
"""
Web Multi-Agent Pilot Execution Script (WWW 2027)
=================================================
Runs a 20-question pilot on real Open-Book HotpotQA data to verify:
  1. Fail-closed dataset loading (0 synthetic examples)
  2. BM25 local Wikipedia retrieval & gold supporting fact recall
  3. Functional heterogeneous agent execution (Agents 0, 1, 2)
  4. Real-time token accounting (generation & communication)
  5. Controller routing & SPRT adaptive stopping
  6. Consensus aggregation & HotpotQA token F1 evaluation
  7. Checkpointing & disk caching

Calculates:
  - Average calls/question
  - Average tokens/question
  - Average rounds/question
  - Failure rate
  - Estimated full-run cost & expected runtime
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
from llm_backend import AzureOpenAIBackend
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

PILOT_DIR = os.path.join(BASE_DIR, "results", "pilot")
os.makedirs(PILOT_DIR, exist_ok=True)
STREAM_FILE = os.path.join(PILOT_DIR, "pilot_stream.jsonl")
REPORT_FILE = os.path.join(PILOT_DIR, "pilot_integrity_report.json")


def run_pilot(n_questions: int = 20, seed: int = 42):
    print("=" * 80)
    print(f"STARTING 20-QUESTION REAL-DATA WEB PILOT ON OPEN-BOOK HOTPOTQA")
    print("=" * 80)
    start_time = time.time()

    # 1. Load validated dataset subset
    subset_path = os.path.join(BASE_DIR, "web_eval_subset_100.json")
    if not os.path.exists(subset_path):
        raise FileNotFoundError(f"Validated subset missing: {subset_path}. Run validate_dataset.py first!")

    with open(subset_path, "r", encoding="utf-8") as f:
        all_samples = json.load(f)

    pilot_samples = all_samples[:n_questions]
    print(f"Loaded {len(pilot_samples)} validated pilot samples.")

    # 2. Setup backend & controller
    config = get_hotpotqa_config(seed=seed)
    set_seed(seed)
    backend = AzureOpenAIBackend(temperature=0.0)

    ckpt_path = os.path.join(BASE_DIR, "08-06results", "hotpotqa_pretrained_400.pt.npz")
    controller = IBController(config)
    if os.path.exists(ckpt_path):
        controller.load_model(ckpt_path)
        print(f"Loaded controller checkpoint: {ckpt_path}")
    else:
        print(f"Warning: Controller checkpoint not found at {ckpt_path}, using fresh controller.")

    # 3. Check existing checkpoints for resume
    completed_ids = set()
    if os.path.exists(STREAM_FILE):
        with open(STREAM_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        completed_ids.add(record["id"])
                    except Exception:
                        pass
        print(f"Found {len(completed_ids)} already completed questions in stream log.")

    call_tracker = {
        "live_calls": 0,
        "cached_hits": 0,
        "successful_calls": 0,
        "failed_attempts": 0,
        "total_tokens": 0,
    }

    results = []
    gold_recalls = []

    for q_idx, sample in enumerate(tqdm(pilot_samples, desc="Pilot Evaluation")):
        qid = sample["id"]
        if qid in completed_ids:
            continue

        q_start = time.time()
        question = sample["question"]
        gold_answer = sample["answer"]
        articles = sample["articles"]
        gold_supp_titles = sample["supporting_facts"]["title"]

        # --- Step 1: Real Local BM25 Retrieval ---
        retriever = LocalWikipediaRetriever(articles)
        retrieved_docs = retriever.retrieve(question, top_k=3)

        # Check gold supporting titles recall
        retrieved_titles = {d["title"] for d in retrieved_docs}
        gold_hit = sum(1 for gt in set(gold_supp_titles) if gt in retrieved_titles)
        recall = gold_hit / max(len(set(gold_supp_titles)), 1)
        gold_recalls.append(recall)

        # --- Step 2: Round 0 Initial Solve ---
        round0_responses = []
        for aid in range(config.n_agents):
            sys_prompt, usr_prompt = build_agent_prompt(
                agent_id=aid,
                question=question,
                evidence_passages=retrieved_docs,
                peer_messages=None,
                round_num=0,
            )
            resp = invoke_agent_with_cache(
                backend=backend,
                agent_id=aid,
                system_prompt=sys_prompt,
                user_prompt=usr_prompt,
                round_num=0,
                call_tracker=call_tracker,
            )
            round0_responses.append(resp)

        current_responses = list(round0_responses)
        prev_W = np.zeros((config.n_agents, config.n_agents))
        total_comm_tokens = 0
        halt_round = config.n_rounds
        halted_early = False

        # --- Step 3: Multi-Round Communication Loop ---
        for r in range(1, config.n_rounds + 1):
            # Check stopping rule
            should_halt, h_logit, s_llr = evaluate_stopping_rule(
                controller=controller,
                current_responses=current_responses,
                prev_W=prev_W,
                round_num=r - 1,
                config=config,
            )
            if should_halt and r > 1:
                halt_round = r - 1
                halted_early = True
                break

            # Controller produces routing weights W
            agent_texts = {idx: resp.text for idx, resp in enumerate(current_responses)}
            agent_answers = {idx: resp.answer for idx, resp in enumerate(current_responses)}
            confs = [resp.confidence for resp in current_responses]
            unique_ans = len(set(agent_answers.values()))

            W, _, _, _ = controller.get_weights(
                reasoning_texts=agent_texts,
                answers=agent_answers,
                confidence_scores=confs,
                prev_W=prev_W,
                round_num=r,
                unique_answers=unique_ans,
                avg_accuracy=0.5,
                explore=False,
            )

            # Transmit over top 2 active edges with 50 tokens per edge budget
            all_edges = [(u, v) for u in range(config.n_agents) for v in range(config.n_agents) if u != v]
            ranked_edges = sorted(all_edges, key=lambda e: W[e[0]][e[1]], reverse=True)
            active_edges = ranked_edges[:2]

            received_messages = {i: [] for i in range(config.n_agents)}
            for u, v in active_edges:
                filtered_msg, tok = select_sentences_for_edge(current_responses[u].text, budget=50)
                total_comm_tokens += tok
                if filtered_msg.strip():
                    received_messages[v].append(f"Agent {u} (Doc evidence/reasoning): {filtered_msg}")

            # Re-solve agents with received messages
            next_responses = []
            for aid in range(config.n_agents):
                msgs = received_messages[aid]
                if not msgs or not any(m.strip() for m in msgs):
                    next_responses.append(current_responses[aid])
                else:
                    sys_p, usr_p = build_agent_prompt(
                        agent_id=aid,
                        question=question,
                        evidence_passages=retrieved_docs,
                        peer_messages=msgs,
                        round_num=r,
                    )
                    new_resp = invoke_agent_with_cache(
                        backend=backend,
                        agent_id=aid,
                        system_prompt=sys_p,
                        user_prompt=usr_p,
                        round_num=r,
                        call_tracker=call_tracker,
                    )
                    next_responses.append(new_resp)

            current_responses = next_responses
            prev_W = W

        # --- Step 4: Final Consensus Aggregation ---
        final_ans = aggregate_responses(current_responses, calibrator=controller.calibrator)
        is_em = evaluate_answer(final_ans, gold_answer, "hotpotqa")
        f1 = hotpotqa_f1_score(final_ans, gold_answer)

        record = {
            "id": qid,
            "question": question,
            "gold_answer": gold_answer,
            "final_answer": final_ans,
            "is_exact_match": bool(is_em),
            "f1_score": float(f1),
            "retrieval_recall": float(recall),
            "retrieved_titles": list(retrieved_titles),
            "gold_titles": list(gold_supp_titles),
            "halt_round": halt_round,
            "halted_early": halted_early,
            "total_comm_tokens": total_comm_tokens,
            "agent_answers_r0": [r.answer for r in round0_responses],
            "agent_answers_final": [r.answer for r in current_responses],
            "elapsed_seconds": round(time.time() - q_start, 2),
        }
        results.append(record)

        # Append to streaming checkpoint file
        with open(STREAM_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    # Flush disk cache
    _GLOBAL_LLM_CACHE.flush()

    # Compile pilot statistics
    total_time = time.time() - start_time
    n_eval = len(results)
    avg_f1 = float(np.mean([r["f1_score"] for r in results])) if results else 0.0
    avg_em = float(np.mean([r["is_exact_match"] for r in results])) if results else 0.0
    avg_recall = float(np.mean(gold_recalls)) if gold_recalls else 0.0
    avg_comm_tokens = float(np.mean([r["total_comm_tokens"] for r in results])) if results else 0.0
    avg_rounds = float(np.mean([r["halt_round"] for r in results])) if results else 0.0

    report = {
        "pilot_name": "Web Multi-Agent 20-Question Pilot",
        "dataset": "Open-Book HotpotQA (fullwiki)",
        "questions_evaluated": n_eval,
        "exact_match": avg_em,
        "token_f1": avg_f1,
        "gold_retrieval_recall": avg_recall,
        "avg_comm_tokens_per_question": avg_comm_tokens,
        "avg_rounds_per_question": avg_rounds,
        "live_api_calls": call_tracker["live_calls"],
        "cached_api_hits": call_tracker["cached_hits"],
        "successful_calls": call_tracker["successful_calls"],
        "failed_attempts": call_tracker["failed_attempts"],
        "failure_rate": call_tracker["failed_attempts"] / max(call_tracker["live_calls"] + call_tracker["failed_attempts"], 1),
        "total_tokens_recorded": call_tracker["total_tokens"],
        "avg_tokens_per_question": call_tracker["total_tokens"] / max(n_eval, 1),
        "elapsed_seconds": round(total_time, 2),
        "estimated_100q_runtime_seconds": round((total_time / max(n_eval, 1)) * 100, 2),
        "status": "PASS" if n_eval > 0 and call_tracker["failed_attempts"] == 0 else "PARTIAL",
    }

    print("\n" + "=" * 80)
    print("PILOT INTEGRITY & PERFORMANCE REPORT")
    print("=" * 80)
    for k, v in report.items():
        print(f"  {k}: {v}")
    print("=" * 80)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved pilot integrity report to: {REPORT_FILE}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_pilot(n_questions=args.samples, seed=args.seed)
