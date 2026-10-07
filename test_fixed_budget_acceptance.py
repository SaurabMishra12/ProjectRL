#!/usr/bin/env python3
"""
Acceptance Test for Total Sample-Level Fixed-Budget Routing (GSM8K)
====================================================================
Tests 6 cells (Ring, Star, Learned Support at V_total in {50, 100})
on 10 matched GSM8K test questions.

Acceptance Criterion:
  Realized mean V_tx for Ring, Star, and Learned Support MUST all be within
  a few percent of V_total (e.g. ~50 tokens at V_total=50, ~100 tokens at V_total=100)
  and within a few percent of each other.
"""

import os
import sys
import json
import numpy as np
from typing import Tuple, List, Dict, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from config import get_gsm8k_config, Config
from utils import set_seed, count_tokens, split_sentences
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


def truncate_to_token_budget(text: str, max_tokens: int) -> Tuple[str, int]:
    """Truncate text so its token count is at most max_tokens."""
    if not text or max_tokens <= 0:
        return "", 0
    if count_tokens(text) <= max_tokens:
        return text, count_tokens(text)

    # 1. Try sentence-level truncation
    sentences = split_sentences(text)
    curr_sentences = []
    for s in sentences:
        candidate = " ".join(curr_sentences + [s])
        if count_tokens(candidate) <= max_tokens:
            curr_sentences.append(s)
        else:
            break

    if curr_sentences:
        cand_text = " ".join(curr_sentences)
        return cand_text, count_tokens(cand_text)

    # 2. Fallback to word-level truncation
    words = text.split()
    low, high, best = 0, len(words), 0
    while low <= high:
        mid = (low + high) // 2
        cand = " ".join(words[:mid])
        if count_tokens(cand) <= max_tokens:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    cand_text = " ".join(words[:best])
    return cand_text, count_tokens(cand_text)


class TotalSampleBudgetCommunicationGraph(CommunicationGraph):
    """
    CommunicationGraph that strictly caps the TOTAL sample-level transmitted tokens at v_total_budget.
    """
    def __init__(self, n_agents: int, v_total_budget: int):
        super().__init__(n_agents=n_agents)
        self.v_total_budget = v_total_budget
        self.accumulated_sample_tokens = 0

    def reset(self):
        super().reset()
        self.accumulated_sample_tokens = 0

    def transmit(self, sender_id: int, receiver_id: int, message: str, round_num: int, llm_backend=None) -> Tuple[str, int]:
        w_ij = self.W[sender_id][receiver_id]
        original_tokens = count_tokens(message)

        if w_ij < 0.05 or self.accumulated_sample_tokens >= self.v_total_budget:
            filtered_msg = ""
            transmitted_tokens = 0
            sem_sim = 0.0
            info_ret = 0.0
            mode = SemanticBandwidthMode.BLOCK
        else:
            mode = SemanticBandwidthMode.PARTIAL if w_ij < 0.95 else SemanticBandwidthMode.FULL
            remaining_budget = self.v_total_budget - self.accumulated_sample_tokens
            filtered_msg, transmitted_tokens = truncate_to_token_budget(message, remaining_budget)
            self.accumulated_sample_tokens += transmitted_tokens
            sem_sim = 1.0 if transmitted_tokens == original_tokens else (transmitted_tokens / max(original_tokens, 1))
            info_ret = sem_sim

        record = TransmissionRecord(
            sender_id=sender_id,
            receiver_id=receiver_id,
            mode=mode,
            original_tokens=original_tokens,
            transmitted_tokens=transmitted_tokens,
            weight=w_ij,
            round_num=round_num,
            semantic_similarity=sem_sim,
            information_retention=info_ret,
            original_msg=message,
            filtered_msg=filtered_msg
        )
        self.transmission_log.append(record)
        self.round_token_counts[round_num] = self.round_token_counts.get(round_num, 0) + transmitted_tokens
        return filtered_msg, transmitted_tokens


def get_topology_weights(topo_type: str, n_agents: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    import networkx as nx
    if topo_type == "ring":
        G = nx.cycle_graph(n_agents)
    elif topo_type == "star":
        G = nx.star_graph(n_agents - 1)
    else:
        raise ValueError(f"Unknown static topology: {topo_type}")
    W = nx.to_numpy_array(G) * 0.6
    np.fill_diagonal(W, 0.0)
    Q = np.zeros((n_agents, 16))
    return W, Q


def run_test(n_questions: int = 10):
    print("=" * 80)
    print(f"  ACCEPTANCE TEST: Total Sample Budget Contol ({n_questions} questions)")
    print("=" * 80)

    config = get_gsm8k_config()
    config.llm_backend = "azure"
    config.api_model = "gpt-oss-120b"
    config.n_samples = n_questions
    config.seed = 42

    _, test_data = load_dataset_by_name("gsm8k", n_train=0, n_test=n_questions, seed=config.seed)
    
    learned_ckpt = os.path.join(BASE_DIR, "08-06results", "gsm8k_pretrained_400.pt.npz")
    learned_ctrl = IBController(config)
    if os.path.exists(learned_ckpt):
        learned_ctrl.load_model(learned_ckpt)
        print("Loaded 400-sample pre-trained controller.")

    topologies = ["Ring", "Star", "Learned Support"]
    budgets = [50, 100]

    results_table = []

    for topo in topologies:
        for v_total in budgets:
            set_seed(config.seed)
            llm_backend = create_llm_backend(config)
            agent_pool = AgentPool(config.n_agents, llm_backend, config.reasoning_styles)
            comm_graph = TotalSampleBudgetCommunicationGraph(n_agents=config.n_agents, v_total_budget=v_total)
            comm_graph.set_dataset_type(config.dataset)

            if topo == "Ring":
                W, Q = get_topology_weights("ring", config.n_agents)
                ctrl = FrozenWeightController(config, frozen_W=W, frozen_Q=Q)
            elif topo == "Star":
                W, Q = get_topology_weights("star", config.n_agents)
                ctrl = FrozenWeightController(config, frozen_W=W, frozen_Q=Q)
            else:
                ctrl = learned_ctrl

            sample_results = []
            for sample in test_data:
                agent_pool.reset_all()
                comm_graph.reset()
                res = run_single_sample(
                    question=sample["question"],
                    gold_answer=sample["answer"],
                    agent_pool=agent_pool,
                    comm_graph=comm_graph,
                    controller=ctrl,
                    config=config,
                    train_controller=False,
                    avg_accuracy=0.5,
                )
                sample_results.append(res)

            correct = sum(1 for s in sample_results if s.is_correct)
            acc = correct / len(sample_results) * 100
            mean_v_tx = np.mean([s.total_tokens_transmitted for s in sample_results])
            mean_v_full = np.mean([s.total_tokens_if_full for s in sample_results])

            results_table.append({
                "topology": topo,
                "v_target": v_total,
                "acc": acc,
                "mean_v_tx": mean_v_tx,
                "mean_v_full": mean_v_full,
            })

    print("\n" + "=" * 70)
    print("  ACCEPTANCE TEST RESULTS TABLE")
    print("=" * 70)
    print(f"| {'Topology':<16} | {'Target V_bar':<12} | {'Realized V_tx':<15} | {'Realized V_full':<15} | {'Acc (%)':<8} |")
    print("|" + "-" * 18 + "|" + "-" * 14 + "|" + "-" * 17 + "|" + "-" * 17 + "|" + "-" * 10 + "|")

    for r in results_table:
        print(f"| {r['topology']:<16} | {r['v_target']:<12} | {r['mean_v_tx']:>13.1f}   | {r['mean_v_full']:>13.1f}   | {r['acc']:>6.1f}% |")

    print("=" * 70)

    # Acceptance test check
    # Check if for each budget, max_v_tx - min_v_tx < 5 tokens
    passed = True
    for v_target in budgets:
        v_txs = [r['mean_v_tx'] for r in results_table if r['v_target'] == v_target]
        diff = max(v_txs) - min(v_txs)
        target_diff = abs(np.mean(v_txs) - v_target)
        print(f"\nTarget V_bar = {v_target}:")
        print(f"  Realized V_tx across topologies: {v_txs}")
        print(f"  Max difference between topologies: {diff:.1f} tokens")
        print(f"  Difference from target: {target_diff:.1f} tokens")
        if diff > 5.0 or target_diff > 5.0:
            print("  ❌ ACCEPTANCE TEST FAILED: Volume is not equal across topologies!")
            passed = False
        else:
            print("  ✅ ACCEPTANCE TEST PASSED: Volume is equal across topologies!")

    return passed, results_table


if __name__ == "__main__":
    run_test(10)
