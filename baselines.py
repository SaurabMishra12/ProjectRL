"""
Baselines: Self-Consistency, Majority Voting, Multi-Agent Debate, Static Routing.

All baselines use the same N agents, same LLM backend, same prompts.
"""

import numpy as np
from typing import List
from tqdm import tqdm

from config import Config
from llm_backend import LLMBackend
from agents import AgentPool, AgentResponse, majority_vote, weighted_majority_vote
from communication import CommunicationGraph
from controller import IBController
from dataset_loaders import load_dataset_by_name, evaluate_answer
from evaluation import SampleResult, ExperimentResults
from pipeline import create_llm_backend
from utils import set_seed, setup_logging

logger = setup_logging()


def run_self_consistency(config: Config) -> ExperimentResults:
    """
    Self-Consistency Baseline: Each agent solves independently (no communication).
    Final answer = majority vote across agents.

    This is equivalent to W=0 for all edges.
    """
    set_seed(config.seed)
    logger.info("Running baseline: Self-Consistency")
    llm_backend = create_llm_backend(config)
    agent_pool = AgentPool(config.n_agents, llm_backend, config.reasoning_styles)

    data = load_dataset_by_name(config.dataset, config.n_samples, config.seed)

    results = ExperimentResults(
        method_name="Self-Consistency",
        dataset_name=config.dataset,
    )

    for sample in tqdm(data, desc="Self-Consistency", disable=len(data) < 5):
        agent_pool.reset_all()

        # Each agent solves independently — no communication
        responses = agent_pool.initial_solve(sample["question"], config.dataset)

        initial_answers = {r.agent_id: r.answer for r in responses}
        initial_correct = {
            r.agent_id: evaluate_answer(r.answer, sample["answer"], config.dataset)
            for r in responses
        }
        total_gen = sum(r.token_count for r in responses)

        # Majority vote
        final = majority_vote(responses)
        is_correct = evaluate_answer(final, sample["answer"], config.dataset)

        sr = SampleResult(
            question=sample["question"],
            gold_answer=sample["answer"],
            final_answer=final,
            is_correct=is_correct,
            initial_answers=initial_answers,
            initial_correct=initial_correct,
            final_answers=initial_answers.copy(),
            total_tokens_transmitted=0,  # no communication
            total_tokens_if_full=0,
            compression_ratio=0.0,
            communication_density=0.0,
            mode_distribution={"BLOCK": config.n_agents * (config.n_agents - 1)},
            n_active_edges=0,
            generation_tokens=total_gen,
        )
        results.sample_results.append(sr)

    return results


def run_majority_voting(config: Config) -> ExperimentResults:
    """
    Majority Voting Baseline: All agents solve independently, take majority vote.
    Uses confidence-weighted voting.
    """
    set_seed(config.seed)
    logger.info("Running baseline: Majority Voting")
    llm_backend = create_llm_backend(config)
    agent_pool = AgentPool(config.n_agents, llm_backend, config.reasoning_styles)

    data = load_dataset_by_name(config.dataset, config.n_samples, config.seed)

    results = ExperimentResults(
        method_name="Majority Voting",
        dataset_name=config.dataset,
    )

    for sample in tqdm(data, desc="Majority Voting", disable=len(data) < 5):
        agent_pool.reset_all()
        responses = agent_pool.initial_solve(sample["question"], config.dataset)

        initial_answers = {r.agent_id: r.answer for r in responses}
        initial_correct = {
            r.agent_id: evaluate_answer(r.answer, sample["answer"], config.dataset)
            for r in responses
        }
        total_gen = sum(r.token_count for r in responses)

        # Weighted majority vote
        final = weighted_majority_vote(responses)
        is_correct = evaluate_answer(final, sample["answer"], config.dataset)

        sr = SampleResult(
            question=sample["question"],
            gold_answer=sample["answer"],
            final_answer=final,
            is_correct=is_correct,
            initial_answers=initial_answers,
            initial_correct=initial_correct,
            final_answers=initial_answers.copy(),
            total_tokens_transmitted=0,
            total_tokens_if_full=0,
            compression_ratio=0.0,
            communication_density=0.0,
            mode_distribution={"BLOCK": config.n_agents * (config.n_agents - 1)},
            n_active_edges=0,
            generation_tokens=total_gen,
        )
        results.sample_results.append(sr)

    return results


def run_multi_agent_debate(config: Config) -> ExperimentResults:
    """
    Multi-Agent Debate: Full communication (W=1.0) for T rounds.
    """
    from pipeline import run_full_communication
    logger.info("Running baseline: Multi-Agent Debate")
    return run_full_communication(config)


def run_static_routing(config: Config) -> ExperimentResults:
    """
    Static Routing: Fixed weight matrix (ring topology, W=0.6) for all rounds.
    No learning, no adaptation.
    """
    set_seed(config.seed)
    logger.info("Running baseline: Static Routing")
    llm_backend = create_llm_backend(config)
    agent_pool = AgentPool(config.n_agents, llm_backend, config.reasoning_styles)
    comm_graph = CommunicationGraph(config.n_agents)
    controller = IBController(config)

    # Fixed ring topology
    static_W = controller.get_static_weights("ring")

    data = load_dataset_by_name(config.dataset, config.n_samples, config.seed)

    results = ExperimentResults(
        method_name="Static Routing",
        dataset_name=config.dataset,
    )

    for sample in tqdm(data, desc="Static Routing", disable=len(data) < 5):
        agent_pool.reset_all()
        comm_graph.reset()

        # Round 0
        r0 = agent_pool.initial_solve(sample["question"], config.dataset)
        initial_answers = {r.agent_id: r.answer for r in r0}
        initial_correct = {
            r.agent_id: evaluate_answer(r.answer, sample["answer"], config.dataset)
            for r in r0
        }
        total_gen = sum(r.token_count for r in r0)

        current = r0
        for t in range(1, config.n_rounds + 1):
            comm_graph.set_weights(static_W)
            texts = {r.agent_id: r.reasoning_text for r in current}
            msgs = comm_graph.get_messages_for_round(texts, t)
            current = agent_pool.communicate_and_solve(
                sample["question"], msgs, config.dataset, t
            )
            total_gen += sum(r.token_count for r in current)

        final = weighted_majority_vote(current)
        is_correct = evaluate_answer(final, sample["answer"], config.dataset)

        sr = SampleResult(
            question=sample["question"],
            gold_answer=sample["answer"],
            final_answer=final,
            is_correct=is_correct,
            initial_answers=initial_answers,
            initial_correct=initial_correct,
            final_answers={r.agent_id: r.answer for r in current},
            total_tokens_transmitted=comm_graph.get_total_tokens_transmitted(),
            total_tokens_if_full=comm_graph.get_total_tokens_if_full(),
            compression_ratio=comm_graph.get_compression_ratio(),
            weight_matrices=[static_W.copy()],
            communication_density=comm_graph.get_communication_density(),
            mode_distribution=comm_graph.get_mode_distribution(),
            n_active_edges=comm_graph.get_active_edges(),
            generation_tokens=total_gen,
        )
        results.sample_results.append(sr)

    results.final_W = static_W
    return results


def run_all_baselines(config: Config) -> List[ExperimentResults]:
    """Run all 4 baselines and return results."""
    logger.info("=" * 50)
    logger.info("Running all baselines")
    logger.info("=" * 50)

    results = []
    results.append(run_self_consistency(config))
    results.append(run_majority_voting(config))
    results.append(run_multi_agent_debate(config))
    results.append(run_static_routing(config))

    return results
