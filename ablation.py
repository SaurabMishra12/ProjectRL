"""
Ablation Studies: Systematically remove components to measure their contribution.

Four ablation conditions:
1. lambda_zero: No information penalty (lambda=0)
2. no_summarize: Summarize mode disabled -> maps to BLOCK
3. no_filter: Filter mode disabled -> maps to BLOCK
4. random_controller: Random W each round (no learning)
"""

import copy
import numpy as np
from typing import List, Dict
from tqdm import tqdm

from config import Config
from communication import CommunicationGraph, SemanticBandwidthMode
from controller import IBController
from pipeline import (
    create_llm_backend,
    run_single_sample,
    run_experiment,
)
from agents import AgentPool, weighted_majority_vote
from dataset_loaders import load_dataset_by_name, evaluate_answer
from evaluation import SampleResult, ExperimentResults
from utils import set_seed, setup_logging

logger = setup_logging()


def run_ablation_lambda_zero(config: Config) -> ExperimentResults:
    """Ablation: Set lambda=0 (no information penalty)."""
    logger.info("Ablation: lambda=0 (no information penalty)")
    abl_config = copy.deepcopy(config)
    abl_config.lambda_info = 0.0
    results = run_experiment(abl_config, method_name="Ablation: lambda=0", train=True)
    return results


def run_ablation_no_summarize(config: Config) -> ExperimentResults:
    """Ablation: Disable SUMMARIZE mode (maps to BLOCK)."""
    logger.info("Ablation: No SUMMARIZE mode (Disabled due to refactor)")
    return ExperimentResults(method_name="Ablation: No Summarize", dataset_name=config.dataset)


def run_ablation_no_filter(config: Config) -> ExperimentResults:
    """Ablation: Disable FILTER mode (maps to BLOCK)."""
    logger.info("Ablation: No FILTER mode (Disabled due to refactor)")
    return ExperimentResults(method_name="Ablation: No Filter", dataset_name=config.dataset)


def run_ablation_random_controller(config: Config) -> ExperimentResults:
    """Ablation: Random weights each round (no learned controller)."""
    logger.info("Ablation: Random Controller")
    set_seed(config.seed)

    llm_backend = create_llm_backend(config)
    agent_pool = AgentPool(config.n_agents, llm_backend, config.reasoning_styles)
    comm_graph = CommunicationGraph(config.n_agents)

    data = load_dataset_by_name(config.dataset, config.n_samples, config.seed)

    results = ExperimentResults(
        method_name="Ablation: Random Controller",
        dataset_name=config.dataset,
    )

    for sample in tqdm(data, desc="Random Ctrl", disable=len(data) < 5):
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
            # Random weights each round
            random_W = np.random.rand(config.n_agents, config.n_agents)
            np.fill_diagonal(random_W, 0)
            comm_graph.set_weights(random_W)

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
            communication_density=comm_graph.get_communication_density(),
            mode_distribution=comm_graph.get_mode_distribution(),
            n_active_edges=comm_graph.get_active_edges(),
            generation_tokens=total_gen,
        )
        results.sample_results.append(sr)

    return results


def run_all_ablations(config: Config) -> List[ExperimentResults]:
    """Run all 4 ablation studies."""
    logger.info("=" * 50)
    logger.info("Running all ablation studies")
    logger.info("=" * 50)

    results = []
    results.append(run_ablation_lambda_zero(config))
    results.append(run_ablation_no_summarize(config))
    results.append(run_ablation_no_filter(config))
    results.append(run_ablation_random_controller(config))

    return results
