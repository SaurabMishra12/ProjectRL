"""
Evaluation Metrics: Accuracy, token cost, communication density,
error propagation rate, and results formatting.
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field
import numpy as np


@dataclass
class SampleResult:
    """Result for a single sample (question)."""
    question: str
    gold_answer: str
    final_answer: str
    is_correct: bool
    exact_match: float = 0.0
    f1_score: float = 0.0

    # Per-agent info
    initial_answers: Dict[int, str] = field(default_factory=dict)
    initial_correct: Dict[int, bool] = field(default_factory=dict)
    final_answers: Dict[int, str] = field(default_factory=dict)
    initial_reasoning: Dict[int, str] = field(default_factory=dict)
    final_reasoning: Dict[int, str] = field(default_factory=dict)
    raw_communications: List[Dict[str, Any]] = field(default_factory=list)

    # Communication stats
    total_tokens_transmitted: int = 0
    total_tokens_if_full: int = 0
    compression_ratio: float = 0.0
    weight_matrices: List[Any] = field(default_factory=list)
    communication_density: float = 0.0
    mode_distribution: Dict[str, int] = field(default_factory=dict)
    n_active_edges: int = 0

    # Token cost (generation + communication)
    generation_tokens: int = 0
    halt_round: int = 3

    # Semantic Compression Metrics
    semantic_similarity: float = 0.0
    token_reduction: float = 0.0
    information_retention: float = 0.0

    # Diagnostic Metrics
    agent_diversity: float = 0.0
    answer_entropy: float = 0.0
    average_delta_w: float = 0.0
    initial_majority_correct: bool = False


@dataclass
class ExperimentResults:
    """Aggregated results for an entire experiment run."""
    method_name: str
    dataset_name: str
    sample_results: List[SampleResult] = field(default_factory=list)
    training_rewards: List[float] = field(default_factory=list)
    training_losses: List[float] = field(default_factory=list)
    final_W: Any = None  # final weight matrix after training


def compute_accuracy(results: ExperimentResults) -> float:
    """Percentage of correct final answers."""
    if not results.sample_results:
        return 0.0
    correct = sum(1 for r in results.sample_results if r.is_correct)
    return correct / len(results.sample_results)


def compute_token_cost(results: ExperimentResults) -> Dict[str, float]:
    """
    Token cost statistics.

    Returns:
        {
            "total_tokens": total across all samples,
            "avg_tokens_per_sample": average per sample,
            "total_comm_tokens": communication tokens only,
            "total_gen_tokens": generation tokens only,
        }
    """
    total_comm = sum(r.total_tokens_transmitted for r in results.sample_results)
    total_gen = sum(r.generation_tokens for r in results.sample_results)
    total = total_comm + total_gen
    n = len(results.sample_results) or 1

    return {
        "total_tokens": total,
        "avg_tokens_per_sample": total / n,
        "total_comm_tokens": total_comm,
        "total_gen_tokens": total_gen,
        "avg_comm_tokens": total_comm / n,
    }


def compute_communication_density(results: ExperimentResults) -> float:
    """
    Average communication density: mean of all W values (excluding diagonal).
    Higher = more communication; lower = sparser.
    """
    if not results.sample_results:
        return 0.0
    densities = [r.communication_density for r in results.sample_results]
    return float(np.mean(densities))


def compute_error_propagation_rate(results: ExperimentResults) -> float:
    """
    Fraction of cases where at least one agent was initially correct,
    but the final answer is incorrect.

    This measures how often communication introduces errors.
    """
    if not results.sample_results:
        return 0.0

    propagation_cases = 0
    eligible_cases = 0

    for r in results.sample_results:
        if not r.initial_correct:
            continue
        # Check if at least one agent was initially correct
        any_initially_correct = any(r.initial_correct.values())
        if any_initially_correct:
            eligible_cases += 1
            if not r.is_correct:
                propagation_cases += 1

    if eligible_cases == 0:
        return 0.0
    return propagation_cases / eligible_cases


def compute_mode_distribution(results: ExperimentResults) -> Dict[str, float]:
    """Average mode distribution across all samples."""
    all_dists = [r.mode_distribution for r in results.sample_results
                 if r.mode_distribution]
    if not all_dists:
        return {}

    combined = {}
    for dist in all_dists:
        for mode, count in dist.items():
            combined[mode] = combined.get(mode, 0) + count

    total = sum(combined.values()) or 1
    return {mode: count / total for mode, count in combined.items()}


def aggregate_metrics(results: ExperimentResults) -> Dict[str, Any]:
    """Compute all metrics for an experiment run."""
    accuracy = compute_accuracy(results)
    token_cost = compute_token_cost(results)
    density = compute_communication_density(results)
    error_prop = compute_error_propagation_rate(results)
    mode_dist = compute_mode_distribution(results)

    n = len(results.sample_results) or 1
    se = np.sqrt(accuracy * (1.0 - accuracy) / n)
    ci95 = 1.96 * se * 100
    return {
        "method": results.method_name,
        "dataset": results.dataset_name,
        "accuracy": accuracy,
        "accuracy_pct": f"{accuracy * 100:.1f}% ± {ci95:.1f}%",
        "token_cost": token_cost,
        "communication_density": density,
        "error_propagation_rate": error_prop,
        "mode_distribution": mode_dist,
        "average_rounds_used": float(np.mean([r.halt_round for r in results.sample_results])) if results.sample_results else 0.0,
        "semantic_similarity": float(np.mean([r.semantic_similarity for r in results.sample_results])) if results.sample_results else 0.0,
        "token_reduction": float(np.mean([r.token_reduction for r in results.sample_results])) if results.sample_results else 0.0,
        "information_retention": float(np.mean([r.information_retention for r in results.sample_results])) if results.sample_results else 0.0,
        "n_samples": len(results.sample_results),
    }


def format_results_table(
    all_metrics: List[Dict[str, Any]],
    dataset_name: str = "",
) -> str:
    """
    Format results as a markdown table.

    Columns: Method | Accuracy | Avg Tokens | Comm Density | Error Prop Rate
    """
    header = (
        f"| Method | Accuracy | Avg Tokens | "
        f"Comm Density | Avg Rounds | Error Prop | Sem. Sim | Info Ret |\n"
        f"|--------|----------|------------|"
        f"-------------|------------|------------|----------|----------|\n"
    )

    rows = []
    for m in all_metrics:
        row = (
            f"| {m['method']:<20s} "
            f"| {m['accuracy_pct']:>8s} "
            f"| {m['token_cost']['avg_tokens_per_sample']:>10.1f} "
            f"| {m['communication_density']:>11.3f} "
            f"| {m['average_rounds_used']:>10.2f} "
            f"| {m['error_propagation_rate']:>10.3f} "
            f"| {m['semantic_similarity']:>8.3f} "
            f"| {m['information_retention']:>8.3f} |"
        )
        rows.append(row)

    title = f"\n### Results — {dataset_name}\n\n" if dataset_name else "\n"
    return title + header + "\n".join(rows) + "\n"


def format_ablation_table(
    all_metrics: List[Dict[str, Any]],
    dataset_name: str = "",
) -> str:
    """Format ablation results table."""
    header = (
        f"| Ablation | Accuracy | Avg Tokens | Density | Mode Dist |\n"
        f"|----------|----------|------------|---------|----------|\n"
    )

    rows = []
    for m in all_metrics:
        mode_str = ", ".join(
            f"{k}:{v:.0%}" for k, v in sorted(m.get("mode_distribution", {}).items())
        )
        row = (
            f"| {m['method']:<25s} "
            f"| {m['accuracy_pct']:>8s} "
            f"| {m['token_cost']['avg_tokens_per_sample']:>10.1f} "
            f"| {m['communication_density']:>7.3f} "
            f"| {mode_str} |"
        )
        rows.append(row)

    title = f"\n### Ablation — {dataset_name}\n\n" if dataset_name else "\n"
    return title + header + "\n".join(rows) + "\n"
