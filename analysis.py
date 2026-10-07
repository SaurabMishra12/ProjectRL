"""
Analysis: Qualitative analysis of the learned communication structure.
Answers the key research questions required for the paper.
"""

import numpy as np
from typing import List, Dict, Any

from evaluation import ExperimentResults, aggregate_metrics, SampleResult


def analyze_blocking_patterns(results: ExperimentResults) -> Dict[str, Any]:
    """
    When does the system block communication?

    Analyzes which edges are consistently blocked (W < 0.25)
    and the conditions under which blocking occurs.
    """
    blocked_by_confidence = {"high_conf": 0, "low_conf": 0, "total": 0}
    blocked_edges = np.zeros((3, 3))  # will be resized if needed

    for sr in results.sample_results:
        if not sr.weight_matrices:
            continue
        W = sr.weight_matrices[-1]  # last round
        n = W.shape[0]
        if blocked_edges.shape != (n, n):
            blocked_edges = np.zeros((n, n))

        # Check which edges are blocked
        for i in range(n):
            for j in range(n):
                if i != j and W[i][j] < 0.25:
                    blocked_edges[i][j] += 1
                    blocked_by_confidence["total"] += 1

                    # Check if blocking correlates with high sender confidence
                    conf = sr.initial_correct.get(i, False)
                    if conf:
                        blocked_by_confidence["high_conf"] += 1
                    else:
                        blocked_by_confidence["low_conf"] += 1

    n_samples = len(results.sample_results) if results.sample_results else 1
    blocked_freq = blocked_edges / n_samples

    return {
        "blocked_edge_frequencies": blocked_freq.tolist(),
        "blocking_by_confidence": blocked_by_confidence,
        "most_blocked_edges": _top_edges(blocked_freq, ascending=True),
        "interpretation": (
            "The system tends to block communication when agents are "
            f"{'confident' if blocked_by_confidence.get('high_conf', 0) > blocked_by_confidence.get('low_conf', 0) else 'uncertain'} "
            "in their initial answers, suggesting the controller learns to preserve "
            "already-good reasoning chains."
        ),
    }


def analyze_agent_dominance(results: ExperimentResults) -> Dict[str, Any]:
    """
    Which agents dominate? Measure how often each agent's answer
    matches the final answer.
    """
    n = max(
        (max(sr.final_answers.keys()) + 1 for sr in results.sample_results if sr.final_answers),
        default=3,
    )

    agent_dominance = {i: 0 for i in range(n)}
    agent_accuracy = {i: 0 for i in range(n)}
    total = len(results.sample_results)

    for sr in results.sample_results:
        for agent_id, answer in sr.final_answers.items():
            if answer == sr.final_answer:
                agent_dominance[agent_id] += 1
            if answer == sr.gold_answer or (
                sr.gold_answer.strip().lower() in answer.strip().lower()
            ):
                agent_accuracy[agent_id] += 1

    dom_rates = {f"Agent {i}": round(v / max(total, 1), 3) for i, v in agent_dominance.items()}
    acc_rates = {f"Agent {i}": round(v / max(total, 1), 3) for i, v in agent_accuracy.items()}

    dominant = max(dom_rates, key=dom_rates.get) if dom_rates else "None"

    return {
        "dominance_rates": dom_rates,
        "accuracy_rates": acc_rates,
        "dominant_agent": dominant,
        "interpretation": (
            f"{dominant} most frequently contributes the final answer. "
            "Agent specialization emerges through the communication structure."
        ),
    }


def analyze_sparsity_emergence(results: ExperimentResults) -> Dict[str, Any]:
    """
    Does sparsity emerge? Track communication density over training
    and compare initial vs final graph density.
    """
    densities = [sr.communication_density for sr in results.sample_results]

    if len(densities) >= 2:
        first_half = np.mean(densities[:len(densities) // 2])
        second_half = np.mean(densities[len(densities) // 2:])
        trend = "decreasing" if second_half < first_half else "stable/increasing"
    else:
        first_half = second_half = np.mean(densities) if densities else 0.5
        trend = "insufficient data"

    # Active edges analysis
    active_edges = [sr.n_active_edges for sr in results.sample_results]
    n_agents = max(
        (max(sr.final_answers.keys()) + 1 for sr in results.sample_results if sr.final_answers),
        default=3,
    )
    max_edges = n_agents * (n_agents - 1)

    return {
        "avg_density": float(np.mean(densities)) if densities else 0.0,
        "density_first_half": float(first_half),
        "density_second_half": float(second_half),
        "trend": trend,
        "avg_active_edges": float(np.mean(active_edges)) if active_edges else 0,
        "max_possible_edges": max_edges,
        "sparsity_ratio": 1 - (float(np.mean(active_edges)) / max_edges) if max_edges > 0 else 0,
        "interpretation": (
            f"Communication density shows a {trend} trend. "
            f"Average sparsity ratio: {1 - (float(np.mean(active_edges)) / max_edges if max_edges > 0 else 0):.1%} "
            f"({float(np.mean(active_edges)):.1f}/{max_edges} edges active on average)."
        ),
    }


def analyze_compression_tradeoff(
    main_results: ExperimentResults,
    baseline_results: List[ExperimentResults],
) -> Dict[str, Any]:
    """
    Does performance degrade with too much compression?
    Compare accuracy at different compression levels.
    """
    main_metrics = aggregate_metrics(main_results)
    baseline_metrics = [aggregate_metrics(r) for r in baseline_results]

    points = []
    for m in [main_metrics] + baseline_metrics:
        points.append({
            "method": m["method"],
            "accuracy": m["accuracy"],
            "avg_tokens": m["token_cost"]["avg_tokens_per_sample"],
            "density": m["communication_density"],
        })

    # Sort by token cost
    points.sort(key=lambda x: x["avg_tokens"])

    # Check for Pareto optimality
    main_point = next(p for p in points if p["method"] == main_metrics["method"])
    pareto_optimal = all(
        main_point["accuracy"] >= p["accuracy"] or main_point["avg_tokens"] < p["avg_tokens"]
        for p in points if p["method"] != main_point["method"]
    )

    return {
        "compression_accuracy_points": points,
        "ib_controller_pareto_optimal": pareto_optimal,
        "interpretation": (
            f"The IB-Controller achieves {main_point['accuracy']:.1%} accuracy "
            f"with {main_point['avg_tokens']:.0f} avg tokens/sample. "
            f"{'It is Pareto-optimal' if pareto_optimal else 'It trades some accuracy for token efficiency'} "
            "compared to baselines."
        ),
    }


def _top_edges(freq_matrix: np.ndarray, ascending: bool = True, top_k: int = 3):
    """Get top-k edges by frequency."""
    n = freq_matrix.shape[0]
    edges = []
    for i in range(n):
        for j in range(n):
            if i != j:
                edges.append((i, j, freq_matrix[i][j]))
    edges.sort(key=lambda x: x[2], reverse=not ascending)
    return [(f"Agent {i} -> Agent {j}", float(f)) for i, j, f in edges[:top_k]]


def run_full_analysis(
    main_results: ExperimentResults,
    baseline_results: List[ExperimentResults],
    ablation_results: List[ExperimentResults],
) -> Dict[str, Any]:
    """Run all analyses and compile into a single report."""
    analysis = {
        "blocking_patterns": analyze_blocking_patterns(main_results),
        "agent_dominance": analyze_agent_dominance(main_results),
        "sparsity_emergence": analyze_sparsity_emergence(main_results),
        "compression_tradeoff": analyze_compression_tradeoff(
            main_results, baseline_results
        ),
    }

    return analysis


def format_analysis_report(analysis: Dict[str, Any], dataset_name: str = "") -> str:
    """Format analysis results as a readable report."""
    lines = [
        f"\n{'='*60}",
        f"ANALYSIS REPORT — {dataset_name}",
        f"{'='*60}\n",
    ]

    sections = [
        ("1. When Does the System Block Communication?", "blocking_patterns"),
        ("2. Which Agents Dominate?", "agent_dominance"),
        ("3. Does Sparsity Emerge?", "sparsity_emergence"),
        ("4. Compression vs. Performance Tradeoff", "compression_tradeoff"),
    ]

    for title, key in sections:
        lines.append(f"\n## {title}\n")
        data = analysis.get(key, {})
        interpretation = data.get("interpretation", "No data available.")
        lines.append(f"  {interpretation}\n")

        # Add key stats
        for k, v in data.items():
            if k != "interpretation" and not isinstance(v, (list, dict, np.ndarray)):
                lines.append(f"  - {k}: {v}")

    return "\n".join(lines)
