"""Reproducible inference over unique questions, never agent/seed replicates.

``analyze_e6`` accepts one dict per unique ``question_id`` and four F1 scores
in [0,1]: ``colocated_nocomm_f1``, ``colocated_comm_f1``,
``distributed_nocomm_f1``, ``distributed_comm_f1``. Repeated assignments must
be averaged within question by a prespecified upstream procedure; duplicate
IDs are rejected here. Missing/failed cells must not be filled with zero.

The primary contrast is the mean of per-question differences-in-differences.
Percentile bootstrap intervals resample whole questions. The sign-flip test
requires independent, symmetric zero-centered question contrasts under H0;
it is not a design-based randomization test for this crossed evaluation.

``power_sensitivity`` uses symmetrized centered pilot residuals and explicit
variance stress tests. Its conditional-normal approximation to the sign-flip
test is for planning, not a replacement for the primary test. Pilot effects
are never used as the assumed true effect. All outputs are JSON serializable.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real
from typing import Any

import numpy as np


SCORE_KEYS = (
    "colocated_nocomm_f1", "colocated_comm_f1",
    "distributed_nocomm_f1", "distributed_comm_f1",
)
_SYMMETRY = (
    "Independent question-level contrasts must be symmetric about zero under H0. "
    "This is a paired sign-flip assumption, not physical randomized treatment allocation."
)


def _positive_int(value: Any, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _validated(rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], np.ndarray]:
    clean = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"row {index} must be a mapping")
        qid = row.get("question_id")
        if not isinstance(qid, str) or not qid.strip():
            raise ValueError(f"row {index} requires a nonempty string question_id")
        if qid in seen:
            raise ValueError(f"duplicate question_id {qid!r}; collapse assignments within question first")
        seen.add(qid)
        values = []
        for key in SCORE_KEYS:
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{qid}: {key} must be a finite numeric score in [0,1]")
            values.append(float(value))
        clean.append((qid, values))
    clean.sort(key=lambda item: item[0])
    return [qid for qid, _ in clean], np.asarray([values for _, values in clean], dtype=float).reshape((-1, 4))


def _sign_flip(values: np.ndarray, resamples: int, rng: np.random.Generator) -> dict[str, Any]:
    n = len(values)
    if n < 2:
        return dict(method="INSUFFICIENT_DATA", p_value=None, resamples=0, assumption=_SYMMETRY)
    # Zeros do not change the null distribution; remove them before enumeration.
    nonzero = values[np.abs(values) > 1e-12]
    count = len(nonzero)
    observed = abs(float(np.sum(values)))
    extreme = 0
    exact = count <= 20
    draws = 2 ** count if exact else resamples
    for offset in range(0, draws, 2048):
        size = min(2048, draws - offset)
        if exact:
            codes = np.arange(offset, offset + size, dtype=np.uint64)[:, None]
            bits = (codes >> np.arange(count, dtype=np.uint64)) & 1
            signs = bits.astype(float) * 2 - 1
        else:
            signs = rng.integers(0, 2, size=(size, count)) * 2 - 1
        null_sums = signs @ nonzero
        extreme += int(np.count_nonzero(np.abs(null_sums) >= observed - 1e-12))
    p = extreme / draws if exact else (extreme + 1) / (draws + 1)
    return dict(method="exact_question_sign_flip" if exact else "monte_carlo_question_sign_flip",
                p_value=float(p), resamples=draws, alternative="two-sided", assumption=_SYMMETRY,
                zero_contrasts_removed=n-count,
                monte_carlo_plus_one_correction=not exact)


def _paired_summary(values: np.ndarray, bootstrap_indices: np.ndarray | None,
                    randomization_resamples: int, rng: np.random.Generator) -> dict[str, Any]:
    n = len(values)
    ci = None
    if bootstrap_indices is not None:
        means = values[bootstrap_indices].mean(axis=1)
        ci = [float(x) for x in np.quantile(means, [.025, .975])]
    return dict(mean=float(np.mean(values)) if n else None,
                sd=float(np.std(values, ddof=1)) if n > 1 else None,
                ci95=ci, ci_method="question_cluster_percentile_bootstrap_exploratory",
                help_harm_unchanged=dict(helped=int(np.sum(values > 1e-12)),
                                        harmed=int(np.sum(values < -1e-12)),
                                        unchanged=int(np.sum(np.abs(values) <= 1e-12))),
                sign_flip_test=_sign_flip(values, randomization_resamples, rng))


def analyze_e6(rows: Sequence[Mapping[str, Any]], *, bootstrap_resamples: int = 10000,
               randomization_resamples: int = 10000, seed: int = 20261015) -> dict[str, Any]:
    """Return cell means, paired effects, exploratory CIs, and sign-flip p-values.

    Scores and returned effects use unit F1; multiply by 100 for F1 points.
    Only ``contrasts.interaction`` is primary. Other p-values are unadjusted,
    descriptive secondary diagnostics, not additional confirmatory findings.
    """
    _positive_int(bootstrap_resamples, "bootstrap_resamples")
    _positive_int(randomization_resamples, "randomization_resamples")
    ids, matrix = _validated(rows)
    n = len(ids)
    rng = np.random.default_rng(seed)
    bootstrap_indices = rng.integers(0, n, size=(bootstrap_resamples, n)) if n >= 2 else None
    colocated = matrix[:, 1] - matrix[:, 0]
    distributed = matrix[:, 3] - matrix[:, 2]
    interaction = distributed - colocated
    contrasts = dict(colocated_comm_effect=colocated, distributed_comm_effect=distributed,
                     interaction=interaction)
    warnings = [
        "Pilot intervals are exploratory; a small deterministic sample does not establish population generality.",
        "A nonsignificant result does not establish equivalence or noninferiority.",
        "Sign-flip inference assumes independent, symmetric question contrasts; crossed evaluations do not supply randomized labels.",
        "Only the interaction is primary; within-placement comparisons are secondary and unadjusted.",
        "Effects concern this frozen evidence assignment and protocol; repeated seeds are not independent questions.",
    ]
    if n < 2:
        warnings.append("INSUFFICIENT_DATA: at least two unique complete questions are required for uncertainty estimates.")
    if n >= 2 and np.std(interaction) <= 1e-12:
        warnings.append("Degenerate pilot interaction variance can produce a zero-width bootstrap interval; it does not establish a precise population effect.")
    return dict(status="EXPLORATORY_PILOT" if n >= 2 else "INSUFFICIENT_DATA",
                n_questions=n, unit="F1 in [0,1]", sampling_unit="unique_question",
                cell_means={key: float(matrix[:, i].mean()) if n else None for i, key in enumerate(SCORE_KEYS)},
                primary_contrast="interaction", contrast_formula="(distributed_comm-distributed_nocomm)-(colocated_comm-colocated_nocomm)",
                contrasts={key: _paired_summary(value, bootstrap_indices, randomization_resamples, rng)
                           for key, value in contrasts.items()},
                per_question=[dict(question_id=qid, colocated_comm_effect=float(colocated[i]),
                                   distributed_comm_effect=float(distributed[i]), interaction=float(interaction[i]))
                              for i, qid in enumerate(ids)],
                bootstrap_resamples=bootstrap_resamples, seed=seed, warnings=warnings)


def _wilson(successes: int, total: int) -> list[float]:
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [max(0., center-half), min(1., center+half)]


def _normal_quantile(probability: float) -> float:
    # Avoid importing stdlib statistics: this study module has that same name
    # and can be imported by a script whose directory is first on sys.path.
    if not 0 < probability < 1:
        raise ValueError("normal probability must be strictly between 0 and 1")
    lower, upper = -40., 40.
    for _ in range(80):
        middle = (lower+upper)/2
        if .5*math.erfc(-middle/math.sqrt(2)) < probability:
            lower = middle
        else:
            upper = middle
    return (lower+upper)/2


def power_sensitivity(rows: Sequence[Mapping[str, Any]], *, effect_grid: Sequence[float] = (.02, .05, .10),
                      sample_sizes: Sequence[int] = (20, 50, 100, 200, 300, 600, 1000, 2000),
                      simulations: int = 2000, alpha: float = .05,
                      variance_multipliers: Sequence[float] = (1., 1.5, 2.),
                      sd_floor: float = .25, seed: int = 20261015) -> dict[str, Any]:
    """Simulate power across scientifically chosen effects and stressed SDs.

    Resample centered empirical residuals, symmetrize using independent random
    signs, rescale to max(pilot SD, sd_floor)*multiplier, and add the assumed
    effect. The decision rule uses the conditional normal approximation to the
    two-sided sign-flip sum: abs(sum(d))/sqrt(sum(d**2)) > z_(1-alpha/2).
    It avoids expensive nested tests but is approximate, especially near n=20.
    Monte Carlo intervals express simulation error only, not pilot-SD error.

    Residual scaling is an explicit moment sensitivity model; any support
    outside the feasible interaction range [-2,2] is reported rather than
    clipped (which would silently change the hypothesized mean/variance).
    Recommendations use the lower Monte Carlo confidence bound and never
    authorize inference. All confirmatory questions must be fresh holdouts.
    """
    _positive_int(simulations, "simulations")
    for n in sample_sizes:
        _positive_int(n, "sample_sizes", 2)
    if not sample_sizes or not effect_grid or not variance_multipliers:
        raise ValueError("nonempty effect_grid, sample_sizes, and variance_multipliers required")
    if not isinstance(alpha, Real) or not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if not isinstance(sd_floor, Real) or not math.isfinite(sd_floor) or sd_floor <= 0:
        raise ValueError("sd_floor must be positive and finite")
    for effect in effect_grid:
        if not isinstance(effect, Real) or not math.isfinite(effect) or not 0 < effect <= 2:
            raise ValueError("effect_grid values must be positive and <=2 in unit F1")
    for multiplier in variance_multipliers:
        if not isinstance(multiplier, Real) or not math.isfinite(multiplier) or multiplier <= 0:
            raise ValueError("variance_multipliers must be positive and finite")
    ids, matrix = _validated(rows)
    pilot_n = len(ids)
    observed = matrix[:, 3] - matrix[:, 2] - matrix[:, 1] + matrix[:, 0]
    observed_sd = float(np.std(observed, ddof=1)) if pilot_n >= 2 else None
    nondegenerate = observed_sd is not None and observed_sd > 1e-12
    if nondegenerate:
        residuals = observed - observed.mean()
        residuals = residuals / math.sqrt(float(np.mean(residuals**2)))
    else:
        residuals = np.asarray([1.])
    base_sd = max(observed_sd or 0., float(sd_floor))
    rng = np.random.default_rng(seed)
    critical = _normal_quantile(1-alpha/2)
    grid = []
    for multiplier in sorted(set(float(x) for x in variance_multipliers)):
        assumed_sd = base_sd * multiplier
        for n in sorted(set(sample_sizes)):
            # Share noise draws across effect sizes to reduce simulation noise
            # when comparing the same variance/sample-size sensitivity scenario.
            sum_noise = np.empty(simulations)
            squared_noise = np.empty(simulations)
            out_of_bounds = {float(effect): 0 for effect in effect_grid}
            for offset in range(0, simulations, 128):
                size = min(128, simulations-offset)
                indices = rng.integers(0, len(residuals), size=(size, n))
                signs = rng.integers(0, 2, size=(size, n)) * 2 - 1
                noise = residuals[indices] * signs * assumed_sd
                sum_noise[offset:offset+size] = noise.sum(axis=1)
                squared_noise[offset:offset+size] = (noise**2).sum(axis=1)
                for effect in out_of_bounds:
                    out_of_bounds[effect] += int(np.count_nonzero(np.abs(noise+effect) > 2))
            for effect in sorted(set(float(x) for x in effect_grid)):
                sums = sum_noise + n*effect
                squares = squared_noise + 2*effect*sum_noise + n*effect**2
                statistic = np.divide(np.abs(sums), np.sqrt(np.maximum(squares, 0)),
                                      out=np.zeros_like(sums), where=squares > 0)
                successes = int(np.count_nonzero(statistic > critical))
                grid.append(dict(effect=effect, n_questions=n, assumed_sd=assumed_sd,
                                 variance_multiplier=multiplier, estimated_power=successes/simulations,
                                 power_ci95=_wilson(successes, simulations), simulations=simulations,
                                 simulated_outside_feasible_range_fraction=out_of_bounds[effect]/(simulations*n),
                                 confirmatory_successful_completions=18*n))
    recommendations = []
    for multiplier in sorted(set(float(x) for x in variance_multipliers)):
        for effect in sorted(set(float(x) for x in effect_grid)):
            for target in (.8, .9):
                candidates = [r for r in grid if r["variance_multiplier"] == multiplier and r["effect"] == effect
                              and r["power_ci95"][0] >= target]
                chosen = min(candidates, key=lambda r: r["n_questions"]) if candidates else None
                approximate_n = math.ceil(((critical + _normal_quantile(target)) * base_sd*multiplier/effect)**2)
                recommendations.append(dict(effect=effect, variance_multiplier=multiplier, target_power=target,
                                            smallest_simulated_n=chosen["n_questions"] if chosen else None,
                                            successful_completions=18*chosen["n_questions"] if chosen else None,
                                            normal_approximation_n=max(2, approximate_n),
                                            status="SIMULATED_GRID_REACHED" if chosen else "NOT_REACHED_IN_GRID"))
    return dict(status="EXPLORATORY_SENSITIVITY" if nondegenerate else "SENSITIVITY_ONLY_DEGENERATE_PILOT",
                n_pilot_questions=pilot_n, observed_interaction_sd=observed_sd, sd_floor=sd_floor,
                base_sd=base_sd, variance_multiplier_applies_to="standard_deviation",
                effect_unit="F1 in [0,1]; .05 means five F1 points",
                alpha=alpha, seed=seed, simulations_per_grid_point=simulations,
                test_method="conditional_normal_approximation_to_two_sided_question_sign_flip",
                residual_model="symmetrized_centered_empirical" if nondegenerate else "rademacher_sd_floor",
                grid=grid, recommendations=recommendations,
                warnings=["Planning sensitivity only, not an API authorization or guarantee of power.",
                          "Conditional normal approximation may be inaccurate for small n or highly discrete residuals; validate candidate n with the exact planned test before confirmatory approval.",
                          "Eight-question pilot variance is uncertain; SD floors and multipliers are sensitivity assumptions, not confidence bounds.",
                          "Monte Carlo intervals quantify simulation noise only, not uncertainty about the population distribution.",
                          "Scaled residuals can exceed feasible interaction support [-2,2]; inspect the reported fraction before using that scenario.",
                          "Use fresh held-out unique questions; pilot-selected design effects cannot be confirmed on the pilot."])
