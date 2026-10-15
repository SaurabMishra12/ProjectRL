"""Exact unconditional power of the planned exact conditional McNemar test.

Refines the deliberately coarse Monte Carlo grid without changing its frozen
source snapshot. No observed selector outcomes or inference calls are used.
"""
from __future__ import annotations
import math
import json
import numpy as np
from scipy.stats import binom, norm
from WEBExtension.causal_study.provider import atomic_json
from .planning import ROOT, csv_rows


def exact_power(n, delta, discordance):
    if type(n) is not int or n < 2 or not 0 <= delta <= discordance <= 1 or discordance == 0:
        raise ValueError('Need n>=2 and 0<=delta<=discordance<=1 with positive discordance')
    # D is the number of discordant questions. Given D, gains are binomial.
    d = np.arange(n + 1)
    critical = binom.ppf(.025, d, .5).astype(int)
    critical -= (binom.cdf(critical, d, .5) > .025).astype(int)
    p_gain = (discordance + delta) / (2 * discordance)
    reject = binom.cdf(critical, d, p_gain) + binom.sf(d - critical - 1, d, p_gain)
    return float(np.dot(binom.pmf(d, n, discordance), reject))


def main():
    directory = ROOT / 'planning_artifacts'
    summary = json.loads((directory / 'summary.json').read_text())
    reference = summary['reference']
    rate = summary['cached_conflict_yield']
    pool = summary['fresh_pool']['gold_free_source_capable_fresh_questions']
    rows = []
    for delta in [.03, .05, .1]:
        for q in [.1, .25, .5]:
            for target in [.8, .9]:
                approx = (norm.ppf(.975) + norm.ppf(target))**2 * (q - delta**2) / delta**2
                n = max(25, 25 * math.ceil(approx / 25))
                while exact_power(n, delta, q) < target:
                    n += 25
                while n > 25 and exact_power(n - 25, delta, q) >= target:
                    n -= 25
                screened = math.ceil(n / rate)
                initial, final = 3 * screened, 2 * n
                total = initial + final
                rows.append({'effect_em_fraction': delta, 'discordance': q, 'target_power': target,
                    'n_questions': n, 'exact_unconditional_power': exact_power(n, delta, q),
                    'method': 'binomial_mixture_of_exact_two_sided_McNemar_rejection',
                    'grid_step_questions': 25, 'not_a_proven_global_minimum': True,
                    'screened_questions_at_point_yield': screened, 'proposal_calls': initial,
                    'finalizer_calls': final, 'total_completions': total,
                    'attempt_ceiling_with_25pct_reserve': math.ceil(1.25 * total),
                    'input_tokens_reference': math.ceil(initial * reference['initial_mean_input_tokens'] +
                         final * summary['finalizer_input_mean_using_e6_overhead']),
                    'output_tokens_reference': math.ceil(initial * reference['initial_mean_output_tokens'] +
                         final * reference['mean_output_tokens']),
                    'serial_hours_reference': total * reference['observed_serial_seconds_per_call'] / 3600,
                    'within_fresh_pool_at_point_yield': screened <= pool, 'paid_authorized': False})
    atomic_json(directory / 'refined_power_targets.json', rows)
    csv_rows(directory / 'refined_power_targets.csv', rows)
    report = ROOT / '05_API_BUDGET_AND_POWER.md'
    marker = '\n## Exact-power refinement after the coarse grid\n'
    text = report.read_text().split(marker)[0]
    lines = [marker, 'The frozen planning module intentionally evaluated a doubling grid. '
        'Its 1,600-question entries above are coarse simulated targets, not minimum sample sizes. '
        'The separate `power_refinement.py` integrates the binomial distribution of discordant questions '
        'and the conditional binomial rejection probability exactly. It searches nearby multiples of25 '
        'without claiming a globally minimal integer sample. These refined scenarios supersede the coarse '
        'targets for budget comparison; both original files and source snapshots are preserved.', '']
    for row in rows:
        if row['effect_em_fraction'] == .05 and row['discordance'] == .25:
            lines.append(f"- Five-point effect, discordance0.25, target{100*row['target_power']:g}%: "
                f"n={row['n_questions']}, exact power={row['exact_unconditional_power']:.4f}; "
                f"screen{row['screened_questions_at_point_yield']:,} source questions; "
                f"{row['proposal_calls']:,} proposal+{row['finalizer_calls']:,} finalizer="
                f"{row['total_completions']:,} completions, suggested attempt ceiling"
                f"{row['attempt_ceiling_with_25pct_reserve']:,}; input-reference"
                f"{row['input_tokens_reference']:,}, output-reference{row['output_tokens_reference']:,}, "
                f"serial reference{row['serial_hours_reference']:.2f}h. Within fresh pool: "
                f"{row['within_fresh_pool_at_point_yield']}.")
    lines += ['', 'All18 refined scenarios are in `planning_artifacts/refined_power_targets.csv`. '
        'The 6,809 source questions would yield about454 mechanically eligible states at the historical '
        'point rate, before semantic qualification. This is an expectation, not a guaranteed eligible pool. '
        'Smaller effects and more discordance need more questions; larger effects can be feasible under '
        'some assumptions. No sensitivity scenario establishes what the actual effect is or authorizes spending.', '']
    report.write_text(text + '\n'.join(lines))
    print(json.dumps([r for r in rows if r['effect_em_fraction'] == .05 and r['discordance'] == .25], indent=2))


if __name__ == '__main__':
    main()
