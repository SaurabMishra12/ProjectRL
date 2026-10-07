# Random Support Baseline: Concentration vs Source Selection

**Experiment**: Pick K=5 random pairs of directed edges (|T|=2),
run each on the same 150 GSM8K questions at V̄ ∈ {50, 100}.

**Question**: Is the learned support's advantage from *concentration*
(any 2 edges suffice) or *source selection* (the specific learned pair matters)?

## Reference Baselines

| Condition | V̄=50 | V̄=100 |
|---|---:|---:|
| NoComm | 78.67% | 78.67% |
| Ring (|T|=6) | 78.67% | 75.33% |
| Learned (|T|=2, 0→1+0→2) | 95.33% | 92.67% |

## Random-2 Results

| Draw | Edges | V̄=50 Acc | V̄=100 Acc |
|---:|---|---:|---:|
| 1 | 0→1 + 1→0 | 80.67% | 80.00% |
| 2 | 1→2 + 2→1 | 79.33% | 82.00% |
| 3 | 0→2 + 2→1 | 86.00% | 79.33% |
| 4 | 2→0 + 2→1 | 80.00% | 78.67% |
| 5 | 0→2 + 2→0 | 76.67% | 79.33% |

## Summary Statistics

### V̄ = 50

- **Random-2 Mean**: 80.53% ± 3.05% [76.67%, 86.00%]
- **Learned Support**: 95.33%
- **Ring**: 78.67%
- **|Random-2 − Learned|**: 14.80 pts
- **|Random-2 − Ring|**: 1.86 pts
- **Verdict**: SOURCE SELECTION (closer to Ring)

### V̄ = 100

- **Random-2 Mean**: 79.87% ± 1.15% [78.67%, 82.00%]
- **Learned Support**: 92.67%
- **Ring**: 75.33%
- **|Random-2 − Learned|**: 12.80 pts
- **|Random-2 − Ring|**: 4.54 pts
- **Verdict**: SOURCE SELECTION (closer to Ring)
