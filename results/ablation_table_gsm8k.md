
### Ablation — gsm8k

| Ablation | Accuracy | Avg Tokens | Density | Mode Dist |
|----------|----------|------------|---------|----------|
| Ablation: lambda=0        |    90.0% |     3491.5 |   0.506 | BLOCK:0%, FILTER:33%, FULL:0%, SUMMARIZE:67% |
| Ablation: No Summarize    |    90.0% |     2907.5 |   0.506 | BLOCK:0%, FILTER:33%, FULL:0%, SUMMARIZE:67% |
| Ablation: No Filter       |    90.0% |     3177.2 |   0.506 | BLOCK:0%, FILTER:33%, FULL:0%, SUMMARIZE:67% |
| Ablation: Random Controller |   100.0% |     3528.9 |   0.500 | BLOCK:27%, FILTER:28%, FULL:25%, SUMMARIZE:20% |
