# Per-Cell Realized Token Volume — 2×2 Factorial Grid

## Summary Table

| n_train | Arm | Description | n | Accuracy | Mean Tokens TX | Mean Tokens Full | Token Reduction | Mean Density |
|---------|-----|-------------|---|----------|----------------|------------------|----------------|--------------|
| 200 | 00 | Static Ring + No Compression | 150 | 95.3% | 366 ± 503 | 629 | 15.2% ± 20.5% | 0.216 |
| 200 | 01 | Learned Topology + No Compression | 150 | 92.0% | 137 ± 179 | 223 | 15.5% ± 20.4% | 0.257 |
| 200 | 10 | Static Ring + Learned Compression | 150 | 94.7% | 384 ± 525 | 680 | 15.7% ± 21.2% | 0.216 |
| 200 | 11 | Learned Topology + Learned Compression (Full SparseComm) | 150 | 98.0% | 125 ± 181 | 189 | 11.2% ± 16.4% | 0.240 |
| 300 | 00 | Static Ring + No Compression | 150 | 94.7% | 369 ± 506 | 648 | 15.6% ± 21.2% | 0.216 |
| 300 | 01 | Learned Topology + No Compression | 150 | 94.0% | 124 ± 172 | 206 | 14.6% ± 20.3% | 0.237 |
| 300 | 10 | Static Ring + Learned Compression | 150 | 94.7% | 355 ± 492 | 638 | 15.7% ± 21.5% | 0.212 |
| 300 | 11 | Learned Topology + Learned Compression (Full SparseComm) | 150 | 92.7% | 123 ± 173 | 199 | 13.2% ± 18.5% | 0.234 |
| 400 | 00 | Static Ring + No Compression | 150 | 93.3% | 366 ± 492 | 653 | 16.0% ± 21.4% | 0.220 |
| 400 | 01 | Learned Topology + No Compression | 150 | 96.7% | 109 ± 172 | 176 | 11.5% ± 17.9% | 0.198 |
| 400 | 10 | Static Ring + Learned Compression | 150 | 95.3% | 344 ± 475 | 609 | 15.4% ± 21.0% | 0.212 |
| 400 | 11 | Learned Topology + Learned Compression (Full SparseComm) | 150 | 93.3% | 59 ± 87 | 187 | 22.6% ± 32.2% | 0.111 |

## Volume Comparisons (Mann-Whitney U, two-sided)

| n_train | Comparison | U statistic | p-value | Interpretation |
|---------|------------|-------------|---------|----------------|
| 200 | Arm_00 vs Arm_10 (compression effect, static topology) | 8415.0 | 0.0002 | Significant (p<0.05) |
| 200 | Arm_01 vs Arm_11 (compression effect, learned topology) | 8954.0 | 0.0022 | Significant (p<0.05) |
| 200 | Arm_00 vs Arm_01 (topology effect, no compression) | 10625.0 | 0.4054 | Not significant |
| 200 | Arm_10 vs Arm_11 (topology effect, with compression) | 10050.0 | 0.1102 | Not significant |
| 300 | Arm_00 vs Arm_10 (compression effect, static topology) | 8629.0 | 0.0005 | Significant (p<0.05) |
| 300 | Arm_01 vs Arm_11 (compression effect, learned topology) | 13800.0 | 0.0007 | Significant (p<0.05) |
| 300 | Arm_00 vs Arm_01 (topology effect, no compression) | 10857.0 | 0.6009 | Not significant |
| 300 | Arm_10 vs Arm_11 (topology effect, with compression) | 10409.0 | 0.2629 | Not significant |
| 400 | Arm_00 vs Arm_10 (compression effect, static topology) | 9504.0 | 0.0201 | Significant (p<0.05) |
| 400 | Arm_01 vs Arm_11 (compression effect, learned topology) | 15037.0 | 0.0000 | Significant (p<0.05) |
| 400 | Arm_00 vs Arm_01 (topology effect, no compression) | 11422.0 | 0.8189 | Not significant |
| 400 | Arm_10 vs Arm_11 (topology effect, with compression) | 10569.0 | 0.3647 | Not significant |

## Mode Distribution (Total Counts)

| n_train | Arm | BLOCK | PARTIAL | FULL |
|---------|-----|-------|---------|------|
| 200 | Static Ring + No Compression | 0 | 972 | 0 |
| 200 | Learned Topology + No Compression | 0 | 360 | 0 |
| 200 | Static Ring + Learned Compression | 0 | 972 | 0 |
| 200 | Learned Topology + Learned Compression (Full SparseComm) | 50 | 177 | 73 |
| 300 | Static Ring + No Compression | 0 | 972 | 0 |
| 300 | Learned Topology + No Compression | 0 | 330 | 0 |
| 300 | Static Ring + Learned Compression | 0 | 954 | 0 |
| 300 | Learned Topology + Learned Compression (Full SparseComm) | 106 | 0 | 212 |
| 400 | Static Ring + No Compression | 0 | 990 | 0 |
| 400 | Learned Topology + No Compression | 0 | 276 | 0 |
| 400 | Static Ring + Learned Compression | 0 | 954 | 0 |
| 400 | Learned Topology + Learned Compression (Full SparseComm) | 200 | 0 | 100 |