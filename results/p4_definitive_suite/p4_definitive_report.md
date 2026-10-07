# Definitive Protocol P4 Evaluation Suite Report

**System / Protocol**: Protocol P4 Redesign (Exact Bandwidth & Support Space Enumeration)
**Date / Timestamp**: 2026-10-03 11:39:08 UTC

---

## Panel A: Exhaustive 15-Support Enumeration (Exact Per-Edge Bandwidth)

Every 2-edge directed support on N=3 agents is evaluated under exact equal-bandwidth allocation:
- At V=50: exactly 25 tokens/edge ceiling (25/25)
- At V=100: exactly 50 tokens/edge ceiling (50/50)

| ID | Support Key | Graph Isomorphism Class | Edges | V=50 Acc [%] | V=100 Acc [%] |
|---:|---|---|---|---:|---:|
| 1 | `S01_OutStar_0` | Out-Star | 0→1 + 0→2 | 73.33% | 76.67% |
| 2 | `S02_Dyad_01` | Reciprocal Dyad | 0→1 + 1→0 | 76.67% | 80.67% |
| 3 | `S03_Relay_012` | Relay Path | 0→1 + 1→2 | 78.00% | 82.00% |
| 4 | `S04_Relay_201` | Relay Path | 0→1 + 2→0 | 73.33% | 76.00% |
| 5 | `S05_InStar_1` | In-Star | 0→1 + 2→1 | 73.33% | 75.33% |
| 6 | `S06_Relay_102` | Relay Path | 0→2 + 1→0 | 76.67% | 82.00% |
| 7 | `S07_InStar_2` | In-Star | 0→2 + 1→2 | 78.67% | 82.00% |
| 8 | `S08_Dyad_02` | Reciprocal Dyad | 0→2 + 2→0 | 73.33% | 77.33% |
| 9 | `S09_Relay_021` | Relay Path | 0→2 + 2→1 | 73.33% | 77.33% |
| 10 | `S10_OutStar_1` | Out-Star | 1→0 + 1→2 | 80.00% | 82.67% |
| 11 | `S11_InStar_0` | In-Star | 1→0 + 2→0 | 75.33% | 78.00% |
| 12 | `S12_Relay_210` | Relay Path | 1→0 + 2→1 | 74.00% | 77.33% |
| 13 | `S13_Relay_120` | Relay Path | 1→2 + 2→0 | 78.67% | 82.67% |
| 14 | `S14_Dyad_12` | Reciprocal Dyad | 1→2 + 2→1 | 78.67% | 81.33% |
| 15 | `S15_OutStar_2` | Out-Star | 2→0 + 2→1 | 73.33% | 75.33% |

### Distribution Statistics across all 15 Supports

**V = 50 tokens**:
- Mean across all supports: **75.78% ± 2.43%** [Min: 73.33%, Max: 80.00%]
- Learned Support (`S01_OutStar_0`): **73.33%**
- Empirical Percentile: **40.0th percentile** (6/15 supports)

**V = 100 tokens**:
- Mean across all supports: **79.11% ± 2.74%** [Min: 75.33%, Max: 82.67%]
- Learned Support (`S01_OutStar_0`): **76.67%**
- Empirical Percentile: **26.7th percentile** (4/15 supports)

## Panel B: Learned Policy Multi-Seed / Checkpoint Robustness

| Checkpoint / Seed | V=50 Acc [%] | V=100 Acc [%] |
|---|---:|---:|
| `PPO_Seed1_400ep` | 90.67% | 92.67% |
| `PPO_Seed2_ep4` | 90.00% | 92.00% |
| `PPO_Seed3_ep3` | 90.67% | 90.00% |
| `PPO_Seed4_300ep` | 94.67% | 94.00% |
| `PPO_Seed5_Arm01` | 93.33% | 91.33% |
| **Mean ± SD** | **91.87% ± 1.81%** | **92.00% ± 1.33%** |
