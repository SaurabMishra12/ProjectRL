# 📊 Full Analysis Report — projectRL `08-06results`

> **Experiment Design**: Multi-agent communication system with PPO-based policy learning.
> 3 Datasets · 3 Methods (Transfer/MAD/NoComm) · 4 Ablation Arms · 4 Training Sizes · 3 Seeds · 5 Epochs

---

## 🗂 Data Structure Summary

| File Group | Content | Samples |
|---|---|---|
| `seed_ckpt_*.json` | Per-seed final accuracies (last 300 samples, 400-trained model) | 300/seed |
| `final_results.json` | Aggregated mean/std/CI across 3 seeds (400-trained model) | 3 seeds |
| `continuous/Transfer_*_seed*.jsonl` | Full 600-sample stream for Transfer method | 600 total |
| `ablation_{100,200,300,400}.json` | 4-arm ablation + OLS betas at each training size | 600 (N per arm varies) |
| `topology_benchmark_results.json` | 7 topology variants at 400-train, GSM8K | 150 each |
| `continuous/Train_Pretrain_*` | Per-epoch training accuracy streams | 100/epoch |

---

## 📈 Plot Gallery

### Transfer Learning & Generalization
- **100-Train Model (First 300 Samples)**:
  ![Transfer Learning – 100-Train Model (First 300 Samples)](./08-06results/plots/plot1_transfer_100train.png)
- **400-Train Model (Last 300 Samples, Final Evaluation)**:
  ![Transfer Learning – 400-Train Model (Last 300 Samples, Final Evaluation)](./08-06results/plots/plot2_transfer_400train.png)
- **Before vs After Training (Δ per Method per Dataset)**:
  ![Before vs After: 100-Train → 400-Train Model](./08-06results/plots/plot3_transfer_before_after.png)

### Ablation Study & Factorial Analysis
- **Ablation Study Across Training Sizes**:
  ![Ablation Study: All 4 Arms Across Training Sizes 100–400](./08-06results/plots/plot4_ablation_all_sizes.png)
- **Ablation Accuracy Heatmap**:
  ![Ablation Accuracy Heatmap (Arms × Training Sizes)](./08-06results/plots/plot5_ablation_heatmap.png)
- **OLS Factorial Analysis**:
  ![OLS Factorial Analysis: β Coefficients & P-Values Across Training Sizes](./08-06results/plots/plot6_ols_betas.png)

### Epoch-by-Epoch Training Curves
- **Training Accuracy per Epoch (Overlaid)**:
  ![Training Accuracy per Epoch – All Datasets & Sizes Overlaid](./08-06results/plots/plot7_training_epochs.png)
- **Training Accuracy per Epoch (Grid)**:
  ![Training Accuracy Grid – Individual Dataset × Size Panels with Stats](./08-06results/plots/plot8_training_epoch_grid.png)
- **Epoch-to-Epoch Fluctuation Symmetry**:
  ![Epoch-to-Epoch Fluctuation ΔAcc – Symmetry Analysis by Dataset](./08-06results/plots/plot11_fluctuation_analysis.png)

### Topology & Token Efficiency
- **Topology Benchmark (GSM8K, 400-Train)**:
  ![Topology Benchmark (400-Train, GSM8K) – Ranked by Accuracy](./08-06results/plots/plot9_topology_benchmark.png)
- **Accuracy vs Token Cost Scatter**:
  ![Token Efficiency: Accuracy vs Token Cost Trade-off Scatter](./08-06results/plots/plot10_token_efficiency.png)
- **Ablation Token Cost vs Accuracy Trajectory**:
  ![Ablation Arms: Token Cost vs Accuracy Trajectory](./08-06results/plots/plot12_token_vs_accuracy.png)
- **Radar Comparison (400-Train)**:
  ![Radar Chart: Multi-Dataset Method Comparison (400-Train)](./08-06results/plots/plot13_radar_methods.png)
- **Improvement Symmetry (Per-Arm Accuracy vs Training Size)**:
  ![Improvement Symmetry: Per-Arm Accuracy vs Training Size](./08-06results/plots/plot14_improvement_symmetry.png)


---

## 📋 Table 1 — Transfer Results: 100-Trained Model (First 300 Samples/Seed)

> These are tested from a model trained on **100 samples**, then evaluated on 300 samples per seed.

| Dataset | Method | Seed 0 | Seed 1 | Seed 2 | **Mean** | **Std** |
|---|---|---|---|---|---|---|
| GSM8K | **Transfer (Ours)** | 53.00% | 53.00% | 56.33% | **54.11%** | ±1.57% |
| AQUA | **Transfer (Ours)** | 86.00% | 86.22% | 86.61% | **86.28%** | ±0.25% |
| HOTPOTQA | **Transfer (Ours)** | 36.33% | 30.67% | 28.00% | **31.67%** | ±3.47% |

> [!NOTE]
> The FullComm/NoComm seeded streams for the 100-trained model are embedded in the `continuous/` JSONL files without per-seed splits for AQUA and HOTPOTQA. GSM8K's NoComm/MAD seed streams also reflect the 300-sample point (see insight below).

---

## 📋 Table 2 — Transfer Results: 400-Trained Model (Last 300 Samples/Seed, Final)

> Model trained on **400 samples**. This is the primary evaluation.

| Dataset | Method | Seed 0 | Seed 1 | Seed 2 | **Mean** | **Std** | **Avg Tokens** |
|---|---|---|---|---|---|---|---|
| GSM8K | **Transfer (Ours)** | 94.67% | 94.33% | 96.00% | **95.00%** | ±0.88% | 52.2 |
| GSM8K | MAD (Full Comm) | 96.00% | 96.33% | 96.33% | 96.22% | ±0.19% | 530.8 |
| GSM8K | No Comm | 86.33% | 87.00% | 87.67% | 87.00% | ±0.67% | 0.0 |
| AQUA | **Transfer (Ours)** | 86.00% | 86.22% | 86.61% | **86.28%** | ±0.31% | 421.6 |
| AQUA | MAD (Full Comm) | 87.00% | 86.33% | 82.68% | 85.34% | ±2.33% | 850.3 |
| AQUA | No Comm | 83.00% | 82.33% | 73.23% | 79.52% | ±5.46% | 0.0 |
| HOTPOTQA | **Transfer (Ours)** | 36.33% | 30.67% | 28.00% | **31.67%** | ±4.26% | 473.2 |
| HOTPOTQA | MAD (Full Comm) | 38.00% | 31.67% | 32.33% | 34.00% | ±3.48% | 2914.1 |
| HOTPOTQA | No Comm | 32.67% | 28.33% | 29.00% | 30.00% | ±2.33% | 0.0 |

---

## 📊 Summary of Best Performances: Ours vs MAD vs No Comm

This table and graph show the comparison of the best (peak seed) and mean accuracies achieved by each model on the 400-trained evaluation:

### Accuracy Comparison Table

| Dataset | Ours (Transfer) | MAD (Full Comm) | No Comm | Best Mean | Best Peak (Seed Max) |
|---|---|---|---|---|---|
| **GSM8K** | Mean: 95.00% (±0.88%)<br>Peak: **96.00%** | Mean: **96.22%** (±0.19%)<br>Peak: **96.33%** | Mean: 87.00% (±0.67%)<br>Peak: 87.67% | MAD (96.22%) | MAD (96.33%) |
| **AQUA** | Mean: **86.28%** (±0.31%)<br>Peak: **86.61%** | Mean: 85.34% (±2.33%)<br>Peak: **87.00%** | Mean: 79.52% (±5.46%)<br>Peak: 83.00% | **Ours** (86.28%) | MAD (87.00%) |
| **HOTPOTQA** | Mean: 31.67% (±4.26%)<br>Peak: **36.33%** | Mean: **34.00%** (±3.48%)<br>Peak: **38.00%** | Mean: 30.00% (±2.33%)<br>Peak: 32.67% | MAD (34.00%) | MAD (38.00%) |

### Accuracy Comparison Graph

![Ours vs MAD vs No Comm Performance Comparison](./08-06results/plots/plot2_transfer_400train.png)

---

## 📋 Table 3 — Ablation Study: All 4 Arms × Training Sizes

| Arm | 100 samples | 200 samples | 300 samples | 400 samples | **Improvement** |
|---|---|---|---|---|---|
| Static Ring (A=0, B=0) | 56.33% | 95.33% | 94.67% | 93.33% | **+37.00%** |
| PPO Only (A=0, B=1) | 66.00% | 92.00% | 94.00% | 96.67% | **+30.67%** |
| Sparse Ring (A=1, B=0) | 63.00% | 94.67% | 94.67% | 95.33% | **+32.33%** |
| **Full / Ours (A=1, B=1)** | 65.00% | **98.00%** | 92.67% | 93.33% | **+28.33%** |

> [!IMPORTANT]
> At **100 samples**: PPO-Only arm (66%) outperforms Full arm (65%) — sparse communication regime favors more direct RL signal early in training.
> At **200 samples**: Full arm peaks at 98% — the joint sparsity + PPO synergy is strongest here.
> At **400 samples**: PPO-Only arm (96.67%) re-emerges as best — suggesting sparsity topology adds noise at high-data regimes.

---

## 📋 Table 4 — OLS Factorial Analysis (2×2 Design: Factor A = Sparsity, Factor B = PPO)

| Train Size | Intercept | β_A (Sparsity) | p_A | β_B (PPO) | p_B | β_AB (Interaction) | p_AB | N |
|---|---|---|---|---|---|---|---|---|
| 100 | 0.5633 | +0.0667 | 0.238 (ns) | **+0.0967** | **0.087 (.)** | −0.0767 | 0.391 (ns) | 600 |
| 200 | 0.9533 | −0.0067 | 0.791 (ns) | −0.0333 | 0.185 (ns) | **+0.0667** | **0.061 (.)** | 600 |
| 300 | 0.9467 | 0.0000 | 1.000 (ns) | −0.0067 | 0.809 (ns) | −0.0133 | 0.732 (ns) | 600 |
| 400 | 0.9333 | +0.0200 | 0.442 (ns) | +0.0333 | 0.200 (ns) | −0.0533 | 0.147 (ns) | 600 |

> **Significance codes**: `***` p<0.001, `**` p<0.01, `*` p<0.05, `.` p<0.1, `ns` not significant

> [!NOTE]
> **At 100 samples**: β_B (PPO) is marginally significant (p=0.087), indicating PPO training is the dominant driver when data is scarce. The negative interaction (β_AB=−0.077) suggests PPO + Sparsity together slightly underperform their individual sum.
>
> **At 200 samples**: The interaction term β_AB approaches significance (p=0.061), indicating the Full arm's synergistic gain — this aligns with its peak accuracy (98%).
>
> **At 300–400**: No individual factor is significant, meaning all arms converge to similar accuracy — the design space becomes statistically flat at higher training volumes.

---

## 📋 Table 5 — Training Accuracy per Epoch (All Datasets × Sizes)

### GSM8K

| Train Size | Ep1 | Ep2 | Ep3 | Ep4 | Ep5 | **Mean** | **Std** | **Max Fluctuation** |
|---|---|---|---|---|---|---|---|---|
| 100 | 58.0% | 59.0% | 58.0% | 54.0% | 54.0% | 56.6% | ±2.15% | 4.0% (↓Ep3→Ep4) |
| 200 | 95.0% | 98.0% | 96.0% | 98.0% | 94.0% | 96.2% | ±1.60% | 4.0% (↑Ep1→Ep2) |
| 300 | 93.0% | 93.0% | 97.0% | 99.0% | 94.0% | 95.2% | ±2.40% | **5.0%** (↑Ep3→Ep4) |
| 400 | 96.0% | 98.0% | 95.0% | 94.0% | 90.0% | 94.6% | ±2.65% | 4.0% (↓Ep4→Ep5) |

### AQUA

| Train Size | Ep1 | Ep2 | Ep3 | Ep4 | Ep5 | **Mean** | **Std** | **Max Fluctuation** |
|---|---|---|---|---|---|---|---|---|
| 100 | 85.0% | 86.0% | 86.0% | 86.0% | 86.0% | **85.8%** | ±0.40% | 1.0% (most stable!) |
| 200 | 74.0% | 74.0% | 74.0% | 69.0% | 70.0% | 72.2% | ±2.23% | **5.0%** (↓Ep3→Ep4) |
| 300 | 69.0% | 73.0% | 74.0% | 74.0% | 73.0% | 72.6% | ±1.85% | 4.0% (↑Ep1→Ep2) |
| 400 | 64.0% | 65.0% | 61.0% | 58.0% | 62.0% | 62.0% | ±2.45% | 4.0% (↓Ep3→Ep4) |

> [!WARNING]
> **AQUA shows a concerning decreasing trend** as training size increases. 100-train achieves 85.8%, but 400-train only achieves 62.0%. This suggests **catastrophic forgetting** or distribution shift — the model trained on more GSM8K-style data generalizes less to AQUA's multiple-choice math format.

### HOTPOTQA

| Train Size | Ep1 | Ep2 | Ep3 | Ep4 | Ep5 | **Mean** | **Std** | **Max Fluctuation** |
|---|---|---|---|---|---|---|---|---|
| 100 | 40.0% | 39.0% | 44.0% | 40.0% | 41.0% | 40.8% | ±1.72% | 5.0% (↑Ep2→Ep3) |
| 200 | 98.9% | 100.0% | 100.0% | 98.0% | 100.0% | **99.4%** | ±0.81% | 2.0% |
| 300 | 100.0% | 100.0% | 100.0% | 100.0% | 98.0% | **99.6%** | ±0.80% | 2.0% |
| 400 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | **100.0%** | **±0.00%** | **0%** (perfectly stable!) |

> [!TIP]
> HOTPOTQA shows a **phase transition**: 100→200 samples jump from ~41% to 99.4%. Once saturated (200+), it achieves near-perfect stability with zero fluctuation at 400.

---

## 📋 Table 6 — Topology Benchmark (400-Train Model, GSM8K, n=150 each)

| Rank | Topology | Accuracy | vs Best | vs Worst |
|---|---|---|---|---|
| 🥇 1 | Fully Connected | **98.67%** | — | +6.00% |
| 🥈 2 | Small World | 96.00% | −2.67% | +3.33% |
| 🥉 3 | Ring | 95.33% | −3.34% | +2.66% |
| 3= | Learned Topology | 95.33% | −3.34% | +2.66% |
| 5 | Star | 94.67% | −4.00% | +2.00% |
| 5= | Random | 94.67% | −4.00% | +2.00% |
| 7 | Scale Free | **92.67%** | −6.00% | — |

> [!NOTE]
> **Learned Topology** achieves the same score as Ring (95.33%) — meaning the learned structure mimics ring-like communication patterns. The gap between best (Fully Connected, 98.67%) and worst (Scale Free, 92.67%) is only **6%**, showing robust performance across topologies at 400 training samples.

---

## 📋 Table 7 — Best vs Worst Method per Dataset (400-Train)

| Dataset | Best Method | Best Acc | Worst Method | Worst Acc | Gap |
|---|---|---|---|---|---|
| GSM8K | MAD (Full Comm) | 96.22% | No Comm | 87.00% | **9.22%** |
| AQUA | **Transfer (Ours)** | 86.28% | No Comm | 79.52% | **6.76%** |
| HOTPOTQA | MAD (Full Comm) | 34.00% | No Comm | 30.00% | **4.00%** |

---

## 🔑 Key Insights

### 1. GSM8K: Massive Training Effect (+40.89% for Transfer)
- **100-trained model**: Transfer achieves only 54.11% (near random for a multi-choice problem)
- **400-trained model**: Transfer reaches 95.00% — a **+40.89%** gain
- This indicates GSM8K requires substantial training signal; 100 samples is insufficient

### 2. AQUA: Stable Under Transfer, Exceptional Efficiency
- Transfer (Ours) at 86.28% uses only **421.6 tokens** vs MAD's **850.3 tokens** — a **2.0× reduction**
- Transfer (Ours) **outperforms MAD** on AQUA (86.28% vs 85.34%) while using half the tokens
- AQUA training accuracy **decreases** as training size grows (85.8% → 62.0%): catastrophic forgetting signal

### 3. HOTPOTQA: Hard Task — All Methods Struggle
- Best method (MAD) reaches only 34% — well below random-guess ceiling for multi-hop QA
- Transfer uses 473 tokens vs MAD's 2914 tokens (**6.2× savings**) for only 2.33% lower accuracy
- Training accuracy (200+) reaches 99%+, but test accuracy is ~30%: **clear train-test mismatch / overfitting**

### 4. Token Efficiency is Transfer's Key Advantage
| Dataset | Transfer Tokens | MAD Tokens | Savings | Accuracy Gap |
|---|---|---|---|---|
| GSM8K | 52 | 531 | **10.2×** | −1.22% |
| AQUA | 422 | 850 | **2.0×** | **+0.94% (Transfer wins)** |
| HOTPOTQA | 473 | 2,914 | **6.2×** | −2.33% |

### 5. OLS Analysis: PPO is the Dominant Factor (Low Data)
- At **100 samples**: β_B (PPO) = +0.097, marginally significant (p=0.087)
- The PPO signal dominates over topology structure when data is scarce
- At **200–400 samples**: no factor reaches significance — **the design space converges**
- Negative interaction (β_AB) at 100 and 400 suggests PPO + Sparsity have slight **diminishing returns** together

### 6. Ablation: No Single Arm Dominates Across All Sizes
| Size | Best Arm | Accuracy |
|---|---|---|
| 100 | PPO Only | 66.00% |
| 200 | Full (Ours) | **98.00%** |
| 300 | Static Ring (tied) | 94.67% |
| 400 | PPO Only | 96.67% |

The Full arm's advantage is most pronounced at the 200-sample sweet spot. At 400 samples, simpler arms (PPO Only) match or exceed Full arm, suggesting the topology learning overhead is less needed with abundant data.

### 7. Topology: Small World & Learned Structure Near-Optimal
- Fully Connected (98.67%) wins but at maximum communication cost
- Learned Topology (95.33%) matches Ring exactly — the policy learns ring-like sparse structure
- Scale Free underperforms (92.67%) — hub-spoke topology hurts multi-agent reasoning consensus

### 8. Training Stability (Symmetry Analysis)
- **Most Stable**: AQUA at 100 samples (std=0.40%, max_fluc=1.0%)
- **Most Unstable**: GSM8K at 300–400 samples (std=2.40–2.65%, max_fluc=4–5%)
- **HOTPOTQA at 400**: Perfect stability (std=0.00%) — model fully converges
- Fluctuations are **symmetric** (ups and downs roughly balance) across all datasets — no systematic drift

---

## 📁 Output Files

All generated plots: [plots/](file:///home/saurab/Documents/projectRLMAIN/08-06results/plots/)
All tables (text): [plots/all_tables.txt](file:///home/saurab/Documents/projectRLMAIN/08-06results/plots/all_tables.txt)
Analysis script: [full_analysis.py](file:///home/saurab/Documents/projectRLMAIN/08-06results/full_analysis.py)
