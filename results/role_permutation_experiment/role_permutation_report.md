# Protocol P4: Role Permutation Experiment Report

## Scientific Objective
Disentangle whether the out-star broadcasting advantage in Protocol P4 is driven by:
1. **Semantic Role** (Analytical persona vs. Creative persona),
2. **Node Index** (Positional bias of Agent 0 vs. Agent 2), or
3. **Role × Index Interaction**.

## Experimental Matrix
| Configuration | Agent 0 Role | Agent 1 Role | Agent 2 Role |
|---|---|---|---|
| **Config A** | Analytical | Step-by-step | Creative |
| **Config B** | Creative | Step-by-step | Analytical |
| **Config C** | Step-by-step | Creative | Analytical |

## Results Across Budgets

### Token Budget V̄ = 50

| Cell Key | Config | Topology Root | Root Role | Accuracy | 95% Wilson CI |
|---|---|---|---|---:|:---:|
| `Config_A_Root0_V50` | Config_A | Root0 (Node 0) | **Analytical** | 76.67% | [69.3%, 82.7%] |
| `Config_A_Root2_V50` | Config_A | Root2 (Node 2) | **Creative** | 76.67% | [69.3%, 82.7%] |
| `Config_B_Root0_V50` | Config_B | Root0 (Node 0) | **Creative** | 82.00% | [75.1%, 87.3%] |
| `Config_B_Root2_V50` | Config_B | Root2 (Node 2) | **Analytical** | 82.00% | [75.1%, 87.3%] |
| `Config_C_Root0_V50` | Config_C | Root0 (Node 0) | **Step_by_step** | 83.33% | [76.6%, 88.4%] |
| `Config_C_Root2_V50` | Config_C | Root2 (Node 2) | **Analytical** | 84.00% | [77.3%, 89.0%] |

#### Causal Decomposition Contrasts
- **Config A (Standard)**: $\Delta(G_0 - G_2) = +0.00$ points
- **Config B (Swapped)**: $\Delta(G_0 - G_2) = +0.00$ points
- **Net Role Effect (Analytical − Creative @ Node 0)**: $-5.33$ points
- **Net Role Effect (Analytical − Creative @ Node 2)**: $+5.33$ points
- **Net Index Effect (Node 0 − Node 2 @ Analytical)**: $-5.33$ points
- **Net Index Effect (Node 0 − Node 2 @ Creative)**: $+5.33$ points

**Causal Verdict**: **ISOMORPHIC_ROOT_INVARIANCE (Zero positional index advantage: |Δ(G0 - G2)| ≤ 0.67 pp, McNemar p = 1.000; team role assignment and epistemic composition govern group performance)**

---

### Token Budget V̄ = 100

| Cell Key | Config | Topology Root | Root Role | Accuracy | 95% Wilson CI |
|---|---|---|---|---:|:---:|
| `Config_A_Root0_V100` | Config_A | Root0 (Node 0) | **Analytical** | 76.67% | [69.3%, 82.7%] |
| `Config_A_Root2_V100` | Config_A | Root2 (Node 2) | **Creative** | 77.33% | [70.0%, 83.3%] |
| `Config_B_Root0_V100` | Config_B | Root0 (Node 0) | **Creative** | 82.00% | [75.1%, 87.3%] |
| `Config_B_Root2_V100` | Config_B | Root2 (Node 2) | **Analytical** | 82.67% | [75.8%, 87.9%] |
| `Config_C_Root0_V100` | Config_C | Root0 (Node 0) | **Step_by_step** | 84.00% | [77.3%, 89.0%] |
| `Config_C_Root2_V100` | Config_C | Root2 (Node 2) | **Analytical** | 84.00% | [77.3%, 89.0%] |

#### Causal Decomposition Contrasts
- **Config A (Standard)**: $\Delta(G_0 - G_2) = -0.67$ points
- **Config B (Swapped)**: $\Delta(G_0 - G_2) = -0.67$ points
- **Net Role Effect (Analytical − Creative @ Node 0)**: $-5.33$ points
- **Net Role Effect (Analytical − Creative @ Node 2)**: $+5.33$ points
- **Net Index Effect (Node 0 − Node 2 @ Analytical)**: $-6.00$ points
- **Net Index Effect (Node 0 − Node 2 @ Creative)**: $+4.67$ points

**Causal Verdict**: **ISOMORPHIC_ROOT_INVARIANCE (Zero positional index advantage: |Δ(G0 - G2)| ≤ 0.67 pp, McNemar p = 1.000; team role assignment and epistemic composition govern group performance)**

---
