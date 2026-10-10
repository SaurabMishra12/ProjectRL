# COMPREHENSIVE EXPERIMENT AUDIT & DEFECT ROOT CAUSE ANALYSIS

**Research Project**: SPARSECOMM: Towards Causally Grounded Communication Efficiency in Multi-Agent Web Systems  
**Target Venue**: The Web Conference (WWW) 2027  
**Date**: October 7, 2026  
**Auditor**: Antigravity Senior Research Team  
**Evaluation Standard**: Zero Fabrication, Real Data Only, Strict Causal Isolation, Fail-Closed Integrity  

---

## 1. Executive Summary & Audit Mandate

This document establishes a complete, forensic audit of the SPARSECOMM codebase, existing experimental results (P1–P5, W1–W4), data loading mechanisms, and execution pipelines. 

### Critical Prior Experimental Deficiencies Uncovered:
1. **Defect in Prior W3 (Adaptive Stopping)**:
   - *Reported Claim*: Adaptive SPRT stopping dramatically reduced communication while maintaining task accuracy under evidence noise.
   - *Audit Finding*: **The claimed early stopping effect was entirely spurious.** In the prior run, `Clean_Fixed`, `Clean_Adaptive`, `Noisy_Fixed`, and `Noisy_Adaptive` all executed exactly $T = 3.0$ communication rounds on 100% of samples (early exit rate $= 0.0\%$).
   - *Root Cause Analysis*: 
     - The stopping evaluation function `evaluate_stopping_rule()` in `web_suite.py` evaluated SPRT against a fixed threshold $\tau_{SPRT} = \log(19) \approx 2.9444$ (calibrated for large-sample hypothesis tests with $\alpha=0.05, \beta=0.05$).
     - For an $N=3$ agent team, even under unanimous agreement ($k=3$), the maximum binomial log-likelihood ratio with $p_1 = 0.90$ is $\Lambda = 3 \times \log(0.9/0.5) \approx 1.7634$. Because $1.7634 < 2.9444$, the SPRT condition was mathematically impossible to trigger!
     - Furthermore, the loaded pretrained controller weights (`08-06results/hotpotqa_pretrained_400.pt.npz`) produced an actor halt logit of $\approx -97.8$, forcing `halt_decision = False` on every sample.
     - Consequently, `should_halt` was permanently `False`, and all adaptive conditions executed the maximum depth ($T=3$) identically to the fixed conditions.

2. **Defect in Prior W4 (Counterfactual Routing Isolation)**:
   - *Reported Claim*: Routing topology has no independent causal effect once communication volume is controlled.
   - *Audit Finding*: **The conditions were NOT volume-matched.** Although all conditions shared a nominal cap of 100 tokens/round, the actual realized transmitted tokens differed by more than 3.5×:
     - `Learned_SparseComm`: Mean 29.96 transmitted tokens (range 0–140)
     - `Random_Routing`: Mean 57.81 transmitted tokens (range 0–132)
     - `Static_Role`: Mean 66.05 transmitted tokens (range 0–140)
     - `Complete_Broadcast`: Mean 103.98 transmitted tokens (range 0–192)
   - *Root Cause Analysis*:
     - In HotpotQA Web multi-hop information seeking, Agent 0 (Web Search Specialist) produced detailed synthesis paragraphs (40–60 tokens), whereas Agent 1 (Evidence Verifier) and Agent 2 (Reasoning Specialist) frequently produced direct answers (5–7 tokens).
     - The learned controller weights selected edges `[(1, 0), (1, 2)]` in 100% of samples, routing solely from Agent 1. As a result, Learned SPARSECOMM transmitted only Agent 1's short text duplicated across 2 edges ($\approx 12$ tokens/round).
     - Static Role selected `[(0, 1), (1, 2)]`, transmitting Agent 0's long text (40–50 tokens) plus Agent 1's text ($\approx 55$ tokens/round).
     - Complete Broadcast had 6 active edges, transmitting from all 3 agents simultaneously ($\approx 104$ tokens).
     - Because downstream task F1 tracked realized volume (F1: 0.452 for 30 tok $\to$ 0.469 for 58 tok $\to$ 0.472 for 66 tok $\to$ 0.496 for 104 tok), the performance differences were heavily confounded by communication volume rather than routing topology.

---

## 2. Comprehensive Repository Map

| File | Purpose | Associated Experiments | Reusable Status | Required Modifications |
|---|---|---|---|---|
| `config.py` | Experiment & model configuration dataclasses | All (P1–P5, W1–W4) | **Fully Reusable** | None. Contains standard parameters ($N=3, T=3$). |
| `controller.py` | PPO Information-Bottleneck (IB) routing controller & Platt calibrator | P1, P2, P4, P5, W1, W4 | **Fully Reusable** | Retain neural architecture. Account for negative halt logit. |
| `communication.py` | Communication graph base classes & transmission records | P1, P2, P4 | **Fully Reusable** | Reuse message logging structure. |
| `p4_definitive_framework.py` | Rigorous bandwidth control, sentence vs word truncation | P4 | **Fully Reusable** | Model for strict token quotas and per-edge bandwidth. |
| `llm_backend.py` | Azure OpenAI & local LLM interfaces with retry & token tracking | All | **Fully Reusable** | Verified working; tracks latency, token usage, failures. |
| `dataset_loaders.py` | Authoritative HuggingFace dataset loaders for HotpotQA/GSM8K | All | **Fully Reusable** | Synthetic generation strictly raises `DatasetIntegrityError`. |
| `validate_dataset.py` | Dataset verification script enforcing 0 synthetic rows | Web Suite | **Fully Reusable** | Verified; produces `dataset_validation_report.json`. |
| `web_suite.py` | Real BM25 Wikipedia retriever, agent prompts, LLM disk cache | W1–W4 | **Partially Reusable** | Calibrate SPRT stopping; standardize edge token quotas. |
| `run_w1_web_cost_accuracy.py` | Cost-accuracy pareto frontier (NoComm vs MAD vs SPARSECOMM) | W1 | **Audited & Validated** | Verified on 100 real HotpotQA questions; retain results. |
| `run_w2_context_rationing.py` | Semantic rationing vs random word truncation under fixed volume | W2 | **Audited & Validated** | Verified on 100 real HotpotQA questions; retain results. |
| `run_w3_adaptive_stopping.py` | SPRT stopping vs fixed depth under clean/noisy evidence | W3 | **Defective (Requires Fix)** | Replace with `run_w3_corrected.py` (calibrated SPRT). |
| `run_w4_counterfactual_routing.py` | Routing topology counterfactual isolation over frozen pool | W4 | **Defective (Requires Fix)** | Replace with `run_w4_corrected.py` (exact volume matching). |

---

## 3. Experiment Map & Status Matrix

| Experiment ID | Task / Domain | Research Question | Independent Variables | Controlled Variables | Audit Verdict |
|---|---|---|---|---|---|
| **P1** | GSM8K Reasoning | Baseline communication efficiency | NoComm vs Complete vs Learned | Question pool, prompt, backbone | **PRESERVED** (Controlled foundation) |
| **P2** | GSM8K Reasoning | Volume alteration under routing | Topology (Sparse vs Dense) | Backbone, prompts | **PRESERVED** (Shows routing changes vol) |
| **P2b** | GSM8K Reasoning | Permutation test of agent roles | Role assignments | Topology, volume | **PRESERVED** (Valid role check) |
| **P4** | GSM8K / AQuA | Volume-matched synthetic control | Edge allocation, semantic mode | Nominal cap = 100 | **PRESERVED** (Synthetic math baseline) |
| **P5** | GSM8K Reasoning | Communication control suite | Budget (25, 50, 100 tok) | Prompts, seeds | **PRESERVED** (Ablation benchmark) |
| **W1** | Open-Book HotpotQA | Web information seeking efficiency | NoComm vs MAD vs SPARSECOMM | 100 real multi-hop questions | **AUDITED & VALIDATED** (Real data, 84.8% reduction) |
| **W2** | Open-Book HotpotQA | Semantic vs random rationing | Full vs Semantic vs Random | Realized volume matched ($\approx 66$ tok) | **AUDITED & VALIDATED** (+2.1% F1 semantic advantage) |
| **W3** | Open-Book HotpotQA | Adaptive stopping effect | Fixed ($T=3$) vs Adaptive SPRT | Clean vs Noisy evidence | **DEFECTIVE $\to$ REPLACED BY W3-CORRECTED** |
| **W4** | Open-Book HotpotQA | Pure causal effect of routing | Static vs Learned vs Random vs Complete | Frozen pool, fixed depth ($T=2$) | **DEFECTIVE $\to$ REPLACED BY W4-CORRECTED** |

---

## 4. Audit of Fallback Paths: Zero Toy/Synthetic Guarantee

A rigorous codebase grep was conducted across all files for: `toy`, `synthetic`, `dummy`, `mock`, `fallback`, `fake`, `emergency`, `sample_data`.

### Findings:
1. `dataset_loaders.py`:
   - Line 24: `def _generate_synthetic_samples(...)` explicitly contains:
     ```python
     raise DatasetIntegrityError(
         "CRITICAL: Synthetic/toy data generation is strictly prohibited by non-negotiable integrity rules. "
         "The experiment must run on real benchmark data or fail closed."
     )
     ```
   - No silent fallback can execute. If the HuggingFace cache or dataset path is missing, a hard exception is raised.
2. `validate_dataset.py`:
   - Actively checks for synthetic math word problem markers (`"Alice has"`, `"Bob gives"`, `"apples"`, `"pencils"`).
   - Any match raises an immediate `DatasetIntegrityError`.
   - Verified: 0 synthetic rows in `web_eval_subset_100.json`.
3. `web_suite.py`:
   - Line 349: `# Word-level fallback if first sentence is longer than budget` is purely an algorithmic token truncation fallback (using binary search) when a single sentence exceeds the token budget; it does NOT generate synthetic data.
4. **Verdict**: **TOY DATA IS IMPOSSIBLE TO SILENTLY ACTIVATE.** The system fails closed.

---

## 5. End-to-End Pipeline Documentation

### A. Data Pipeline
```
Local Wikipedia Corpus (fullwiki validation split)
  │
  ▼
BM25 Retriever (LocalWikipediaRetriever: k1=1.5, b=0.75)
  │ (Clean: top-2 articles; Distractor: all 10 candidate articles)
  ▼
Agent Round 0 Initial Reasoning (Specialized roles: Search, Verifier, Reasoner)
  │
  ▼
Frozen Information Pool (Saved & immutable; message SHA-256 hashes verified)
  │
  ▼
Communication Round 1..T (Strict volume quota / Edge routing matrix)
  │
  ▼
Agent Re-solving with peer messages
  │
  ▼
Stopping Rule Evaluation (SPRT LLR vs Calibrated threshold tau=1.386)
  │
  ▼
Aggregator (Platt-calibrated weighted majority vote)
  │
  ▼
Evaluator (Exact Match & Token-level F1 against official gold answers)
  │
  ▼
Immutable Output Logging (Per-question raw JSONL record stream)
```

### B. Model & Inference Pipeline
- **Backbone Model**: `gpt-oss-120b` deployed on Azure OpenAI.
- **Provider Endpoint**: `https://vaibhavka43-0128-resource.openai.azure.com/` (API version `2024-02-15-preview`).
- **Generation Parameters**: `temperature=0.0`, `top_p=1.0`, `max_tokens=512`.
- **System Prompts**: Functionally heterogeneous roles (Agent 0: Retrieval, Agent 1: Evidence Verification, Agent 2: Multi-Hop Reasoning).
- **Persistent Disk Cache**: `cache/web_llm_cache.json` indexed by SHA-256 hash of `agent_{id}|r_{round}|sys_{prompt}|usr_{prompt}`.
- **Accounting**: Live API calls, cache hits, token usage, latencies, and API failures are strictly tracked.

---

## 6. Action Plan for Corrected Implementations

1. **Implement `run_w3_corrected.py`**:
   - Calibrate the SPRT stopping rule for an $N=3$ agent deliberation team ($\tau = 1.386$, equivalent to 4:1 odds).
   - Allow consensus-based early exit after Round 1 or Round 2 when unanimous ($k=3$) or high-confidence majority ($k=2, \text{conf} \ge 80\%$).
   - Record exact round distribution, stopping reasons, token counts, latency, and task accuracy.
2. **Implement `run_w4_corrected.py`**:
   - Enforce **strict realized volume matching**: standardize communicated payload length across conditions for every question ($2 \times L_{min}$ tokens across active edges, matching with $\le 1\%$ relative tolerance, 0.00% across 2-edge topologies).
   - Evaluate Static Role, Learned SPARSECOMM, Random Routing, and Complete Broadcast over the identical frozen message pool at fixed depth $T=2$.
   - Compute paired bootstrap/permutation statistics, 95% confidence intervals, and effect sizes.
