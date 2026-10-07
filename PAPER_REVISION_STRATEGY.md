# PAPER REVISION STRATEGY & RESPONSE ACTION PLAN
**Target Paper**: *"Routing or Rationing? Decomposing Communication Control in Multi-Agent Language Models"* (ACL Anonymous Submission)

---

## Executive Summary

This document provides an exhaustive, actionable strategy resolving all reviewer criticisms and code audit findings. The plan is organized into three main pillars:

1. **Four Zero-Cost Correctness Fixes**: Resolving mathematical and state-specification errors in the paper text (pure gain, zero reviewer penalty).
2. **Framing & De-Hedging Overhaul**: Consolidating caveats into the post-Conclusion *Limitations* section (outside ACL page limits), removing self-editorializing text, and tightly scoping claims.
3. **Ranked Experiment Priority Matrix**: A cost-benefit framework (Reviewer Convergence vs. API Cost) for targeted re-runs, including a minimal fixed-budget routing experiment.

---

## 1. Category Errors & Correctness Bugs (Pure Gain)

The following four items represent technical corrections. Fixing them involves zero risk of reviewer penalty and immediately strengthens the paper's rigor.

### Bug 1: "Running Accuracy" in Controller State (§3.3)
- **Issue**: Reviewers expressed concern that if the controller state consumes gold labels at evaluation time, evaluation is invalid.
- **Code Audit Evidence**: Inspection of `pipeline.py` (lines 504–637) confirms that during evaluation (`train_controller=False`), `avg_accuracy` is passed as a static scalar (the final epoch accuracy from training, or `0.5` for random baseline). It **never** accesses test-set gold labels.
- **Actionable Fix**: Clarify in one sentence in §3.3:
  > *"The scalar state feature $s_{\text{acc}}$ represents the controller's historical accuracy over the training split and remains fixed during evaluation, ensuring test-set evaluation never accesses ground-truth labels."*

### Bug 2: Proposition 1 Off-Diagonal Matrix Norm Mismatch
- **Issue**: Volume is defined over off-diagonal communication ($\text{Vol}(W) = \sum_{i \neq j} W_{ij}$), but Proposition 1's statement used the full-matrix $\ell_1$ norm $\|W\|_1 = \sum_{i,j} |W_{ij}|$.
- **Actionable Fix**: Replace $\|W\|_1$ with the off-diagonal $\ell_1$ norm $\|W\|_1^{\text{off}} = \sum_{i \neq j} |W_{ij}|$ in the statement and proof sketch of Proposition 1.

### Bug 3: Minimal Detectable Effect (MDE) Misinterpretation
- **Issue**: Stating *"Any effect larger than ~15 points would have shown up"* conflates 80% power under null hypothesis testing with an empirical confidence bound.
- **Actionable Fix**: Replace MDE phrasing with explicit 95% paired question-clustered bootstrap confidence intervals $[-\Delta_{\text{lower}}, +\Delta_{\text{upper}}]$ and non-inferiority margins.

### Bug 4: Malformed PPO Clip Operator Syntax (§3.4, Eq. 6)
- **Issue**: The text currently writes $\text{clip}(\rho, 1 \pm \varepsilon)$ using a single bound argument.
- **Actionable Fix**: Correct the notation to standard two-argument interval syntax:
  $$\text{clip}\left(\rho_{ij}, 1-\varepsilon, 1+\varepsilon\right)$$

---

## 2. Framing & De-Hedging Strategy

To maximize reviewer convergence and paper impact, the manuscript framing will be systematically restructured:

1. **Single-Location Caveats (Post-Conclusion Limitations)**:
   - All caveats, bounds, and scope limits will be stated **once**, precisely, in the dedicated *Limitations* section (which is outside the ACL 8-page main text limit).
   - Remove redundant hedging from the Abstract, Introduction, and Results sections to prevent reviewer pile-on.

2. **Strip Self-Editorializing**:
   - Eliminate meta-narratives (e.g., avoid writing anything resembling *"this is a negative result plus a proposal"*).
   - Present the contribution purely objectively as a *controlled empirical decomposition isolating communication topology from communication volume*.

3. **Tight Claim Scoping**:
   - Scope all claims strictly to the empirical evidence: a controlled comparison against fully-connected debate (MAD) on equal compute/budget grounds.
   - Avoid claiming broad state-of-the-art communication efficiency across all scale regimes.

---

## 3. Experiment Priority Under Budget Constraints

Experiments are ranked by **Reviewer Convergence / API Cost**:

| Rank | Item | API Cost | Cited By | ROI & Action Plan |
| :---: | :--- | :---: | :---: | :--- |
| **1** | **Specify/audit "running accuracy"** | **$0** (Code Audit) | All 4 | **Immediate Win**: Clarify in §3.3 that $s_{\text{acc}}$ is frozen from training. |
| **2** | **Paired question-clustered bootstrap & non-inferiority test** | **$0** (Uses existing logs) | R2, R4 | **High ROI**: Re-analyze archived `.jsonl` streams; replace 3-seed Welch $t$-test. |
| **3** | **Extract per-cell realized volume from 2×2 grid** | **$0** (Uses existing logs) | R1, R4 | **High ROI**: Compute token volume distribution across factorial arms from logs. |
| **4** | **Minimal fixed-budget routing experiment** | **Low** (1 targeted run) | All 4 | **Core Empirical Addition**: Run GSM8K fixed-budget routing comparison. |
| **5** | **Filtering vs. random/truncation at matched budget** | **Medium** (2nd spend) | R2, R4 | Secondary experiment if API budget permits after #4. |
| **6** | **Trained $N=5$ controller & AgentPrune / AgentDropout baselines** | **High** (Expensive) | R2, R4 | **Skip**: Address via claim-scoping against fully-connected debate instead. |

---

## 4. Log Archive & Data Dependencies

Items #2 and #3 rely on per-question log availability:
- **Status**: Confirmed present in `08-06results/continuous/*.jsonl` (e.g., `Multi-Agent Debate_gsm8k_stream.jsonl`, `No Communication_gsm8k_stream.jsonl`, `Arm_01_PPOOnly_gsm8k_stream.jsonl`, `Arm_11_Full_gsm8k_stream.jsonl`).
- **Reviewer 4 Response**: Export clean row-level prediction CSVs into the archived repository data package so bootstrap resampling and realized volume extraction can be 100% independently reproduced without re-running evaluation scripts.

---

## 5. Minimal Fixed-Budget Routing Experiment Design (Item #4)

Rather than proposing a massive full-factorial sweep, run a lean, high-impact instance:

- **Dataset**: GSM8K ($n=150$ or $n=400$ matched questions, shortest trace length).
- **Topologies**: Ring, Star, Random, and Learned Support.
- **Fixed Token Budgets**: $V_1 = 100$, $V_2 = 200$ tokens/round (truncated at allocation).
- **Outcome**: Converts *"we propose an experiment"* into *"we ran an initial instance of it"*, directly removing Reviewer 2's primary blocker.
