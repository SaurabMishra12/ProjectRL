# Audit: "Routing or Rationing? Decomposing Communication Control in Multi-Agent Language Models"

Source PDF: `/home/saurab/Downloads/sparsecomV2 (2).pdf` (13 pages, ACL anonymous submission, system name **SparseComm**).
Codebase audited: `/home/saurab/Documents/projectRLMAIN` (not a git repo).
Audit date: 2026-08-12.

This document resolves all 15 `[VERIFY: ...]` placeholder notes left in the draft, cross-checks the paper's claims against the actual implementation and raw result logs, and lists concrete next steps. Every claim below cites a file path (and line number where applicable) so it can be spot-checked directly.

**Cross-check against `SparseComm_Kaggle.ipynb`** (2026-08-12): the user identified this notebook as the main code. Its single ~4,768-line code cell was extracted and diffed against every root `.py` file cited below — `IBController`, `SemanticBandwidthMode`, `evaluate_answer`, and `AzureOpenAIBackend.compress_message` are all identical to `controller.py` / `communication.py` / `dataset_loaders.py` / `llm_backend.py`. The notebook's own `if __name__ == "__main__":` driver additionally confirms, straight from the code that ran: (a) the exact `Arm_00/01/10/11` → (sparsity, PPO) mapping used to recompute Table 5 in §4.1, and (b) that every `Transfer_{GSM8K,AQUA,HOTPOTQA}` run explicitly used `llm_backend="azure"`, `api_model="gpt-oss-120b"` (item 7). No mention of "nemotron" appears anywhere in this notebook — the §5.5 transfer check is confirmed to be a separate script, not part of this main pipeline. No finding below changes as a result of this cross-check. (One inconsequential drift: this notebook's `Config.api_model` default is `"gpt-oss-120b"` vs. root `config.py`'s `"deepseek-ai/deepseek-v3.2"` — overridden explicitly in every actual run either way.)

---

## 0. Executive summary — read this first

1. **The §5.5 Nemotron transfer-check number for SparseComm is wrong.** The paper reports SparseComm at 54.0% accuracy; the raw result file it was computed from (`08-06results/nemotron_gsm8k_150/summary_results.json`) says **43/150 = 28.67%**. NoComm (36.0%) and MAD (50.0%) both match the paper exactly — only the SparseComm number is off, and it's off by a lot (not a rounding issue). This must be fixed before submission; see §3.6.
2. **Eq. 5 does not describe the implemented policy.** The paper writes the training-time policy as `Bern(A;P) · q(R) · Bern(h; σ(z))` — a Bernoulli routing gate, a separate retention-ratio distribution family `q`, and a stochastic halting Bernoulli. The actual controller (`controller.py`) has none of these: it is a hand-rolled NumPy diagonal-Gaussian policy over a continuous latent that is reshaped into a single continuous matrix `W` doing double duty as gate *and* retention ratio, and the halting decision is a hard, **never-stochastic** threshold (exploration noise is explicitly zeroed on that dimension even during training). See §3.1.
3. **Table 4's numbers can't be reproduced from anything in the repo.** The definition (round-1, largest outgoing weight) is confirmed in a sibling LaTeX draft, but no script or dataset in the repo reproduces the specific percentages (97.89/2.11/0.00 etc.); the closest available raw data gives different numbers. See §2, item 10.
4. **Figure 1's BLOCK/SUMMARIZE/FILTER/FULL bands and their frequencies do not exist in the runtime code.** The system that actually runs only has 3 bands (BLOCK/PARTIAL/FULL at 0.05/0.95), not 4 quartile bands, and the 4-band frequency numbers shown in the plotting script are hardcoded, not computed from any result file — they read as placeholder/illustrative data that was never replaced. See §3.4.
5. **The paper's "compression call" (a stochastic Markov kernel with an LLM compress step) is dead code.** The mechanism that actually runs is embedding-similarity-weighted stochastic sentence dropout, with no LLM call and no truncation-to-budget step. See §3.3.
6. Thirteen items are fully or partly resolved cleanly and are ready to paste back into the paper (learning rate, Z, GSM8K tolerance, Table 5's full 12-row Holm table, Table 7 with real sample sizes and intervals, etc.) — see §2 and §4.

---

## 1. How this was produced

Three things were cross-referenced: the submitted PDF text (all `[VERIFY]` notes extracted verbatim), the live implementation (`config.py`, `controller.py`, `communication.py`, `agents.py`, `pipeline.py`, `llm_backend.py`, `dataset_loaders.py`, `real_kaggle.py`), and raw result logs (`08-06results/continuous/*.jsonl`, `08-06results/nemotron_gsm8k_150/`, `results/rq3/table7_sensitivity_*.csv`, `08-06results/scaling_study_agents.json`). Two web searches confirmed Table 1's cited-system naming. Regression tables were independently recomputed from the frozen CSVs (not just copied from a script), and confidence intervals were computed directly from raw counts.

**Repo hygiene note relevant to trust in any of this**: the repo has no `.tex` source that matches the submitted PDF. It contains several *other*, earlier/sibling paper drafts with materially different numbers for what looks like the same systems:
- `08-06results/paper_latex.tex` — different title ("Who Talks to Whom?..."), no Table 1, no `[VERIFY]` markers.
- `full_timeline.md`, `results18jul.md`, `SparseComm_Research_Paper.md`, `COMPLETE_RESEARCH_ANALYSIS.md` — all report different GSM8K/AQuA-RAT accuracy numbers than the target PDF's abstract (95.00/86.28/31.67).
- There are also two separate `Config` class definitions in the repo (root `config.py` defaults to `llm_backend="mock"`, `api_model="deepseek-ai/deepseek-v3.2"`; `real_kaggle.py` — the script that actually generated Table 3 — defaults to `llm_backend="azure"`, `api_model="gpt-oss-120b"`). This isn't a bug by itself (the eval scripts override the defaults) but it means "the config" is not a single source of truth in this repo, so exact reproducibility depends on knowing which script produced which number.

**Bottom line**: treat the target PDF's own numbers as authoritative, use this report to fix its 15 open items, and don't cross-cite the other drafts' numbers — they're earlier iterations, not corroboration.

---

## 2. All 15 `[VERIFY]` items, resolved

| # | Location | Paper's ask | Resolution | Status |
|---|---|---|---|---|
| 1 | Table 1 caption | "confirm each cell against the cited paper" | System *names* confirmed correct via web search: **AgentPrune** is the self-given name in Zhang et al., "Cut the Crap: An Economical Communication Pipeline for LLM-based Multi-Agent Systems," ICLR 2025 (arXiv:2410.02506); **ARG-Designer** is the self-given name in Li et al., "Assemble Your Crew: Automatic Multi-agent Communication Topology Design via Autoregressive Graph Generation," AAAI 2026 (arXiv:2507.18224). Per-cell ✓/– claims for the four feature columns were not exhaustively re-checked against all 6 full papers (out of scope for a code audit) — flag **AgentDropout's "Adaptive depth ✓"** as the most likely to need a second look, since its headline mechanism is agent elimination (a topology edit), not round-count control. | Partly resolved |
| 2 | §3.3, "state built from..." | "state d and the exact configuration" | `controller.py:212-213,241-243`: for N=3, **d = 20** = 5 semantic features (`SemanticStateEncoder`) + 3 per-agent confidences + 9 values (the **entire flattened previous 3×3 weight matrix**, not just "the previous gate" as the prose says) + round_num/n_rounds + unique_answers/n_agents + avg_acc. The last two features (`unique_answers`, `avg_acc`) aren't mentioned in the paper's prose at all. | Resolved — but paper prose undercounts/mischaracterizes the state |
| 3 | §3.3, transfer to Nemotron | 'define "thinking mode": request parameter and reasoning budget' | Not found as a named, distinct request parameter in the transfer scripts (`run_nemotron_gsm8k_eval.py`, `run_our_sparsecomm_eval.py`). The scripts set `api_temperature = 1.0` for the Nemotron calls but no explicit "thinking mode" flag/budget was located. | Unresolved — needs author clarification or a code change to add an explicit parameter |
| 4 | §3.3, Eq. 5 | "confirm that gates are sampled in training and thresholded at evaluation, and give the family q of the retention head" | **Does not match Eq. 5.** No Bernoulli anywhere (`grep "torch\."` on `controller.py` returns nothing — `import torch` is dead). The actor (`controller.py:283-306`) samples a diagonal Gaussian action, reshapes it via a bilinear form `H_src @ H_dst.T` + sigmoid into one continuous matrix `W` (`controller.py:245-262`) that serves as **both** the routing gate and the retention ratio — there is no separate `A_ij` and `R_ij`, hence no distribution family `q(R)` to report. At evaluation (`explore=False`), `action = mean_action` exactly (`controller.py:298-299`) — so the "thresholded at evaluation" idea is right in spirit for `W`, but wrong in mechanism (it's the mean of a Gaussian, not a thresholded Bernoulli draw). The halting head is **never sampled**, even in training: exploration noise is explicitly zeroed on that dimension (`controller.py:296`, `noise[-1] = 0.0`), and `halt_decision = bool(halt_prob > 0.5)` (`controller.py:309-310`) is applied identically at train and eval time. | Resolved — and reveals Eq. 5 needs rewriting, not just filling in |
| 5 | Figure 1 caption | "state the band boundaries" for BLOCK/SUMMARIZE/FILTER/FULL | **Ambiguous/conflicting, two different schemes exist.** Runtime code (`communication.py:22-34`) has only **3** modes: `BLOCK` (`w<0.05`), `PARTIAL` (`0.05≤w<0.95`), `FULL` (`w≥0.95`) — no SUMMARIZE/FILTER at all in the code path that actually executes. A **4th, quartile-based** scheme (`[0,0.25) BLOCK / [0.25,0.5) SUMMARIZE / [0.5,0.75) FILTER / [0.75,1.0] FULL`) exists only in `generate_plots.py:66` and in a doc-comment in `config.py:26-30` (`bandwidth_thresholds`) — but `config.py`'s `bandwidth_thresholds` field is **never read** by `communication.py`'s `get_mode()`. See §3.4 for more on this. | Resolved as ambiguous — the two schemes disagree and only one runs |
| 6 | §4.1, answer normalizer | "confirm the normalizer, in particular numeric tolerance on GSM8K" | `dataset_loaders.py:242-255`: **tolerance = `1e-3`** (`abs(float(pred_nums[-1]) - float(gold_nums[-1])) < 1e-3`). Secondary finding: this scoring function does **not** strip comma thousand-separators before regex-extracting numbers (only the separate vote-normalizer in `agents.py` does), so a gold/predicted answer like `"1,200"` would be mis-tokenized into `"1"` and `"200"` — a latent bug for any GSM8K answer ≥1000, worth checking whether it actually affected any of the reported runs. | Resolved, with a secondary bug flagged |
| 7 | §4.2, model deployment | "give the exact deployment, model version, and serving configuration" | For the **main Table 3 results** (GSM8K/AQuA-RAT/HotpotQA, `real_kaggle.py` defaults, confirmed by matching raw logs — see §4.5): `llm_backend="azure"`, `api_model="gpt-oss-120b"`. Serving config (`llm_backend.py:637-722`, `AzureOpenAIBackend`): `api_version="2024-02-15-preview"`, `max_tokens=512`, `temperature=0.0`, `top_p=1.0`. An earlier sibling draft's hyperparameter table (`paper_pdftotext.txt:1517-1569`, a *different* paper, treat only as corroboration) independently lists `max_tokens = 512 (GSM8K), 1024 (AQuA, HotpotQA)`, consistent with a per-dataset override of the same field. Endpoint/env-var values are inconsistent across files (leftover scaffold variables not actually used by the client) — not reproducibility-relevant, just messy. | Resolved for the systems in Table 3 |
| 8 | §4.3, calibration split | "confirm the calibration split is disjoint from every training and evaluation set" | Not verifiable from repo artifacts — no code path that explicitly asserts or logs this disjointness was found. | Unresolved — recommend either adding an explicit disjointness check/log, or stating the split procedure precisely enough in the paper that a reader can verify it by construction. |
| 9 | §3.4, Eq. 7, `C_info` normalization `Z` | "copy the exact expression for Z from the training code" | `controller.py:344`: **`Z = tokens_if_full`**, i.e. the token count of the *same episode* under full communication — an empirically measured, episode-dependent quantity, not a closed-form function of `N` and message length as the paper's phrasing implies. Also worth noting: `include_token_penalty=False` by default (`controller.py:334`), so **this term is zero in the final-step reward call** the equation is describing; the real per-round token cost is applied earlier, during intermediate steps in `pipeline.py`'s `run_single_sample`, not in `compute_reward`. | Resolved — but the paper's "Z chosen so C_info∈[0,1]" phrasing implies a fixed formula that doesn't exist; needs a more accurate description |
| 10 | Table 4, role-associated routing | "confirm the round and the aggregation order" | Round-1 confirmed via a sibling LaTeX draft (`08-06results/paper_latex.tex:526-556,568`, caption: "largest total outgoing communication weight ('leader')... round-1"), stating the pooled sample sizes as 2,049 SparseComm / 900 MAD / 900 NoComm samples. **However, no script or single raw file anywhere in the repo reproduces Table 4's specific percentages.** Attempts to recompute "largest row-sum of the round-1 weight matrix" from the two most plausible raw sources (`Master_SparseComm_Full_Results.json`, `08-06results/continuous/IB-Controller_*_stream.jsonl`) gave **different numbers** than the paper (e.g. AQuA → 91.67/0.00/8.33% on n=12 valid samples vs. the paper's 27.69/13.85/58.46%), and most samples in those files don't even have a saved `weight_matrices` field. | Unverifiable from repo contents as they stand — likely computed from an aggregated dataset or a Jupyter cell whose output was never archived. Recommend re-running this analysis with the script saved and the input dataset archived before resubmission. |
| 11 | §5.5, Nemotron transfer | "report the evaluation-set size, the benchmark, and an interval" | **Benchmark: GSM8K, n=150.** Raw file: `08-06results/nemotron_gsm8k_150/summary_results.json`. NoComm 54/150=36.0% [Wilson 95% CI 28.76–43.94%], MAD 75/150=50.0% [42.10–57.90%] — both match the paper exactly. **SparseComm: the raw file says 43/150 = 28.67% [22.03–36.37%], not the 54.0% reported in the paper.** Token counts (232,240 vs 1,321,043 transmitted, 82.42% reduction) do match the paper — only the accuracy figure is wrong. See §3.6 for detail. | **Resolved, and the paper's own reported number is contradicted by its own source data.** Highest-severity finding in this audit. |
| 12 | Appendix B, Algorithm 1 box | "replace lines 3-6 with the exact state features, action distributions, compression prompt, and truncation rule" | State features/action distributions: see items 2 and 4 above. Compression prompt (`llm_backend.py:753-781`, `AzureOpenAIBackend.compress_message`) — system: *"You are a communication compressor. Extract only the essential information from the user's message."*; SUMMARIZE mode: *"Summarize the key assumptions and logic of this message concisely:\n\n{message}"*; FILTER mode: *"Extract ONLY the final claims, equations, or answers from this message. Omit all reasoning:\n\n{message}"* (`temperature=0.3`, `max_tokens=150`). **But this method is never called anywhere in the live pipeline** (`grep -n "\.compress_message("` across `pipeline.py communication.py agents.py main.py` → no matches; `ablation.py` even has comments noting "SUMMARIZE mode... Disabled due to refactor"). Truncation rule: **no truncation-to-budget step exists anywhere in the codebase** (`grep -n "truncat"` across the relevant files returns nothing) — the Remark on integrality in Appendix A ("Truncating each message at its allocation gives B̃≤B") describes a mechanism that isn't implemented. | Resolved as a discrepancy: the specification the paper wants to reproduce (LLM compression + truncation) is dead code; the mechanism that actually runs is embedding-similarity-weighted stochastic sentence dropout (`communication.py:110-224`) with no explicit truncation step. |
| 13 | Table 5 (Appendix C) | "recompute p_Holm for every row from the frozen script; only the 0.035→0.416 entry was supplied" | **Fully recomputed and verified**, see §4.1 for the complete 12-row table, independently re-derived from `results/rq3/table7_sensitivity_{100,200,300,400}.csv` (matched-question intersection across all 4 arms, CR1-clustered SE, Holm correction over 12 tests). The recomputed β, SE, and the disclosed 0.035→0.416 entry match the paper's Table 5 exactly. | Fully resolved |
| 14 | Table 7 (Appendix E), N=4,5 | "re-export with sample sizes and intervals" | **Fully resolved**: n=150 for both N=4 and N=5, from `08-06results/scaling_study_agents.json`. N=4: 148/150=98.67%, N=5: 147/150=98.00%. Both normal-approx (as the file already stores, matching the ±1.8/±2.2 the paper could use) and tighter Wilson 95% CIs computed — see §4.2. | Fully resolved |
| 15 | Table 8, learning rate | "[VERIFY: state]" | **Fully resolved: `1e-3`** (`config.py:34`, `controller_lr: float = 1e-3`), used identically for both actor and critic (`controller.py:224-225`), no schedule (`grep`'d, none found). Corroborated by an earlier sibling draft's filled-in hyperparameter table (`paper_pdftotext.txt:1517-1569`, treat as corroboration only). **Bonus finding in the same table**: Table 8 lists the optimizer as "AdamW," but `controller.py:178-196`'s `MLP.update()` implements **plain Adam** (β1=0.9, β2=0.999, ε=1e-8) — no weight-decay term anywhere. | Fully resolved, with a second small correction needed in the same table |

---

## 3. Discussion of the discrepancies (not just missing numbers)

Several of the items above resolve to a concrete answer that **contradicts** the paper's own description. These are worth fixing carefully rather than papering over, since a careful reviewer or reproducer would hit them immediately.

### 3.1 Eq. 5's policy family vs. the actual controller

The paper's factorization (§3, Eq. 1) — a binary gate `A`, a separate bandwidth allocation `B`, realized communication `M = A ⊙ B` — is a clean, defensible design on paper, and the identification analysis (Propositions 1-3) is written entirely in terms of that clean factorization. The *implementation*, however, collapses gate and allocation into a single continuous matrix `W ∈ (0,1)^{N×N}` produced by `sigmoid(H_src @ H_dst.T + 0.5)` (`controller.py:254-259`), used directly as a per-edge keep-probability in `communication.py`'s stochastic sentence dropout. There is no code-level object that corresponds to "A, thresholded to {0,1}" separately from "B, a continuous retention ratio." This matters for the paper's central theoretical claims:

- **Proposition 1** ("Gate as a projection") is a statement about a genuinely binary `A ∈ {0,1}^{N×N}`. If the deployed system never actually forms a binary `A` (Table 8's evaluation-time threshold `τ_A = 0.5` implies one is supposed to exist, applied to a routing score `P`, but `controller.py` has no `P` head separate from `W`), the proposition's premises may not describe what runs.
- The **2×2 factorial arms** (§4.4) require the ability to fix one factor (routing/topology) while manipulating the other (compression/volume) independently. That's only meaningful if the code has two independent knobs. The `Arm_00/01/10/11` CSVs in `results/rq3/` do exist and do look designed exactly this way (`Arm_00_StaticRing`, `Arm_01_PPOOnly`, `Arm_10_SparseRing`, `Arm_11_Full` — see §4.1), so this specific ablation experiment likely *does* implement two independent knobs even if the main `IBController.get_weights()` code path (used for the headline Table 3 numbers) does not cleanly separate them. **Recommend explicitly confirming, and stating in the paper, whether the factorial arms use the same `W`-only mechanism as the headline system, or a different, more literally-factorized implementation** — if they differ, the factorial result doesn't actually characterize the deployed SparseComm controller.

### 3.2 The state vector is richer (and different) than described

`s_r`'s 20 dimensions include the **entire previous 3×3 weight matrix** (9 of 20 dimensions), not "the previous gate `A_{r-1}`" (which in the paper's own Eq. 1 notation is a *separate* object from the continuous allocation). It also includes `unique_answers` and `avg_acc`, which are not mentioned anywhere in the prose. This is a straightforward fix — just describe the actual 20-d feature vector — but should be done precisely, since a reproducer building the state vector from the paper's prose alone would build the wrong thing (they'd feed in a 1-bit-per-edge previous gate, not the full continuous matrix).

### 3.3 The compression mechanism in the paper doesn't run

The paper frames compression as "a compression call, which is a stochastic Markov kernel κ on token strings with target ratio R" (§3.1) and discusses its integrality/truncation properties in Appendix A's Remark. The actual, executing mechanism (`communication.py:110-224`) is **per-sentence stochastic dropout weighted by cosine similarity to the extracted answer** (`p_keep = min(1.0, w_ij * normalized_sims[i])`), with **no LLM call and no explicit token-count truncation anywhere in the codebase**. The LLM-based `compress_message()` method exists (prompt text quoted in item 12 above) but is never invoked — `ablation.py` has developer comments confirming it was disabled during a refactor and never re-enabled. This is a correctness issue for a reproducer, not a style issue: someone trying to rebuild "the compression call" from the paper's math would build something that doesn't correspond to what generated the reported numbers.

### 3.4 Figure 1's bands and frequencies look like leftover placeholder data

`generate_plots.py:66-68` hardcodes:
```python
modes = ['BLOCK\n[0, 0.25)', 'SUMMARIZE\n[0.25, 0.50)', 'FILTER\n[0.50, 0.75)', 'FULL\n[0.75, 1.0]']
gsm8k_freq = [0.0, 31.7, 68.3, 0.0]
aqua_freq = [0.0, 50.1, 49.9, 0.0]
```
No `grep` for these numbers, or for any computation resembling them, turns up a source result file — they appear to be either illustrative placeholder values from early development or numbers computed once and never re-derivable. Combined with the fact that the runtime code has 3 modes, not 4 (§3.4 above / item 5), **Figure 1 as captioned describes a mechanism the code doesn't implement, using numbers that can't be traced to a result file.** This is worth either fixing at the source (compute real BLOCK/PARTIAL/FULL frequencies from the actual `mode_distribution` fields already logged in files like `08-06results/scaling_study_agents.json`) or removing/rewriting the figure.

### 3.5 Optimizer mismatch

Small but easy to fix: Table 8 says AdamW; `controller.py:178-196` implements plain Adam with no weight-decay term. Either add weight decay to match the table, or change the table to say "Adam."

### 3.6 The Nemotron transfer-check accuracy is the paper's most serious problem

This deserves its own paragraph because it's qualitatively different from the rest: it isn't a missing detail, it's a **result that contradicts its own source data**.

`08-06results/nemotron_gsm8k_150/summary_results.json` (verified directly, not just via subagent report):
```json
"No Communication":        {"accuracy": 0.36,              "correct_count": 54, "total_count": 150},
"Multi-Agent Debate (MAD)":{"accuracy": 0.5,                "correct_count": 75, "total_count": 150},
"Our SparseComm (400-sample Model)": {"accuracy": 0.28667, "correct_count": 43, "total_count": 150}
```
The paper (§5.5): *"SparseComm reached 54.0% against 50.0% for MAD and 36.0% for NoComm."* MAD's 50.0% and NoComm's 36.0% are exact matches to the file above. SparseComm's number in the paper (54.0%) is not 28.67% under any reasonable rounding or subset — it looks like it may have been transcribed from the wrong field, confused with MAD's number, or come from a different (unarchived) run. The repo has diagnostic scripts (`08-06results/analyze_nemotron_details.py`, `08-06results/diagnose_nemotron.py`) apparently written specifically because the authors already noticed SparseComm underperforming here — worth checking what those scripts concluded, since the underlying finding (a controller trained on gpt-oss-120b **losing to NoComm** when transferred to Nemotron 3 Ultra, 28.67% vs 36.0%) is actually a legitimate and interesting negative result about transfer robustness, and may be more scientifically honest than the currently-reported 54.0%. **Recommend: (a) fix the number to match the archived raw data, or (b) if 54.0% comes from a real, different run, locate and archive that run's raw file before resubmitting — do not leave this unresolved.**

---

## 4. Ready-to-paste numbers

### 4.1 Table 5 — full 12-row Holm-corrected table (independently re-derived)

Recomputed via saturated OLS `y = β0 + βC·Compression + βT·Topology + βCT·Interaction`, CR1-clustered SE (cluster = question_id, matched across all 4 arms), from `results/rq3/table7_sensitivity_{100,200,300,400}.csv`. G = number of question clusters, matching the paper's stated cluster counts (100 at n_train=100, 50 at n_train=200/300/400).

| n_train | term | β̂ | CR1 SE | p | p_Holm |
|---|---|---|---|---|---|
| 100 | Compression | 0.0500 | 0.0559 | 0.3736 | 1.000 |
| 100 | Topology | 0.0800 | 0.0583 | 0.1729 | 1.000 |
| 100 | Interaction | -0.0600 | 0.0792 | 0.4505 | 1.000 |
| 200 | Compression | -0.0800 | 0.0485 | 0.1056 | 1.000 |
| 200 | Topology | -0.0600 | 0.0447 | 0.1856 | 1.000 |
| **200** | **Interaction** | **0.1400** | **0.0644** | **0.0347** | **0.4161** |
| 300 | Compression | -0.0400 | 0.0495 | 0.4232 | 1.000 |
| 300 | Topology | -0.0200 | 0.0454 | 0.6617 | 1.000 |
| 300 | Interaction | 0.0200 | 0.0788 | 0.8007 | 1.000 |
| 400 | Compression | 0.0400 | 0.0495 | 0.4232 | 1.000 |
| 400 | Topology | 0.0400 | 0.0403 | 0.3259 | 1.000 |
| 400 | Interaction | -0.1000 | 0.0594 | 0.0984 | 1.000 |

(Intercepts: 0.5800/0.0498, 0.9800/0.0202, 0.9600/0.0282, 0.9200/0.0391 at n=100/200/300/400 respectively, all p<0.001 — matches the paper's disclosed intercept row exactly, confirming this is the correct frozen dataset/methodology.)

### 4.2 Table 7 — N=4,5 with real sample sizes and intervals

Source: `08-06results/scaling_study_agents.json`, GSM8K, n=150 for each N.

| N | Correct/n | Accuracy | Normal-approx (as originally stored) | Wilson 95% CI |
|---|---|---|---|---|
| 4 | 148/150 | 98.67% | ±1.8% | [95.27%, 99.63%] |
| 5 | 147/150 | 98.00% | ±2.2% | [94.29%, 99.32%] |

(N=2, not in the paper's Table 7 but present in the same file, for completeness: 147/150 = 98.00%, Wilson [94.29%, 99.32%].)

### 4.3 §5.5 Nemotron transfer — with intervals (see §3.6 for the accuracy caveat)

n=150, GSM8K, from `08-06results/nemotron_gsm8k_150/summary_results.json`:

| Method | Correct/n | Accuracy | Wilson 95% CI |
|---|---|---|---|
| NoComm | 54/150 | 36.00% | [28.76%, 43.94%] |
| MAD | 75/150 | 50.00% | [42.10%, 57.90%] |
| SparseComm (raw, contradicts paper's 54.0%) | 43/150 | 28.67% | [22.03%, 36.37%] |

### 4.4 Other filled-in values

- **Learning rate (Table 8)**: `1e-3` (both actor and critic, no schedule).
- **Optimizer (Table 8)**: correct to "Adam" (not AdamW) unless weight decay is added to match the table.
- **Z (§3.4, Eq. 7)**: not a closed form; `Z = tokens_if_full`, the measured full-communication token count for that episode. Suggest rewording: *"Z is the token count of the same episode under full communication, measured at runtime."*
- **GSM8K numeric tolerance (§4.1)**: `1e-3`.
- **Model/serving config (§4.2)**: `gpt-oss-120b` via Azure OpenAI, `api_version=2024-02-15-preview`, `max_tokens=512`, `temperature=0.0`, `top_p=1.0`.
- **State dimension d (§3.3)**: 20 (for N=3); see item 2 in §2 for the full feature list.


### 4.5 Table 3 raw-log cross-check (not a `[VERIFY]` item, done as a sanity check)

All headline Table 3 numbers were traced to raw per-question `.jsonl` streams and reproduce exactly:
- GSM8K: MAD 96.00/96.33/96.33% and NoComm 86.33/87.00/87.67% from `Multi-Agent Debate_gsm8k_stream.jsonl` / `No Communication_gsm8k_stream.jsonl`; SparseComm 94.67/94.33/96.00% (mean 95.00%, exact match) from the **rows 300-599** of `Transfer_GSM8K_seed{0,1,2}_gsm8k_stream.jsonl` (rows 0-299 in the same files are a different, 100-sample-trained checkpoint at 53-56% — make sure whichever script re-derives Table 3 uses the 400-sample-trained rows, not the whole file).
- AQuA-RAT and HotpotQA: all three systems' three-seed numbers reproduce exactly from the corresponding `*_aqua_stream.jsonl` / `*_hotpotqa_stream.jsonl` files.
- One data-hygiene note: `Transfer_GSM8K_seed1_gsm8k_stream.jsonl` has 849 rows instead of a clean 600 (both rows 300:600 and rows 549:849 independently give 94.33%, i.e. duplicated/appended logging) — doesn't change the reported mean, but worth cleaning up before archiving.

---

## 5. Suggested next experiments / directions

These are prioritized by what the paper *itself* already identifies as needed but unrun, plus what this audit surfaced independently.

1. **Run the Proposition 3 fixed-budget comparison.** This is the paper's own headline "what stays open" item — Proposition 2 proves the current 2×2 design *cannot* isolate a topology effect from a volume-controlled effect, and §6/Limitations spell out exactly what experiment would: budgets `V̄ ∈ {50, 100, 200, 400}` tokens/round, topologies `{ring, star, random, complete, learned support}`, truncating each message at its allocation, ~400 questions/cell for a 5-point main effect. Right now the paper's entire "no detectable component effect" story rests on an admittedly-underpowered design; this experiment is what would let the paper make a real topology-vs-volume claim instead of a bounded null.
2. **Fix or re-run the Nemotron transfer check (§3.6) before anything else** — this is a correctness blocker, not a nice-to-have.
3. **More seeds for Protocol P1.** The paper's own power analysis says 6 seeds (vs. the current 3) would resolve the 1.22-point GSM8K accuracy gap between SparseComm and MAD at 80% power. Given how central "the saving is free" is to the abstract, this is worth the extra compute.
4. **More matched questions for Protocol P2.** The current 100/50-question checkpoints can only detect effects ≥11.5-22.5 points; getting down to a genuinely interesting 5-point effect needs roughly 255 (topology), 385 (compression), and 555 (interaction) additional matched questions per the paper's own math.
5. **Benchmark-contamination / calibration-split-leakage check** — explicitly listed in Limitations as untested (`§Limitations, "Setting"`), and directly relevant given HotpotQA closed-book training accuracy hits 100% at 400 samples while held-out accuracy stays at ~32% (Appendix G) — the paper already interprets this as memorization; a formal contamination check would strengthen that claim rather than leave it as a caveat.
6. **Decide what the compression mechanism actually is, and make the paper and code agree** (§3.3). Either (a) wire up the existing but currently-dead `compress_message()` LLM call so the system matches the paper's "stochastic Markov kernel" framing, or (b) rewrite the methods section and Appendix A's integrality remark to describe the sentence-dropout mechanism that actually runs. Right now neither the paper nor a reproducer following it would get the system that generated the numbers.
7. **Archive Table 4's computation.** Since its provenance can't currently be traced (§2 item 10), save the exact script and input dataset that produces the leadership percentages, so the table can survive a reviewer's request for the underlying code.
8. **Double-check the GSM8K comma-stripping edge case (§2 item 6)** — worth a quick grep of the GSM8K test set for any gold answers ≥1000 to confirm this didn't silently affect any reported accuracy number.
9. **Reconcile Eq. 5 / Proposition 1's binary-gate formalism with the single continuous-`W` implementation (§3.1)** — either implement a literal binary `A` split from a continuous `B` (matching Eq. 1's `M = A⊙B` factorization) so the propositions describe the deployed system, or explicitly state in the paper that the deployed system uses a single continuous channel-weight and that Eq. 1's clean factorization is a design *target* realized more literally only in the 2×2 factorial arms (which do appear to implement two separate knobs, per the `Arm_00/01/10/11` naming in `results/rq3/`).

---

## 6. File index (for spot-checking)

| Claim area | File(s) |
|---|---|
| Controller / policy / state / reward | `controller.py` |
| Communication mechanism / bands | `communication.py` |
| Compression prompt (dead code) | `llm_backend.py:753-789` (and duplicated at `:496-529`) |
| Azure serving config | `llm_backend.py:637-722` |
| Answer normalization | `dataset_loaders.py:242-320` |
| Config defaults, learning rate | `config.py` |
| Main-results generator (Table 3) | `real_kaggle.py` |
| Nemotron transfer raw data | `08-06results/nemotron_gsm8k_150/summary_results.json`, `our_results.json`, `mad_results.json`, `nocomm_results.json` |
| Table 5 raw data | `results/rq3/table7_sensitivity_{100,200,300,400}.csv` |
| Table 7 (N=4,5) raw data | `08-06results/scaling_study_agents.json` |
| Table 3 raw per-question streams | `08-06results/continuous/*_stream.jsonl` |
| Table 4 (unverifiable) closest sources | `Master_SparseComm_Full_Results.json`, `08-06results/continuous/IB-Controller_*_stream.jsonl` |
| Figure 1 band frequencies (placeholder data) | `generate_plots.py:66-68` |
| Sibling/earlier drafts (do not cite as source of truth) | `08-06results/paper_latex.tex`, `full_timeline.md`, `results18jul.md`, `SparseComm_Research_Paper.md`, `COMPLETE_RESEARCH_ANALYSIS.md` |
