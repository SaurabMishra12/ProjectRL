# EXPERIMENTAL DESIGN & CAUSAL IDENTIFICATION PROTOCOL (CORRECTED SUITE)

**Venue**: The Web Conference (WWW) 2027  
**Framework**: SPARSECOMM Counterfactual Communication Decomposition  
**Author**: Senior Research Team  
**Status**: APPROVED FOR EXECUTION  

---

## 1. Central Scientific Question & Causal Framework

The central organizing question of this work is:
$$\text{"When multi-agent LLM communication becomes more efficient, what actually causes the improvement: routing/topology, message/content rationing, adaptive stopping, or interactions among these factors?"}$$

In previous literature (and prior runs of this project), multi-agent communication improvements were conflated across:
1. **Communication Volume** ($V$): Total number of tokens transmitted across the agent network.
2. **Content Rationing** ($C$): How transmitted information is selected (semantic sentence extraction vs. arbitrary truncation).
3. **Adaptive Stopping** ($S$): Stopping communication dynamically upon reaching consensus vs. running fixed depths.
4. **Routing Topology** ($R$): The directed graph of who transmits to whom.

To isolate these factors without confounding, we formalize the system response as:
$$Y = f(R, V, C, S, \mathbf{X})$$
where $\mathbf{X}$ represents the input query, retrieved evidence, and underlying initial reasoning.

---

## 2. Experiment W3-Corrected: Isolating Adaptive Stopping

### 2.1 Scientific Objective
To determine whether **adaptive stopping independently improves the communication-quality tradeoff** when routing topology, message rationing, and retrieval evidence are held constant.

### 2.2 Factorial Design ($2 \times 2$)
- **Factor 1: Stopping Policy ($S$)**
  - **Condition A (Fixed Depth)**: Communication proceeds for exactly $T = 3$ rounds for every question.
  - **Condition B (Adaptive Stopping)**: Communication halts dynamically at round $r \in \{1, 2, 3\}$ upon satisfying the sequential consensus condition.
- **Factor 2: Retrieval Noise ($\mathbf{X}_{noise}$)**
  - **Clean Evidence**: Top-2 BM25 retrieved passages from candidate Wikipedia articles.
  - **Distractor Noise**: All 10 candidate Wikipedia articles (containing 8 distractor articles).

### 2.3 Mathematical Formulation: Calibrated Wald's SPRT for $N=3$ Agents
In an $N=3$ agent team deliberation, let $a_i^{(r)}$ be the normalized candidate answer produced by agent $i$ at round $r$, and let $k$ be the count of agents agreeing on the majority candidate answer ($k \in \{1, 2, 3\}$).

Under Wald's Sequential Probability Ratio Test (SPRT):
- $H_1$: The majority answer is correct ($p_1 = 0.90$).
- $H_0$: The majority answer is random noise ($p_0 = 0.50$).

The binomial log-likelihood ratio is:
$$\Lambda(k) = k \ln \left(\frac{p_1}{p_0}\right) + (N - k) \ln \left(\frac{1 - p_1}{1 - p_0}\right)$$

For $N=3$ agents:
- **Unanimous Agreement ($k=3$)**:
  $$\Lambda(3) = 3 \ln\left(\frac{0.9}{0.5}\right) + 0 = 3 \times 0.5878 \approx 1.7634$$
- **Split Majority ($k=2$)**:
  $$\Lambda(2) = 2 \ln(1.8) + 1 \ln(0.2) = 2(0.5878) - 1.6094 \approx -0.4338$$
- **Total Disagreement ($k=1$)**:
  $$\Lambda(1) = 1 \ln(1.8) + 2 \ln(0.2) = 0.5878 - 3.2189 \approx -2.6311$$

**Calibrated Halting Rule**:
Halting occurs at round $r < T_{max}$ if and only if:
1. **Unanimous Consensus**: $k = 3$ and $\Lambda(3) > \tau_{SPRT}$ with $\tau_{SPRT} = \ln(4) \approx 1.3863$ (odds ratio 4:1).
2. **High-Confidence Majority Consensus**: $k = 2$, round $r \ge 2$, and average calibrated confidence $\bar{c}_{maj} \ge 80\%$.
3. **Safety Fallback**: Reaches maximum depth $T_{max} = 3$.

### 2.4 Controlled Variables
- Routing topology: Held constant at static role pipeline $(0 \to 1)$ and $(1 \to 2)$.
- Message rationing rule: Sentence-preserving semantic rationing up to 50 tokens per active edge.
- Evidence corpus & questions: Identical.
- Model backend: `gpt-oss-120b`, temperature 0.0.

### 2.5 Verification Metrics
- Realized rounds distribution ($T \in \{1, 2, 3\}$).
- Early exit rate: $\text{Rate}_{early} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(T_i < 3)$.
- Total transmitted communication tokens.
- Exact Match (EM) and token-level F1 score.
- API and cache latencies measured separately.

---

## 3. Experiment W4-Corrected: Isolating Routing Topology

### 3.1 Scientific Objective
To determine whether **communication routing topology independently improves multi-agent task performance** once communication volume, content, and depth are strictly controlled.

### 3.2 The Frozen Information Pool Protocol
For each question:
1. **Step 1 (Retrieval)**: Real BM25 retrieval over candidate articles is performed once.
2. **Step 2 (Initial Generation)**: Round 0 initial responses $M_0, M_1, M_2$ are generated once by the three specialized agents.
3. **Step 3 (Freezing)**: The pool $\mathcal{P} = \{M_0, M_1, M_2\}$ is frozen. Its SHA-256 hash is recorded.
4. **Step 4 (Intervention)**: All counterfactual routing topologies operate over this exact frozen pool. No agent regenerates Round 0 messages.

### 3.3 Routing Conditions Evaluated
1. **Condition A: Static Role-Aligned Routing**
   - Active directed edges: $E = \{(0, 1), (1, 2)\}$ (Web Search $\to$ Verifier $\to$ Reasoner).
2. **Condition B: Learned SPARSECOMM Routing**
   - Active directed edges: Top-2 edges selected by the Information-Bottleneck (IB) controller adjacency weights $W_{u,v}$.
3. **Condition C: Random Routing**
   - Active directed edges: 2 directed edges sampled uniformly at random from the 6 possible off-diagonal edges (fixed seed per question/round).
4. **Condition D: Complete Broadcast**
   - Active directed edges: All 6 directed edges connecting all agent pairs.

### 3.4 Strict Realized Volume Matching Mechanism
To eliminate the severe volume confound discovered in the prior run (where volumes varied between 30 and 104 tokens):
1. **Per-Question Token Quota ($L_q$)**:
   For question $q$, let $l_u = \text{count\_tokens}(M_u)$ for $u \in \{0, 1, 2\}$.
   Define the standardized transmission quota:
   $$L_q = \min(l_0, l_1, l_2)$$
   If $L_q = 0$, communication is naturally 0 tokens.
2. **Standardized Truncation**:
   Each agent $u$'s transmitted payload is truncated using sentence-preserving / exact token matching to at most $L_q$ tokens:
   $$m_u = \text{truncate}(M_u, L_q)$$
3. **Realized Volume Guarantees**:
   - For any 2-edge topology (Static, Learned, Random):
     $$\text{Volume}_{total} = \text{tok}(m_{u_1}) + \text{tok}(m_{u_2}) = L_q + L_q = 2 L_q$$
   - Realized token difference across Static, Learned, and Random: **0.000% (Strictly Identical to the Token)**.
   - For Complete Broadcast: Evaluated under both volume-matched budget ($2 L_q$ distributed across edges) and unconstrained broadcast ($6 L_q$) to provide an explicit upper-bound reference.
4. **Fixed Depth**:
   Communication depth is strictly fixed to $T = 2$ rounds for all routing conditions. No condition is permitted to halt early in W4.

### 3.5 Statistical Hypothesis Testing & Precision
- **Primary Comparisons**:
  1. Learned SPARSECOMM vs. Static Role
  2. Learned SPARSECOMM vs. Random Routing
  3. Static Role vs. Random Routing
- **Statistical Tests**:
  - Point estimate of mean difference: $\Delta_{\text{F1}} = \bar{\text{F1}}_A - \bar{\text{F1}}_B$
  - 95% Confidence Intervals via non-parametric studentized bootstrap (10,000 resamples).
  - Paired permutation test (10,000 permutations) for exact $p$-values.
  - Cohen's $d$ effect size.
- **Decision Rules (Anti-Sycophancy)**:
  - If $p > 0.05$ and the 95% CI contains 0 and is narrow ($|\text{CI width}| \le 0.08$): Verdict = **EVIDENCE AGAINST PRACTICALLY MEANINGFUL EFFECT**.
  - If $p > 0.05$ but the 95% CI is wide ($|\text{CI width}| > 0.12$): Verdict = **INCONCLUSIVE (INSUFFICIENT PRECISION)**.
  - If $p < 0.05$ and $\Delta_{\text{F1}} > 0.03$: Verdict = **SUPPORTED**.

---

## 4. Automated Integrity & Audit Gates

1. **Gate 1: Dataset Integrity** (Fail-closed; verified 0 synthetic rows).
2. **Gate 2: Real Retrieval Verification** (BM25 over real Wikipedia passages).
3. **Gate 3: Message Pool Immutability** (Identical Round 0 hashes across W4 conditions).
4. **Gate 4: Realized Volume Verification** (W4 token difference $\le 1\%$).
5. **Gate 5: Stopping Early-Exit Verification** (W3 adaptive early exit rate $> 0$).
6. **Gate 6: Independent Recomputation** (All metrics verified from raw JSONL streams).
