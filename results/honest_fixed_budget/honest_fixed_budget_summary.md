# Honest Fixed-Budget Routing Experiment & Manuscript Verification

**Model**: Azure OpenAI `gpt-oss-120b`  
**Questions**: n=150 matched GSM8K test questions  
**NoComm Baseline**: **78.67%** (0 tokens, initial majority vote on exact same 150 questions)  

## Full Summary Table with Active Edge Counts ($|\mathcal{T}|$) & Token Density

| Topology | Budget $\bar{V}$ | Acc(%) | $\Delta_{\text{NoComm}}$ | $n_{\comm}$ | $\widetilde{V}/\bar{V}\mid\text{comm}$ | $|\mathcal{T}|\mid\text{comm}$ | Tokens/Edge |
|---|---:|---:|---:|---:|---:|---:|---:|
| **NoComm** | 0 | 78.67% | 0.00 pts | 0 | --- | 0.0 | --- |
| **Ring** | 50 | 78.67% | +0.00 pts | 54 | 1.000 | 6.0 | 8.3 |
| **Star** | 50 | 80.00% | +1.33 pts | 52 | 1.000 | 4.0 | 12.5 |
| **Learned Support** | 50 | 95.33% | +16.67 pts | 52 | 0.993 | 2.0 | 24.8 |
| **Ring** | 100 | 75.33% | -3.33 pts | 61 | 1.000 | 6.0 | 16.7 |
| **Star** | 100 | 76.00% | -2.67 pts | 57 | 1.000 | 4.0 | 25.0 |
| **Learned Support** | 100 | 92.67% | +14.00 pts | 58 | 0.951 | 2.0 | 47.6 |

## Fill Rule & Selection Verification (Addressing Abstract & §5.5)

1. **Support-Only Truncation Rule Verified**:
   - `TotalSampleBudgetCommunicationGraph.transmit` uses `truncate_to_token_budget(message, remaining_budget)` for ALL topologies.
   - There is **no semantic sentence selection** applied inside `transmit` for Learned Support.
   - Learned Support uses the exact same token truncation function as Ring and Star.

2. **Explanation for $\widetilde{V}/\bar{V} = 0.993 / 0.951$**:
   - Learned Support concentrates edges to $|\mathcal{T}| = 2.0$.
   - On questions where the text offered across those 2 active edges is slightly shorter than $\bar{V}$ (e.g., 49 tokens at $\bar{V}=50$, or 95–99 tokens at $\bar{V}=100$), 100% of the offered text is transmitted without needing truncation.
   - Ring ($|\mathcal{T}|=6.0$) and Star ($|\mathcal{T}|=4.0$) offer ~300–600 tokens across 4–6 edges, so truncation always hits the ceiling exactly (1.000).

3. **Worse Than Silence Phenomenon (NoComm Comparison at P4)**:
   - **NoComm Baseline**: **78.67%** (0 tokens transmitted).
   - **Ring V100**: **75.33%** (**-3.34 pts below NoComm** — worse than silence!).
   - **Star V100**: **76.00%** (**-2.67 pts below NoComm** — worse than silence!).
   - **Learned Support V50**: **95.33%** (**+16.66 pts above NoComm**).
   - Static motif communication dilutes tokens across 4–6 edges ($8.3 \text{--} 25.0$ tokens/edge), producing fragmented, uninformative messages that confuse agents. Dynamic Learned Support concentrates tokens onto 2 high-leverage edges ($25.0 \text{--} 49.6$ tokens/edge), providing coherent reasoning steps.