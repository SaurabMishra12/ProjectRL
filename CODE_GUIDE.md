# SparseComm Kaggle Notebook - Code Guide & Comments

## Executive Summary
This notebook implements **Sparse Semantic Communication with PPO**, a multi-agent system where agents solve problems together with learned, bandwidth-efficient communication. A PPO controller learns which agents should communicate, what to transmit (semantic filtering), and when to stop (early halting).

---

## Notebook Structure & Key Components

### 1. **Configuration & Utilities** (Lines 1–300)
- **`Config` class**: Centralized hyperparameter management
  - System: seed, device (CPU/GPU)
  - Agents: count, reasoning styles (analytical/creative/etc.)
  - Communication: rounds (T), bandwidth thresholds, compression modes
  - Controller: architecture (hidden dims), PPO settings, learning rate
  - LLM backend: mock (testing), NVIDIA, Azure, HuggingFace
  - Dataset: GSM8K, HotpotQA, MATH, TruthfulQA
  - Reward: information cost weight, cycle penalty, halt cost

- **Utility functions**:
  - `count_tokens()`: Accurate token counting via tiktoken (GPT-4 encoding)
  - `set_seed()`: Reproducible randomness (numpy, torch, random)
  - `setup_logging()`: Configure logging for experiments
  - `save_json()`, `load_json()`: I/O helpers

---

### 2. **Metrics & Results** (Lines 300–600)
- **`SampleResult`**: Result for one question
  - Tracks: is_correct, initial/final answers, reasoning texts, token costs
  - Semantic metrics: similarity, token reduction, information retention
  - Communication patterns: mode distribution, active edges

- **`ExperimentResults`**: Aggregated results for full run
  - Collections of sample_results, training rewards/losses, final weights

- **Metric functions**:
  - `compute_accuracy()`: Fraction of correct final answers
  - `compute_token_cost()`: Breakdown of generation vs communication tokens
  - `compute_communication_density()`: Fraction of active communication edges
  - `compute_error_propagation_rate()`: Cases where correct agents led to wrong final answer
  - `aggregate_metrics()`: Combines all metrics for reporting

---

### 3. **LLM Backends** (Lines 600–1500)
**Why multiple backends?**
- **Mock**: Fast, deterministic testing of RL pipeline (NOT for accuracy)
- **Real backends** (NVIDIA, Azure, HuggingFace): Actual task evaluation

#### **`LLMBackend` (Abstract Base Class)**
- `generate()`: Produce response to question with optional context
- `compress_message()`: Semantically compress message based on bandwidth mode
- `_extract_answer()`: Parse final answer from LLM output
- `_estimate_confidence()`: Extract confidence score from response text

#### **`MockLLMBackend`** (For testing RL, NOT evaluation)
- **Key property**: Deterministic hash-based correctness
  - Given (question, agent_id, context), always produces same answer
  - NOT real task performance; must never report accuracy
- Produces realistic token counts and chain-of-thought reasoning
- Configurable per-agent accuracy with diversity
- **CRITICAL WARNING**: Transfer to real LLMs required for paper

#### **`NvidiaDeepSeekBackend`** (Real evaluation)
- Uses NVIDIA's OpenAI-compatible API with DeepSeek v3.2
- Exponential backoff retry for rate limits
- Returns actual task performance

#### **`AzureOpenAIBackend`** (Real evaluation)
- Uses Azure OpenAI (GPT models)
- Requires credentials in environment
- Enterprise-grade reliability

#### **`HuggingFaceLocalBackend`** (Real evaluation)
- Loads local models (e.g., Mistral-7B)
- Enables offline evaluation and custom tuning

---

### 4. **Multi-Agent Framework** (Lines 1500–2200)
#### **`Agent`** (Single LLM-powered agent)
- **Attributes**:
  - agent_id, llm_backend, reasoning_style
  - history: List of previous responses
  
- **Key methods**:
  - `solve()`: Answer question with optional context from other agents
    - Compresses received messages (key phrases only)
    - Builds prompt, calls LLM backend
    - Computes confidence (base LLM + consistency check across rounds)
  - `_build_prompt()`: Construct full prompt with style instructions
  - `_compute_confidence()`: Boost confidence if answer matches previous rounds
  - `reset()`: Clear history for new question

#### **`AgentPool`** (Manages N agents)
- **Key methods**:
  - `initial_solve()`: All agents solve independently (Round 0)
    - Executes in parallel via ThreadPoolExecutor
    - Returns agent responses in stable order
  - `communicate_and_solve()`: Agents solve with filtered peer messages (Round 1..T)
    - Implements exponential backoff for rate limit errors
    - Parallel execution with proper ordering
  - `reset_all()`: Reset all agents for new question
  - `get_confidence_vector()`: Collect latest confidence from each agent

#### **Voting functions**:
- `majority_vote()`: Simple majority over answers
- `weighted_majority_vote()`: Confidence-weighted majority

---

### 5. **Communication System** (Lines 2200–3000)
#### **`StructuredSemanticState`** (Parsed reasoning representation)
- `claims`: Factual statements
- `evidence`: Supporting facts/numbers
- `assumptions`: Premises
- `intermediate_conclusions`: Derived facts
- `final_answer`: The answer
- `to_formatted_string()`: Convert to structured text (e.g., "[FINAL ANSWER]: X")

#### **`HeuristicSemanticParser`** (Rule-based parsing)
- `parse()`: Extract claims, evidence, assumptions from reasoning text
  - Uses keyword heuristics (e.g., "therefore" → conclusion)
  - Groups sentences by category

#### **`SemanticBandwidthMode`** (Enum for compression levels)
- BLOCK (W < 0.25): Send nothing
- SUMMARIZE (0.25 ≤ W < 0.5): Only conclusions + final answer
- FILTER (0.5 ≤ W < 0.75): Conclusions + evidence + final answer
- FULL (W ≥ 0.75): Send complete structured reasoning

#### **`TransmissionRecord`** (Logging for one message transmission)
- Tracks: sender, receiver, mode, tokens (original vs transmitted)
- Weight, round, semantic similarity, information retention

#### **`CommunicationGraph`** (Learnable communication topology)
- **Learnable parameters**:
  - W: (N×N) edge weights in [0, 1] per agent pair (diagonal = 0)
  - Q: (N×query_dim) semantic query vectors
  
- **Key methods**:
  - `set_weights()`: Apply W and Q from policy
  - `transmit()`: Filter message from sender to receiver
    - Uses W[sender][receiver] to determine bandwidth mode
    - Parses message → filters by mode → returns compressed + token count
    - Computes semantic similarity & information retention via embeddings
  - `get_messages_for_round()`: Route all agent messages through graph
    - Creates received_messages dict: {receiver_id: [filtered messages]}
    - Prevents self-loops (no self-reflection)
  - `get_communication_density()`: Active edges / total possible
  - `count_cycles()`: Detect feedback loops in W
  - `reset()`: Clear transmission log and reset W to 0.5
  - `disable_mode()` / `restore_all_modes()`: Ablation support

---

### 6. **Semantic State Encoding** (Lines 3000–3300)
#### **`SemanticStateEncoder`** (Feature extraction for PPO)
- **5-dimensional feature vector**:
  1. Shannon entropy: Measure of disagreement (higher = more diverse answers)
  2. Jaccard similarity: Reasoning overlap between agents
  3. Contradiction score: Fraction of agents disagreeing on answer
  4. Novelty score: Vocabulary richness across agents
  5. Consensus strength: Strength of majority answer

- **Why these features?** Ground PPO decision-making in language semantics, not just raw signals
- Falls back gracefully if SentenceTransformer unavailable

---

### 7. **PPO Controller** (Lines 3300–4200)
#### **`MLP`** (2-hidden-layer NumPy neural network)
- Used for: Actor (π) and Critic (V) in PPO
- **Components**:
  - W1, b1: Input → hidden1
  - W2, b2: Hidden1 → hidden2
  - W3, b3: Hidden2 → output
- **Activation**: ReLU (hidden), optional sigmoid (output for bounded predictions)
- **Optimizer**: AdamW with gradient clipping (max_norm=1.0)
- **Methods**:
  - `forward()`: Compute network output + cache for backprop
  - `backward()`: Compute gradients via backpropagation
  - `update()`: Apply AdamW parameter update

#### **`IBController`** (Core PPO learning algorithm)
- **Learns**:
  - W: Which agents should communicate (0..1 per edge)
  - Q: Semantic query vectors (content-aware filtering)
  - Halt decision: Whether to stop early or continue rounds

- **Key hyperparameters**:
  - lambda_info: Weight on communication cost penalty
  - gamma_loop: Penalty for cycles in W
  - gamma_discount: Future reward discount (0.99)
  - ppo_epsilon: Trust region size (0.2)
  - ppo_epochs: Update iterations per batch

- **Key methods**:
  - `get_weights()`: Sample action from policy
    - Forward pass through actor: outputs mean and log_std
    - Sample noise + compute log_prob for PPO
    - Split action into: W, Q, halt_logit via sigmoid
    - Halt decision: high consensus + good confidence → halt
  
  - `compute_reward()`: Calculate reward for this step
    - Task reward: +1 if correct, -0.5 if wrong (only at final step)
    - Shaping reward: confidence improvement + entropy reduction
    - Information cost: tokens_transmitted / baseline_max
    - KL sparsity: Divergence from prior (p=0.1)
    - Cycle penalty: Discourage feedback loops
    - Halt penalty: Discourage using all rounds
  
  - `store_transition()`: Buffer transition for batch update
  
  - `update_policy()`: PPO update step
    - Compute advantages via GAE (λ=0.95)
    - For each PPO epoch:
      - For each minibatch:
        - Forward pass: re-evaluate π and V
        - Compute PPO clipped surrogate loss
        - Entropy regularization (prevent collapse)
        - Backprop + update actor and critic
    - Return average loss
  
  - `save_model()` / `load_model()`: Persist weights to .npz file

---

### 8. **Dataset Loaders** (Lines 4200–5000)
**Design principle**: Strict train/test split (no data leakage)

- `load_gsm8k()`: Math word problems → (train, test)
- `load_hotpotqa()`: Multi-hop questions → (train, validation)
- `load_math()`: Google Deepmind AQuA → (train, test)
- `load_truthfulqa()`: Multiple choice (hard-partitioned validation)
- `load_mbpp()`: Coding tasks → (train, test)
- `load_dataset_by_name()`: Factory to load any dataset

- `evaluate_answer()`: Dataset-specific answer matching
  - GSM8K: numeric comparison (last number)
  - HotpotQA: token F1 with stopword removal
  - Math/TruthfulQA: alphanumeric match
  - MBPP: function signature heuristic

---

### 9. **Experiment Pipeline** (Lines 5000–6000)
#### **`run_single_sample()`** (Execute one question)
1. Initialize agents and communication graph
2. **Round 0**: All agents solve independently
3. **Rounds 1..T**:
   - Controller generates W, Q, halt decision
   - Communication graph routes and filters messages
   - Agents re-solve with filtered context
   - Check halt condition: if confident, break early
4. **Final answer**: Weighted majority vote
5. **Compute reward**: task_correct + shaping - communication_cost - sparsity_penalty - ...
6. **Store transition**: (state, action, log_prob, reward, done)
7. Return detailed SampleResult with all metrics

#### **`run_experiment()`** (Full train + eval)
1. Load dataset (strict split)
2. Initialize LLM backend, agents, graph, controller
3. **Training phase**:
   - For each epoch:
     - For each training sample:
       - Run single_sample with train=True
       - Accumulate transitions
     - PPO update: compute advantages, run policy gradient steps
   - Early stopping if loss doesn't improve
   - Save model
4. **Evaluation phase**:
   - For each test sample:
     - Run single_sample with train=False
     - Collect results
5. Return ExperimentResults

#### **Baseline experiments**:
- `run_full_communication()`: W=1 everywhere (upper bound)
- `run_no_communication()`: W=0 everywhere (lower bound)
- `run_experiment_with_random_weights()`: Fixed random W (no learning)

#### **`run_multi_seed_experiment()`** (Statistical rigor)
- Run experiment N times with different seeds
- Compute: mean ± std, 95% CI
- Enables significant reporting (not single-seed noise)

---

### 10. **Main Execution** (Lines 5500–end)
**Three-phase approach** (cost-effective for API evaluation):

#### **Phase 1: Pre-training** (100 samples/dataset, 3 epochs)
- Train universal controller on all datasets
- Share learned weights across datasets (transfer learning)

#### **Phase 2: Transfer Evaluation** (300 test samples/dataset, 1 seed)
- Evaluate trained controller on hold-out test sets
- Measure accuracy, token costs, communication density

#### **Phase 3: Baselines** (300 test samples/dataset)
- Full communication (debate)
- No communication (independent agents)

#### **Phase 4: Ablation Studies**
- Sparse-only (fixed weights, no PPO)
- PPO-only (no sparsity penalty)

---

## Key Design Principles

### 1. **Information Bottleneck Framework**
- Goal: Minimize task loss while minimizing communication
- Reward = Task correctness - λ × Communication cost - γ × Cycle penalty
- PPO learns W (topology) to maximize this trade-off

### 2. **Semantic Compression**
- Not just token truncation; extract high-level semantics
- BLOCK: No information
- SUMMARIZE: Only conclusions
- FILTER: Conclusions + evidence
- FULL: Complete reasoning
- Semantic similarity measured via embeddings (if available)

### 3. **Adaptive Halting**
- Early termination when agents reach consensus + high confidence
- Saves communication rounds (and thus tokens)

### 4. **Deterministic Testing (Mock Backend)**
- Mock NEVER used for accuracy reporting
- Used only to verify RL pipeline logic
- Real evaluation requires NVIDIA or Azure backend

### 5. **Reproducibility**
- Strict train/test split (no leakage)
- Random seed control
- Model checkpointing (.npz format)

---

## Common Pitfalls & Fixes

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| High CI from single seed | n_seeds=1 with 95% CI reporting | Use n_seeds ≥ 3 |
| API costs exploding | Training on too many samples | Start with n_train=30, n_test=15 |
| Accuracy below expected | Using MockLLMBackend for eval | Use azure/nvidia backend |
| Communication density always 0.5 | W not reset properly | Zero diagonal in reset() |
| Gradient explosion in MLP | Large state features | Normalize features, clip gradients |
| Stale imports of deleted backends | File reorganization | All backends in one notebook cell |

---

## Performance Profiling

**Typical timing** (Mock backend, 3 agents, 3 rounds, N=50 samples):
- Per sample: ~2-3 seconds
- Per epoch (30 samples): ~60-90 seconds
- Full train (10 epochs): ~10-15 minutes

**API costs** (Azure/NVIDIA backend):
- Per sample: ~10-20 tokens generated + ~5-10 transmitted
- 300 test samples: ~5000-7000 API tokens ≈ $0.10-0.20 per dataset

---

## Next Steps for Readers

1. **Understand the reward function** (`IBController.compute_reward()`)
2. **Trace one full sample** through `run_single_sample()`
3. **Review MLP gradients** in `IBController.update_policy()`
4. **Check semantic features** in `SemanticStateEncoder.encode_state()`
5. **Run toy experiment** with n_samples=15 to verify end-to-end

---

*Last updated: June 11, 2026*
