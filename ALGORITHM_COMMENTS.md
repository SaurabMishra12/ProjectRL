# Critical Algorithms - Detailed Comments

## 1. MLP Backward Pass (Gradient Computation)

```python
def backward(self, cache: dict, grad_output: np.ndarray) -> dict:
    """Backpropagation through 2-hidden-layer network."""
    
    # Unpack cached values from forward pass
    x, a1, a2, z1, z2, z3, out = cache['x'], cache['a1'], cache['a2'], cache['z1'], cache['z2'], cache['z3'], cache['out']

    # === Layer 3 (Output) ===
    dz3 = grad_output
    if self.use_sigmoid:
        # Sigmoid derivative: σ'(z) = σ(z)(1 - σ(z)) = out(1 - out)
        dz3 = grad_output * out * (1 - out)

    # Compute weight and bias gradients for layer 3
    dW3 = np.outer(a2, dz3)  # Shape: (hidden_dim, output_dim)
    db3 = dz3                  # Shape: (output_dim,)
    da2 = dz3 @ self.W3.T     # Propagate gradient back to hidden layer 2

    # === Layer 2 (Hidden) ===
    # Apply ReLU derivative: ReLU'(z) = 1 if z > 0, else 0
    dz2 = da2 * self._relu_grad(z2)
    
    dW2 = np.outer(a1, dz2)   # Shape: (hidden_dim, hidden_dim)
    db2 = dz2                  # Shape: (hidden_dim,)
    da1 = dz2 @ self.W2.T     # Propagate back to hidden layer 1

    # === Layer 1 (Hidden) ===
    dz1 = da1 * self._relu_grad(z1)
    
    dW1 = np.outer(x, dz1)    # Shape: (input_dim, hidden_dim)
    db1 = dz1                  # Shape: (hidden_dim,)

    return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}
```

**Why this matters:**
- Gradient shapes must match parameter shapes exactly
- np.outer(a, b) produces correct outer product for weight matrices
- Scalar biases get column vector gradients (automatically broadcast in update)

---

## 2. PPO Update Step (Policy Gradient)

```python
def update_policy(self) -> float:
    """Proximal Policy Optimization with Generalized Advantage Estimation."""
    
    if not self.saved_transitions: return 0.0
    n_steps = len(self.saved_transitions)

    # Extract rewards and values from transition buffer
    rewards  = np.array([t['reward'] for t in self.saved_transitions], dtype=np.float64)
    values   = np.array([t['value']  for t in self.saved_transitions], dtype=np.float64)

    # === STEP 1: Compute advantages via GAE (λ=0.95) ===
    # GAE reduces variance vs Monte Carlo returns while keeping low bias
    # Formula: A_t = Σ(γλ)^l δ_{t+l} where δ_t = r_t + γV(s_{t+1}) - V(s_t)
    
    advantages = np.zeros(n_steps, dtype=np.float64)
    returns    = np.zeros(n_steps, dtype=np.float64)
    gae        = 0.0
    next_value = 0.0  # Terminal bootstrap: V(terminal) = 0
    
    for t in reversed(range(n_steps)):
        is_done = self.saved_transitions[t].get('done', False)
        # At terminal, mask out future value (no successor state)
        mask = 0.0 if is_done else 1.0
        
        # Compute TD error
        delta = rewards[t] + self.gamma_discount * next_value * mask - values[t]
        
        # Accumulate GAE with exponential decay (gae_lambda = 0.95)
        gae = delta + self.gamma_discount * self.gae_lambda * gae * mask
        
        advantages[t] = gae
        returns[t] = gae + values[t]  # V-target = Advantage + Baseline
        next_value = values[t]

    # Normalize advantages for numerical stability
    if np.std(advantages) > 1e-8:
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

    # === STEP 2: PPO updates (multiple epochs over batch) ===
    total_actor_loss, total_critic_loss = 0.0, 0.0

    for epoch in range(self.ppo_epochs):  # Typically 3 epochs
        # Shuffle transitions for minibatch sampling
        indices = np.random.permutation(n_steps)
        batch_size = 16

        for start_idx in range(0, n_steps, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]

            # === STEP 3: Actor update (policy gradient with PPO clipping) ===
            for i in batch_indices:
                feat         = self.saved_transitions[i]['features']
                action       = self.saved_transitions[i]['action']
                old_log_prob = self.saved_transitions[i]['log_prob']
                old_sigma    = self.saved_transitions[i].get('sigma', np.full_like(action, 0.2))
                adv          = advantages[i]
                ret          = returns[i]

                # Re-evaluate policy and value function on current batch
                raw_out_new, acache = self.actor.forward(feat)
                v_pred,     vcache  = self.critic.forward(feat)

                # Parse actor output: [mean_action | log_std]
                mean_a_new   = raw_out_new[:self.mean_dim]
                log_std_new  = np.clip(raw_out_new[self.mean_dim:], -3.0, -0.5)
                sigma_new    = np.exp(log_std_new)

                # Compute new log probability (Gaussian)
                # log π(a|s) = -0.5 * Σ((a-μ)/(σ))^2 - Σ(log σ) - 0.5*log(2π)
                new_log_prob = -0.5 * np.sum(
                    ((action - mean_a_new) / (sigma_new + 1e-8))**2
                    + 2 * log_std_new + np.log(2 * np.pi)
                )

                # Probability ratio: π_new(a|s) / π_old(a|s)
                ratio = np.exp(np.clip(new_log_prob - old_log_prob, -10, 10))

                # PPO clipping: limit ratio to [1-ε, 1+ε]
                surr1 = ratio * adv
                surr2 = np.clip(ratio, 1.0 - self.ppo_epsilon, 1.0 + self.ppo_epsilon) * adv
                
                # Entropy regularization (prevent premature convergence)
                entropy = 0.5 * np.sum(2 * log_std_new + np.log(2 * np.pi * np.e))
                entropy_coef = 0.01
                
                actor_loss = -min(surr1, surr2) - entropy_coef * entropy

                # Compute gradient of log π w.r.t. mean and log_std
                # When clipped: gradient = 0 if ratio outside trust region and moving wrong direction
                if adv >= 0:
                    clipped = ratio > (1.0 + self.ppo_epsilon)  # Clip if too optimistic
                else:
                    clipped = ratio < (1.0 - self.ppo_epsilon)  # Clip if too pessimistic
                
                d_ratio = 0.0 if clipped else adv
                d_log_prob = d_ratio * ratio
                
                # Gradient for mean head: ∂log π / ∂μ = (a - μ) / σ²
                d_mean = d_log_prob * (action - mean_a_new) / (sigma_new ** 2 + 1e-8)
                
                # Gradient for log_std head: ∂log π / ∂log σ = ((a-μ)/σ)² - 1
                d_log_std = d_log_prob * ((action - mean_a_new)**2 / (sigma_new**2 + 1e-8) - 1.0)
                d_log_std += entropy_coef * np.ones_like(log_std_new)  # Entropy bonus
                
                # Combine gradients for both actor heads
                d_out = np.concatenate([-d_mean / len(batch_indices),
                                        -d_log_std / len(batch_indices)])
                
                agrad = self.actor.backward(acache, d_out)

                # === STEP 4: Critic update (value function regression) ===
                # L_critic = 0.5 * (V_pred - V_target)²
                critic_loss = 0.5 * (ret - v_pred[0])**2
                vgrad = self.critic.backward(vcache, np.array([(v_pred[0] - ret) / len(batch_indices)]))

                # Accumulate losses for logging
                total_actor_loss  += abs(actor_loss)
                total_critic_loss += critic_loss

            # Apply accumulated gradients to actor and critic
            self.actor.update(actor_grads)
            self.critic.update(critic_grads)

    avg_loss = (total_actor_loss + self.ppo_v_coef * total_critic_loss) / (n_steps * self.ppo_epochs)
    return avg_loss
```

**Why this matters:**
- **GAE**: Low-bias, low-variance advantage estimate (industry standard)
- **PPO Clipping**: Prevents catastrophic policy updates by limiting step size
- **Entropy regularization**: Maintains exploration throughout training
- **Minibatch updates**: Reduces variance compared to single-sample updates

---

## 3. Reward Computation (Information Bottleneck)

```python
def compute_reward(
    self,
    task_correct: bool,          # Did agents get the right answer?
    tokens_transmitted: int,      # Tokens sent via communication
    tokens_if_full: int,          # Tokens if all communication was full
    compression_ratio: float,     # tokens_transmitted / tokens_if_full
    n_cycles: int,                # Feedback loops in communication graph
    current_W: np.ndarray,        # Edge weight matrix (N×N)
    halt_round: int = 0,          # Which round did agents halt?
    confidence_improvement: float = 0.0,  # How much did confidence improve?
    answer_entropy_reduction: float = 0.0,  # Did agents reach consensus?
    is_final_step: bool = True,   # Is this the final evaluation step?
) -> float:
    """Compute reward signal for this step of problem-solving.
    
    Reward = Task correctness + Communication efficiency + Sparsity
    
    Goal: Maximize accuracy while minimizing communication cost and cycles.
    """
    
    # === 1. TASK REWARD: Core objective (only at final step) ===
    # Positive if correct, negative if wrong, zero for intermediate steps
    task_rew = (1.0 if task_correct else -0.5) if is_final_step else 0.0

    # === 2. SHAPING REWARDS: Intermediate guidance signals ===
    # Help agent learn to improve confidence and consensus during intermediate rounds
    shaping_reward = 0.0
    if is_final_step:
        # Confidence increased? → Small bonus (helps policy learn to converge)
        shaping_reward += confidence_improvement * 0.3
        # Agents reached consensus (entropy decreased)? → Small bonus
        shaping_reward += answer_entropy_reduction * 0.2
        # Constant term: small bonus just for completing
        shaping_reward += 0.05

    # === 3. INFORMATION COST: Communication efficiency penalty ===
    # Penalize excessive token transmission
    # Normalize by max possible transmission to make cost bounded
    n_rounds = getattr(self.config, 'n_rounds', 3)
    baseline_max = max(
        tokens_if_full,  # How many tokens WOULD be sent if W=1 everywhere
        self.n_agents * (self.n_agents - 1) * 50 * n_rounds,  # Heuristic maximum
        1.0
    )
    info_cost = tokens_transmitted / baseline_max  # In [0, 1]
    
    # Apply weight: -lambda_info * info_cost
    # lambda_info = 0.3 (from config): penalize 30% of task reward per communication unit

    # === 4. KL SPARSITY PENALTY: Encourage sparse W ===
    # Goal: W should be mostly 0 (off) or 1 (on), not 0.5 (undecided)
    # Use KL divergence from prior distribution
    
    prior_p = 0.1  # Target: 10% of edges ON, 90% OFF
    W_clip = np.clip(current_W, 1e-5, 1.0 - 1e-5)  # Avoid log(0)
    
    # KL(W || Bernoulli(0.1))
    kl_divergence = float(np.sum(
        W_clip * np.log(W_clip / prior_p) +              # P(W=1) term
        (1 - W_clip) * np.log((1 - W_clip) / (1 - prior_p))  # P(W=0) term
    ))
    
    # Coefficient: -0.08 * kl_divergence (learned weight from ablation)

    # === 5. CYCLE PENALTY: Discourage feedback loops ===
    # Cycles can cause information to spiral and distort consensus
    # Normalize by total possible edges
    loop_penalty = n_cycles / max(self.n_agents * (self.n_agents - 1) // 2, 1)
    
    # Weight: -gamma_loop * loop_penalty (gamma_loop = 0.1 from config)

    # === 6. HALT PENALTY: Discourage wasting communication rounds ===
    # Encourage early termination if confident
    halt_penalty = halt_round * 0.1

    # === COMBINE ALL TERMS ===
    # Weighted sum with learned hyperparameters (lambda_info, gamma_loop)
    return (
        task_rew                              # Primary objective
        + shaping_reward                      # Intermediate learning signal
        - self.lambda_info * info_cost        # Information bottleneck penalty
        - 0.08 * kl_divergence                # Sparsity preference
        - self.gamma_loop * loop_penalty      # Cycle avoidance
        - halt_penalty                        # Encourage early halting
    )
```

**Why this matters:**
- **Multi-objective trade-off**: Balance accuracy vs efficiency
- **Shapingrewards**: Help agent learn useful behaviors (confidence, consensus)
- **KL sparsity**: Prevents dense medium-weight edges that waste computation
- **Cycle penalty**: Ensures information flows without circular distortion

---

## 4. Semantic State Encoding (Feature Extraction)

```python
def encode_state(self, structured_states: Dict[int, StructuredSemanticState]) -> np.ndarray:
    """Extract 5-dim feature vector from multi-agent reasoning state.
    
    Features ground PPO's decision-making in language semantics, not just task signals.
    """
    
    if not structured_states:
        return np.zeros(5, dtype=np.float64)

    answers = [str(s.final_answer).lower().strip() for s in structured_states.values()]
    texts = [s.to_formatted_string() for s in structured_states.values()]

    # === FEATURE 1: Shannon Entropy (Disagreement) ===
    # Measure: How much do agents disagree on the final answer?
    # High entropy = agents suggest different answers → need more discussion
    # Low entropy = agents agree → can halt early
    
    if len(answers) > 0:
        unique, counts = np.unique(answers, return_counts=True)
        probs = counts / len(answers)
        shannon_entropy = -np.sum(probs * np.log2(probs + 1e-8))
        shannon_entropy = shannon_entropy / 1.58  # Normalize by max entropy for 3 agents
    else:
        shannon_entropy = 0.0

    # === FEATURE 2: Jaccard Similarity (Reasoning Overlap) ===
    # Measure: How much reasoning overlap is there between agents?
    # High = agents reason similarly → lots of communication → high W
    # Low = agents have different perspectives → selective communication
    
    if len(texts) >= 2:
        text_sets = [set(t.lower().split()) for t in texts]
        pairwise_jaccard = []
        for i in range(len(text_sets)):
            for j in range(i + 1, len(text_sets)):
                intersection = len(text_sets[i] & text_sets[j])
                union = len(text_sets[i] | text_sets[j])
                pairwise_jaccard.append(intersection / max(union, 1))
        jaccard_similarity = float(np.mean(pairwise_jaccard))
    else:
        jaccard_similarity = 1.0

    # === FEATURE 3: Contradiction Score ===
    # Measure: How many agents disagree on the answer?
    # 0 = perfect consensus, 1 = all different answers
    unique_answers = len(set(answers))
    contradiction_score = 1.0 - (unique_answers / max(len(answers), 1))
    # (If 3 agents all agree: 1 - 1/3 = 0.67... close to consensus)

    # === FEATURE 4: Novelty Score (Vocabulary Richness) ===
    # Measure: How diverse is the combined reasoning?
    # High = agents bring different ideas → benefit from communication
    # Low = agents think similarly → less to communicate
    
    all_words: set = set()
    for t in texts:
        all_words.update(t.lower().split())
    avg_words_per_agent = float(np.mean([len(t.split()) for t in texts])) if texts else 0
    novelty_score = len(all_words) / max(avg_words_per_agent * len(texts), 1)
    novelty_score = min(1.0, novelty_score / 2.0)  # Normalize to [0, 1]

    # === FEATURE 5: Consensus Strength ===
    # Measure: How strong is the majority answer?
    # 1.0 = unanimous, 0.33 = split 3-way, 0 = ties (rare)
    if unique_answers == 1:
        consensus_strength = 1.0  # Perfect agreement
    else:
        answer_counts = Counter(answers)
        majority_count = answer_counts.most_common(1)[0][1]
        consensus_strength = majority_count / len(answers)

    # Combine into feature vector (5 dimensions for PPO input)
    features = np.array([
        shannon_entropy,         # Disagreement (0-1)
        jaccard_similarity,      # Reasoning overlap (0-1)
        contradiction_score,     # Answer divergence (0-1)
        novelty_score,           # Vocabulary diversity (0-1)
        consensus_strength,      # Majority agreement (0-1)
    ], dtype=np.float64)
    
    return np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
```

**Why this matters:**
- **Semantic grounding**: Features capture meaning, not just raw signals
- **Interpretability**: Can understand what the controller "sees"
- **Stability**: Robust to outliers and missing data

---

*Last updated: June 11, 2026*
