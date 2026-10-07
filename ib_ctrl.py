class IBController:
    """PPO Controller with native Actor-Critic MLPs and Semantic State Awareness."""

    def __init__(self, config: Config):
        self.config = config
        self.n_agents = config.n_agents
        self.query_dim = getattr(config, 'query_dim', 16)
        
        self.semantic_encoder = SemanticStateEncoder(config.n_agents)
        self.parser = HeuristicSemanticParser()
        
        # Input: 5 (semantic_state) + confs (N) + W (N^2) + round_num (1) + unique_answers (1) + avg_acc (1)
        self.input_dim = 5 + self.n_agents + (self.n_agents * self.n_agents) + 3
        
        self.k_dim = 2  # The sub-task bottleneck dimension
        # Actor outputs: H_src and H_dst (N*k*2), Q (N*query_dim), halt_logit (1), log_std (same dim)
        self.mean_dim = (self.n_agents * self.k_dim * 2) + (self.n_agents * self.query_dim) + 1
        self.action_dim = self.mean_dim * 2  # second half is log_std

        # GAE lambda for advantage estimation (standard PPO setting)
        self.gae_lambda = getattr(config, 'gae_lambda', 0.95)

        # PPO components
        self.actor = MLP(self.input_dim, config.controller_hidden_dim, self.action_dim, lr=config.controller_lr, use_sigmoid=False)
        self.critic = MLP(self.input_dim, config.controller_hidden_dim, 1, lr=config.controller_lr)

        self.lambda_info = config.lambda_info
        self.gamma_loop = config.gamma_loop
        self.gamma_discount = config.gamma_discount
        self.ppo_epsilon = getattr(config, 'ppo_clip_epsilon', 0.2)
        self.ppo_epochs = getattr(config, 'ppo_epochs', 3)
        self.ppo_v_coef = getattr(config, 'ppo_value_coef', 0.5)

        self.episode_rewards: List[float] = []
        self.episode_losses: List[float] = []
        self.saved_transitions: List[dict] = []

    def build_features(self, semantic_state: np.ndarray, confs: List[float], prev_W: np.ndarray, round_num: int, unique_answers: int = 1, avg_acc: float = 0.5) -> np.ndarray:
        feats = semantic_state.tolist() + list(confs) + prev_W.flatten().tolist() + [round_num / max(self.config.n_rounds, 1), unique_answers / self.n_agents, avg_acc]
        return np.array(feats, dtype=np.float64)

    def _split_action(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        N, k = self.n_agents, self.k_dim
        
        # Now we slice out 2 * N * k elements
        h_flat = action[:N * k * 2]
        q_flat = action[N * k * 2 : N * k * 2 + N * self.query_dim]
        halt_logit = action[-1]
        
        # Give full N x k matrices to both Source and Destination
        H_src = h_flat[:N * k].reshape(N, k)
        H_dst = h_flat[N * k:].reshape(N, k)
        W_raw = H_src @ H_dst.T - 2.0
        pass # np.fill_diagonal(W_raw, -500.0)  # Pre-sigmoid elimination
        W = 1 / (1 + np.exp(-np.clip(W_raw, -500, 500)))
        np.fill_diagonal(W, 1.0)  # Hard-zero: ensures density/state features are clean
        
        Q = q_flat.reshape(N, self.query_dim)
        return W, Q, halt_logit

    def get_weights(self, reasoning_texts: Dict[int, str], answers: Dict[int, str], confidence_scores: List[float], prev_W: np.ndarray, round_num: int, unique_answers: int = 1, avg_accuracy: float = 0.5, explore: bool = True):
        # 1. Parse text into structures
        structured_states = {i: self.parser.parse(reasoning_texts.get(i, ""), answers.get(i, "")) for i in range(self.n_agents)}
        
        # 2. Extract exactly 5 metrics
        semantic_state = self.semantic_encoder.encode_state(structured_states)
        features = self.build_features(semantic_state, confidence_scores, prev_W, round_num, unique_answers, avg_accuracy)
        
        raw_out, actor_cache = self.actor.forward(features)
        value, critic_cache = self.critic.forward(features)

        # Split actor output into mean and log_std (learnable exploration)
        mean_action = raw_out[:self.mean_dim]
        log_std_raw  = raw_out[self.mean_dim:]
        # Clamp log_std: allow network to reduce noise as it converges
        # log_std in [-3, -0.5] => sigma in [0.05, 0.61]
        log_std = np.clip(log_std_raw, -3.0, -0.5)
        sigma = np.exp(log_std)  # per-dimension, learned

        if explore:
            noise = np.random.randn(*mean_action.shape)
            noise[-1] = 0.0  # Remove exploration noise from the discrete halt logit
            action = mean_action + noise * sigma
        else:
            action = mean_action

        # Gaussian log-prob (per-dimension sigma)
        log_prob = -0.5 * np.sum(
            ((action - mean_action) / (sigma + 1e-8))**2 + 2 * log_std + np.log(2 * np.pi)
        )

        W, Q, halt_logit = self._split_action(action)
        halt_decision = (1 / (1 + np.exp(-np.clip(halt_logit, -500, 500)))) > 0.5

        meta = {
            'features': features,
            'action': action,
            'log_prob': log_prob,
            'log_std': log_std,
            'sigma': sigma,
            'value': value[0],
            'actor_cache': actor_cache,
            'mean_action': mean_action
        }
        return W, Q, halt_decision, meta

    def compute_reward(self, task_correct: bool, tokens_transmitted: int, tokens_if_full: int, compression_ratio: float, n_cycles: int, halt_round: int = 0) -> float:
        task_rew = 1.0 if task_correct else -0.5
        # Use the ACTUAL full-communication token count as the denominator.
        # This makes the reward invariant to LLM backend verbosity.
        # Fallback to a conservative estimate if tokens_if_full is 0 (e.g. no communication rounds ran).
        n_rounds = getattr(self.config, 'n_rounds', 3)
        baseline_max = max(tokens_if_full, self.n_agents * (self.n_agents - 1) * 50 * n_rounds, 1.0)
        info_cost = tokens_transmitted / baseline_max
        loop_penalty = n_cycles / max(self.n_agents * (self.n_agents - 1) // 2, 1)
        halt_penalty = halt_round * 0.15  # Scaled penalty for taking more rounds
        return task_rew - self.lambda_info * info_cost - self.gamma_loop * loop_penalty - halt_penalty

    def store_transition(self, meta, reward: float, done: bool = False):
        if meta:
            meta['reward'] = reward
            meta['done'] = done
            self.saved_transitions.append(meta)

    def update_policy(self) -> float:
        if not self.saved_transitions: return 0.0

        n_steps = len(self.saved_transitions)

        rewards  = np.array([t['reward'] for t in self.saved_transitions], dtype=np.float64)
        values   = np.array([t['value']  for t in self.saved_transitions], dtype=np.float64)

        # --- Generalised Advantage Estimation (GAE, lambda=0.95) ---
        # Reduces variance vs pure MC returns while keeping low bias.
        # Terminal steps bootstrap from V=0 (no future value).
        advantages = np.zeros(n_steps, dtype=np.float64)
        returns    = np.zeros(n_steps, dtype=np.float64)
        gae        = 0.0
        next_value = 0.0  # terminal bootstrap
        for t in reversed(range(n_steps)):
            # Check done AFTER computing delta so the terminal step
            # correctly uses next_value=0 (no successor).
            is_done = self.saved_transitions[t].get('done', False)
            # At terminal steps, next_value should be 0 (no future).
            # Reset gae accumulator to prevent cross-episode bleed.
            mask = 0.0 if is_done else 1.0
            delta       = rewards[t] + self.gamma_discount * next_value * mask - values[t]
            gae         = delta + self.gamma_discount * self.gae_lambda * gae * mask
            advantages[t] = gae
            returns[t]    = gae + values[t]  # V-target for critic
            next_value  = values[t]

        # Normalize advantages for numerical stability
        if np.std(advantages) > 1e-8:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        total_actor_loss, total_critic_loss = 0.0, 0.0

        for epoch in range(self.ppo_epochs):
            indices    = np.random.permutation(n_steps)
            batch_size = 16

            for start_idx in range(0, n_steps, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]

                actor_grads  = {n: np.zeros_like(getattr(self.actor,  n)) for n in ['W1','b1','W2','b2','W3','b3']}
                critic_grads = {n: np.zeros_like(getattr(self.critic, n)) for n in ['W1','b1','W2','b2','W3','b3']}

                for i in batch_indices:
                    feat         = self.saved_transitions[i]['features']
                    action       = self.saved_transitions[i]['action']
                    old_log_prob = self.saved_transitions[i]['log_prob']
                    old_sigma    = self.saved_transitions[i].get('sigma', np.full_like(action, 0.2))
                    adv          = advantages[i]
                    ret          = returns[i]

                    # --- Forward re-evaluation ---
                    raw_out_new, acache = self.actor.forward(feat)
                    v_pred,     vcache  = self.critic.forward(feat)

                    mean_a_new   = raw_out_new[:self.mean_dim]
                    log_std_new  = np.clip(raw_out_new[self.mean_dim:], -3.0, -0.5)
                    sigma_new    = np.exp(log_std_new)

                    new_log_prob = -0.5 * np.sum(
                        ((action - mean_a_new) / (sigma_new + 1e-8))**2
                        + 2 * log_std_new + np.log(2 * np.pi)
                    )
                    ratio  = np.exp(np.clip(new_log_prob - old_log_prob, -10, 10))

                    surr1  = ratio * adv
                    surr2  = np.clip(ratio, 1.0 - self.ppo_epsilon, 1.0 + self.ppo_epsilon) * adv
                    entropy = 0.5 * np.sum(2 * log_std_new + np.log(2 * np.pi * np.e))
                    entropy_coef = 0.01
                    actor_loss = -min(surr1, surr2) - entropy_coef * entropy

                    # --- Actor backward (mean head) ---
                    # PPO clipping: zero the policy gradient when the ratio
                    # has crossed the trust region boundary in the direction
                    # that would IMPROVE the surrogate (making it diverge).
                    # When adv >= 0: clip if ratio > 1+eps (too optimistic)
                    # When adv <  0: clip if ratio < 1-eps (too pessimistic)
                    if adv >= 0:
                        clipped = ratio > (1.0 + self.ppo_epsilon)
                    else:
                        clipped = ratio < (1.0 - self.ppo_epsilon)
                    d_ratio = 0.0 if clipped else adv
                    d_log_prob = d_ratio * ratio
                    d_mean     = d_log_prob * (action - mean_a_new) / (sigma_new ** 2 + 1e-8)

                    # Gradient of log_std head: entropy-regularised to prevent collapse
                    entropy_coef = 0.01
                    d_log_std    = d_log_prob * ((action - mean_a_new)**2 / (sigma_new**2 + 1e-8) - 1.0)
                    d_log_std   += entropy_coef * np.ones_like(log_std_new)  # entropy bonus

                    # Combine gradients for both heads of the single actor network
                    d_out = np.concatenate([-d_mean / len(batch_indices),
                                            -d_log_std / len(batch_indices)])
                    agrad = self.actor.backward(acache, d_out)
                    for k in actor_grads: actor_grads[k] += agrad[k]

                    # --- Critic backward (fits GAE V-targets) ---
                    critic_loss = 0.5 * (ret - v_pred[0])**2
                    vgrad = self.critic.backward(vcache, np.array([(v_pred[0] - ret) / len(batch_indices)]))
                    for k in critic_grads: critic_grads[k] += vgrad[k]

                    total_actor_loss  += abs(actor_loss)
                    total_critic_loss += critic_loss

                self.actor.update(actor_grads)
                self.critic.update(critic_grads)

        avg_loss = (total_actor_loss + self.ppo_v_coef * total_critic_loss) / (n_steps * self.ppo_epochs)
        if hasattr(self, 'episode_losses'):
            self.episode_losses.append(avg_loss)
            # FIX: Only log the terminal reward for plotting
            self.episode_rewards.extend([t['reward'] for t in self.saved_transitions if t.get('done', False)])

        self.saved_transitions = []
        return avg_loss

    