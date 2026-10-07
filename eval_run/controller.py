"""
Information Bottleneck Controller: Pure NumPy PPO implementation.

Features a continuous policy mechanism outputting W (Edge Constraints)
and Q (Semantic Masking Queries). Uses separate Actor-Critic MLPs.
"""

import numpy as np
import torch
import math
import re
from typing import List, Optional, Tuple, Dict
from collections import Counter
from config import Config
from communication import StructuredSemanticState
from agents import _normalize_for_vote, PlattCalibrator


class SemanticStateEncoder:
    """Encodes multi-agent reasoning state into 5 interpretable features for PPO controller.
    
    Features: Shannon entropy (disagreement), Jaccard similarity (overlap), 
    contradiction score, novelty score, consensus strength.
    These ground the policy network's decision-making in language semantics.
    """
    def __init__(self, n_agents: int):
        self.n_agents = n_agents

    def encode_state(self, structured_states: Dict) -> np.ndarray:
        """
        Information-theoretic state representation (ACL-worthy, zero API cost).
        Measures: disagreement, reasoning overlap, contradiction, novelty, consensus.
        No SentenceTransformer needed - pure mathematics.
        """
        if not structured_states:
            return np.zeros(5, dtype=np.float64)

        # V3-3: normalize answers before entropy so '24'/'24.0'/'24.00' collapse to one key
        _dataset_type = getattr(self, '_dataset_type', 'gsm8k')
        answers = []
        texts = []
        for s in structured_states.values():
            if hasattr(s, 'final_answer'):
                answers.append(_normalize_for_vote(str(s.final_answer).lower().strip(), _dataset_type))
            else:
                answers.append('')
            if hasattr(s, 'to_formatted_string'):
                texts.append(s.to_formatted_string())
            else:
                texts.append(str(s))

        # 1. Shannon Entropy (disagreement measure)
        if len(answers) > 0:
            unique, counts = np.unique(answers, return_counts=True)
            probs = counts / len(answers)
            shannon_entropy = -np.sum(probs * np.log2(probs + 1e-8))
            # Normalize: max entropy = log2(n_agents), generalized for any N (Bug #3 fix)
            shannon_entropy = shannon_entropy / max(np.log2(self.n_agents), 1e-8)
        else:
            shannon_entropy = 0.0

        # 2. Jaccard Similarity (reasoning overlap)
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

        # 3. Answer Diversity (fraction unique; HIGH=disagreement). V3-4 fix: was inverted.
        unique_answers = len(set(answers))
        answer_diversity = unique_answers / max(len(answers), 1)

        # 4. Answer Length Variance (CoV of reasoning length — proxy for confidence spread)
        # V3-4: replaces novelty_score which was redundant with consensus_strength
        if texts:
            lengths = np.array([len(t.split()) for t in texts], dtype=float)
            mean_len = lengths.mean()
            length_variance = float(min(1.0, np.std(lengths) / (mean_len + 1e-8)))
        else:
            length_variance = 0.0

        # 5. Consensus Strength
        if unique_answers == 1:
            consensus_strength = 1.0
        else:
            answer_counts = Counter(answers)
            majority_count = answer_counts.most_common(1)[0][1]
            consensus_strength = majority_count / len(answers)

        features = np.array([
            shannon_entropy,
            jaccard_similarity,
            answer_diversity,    # V3-4: was contradiction_score (semantically inverted)
            length_variance,     # V3-4: replaces novelty_score (was redundant)
            consensus_strength,
        ], dtype=np.float64)
        return np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)


class MLP:
    """Standard NumPy 2-hidden-layer MLP with AdamW optimizer and gradient clipping.
    
    Used for both actor (policy π) and critic (value V) networks in PPO controller.
    Implements ReLU activations, handles sigmoid output for bounded predictions.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, lr: float = 1e-3, use_sigmoid: bool = False):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        self.use_sigmoid = use_sigmoid

        scale1 = np.sqrt(2.0 / (input_dim + hidden_dim))
        scale2 = np.sqrt(2.0 / (hidden_dim + hidden_dim))
        scale3 = np.sqrt(2.0 / (hidden_dim + output_dim))

        self.W1 = np.random.randn(input_dim, hidden_dim) * scale1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * scale2
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.randn(hidden_dim, output_dim) * scale3
        self.b3 = np.zeros(output_dim)

        self.m = {n: np.zeros_like(getattr(self, n)) for n in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']}
        self.v = {n: np.zeros_like(getattr(self, n)) for n in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']}
        self.t = 0

    @staticmethod
    def _relu(x): return np.maximum(0, x)

    @staticmethod
    def _relu_grad(x): return (x > 0).astype(float)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, dict]:
        z1 = x @ self.W1 + self.b1
        a1 = self._relu(z1)

        z2 = a1 @ self.W2 + self.b2
        a2 = self._relu(z2)

        z3 = a2 @ self.W3 + self.b3
        out = z3
        
        if self.use_sigmoid:
            out = np.where(z3 >= 0, 1 / (1 + np.exp(-np.clip(z3, -500, 500))), np.exp(np.clip(z3, -500, 500)) / (1 + np.exp(np.clip(z3, -500, 500))))

        cache = {'x': x, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2, 'z3': z3, 'out': out}
        return out, cache

    def backward(self, cache: dict, grad_output: np.ndarray) -> dict:
        x, a1, a2, z1, z2, z3, out = cache['x'], cache['a1'], cache['a2'], cache['z1'], cache['z2'], cache['z3'], cache['out']

        dz3 = grad_output
        if self.use_sigmoid:
            dz3 = grad_output * out * (1 - out)

        dW3 = np.outer(a2, dz3)
        db3 = dz3
        da2 = dz3 @ self.W3.T

        dz2 = da2 * self._relu_grad(z2)
        dW2 = np.outer(a1, dz2)
        db2 = dz2
        da1 = dz2 @ self.W2.T

        dz1 = da1 * self._relu_grad(z1)
        dW1 = np.outer(x, dz1)
        db1 = dz1

        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2, 'W3': dW3, 'b3': db3}

    def update(self, grads: dict, max_norm: float = 1.0):
        self.t += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        total_norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            grads = {k: v * clip_coef for k, v in grads.items()}

        for name in grads:
            g = grads[name]
            self.m[name] = beta1 * self.m[name] + (1 - beta1) * g
            self.v[name] = beta2 * self.v[name] + (1 - beta2) * (g ** 2)
            m_hat = self.m[name] / (1 - beta1 ** self.t)
            v_hat = self.v[name] / (1 - beta2 ** self.t)
            
            param = getattr(self, name)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
            setattr(self, name, param)


SPARSITY_PRIOR: float = 0.1


class IBController:
    """PPO Controller with native Actor-Critic MLPs and Semantic State Awareness."""

    def __init__(self, config: Config):
        self.config = config
        self.n_agents = config.n_agents
        self.query_dim = getattr(config, 'query_dim', 16)
        
        self.semantic_encoder = SemanticStateEncoder(config.n_agents)
        
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

        self.lambda_kl = getattr(config, 'lambda_kl', 0.01)
        self.lambda_info = config.lambda_info
        self.gamma_loop = config.gamma_loop
        self.gamma_discount = config.gamma_discount
        self.ppo_epsilon = getattr(config, 'ppo_clip_epsilon', 0.2)
        self.ppo_epochs = getattr(config, 'ppo_epochs', 3)
        self.ppo_v_coef = getattr(config, 'ppo_value_coef', 0.5)

        self.episode_rewards: List[float] = []
        self.episode_losses: List[float] = []
        self.saved_transitions: List[dict] = []
        # Fix 4: adaptive_prior removed; SPARSITY_PRIOR = 0.1 constant used
        self.calibrator: PlattCalibrator = PlattCalibrator()  # Improvement VI: Platt scaling

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
        W_raw = H_src @ H_dst.T + 0.5
        np.fill_diagonal(W_raw, -500.0)  # Pre-sigmoid elimination
        W = 1 / (1 + np.exp(-np.clip(W_raw, -500, 500)))
        np.fill_diagonal(W, 0.0)  # Hard-zero: ensures density/state features are clean
        
        Q = q_flat.reshape(N, self.query_dim)
        return W, Q, halt_logit

    def get_weights(self, reasoning_texts: Dict[int, str], answers: Dict[int, str], confidence_scores: List[float], prev_W: np.ndarray, round_num: int, unique_answers: int = 1, avg_accuracy: float = 0.5, explore: bool = True):
        # 1. Build structured states directly from reasoning_texts and answers
        # Bug #2 Fix: removed self.parser (HeuristicSemanticParser was deleted).
        # Feed text and answer directly into StructuredSemanticState.
        dataset_type = getattr(self.config, 'dataset', 'gsm8k')
        self.semantic_encoder._dataset_type = dataset_type
        _norm_answers = {i: _normalize_for_vote(answers.get(i, ''), dataset_type) for i in range(self.n_agents)}
        structured_states = {
            i: StructuredSemanticState(
                final_answer=_norm_answers.get(i, ''),
                reasoning_text=reasoning_texts.get(i, '')
            )
            for i in range(self.n_agents)
        }

        # 2. Extract exactly 5 metrics
        semantic_state = self.semantic_encoder.encode_state(structured_states)
        features = self.build_features(semantic_state, confidence_scores, prev_W, round_num, unique_answers, avg_accuracy)
        
        raw_out, actor_cache = self.actor.forward(features)
        value, critic_cache = self.critic.forward(features)

        # Split actor output into mean and log_std (learnable exploration)
        mean_action = raw_out[:self.mean_dim]
        log_std_raw  = raw_out[self.mean_dim:]
        # Clamp log_std: allow network to explore initially, then reduce noise as it converges
        # log_std in [-2.0, 0.5] => sigma in [0.13, 1.64]
        log_std = np.clip(log_std_raw, -2.0, 0.5)
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

        # Smart early halting: purely based on controller's learned logit
        halt_prob = 1 / (1 + np.exp(-np.clip(halt_logit, -500, 500)))
        halt_decision = bool(halt_prob > 0.5)

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

    def compute_reward(
        self,
        task_reward: float,
        tokens_transmitted: int,
        tokens_if_full: int,
        compression_ratio: float,
        n_cycles: int,
        current_W: np.ndarray,
        halt_round: int = 0,
        is_final_step: bool = True,
        include_token_penalty: bool = False,  # Fix2: False by default — intermediate steps handle per-round tokens
    ) -> float:
        # Task reward: I(Z; Y) - accuracy signal
        task_rew = (task_reward * 1.5 - 0.5) if is_final_step else 0.0

        # Fix 2: Shaping rewards removed (see compute_reward docstring)

        # Fix2: Token penalty is applied per-round in run_single_sample intermediate steps.
        # include_token_penalty=False (default) avoids double-counting the total token cost
        # in the final compute_reward call on top of already-penalised intermediate steps.
        info_cost = (tokens_transmitted / max(float(tokens_if_full), 1.0)) if include_token_penalty else 0.0

        # KL Sparsity Penalty: encourage W ~ Bernoulli(adaptive_prior) — Tier4 #22
        # Prior adapts via EMA of observed W during training.
        prior_p = SPARSITY_PRIOR  # Fix 4: fixed constant
        W_clip = np.clip(current_W, 1e-5, 1.0 - 1e-5)
        # Exclude diagonal entirely from KL computation using boolean mask
        off_diag_mask = ~np.eye(W_clip.shape[0], dtype=bool)
        W_off = W_clip[off_diag_mask]
        
        kl_raw = float(np.sum(
            W_off * np.log(W_off / prior_p) +
            (1 - W_off) * np.log((1 - W_off) / (1 - prior_p))
        ))
        n_edges = max(self.n_agents * (self.n_agents - 1), 1)
        kl_divergence = kl_raw / n_edges  # Per-edge normalized KL

        # Cycle penalty
        loop_penalty = n_cycles / max(self.n_agents * (self.n_agents - 1) // 2, 1)

        # Halt penalty (discourage using all rounds)
        halt_penalty = halt_round * 0.1

        return (
            task_rew
            - self.lambda_info * info_cost
            - self.lambda_kl * kl_divergence
            - self.gamma_loop * loop_penalty
            - halt_penalty
        )

    def store_transition(self, meta, reward: float, done: bool = False):
        if meta:
            meta['reward'] = reward
            meta['done'] = done
            self.saved_transitions.append(meta)

    # Fix 4: update_adaptive_prior() removed. SPARSITY_PRIOR=0.1 used.
    def update_policy(self) -> float:
        if not self.saved_transitions: return 0.0

        n_steps = len(self.saved_transitions)

        rewards  = np.array([t['reward'] for t in self.saved_transitions], dtype=np.float64)
        values   = np.array([t['value']  for t in self.saved_transitions], dtype=np.float64)

        # --- Generalised Advantage Estimation (GAE, lambda=0.95) ---
        advantages = np.zeros(n_steps, dtype=np.float64)
        returns    = np.zeros(n_steps, dtype=np.float64)
        gae        = 0.0
        next_value = 0.0  # terminal bootstrap
        for t in reversed(range(n_steps)):
            is_done = self.saved_transitions[t].get('done', False)
            mask = 0.0 if is_done else 1.0
            if is_done:
                next_value = 0.0
                gae = 0.0
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
                    # Bug #6 Fix: unify PPO log_std clamp with sampling clamp [-2.0, 0.5]
                    log_std_new  = np.clip(raw_out_new[self.mean_dim:], -2.0, 0.5)
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
                    d_log_prob_wrt_mean = (action - mean_a_new) / (sigma_new ** 2 + 1e-8)
                    clipped_ratio = np.clip(ratio, 1.0 - self.ppo_epsilon, 1.0 + self.ppo_epsilon)
                    use_clipped = (ratio > 1.0 + self.ppo_epsilon and adv > 0) or (ratio < 1.0 - self.ppo_epsilon and adv < 0)
                    d_mean = adv * (clipped_ratio if use_clipped else ratio) * d_log_prob_wrt_mean

                    # Gradient of log_std: uses unclipped (adv * ratio), not clipped d_log_prob
                    d_log_std    = (adv * ratio) * ((action - mean_a_new)**2 / (sigma_new**2 + 1e-8) - 1.0)
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
            self.episode_rewards.extend([t['reward'] for t in self.saved_transitions if t.get('done', False)])

        self.saved_transitions = []
        return avg_loss

    def get_static_weights(self, topology: str = "ring") -> Tuple[np.ndarray, np.ndarray]:
        N = self.n_agents
        W = np.zeros((N, N))
        if topology == "ring":
            for i in range(N):
                W[i][(i + 1) % N] = 0.6
                W[(i + 1) % N][i] = 0.6
        else:
            W = np.ones((N, N)) * 0.5
            np.fill_diagonal(W, 0)
        return W, np.zeros((N, self.query_dim))

    def get_random_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        W = np.random.rand(self.n_agents, self.n_agents)
        np.fill_diagonal(W, 0)
        Q = np.random.randn(self.n_agents, self.query_dim)
        return W, Q

    def save_model(self, filepath: str):
        """Save actor, critic MLP weights + full calibrator state."""
        import os, pickle
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        np.savez(filepath,
                 actor_W1=self.actor.W1, actor_b1=self.actor.b1,
                 actor_W2=self.actor.W2, actor_b2=self.actor.b2,
                 actor_W3=self.actor.W3, actor_b3=self.actor.b3,
                 critic_W1=self.critic.W1, critic_b1=self.critic.b1,
                 critic_W2=self.critic.W2, critic_b2=self.critic.b2,
                 critic_W3=self.critic.W3, critic_b3=self.critic.b3,
                 mean_dim=np.array(self.mean_dim),
                 calibrator_fitted=np.array(self.calibrator._fitted) if self.calibrator is not None else np.array(False))
        # Bug #8 Fix: persist full calibrator state (IR model, per-agent stats) via pickle
        calib_path = filepath.replace('.npz', '_calibrator.pkl')
        with open(calib_path, 'wb') as f:
            if self.calibrator is not None:
                pickle.dump({
                    'ir': self.calibrator.ir if self.calibrator._fitted else None,
                    '_fitted': self.calibrator._fitted,
                    '_agent_vars': self.calibrator._agent_vars,
                    '_agent_historical_acc': self.calibrator._agent_historical_acc,
                }, f)
            else:
                pickle.dump(None, f)

    def load_model(self, filepath: str):
        """Load actor/critic weights and full calibrator state."""
        import pickle
        data = np.load(filepath)
        self.actor.W1 = data['actor_W1']
        self.actor.b1 = data['actor_b1']
        self.actor.W2 = data['actor_W2']
        self.actor.b2 = data['actor_b2']
        self.actor.W3 = data['actor_W3']
        self.actor.b3 = data['actor_b3']
        self.critic.W1 = data['critic_W1']
        self.critic.b1 = data['critic_b1']
        self.critic.W2 = data['critic_W2']
        self.critic.b2 = data['critic_b2']
        self.critic.W3 = data['critic_W3']
        self.critic.b3 = data['critic_b3']
        # Bug #8 Fix: restore full calibrator from pickle if present
        calib_path = filepath.replace('.npz', '_calibrator.pkl')
        import os
        if os.path.exists(calib_path):
            with open(calib_path, 'rb') as f:
                calib_data = pickle.load(f)
            if calib_data is not None:
                if calib_data.get('ir') is not None:
                    self.calibrator.ir = calib_data['ir']
                self.calibrator._fitted = calib_data.get('_fitted', False)
                self.calibrator._agent_vars = calib_data.get('_agent_vars', {})
                self.calibrator._agent_historical_acc = calib_data.get('_agent_historical_acc', {})


class FrozenWeightController(IBController):
    """Bug #1 fix: Subclass that returns a fixed W instead of calling the MLP.

    run_experiment_with_random_weights was broken because run_single_sample calls
    controller.get_weights() each round, overwriting comm_graph.W.
    This subclass short-circuits that by returning the frozen W unchanged.
    """
    def __init__(self, config: Config, frozen_W: np.ndarray, frozen_Q: np.ndarray):
        super().__init__(config)
        self._frozen_W = frozen_W.copy()
        self._frozen_Q = frozen_Q.copy()

    def get_weights(self, reasoning_texts=None, answers=None, confidence_scores=None,
                    prev_W=None, round_num=0, unique_answers=1, avg_accuracy=0.5, explore=False):
        """Return frozen W unchanged — never overwrites the ablation topology."""
        dummy_meta = {
            'features': np.zeros(self.input_dim),
            'action': np.zeros(self.mean_dim),
            'log_prob': 0.0,
            'log_std': np.zeros(self.mean_dim),
            'sigma': np.ones(self.mean_dim),
            'value': 0.0,
            'actor_cache': None,
            'mean_action': np.zeros(self.mean_dim),
        }
        return self._frozen_W.copy(), self._frozen_Q.copy(), False, dummy_meta
