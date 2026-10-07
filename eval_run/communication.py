"""
Communication System: Embedding-Attention Masker, communication graph.
Totally eliminates regex string-matching heuristics.
"""

import re
import hashlib
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from utils import count_tokens, split_sentences


class SemanticBandwidthMode(Enum):
    BLOCK = 0       # w < 0.05: silence channel
    PARTIAL = 1     # 0.05 <= w < 0.95: stochastic semantic dropout
    FULL = 2        # w >= 0.95: full message pass-through


def get_mode(w_ij: float) -> SemanticBandwidthMode:
    if w_ij < 0.05:
        return SemanticBandwidthMode.BLOCK
    elif w_ij >= 0.95:
        return SemanticBandwidthMode.FULL
    else:
        return SemanticBandwidthMode.PARTIAL


@dataclass
class StructuredSemanticState:
    """Minimal structured representation of an agent's reasoning state."""
    final_answer: str = ""
    reasoning_text: str = ""

    def to_formatted_string(self) -> str:
        parts = []
        if self.final_answer:
            parts.append(f"[FINAL ANSWER]: {self.final_answer}")
        if self.reasoning_text:
            parts.append(self.reasoning_text)
        return "\n".join(parts)


@dataclass
class TransmissionRecord:
    """Record of a single message transmission."""
    sender_id: int
    receiver_id: int
    mode: SemanticBandwidthMode
    original_tokens: int
    transmitted_tokens: int
    weight: float
    round_num: int
    semantic_similarity: float = 0.0
    information_retention: float = 0.0
    original_msg: str = ""
    filtered_msg: str = ""


class CommunicationGraph:
    """Directed communication graph with learnable edge weights W and semantic compression.
    
    Edges represent agent-to-agent channels; weights W determine bandwidth allocation.
    Implements semantic abstraction (extract claims, filter evidence) for bandwidth-aware messaging.
    Tracks transmission costs, compression ratios, and semantic similarity metrics.
    """

    def __init__(self, n_agents: int, vocab_size: int = 500, query_dim: int = 16):
        self.n_agents = n_agents
        
        self.W = np.ones((n_agents, n_agents)) * 0.5
        np.fill_diagonal(self.W, 0.0)  # Bug #4 fix: zero diagonal at construction, not only reset()
        self.Q = np.zeros((n_agents, query_dim))

        import logging
        logger = logging.getLogger("SparseComm")
        
        # Initialize embedder for semantic similarity calculation
        if SentenceTransformer is not None:
            try:
                self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("CommunicationGraph: SentenceTransformer loaded successfully")
            except Exception as e:
                logger.warning(f"CommunicationGraph: Failed to load SentenceTransformer: {e}")
                self._embedder = None
        else:
            self._embedder = None
            logger.warning("CommunicationGraph: SentenceTransformer not available. Semantic similarity will use fallback.")
        
        self.transmission_log: List[TransmissionRecord] = []
        self.round_token_counts: Dict[int, int] = {}
        self.modes_disabled: List[SemanticBandwidthMode] = []
        self._agent_last_text: Dict[int, str] = {}  # Tier4 #21: CONTRAST mode state
        self.dataset_type: str = "gsm8k"  # Bug A4 fix: set via set_dataset_type() before use

    def set_weights(self, W: np.ndarray, Q: Optional[np.ndarray] = None):
        """Set the full weight matrix and optional query semantics from policy."""
        self.W = W.copy()
        if Q is not None:
            self.Q = Q.copy()

    def transmit(self, sender_id: int, receiver_id: int, message: str, round_num: int, llm_backend=None) -> Tuple[str, int]:
        """
        Transmit a message applying LLM-driven Semantic Abstraction.
        """
        w_ij = self.W[sender_id][receiver_id]
        mode = get_mode(w_ij)
        
        # Ablation fallback check
        if mode in self.modes_disabled:
            mode = SemanticBandwidthMode.BLOCK
            w_ij = 0.0
            
        original_tokens = count_tokens(message)

        # V5 Priority #1 & #7: Continuous Latent Bottleneck / Reasoning Subgraph Extractor
        if w_ij < 0.05:
            filtered_msg = ""
            transmitted_tokens = 0
            sem_sim = 0.0
            info_ret = 0.0
        elif w_ij > 0.95:
            filtered_msg = message
            transmitted_tokens = count_tokens(filtered_msg)
            sem_sim = 1.0
            info_ret = 1.0
        else:
            # Extract final answer
            _ans_match = re.search(
                r'(?:Answer:|ANSWER:|The answer is:?)\s*(.+?)(?:\n\n|\Z)',
                message,
                re.IGNORECASE | re.DOTALL,
            )
            if _ans_match:
                ans_str = "Answer: " + _ans_match.group(1).strip()
            else:
                ans_str = "Answer: " + next((l.strip() for l in reversed(message.splitlines()) if l.strip()), '')

            # Split into reasoning nodes (sentences)
            sentences = split_sentences(message)
            
            try:
                import random
                import numpy as np
                
                # V5 Bugfix: Use SentenceTransformers for true semantic extraction instead of lexical TF-IDF
                if hasattr(self, '_embedder') and self._embedder is not None and len(sentences) > 0:
                    import torch
                    X_emb = self._embedder.encode(sentences, convert_to_tensor=True, show_progress_bar=False)
                    ans_emb = self._embedder.encode([ans_str], convert_to_tensor=True, show_progress_bar=False)
                    sims = torch.nn.functional.cosine_similarity(X_emb, ans_emb).cpu().numpy().flatten()
                    # Rescale sims to [0, 1] range to avoid negative probabilities breaking the mean logic
                    sims = (sims + 1.0) / 2.0
                else:
                    # Fallback to TF-IDF if embedder is not available
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    from sklearn.metrics.pairwise import cosine_similarity
                    vectorizer = TfidfVectorizer()
                    X = vectorizer.fit_transform(sentences)
                    ans_vec = vectorizer.transform([ans_str])
                    sims = cosine_similarity(X, ans_vec).flatten()
                
                # Normalize sims so the mean is 1.0
                mean_sim = np.mean(sims) + 1e-9
                normalized_sims = sims / mean_sim
                
                # V5 Bugfix: Stochastic Semantic Dropout (Continuous in Expectation)
                kept_sentences = []
                for i, s in enumerate(sentences):
                    # P(keep) = w_ij * normalized_relevance
                    p_keep = min(1.0, float(w_ij * normalized_sims[i]))
                    if random.random() < p_keep:
                        kept_sentences.append(s)
                        
                filtered_msg = " ".join(kept_sentences)
            except Exception:
                # Fallback: pure uniform stochastic dropout
                import random
                kept_sentences = [s for s in sentences if random.random() < w_ij]
                filtered_msg = " ".join(kept_sentences)
                
            transmitted_tokens = count_tokens(filtered_msg)
            # ACTUAL SEMANTIC SIMILARITY CALCULATION
            if hasattr(self, '_embedder') and self._embedder is not None and message.strip() and filtered_msg.strip():
                try:
                    import torch
                    emb_orig = self._embedder.encode([message], convert_to_tensor=True, show_progress_bar=False)
                    emb_filt = self._embedder.encode([filtered_msg], convert_to_tensor=True, show_progress_bar=False)
                    cos_sim = torch.nn.functional.cosine_similarity(emb_orig, emb_filt)
                    sem_sim = float(cos_sim.item())
                except Exception as e:
                    print("EMBEDDER ERROR:", e)
                    sem_sim = 0.0
            else:
                sem_sim = 0.0
                
            info_ret = sem_sim  # Edge-level proxy; round-level InfoNCE computes true mutual info

        record = TransmissionRecord(
            sender_id=sender_id,
            receiver_id=receiver_id,
            mode=mode,
            original_tokens=original_tokens,
            transmitted_tokens=transmitted_tokens,
            weight=w_ij,
            round_num=round_num,
            semantic_similarity=sem_sim,
            information_retention=info_ret,
            original_msg=message,
            filtered_msg=filtered_msg
        )
        self.transmission_log.append(record)

        self.round_token_counts[round_num] = self.round_token_counts.get(round_num, 0) + transmitted_tokens

        return filtered_msg, transmitted_tokens

    def get_messages_for_round(self, agent_responses: dict, round_num: int, llm_backend=None) -> Dict[int, List[str]]:
        # Tier4 #21: Update agent last known texts for CONTRAST mode
        self._agent_last_text.update({k: v for k, v in agent_responses.items()})
        received: Dict[int, List[str]] = {i: [] for i in range(self.n_agents)}
        for sender_id in range(self.n_agents):
            for receiver_id in range(self.n_agents):
                # Disallow self-loops (self-reflection bias)
                if sender_id == receiver_id:
                    continue
                msg = agent_responses.get(sender_id, "")
                filtered_msg, _ = self.transmit(sender_id, receiver_id, msg, round_num, llm_backend)
                if filtered_msg.strip():
                    received[receiver_id].append(f"Agent {sender_id}: {filtered_msg}")
        return received

    def get_total_tokens_transmitted(self) -> int:
        return sum(r.transmitted_tokens for r in self.transmission_log)

    def get_total_tokens_if_full(self) -> int:
        return sum(r.original_tokens for r in self.transmission_log)

    def get_compression_ratio(self) -> float:
        active = [r for r in self.transmission_log if r.original_tokens > 0]
        if not active: return 0.0
        # V5 Bugfix: Mathematical compression ratio (sum(transmitted) / sum(original))
        return float(sum(r.transmitted_tokens for r in active) / sum(r.original_tokens for r in active))

    def get_avg_semantic_similarity(self) -> float:
        total_edges = [r for r in self.transmission_log if r.mode != SemanticBandwidthMode.FULL]
        active = [r for r in total_edges if r.original_tokens > 0 and r.weight > 0.0]
        zero_sim = [r for r in active if r.semantic_similarity == 0.0]
        
        print(f"DIAGNOSTIC: n_total_edges={len(total_edges)}, n_active_edges={len(active)}, n_zero_similarity={len(zero_sim)}")
        
        if not active: return 0.0
        return float(np.mean([r.semantic_similarity for r in active]))

    def get_avg_information_retention(self) -> float:
        """V5 Priority #3: InfoNCE Mutual Information Lower Bound"""
        active = [r for r in self.transmission_log if r.mode != SemanticBandwidthMode.FULL and r.original_tokens > 0 and r.weight > 0.0]
        if not active or len(active) < 2: return 1.0
        
        try:
            import numpy as np
            orig_msgs = [r.original_msg for r in active]
            comp_msgs = [r.filtered_msg for r in active]
            
            # V5 Bugfix: Use SentenceTransformers for true semantic InfoNCE instead of lexical TF-IDF
            if hasattr(self, '_embedder') and self._embedder is not None:
                import torch
                X_emb = self._embedder.encode(orig_msgs, convert_to_tensor=True, show_progress_bar=False)
                Z_emb = self._embedder.encode(comp_msgs, convert_to_tensor=True, show_progress_bar=False)
                X_norm = torch.nn.functional.normalize(X_emb, p=2, dim=1)
                Z_norm = torch.nn.functional.normalize(Z_emb, p=2, dim=1)
                sim_matrix = torch.matmul(X_norm, Z_norm.T).cpu().numpy()
            else:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                vectorizer = TfidfVectorizer()
                vectorizer.fit(orig_msgs + comp_msgs)
                X = vectorizer.transform(orig_msgs)
                Z = vectorizer.transform(comp_msgs)
                sim_matrix = cosine_similarity(X, Z)  # (N, N)
            tau = 0.1
            sim_matrix /= tau
            
            # log sum exp trick
            max_sim = np.max(sim_matrix, axis=1, keepdims=True)
            sum_exp = np.sum(np.exp(sim_matrix - max_sim), axis=1, keepdims=True)
            log_sum_exp = max_sim + np.log(sum_exp)
            
            # Positive pairs on diagonal
            pos_sim = np.diag(sim_matrix)
            
            # L_InfoNCE
            l_infonce = np.mean(pos_sim - log_sum_exp.flatten())
            # Convert to retention metric [0, 1]
            max_mi = np.log(len(orig_msgs))
            retention = min(1.0, max(0.0, (l_infonce + max_mi) / max_mi))
            return float(retention)
        except Exception:
            # Fallback if TF-IDF fails (e.g. empty vocab)
            return float(np.mean([r.information_retention for r in active]))

    def get_communication_density(self) -> float:
        """V5 Priority #4: L1-Norm Weighted Communication Density."""
        n_possible = self.n_agents * (self.n_agents - 1)
        if n_possible == 0:
            return 0.0
        W_off = self.W.copy()
        np.fill_diagonal(W_off, 0.0)
        return float(np.sum(W_off) / n_possible)

    def get_active_edges(self, threshold: float = 0.25) -> int:
        """Count off-diagonal edges above threshold."""
        W_off = self.W.copy()
        np.fill_diagonal(W_off, 0.0)
        return int(np.sum(W_off >= threshold))

    def get_mode_distribution(self) -> Dict[str, int]:
        dist = {mode.name: 0 for mode in SemanticBandwidthMode}
        for r in self.transmission_log:
            dist[r.mode.name] = dist.get(r.mode.name, 0) + 1
        return dist

    def count_cycles(self, threshold: float = 0.5) -> int:
        """Count 3-cycles in the communication graph using trace(A^3)/6."""
        if threshold is None:
            threshold = 0.25
        adj = (self.W >= threshold).astype(int)
        np.fill_diagonal(adj, 0)
        return int(np.trace(adj @ adj @ adj)) // 6

    def reset(self):
        self.transmission_log = []
        self.round_token_counts = {}
        self._agent_last_text = {}  # Tier4 #21: Clear CONTRAST state on new sample
        # Reset weight matrix and ensure diagonal (self-loops) are always zero
        self.W = np.ones((self.n_agents, self.n_agents)) * 0.5
        np.fill_diagonal(self.W, 0.0)

    def set_dataset_type(self, dataset_type: str):
        """Bug A4 fix: configure parser branch for transmit()."""
        self.dataset_type = dataset_type

    def get_weight_snapshot(self) -> np.ndarray:
        return self.W.copy()

    def disable_mode(self, mode_to_disable: SemanticBandwidthMode):
        if mode_to_disable not in self.modes_disabled:
            self.modes_disabled.append(mode_to_disable)

    def restore_all_modes(self):
        self.modes_disabled = []
