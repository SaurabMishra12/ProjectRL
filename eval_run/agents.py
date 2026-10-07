"""
Multi-Agent Framework: Agent class, AgentPool, and response aggregation.
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from llm_backend import LLMBackend, LLMResponse


@dataclass
class AgentResponse:
    """Response from a single agent for a single round."""
    agent_id: int
    answer: str
    confidence: float
    reasoning_text: str
    token_count: int
    round_num: int


class Agent:
    """An LLM-powered agent that solves tasks and incorporates peer communication.
    
    Each agent maintains a reasoning history, generates answers independently,
    then refines answers using filtered messages from other agents.
    Confidence is boosted if answers remain consistent across rounds.
    """

    def __init__(
        self,
        agent_id: int,
        llm_backend: LLMBackend,
        reasoning_style: str = "analytical",
    ):
        self.agent_id = agent_id
        self.llm_backend = llm_backend
        self.reasoning_style = reasoning_style
        self.history: List[AgentResponse] = []

    def solve(
        self,
        question: str,
        context_messages: Optional[List[str]] = None,
        dataset_type: str = "gsm8k",
        round_num: int = 0,
    ) -> AgentResponse:
        """
        Solve a question, optionally incorporating context from other agents.

        Args:
            question: The question to solve
            context_messages: List of messages from other agents (after filtering)
            dataset_type: "gsm8k" or "hotpotqa"
            round_num: Current communication round

        Returns:
            AgentResponse with answer, confidence, reasoning, and token count
        """
        # Build context string
        context = ""
        if context_messages:
            valid = [m for m in context_messages if m.strip()]
            if valid:
                context = "Key points from other agents:\n" + "\n".join(valid)

        # Build prompt
        prompt = self._build_prompt(question, context, dataset_type)

        # Get LLM response
        llm_response = self.llm_backend.generate(
            prompt=prompt,
            question=question,
            context=context,
            dataset_type=dataset_type,
            agent_id=self.agent_id,
        )

        # Compute confidence (combine LLM confidence with consistency check)
        confidence = self._compute_confidence(llm_response)

        response = AgentResponse(
            agent_id=self.agent_id,
            answer=llm_response.answer,
            confidence=confidence,
            reasoning_text=llm_response.text,
            token_count=llm_response.token_count,
            round_num=round_num,
        )
        self.history.append(response)
        return response

    def _build_prompt(
        self, question: str, context: str, dataset_type: str
    ) -> str:
        """Build the full prompt for the LLM."""
        style_instructions = {
            "analytical": "Analyze the problem systematically, identifying all relevant quantities and relationships.",
            "step_by_step": "Break the problem into clear, numbered steps and solve each one.",
            "creative": "Consider multiple approaches and choose the most elegant solution.",
            "careful": "Double-check each step of your reasoning before proceeding.",
            "efficient": "Find the most direct path to the solution.",
        }

        instruction = style_instructions.get(
            self.reasoning_style, style_instructions["analytical"]
        )

        if dataset_type == "gsm8k":
            task = "Solve this math word problem. Show your work and provide a numeric answer."
        else:
            task = "Answer this multi-hop question by reasoning through the connected facts."

        parts = [instruction, task, f"Question: {question}"]
        if context:
            parts.append(f"\n{context}")
        parts.append("\nProvide your reasoning and final answer.")

        return "\n\n".join(parts)

    def _compute_confidence(self, llm_response: LLMResponse) -> float:
        """
        Compute confidence based on LLM's internal estimate and
        consistency with previous answers.
        """
        base_confidence = llm_response.confidence

        if not self.history:
            return base_confidence

        # Fix 1: Consistency bonus removed (see _compute_confidence for details)
        return base_confidence

    def reset(self):
        """Reset agent history for a new question."""
        self.history = []


class AgentPool:
    """Manages a pool of N agents with coordinated multi-round problem solving.
    
    Handles parallel execution of agents (ThreadPoolExecutor) and message routing.
    Supports initial independent solving (Round 0) and iterative communication rounds.
    """

    def __init__(
        self,
        n_agents: int,
        llm_backend: LLMBackend,
        reasoning_styles: Optional[List[str]] = None,
    ):
        if reasoning_styles is None:
            reasoning_styles = ["analytical", "step_by_step", "creative",
                                "careful", "efficient"]

        self.agents: List[Agent] = []
        for i in range(n_agents):
            style = reasoning_styles[i % len(reasoning_styles)]
            self.agents.append(Agent(i, llm_backend, style))

        self.n_agents = n_agents

    def initial_solve(
        self, question: str, dataset_type: str = "gsm8k"
    ) -> List[AgentResponse]:
        """All agents solve the question independently (Round 0) in parallel."""
        futures = {}
        with ThreadPoolExecutor(max_workers=self.n_agents) as executor:
            for agent in self.agents:
                futures[executor.submit(
                    agent.solve, question,
                    None, dataset_type, 0
                )] = agent.agent_id

        # Reconstruct in stable agent_id order
        result_map = {}
        for future, agent_id in futures.items():
            result_map[agent_id] = future.result(timeout=120)
        return [result_map[i] for i in sorted(result_map)]

    def communicate_and_solve(
        self,
        question: str,
        received_messages: Dict[int, List[str]],
        dataset_type: str = "gsm8k",
        round_num: int = 1,
    ) -> List[AgentResponse]:
        """Execute agents in parallel with rate limit handling."""
        import time
        
        logger = logging.getLogger("SparseComm")  # ← USE EXISTING LOGGER
        
        max_retries = 3
        base_delay = 2.0  # seconds
        
        for attempt in range(max_retries):
            try:
                futures = {}
                with ThreadPoolExecutor(max_workers=self.n_agents) as executor:
                    for agent in self.agents:
                        messages = received_messages.get(agent.agent_id, [])
                        futures[executor.submit(
                            agent.solve, question, messages, dataset_type, round_num
                        )] = agent.agent_id

                result_map = {}
                for future, agent_id in futures.items():
                    result_map[agent_id] = future.result(timeout=120)
                return [result_map[i] for i in sorted(result_map)]
                
            except Exception as e:
                if "rate limit" in str(e).lower() or "429" in str(e):
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Rate limit hit. Retrying in {delay}s... (attempt {attempt+1}/{max_retries})")
                    time.sleep(delay)
                else:
                    raise  # Re-raise non-rate-limit errors
        
        raise RuntimeError(f"Failed after {max_retries} retries due to rate limiting")

    def reset_all(self):
        """Reset all agents for a new question."""
        for agent in self.agents:
            agent.reset()

    def get_confidence_vector(self) -> List[float]:
        """Get the latest confidence score from each agent."""
        confidences = []
        for agent in self.agents:
            if agent.history:
                confidences.append(agent.history[-1].confidence)
            else:
                confidences.append(0.5)
        return confidences


def _normalize_for_vote(answer: str, dataset_type: str) -> str:
    """Voting improvement I: numeric normalization before vote aggregation.
    Prevents "24", "24.0", "24.00", "1,200" being counted as different answers.
    """
    if dataset_type == "gsm8k":
        answer = answer.replace(",", "").strip()
        nums = re.findall(r"-?\d+\.?\d*", answer)
        if nums:
            try:
                v = float(nums[-1])
                return str(int(v)) if v == int(v) else f"{v:.4f}".rstrip("0").rstrip(".")
            except (ValueError, OverflowError):
                pass
    return answer.lower().strip()


class PlattCalibrator:
    """V5 Priority #2: Isotonic Calibration (renamed class internally to avoid breaking save_model signatures).
    State-of-the-art non-parametric calibration wrapping sklearn IsotonicRegression.
    """
    def __init__(self):
        from sklearn.isotonic import IsotonicRegression
        self.ir = IsotonicRegression(out_of_bounds='clip')
        self._fitted = False
        self._train_confs: List[float] = []
        self._train_labels: List[bool] = []
        # Dummy a/b for backward compatibility with save/load format
        self.a = 1.0; self.b = 0.0
        # V5 Priority #5 Fix: Empirical Variance tracking
        self._agent_sq_errors: Dict[int, List[float]] = {}
        self._agent_vars: Dict[int, float] = {}
        # V5 Bugfix: Historical agent accuracy overrides self-reported confidence
        self._agent_corrects: Dict[int, List[bool]] = {}
        self._agent_historical_acc: Dict[int, float] = {}

    def accumulate(self, agent_id: int, conf: float, is_correct: bool):
        self._train_confs.append(conf)
        self._train_labels.append(is_correct)
        if agent_id not in self._agent_corrects:
            self._agent_corrects[agent_id] = []
        self._agent_corrects[agent_id].append(is_correct)
        
    def accumulate_numeric(self, agent_id: int, y_pred: float, y_true: float):
        """Track squared errors to compute true empirical variance \sigma_i^2"""
        if agent_id not in self._agent_sq_errors:
            self._agent_sq_errors[agent_id] = []
        # V5 Bugfix: Use scale-invariant relative squared error to prevent large numbers from dominating
        rel_err = (y_pred - y_true) / max(abs(y_true), 1e-4)
        # Clip relative error at 10.0 (1000%) to prevent division by near-zero blowing up the metric
        rel_err = max(-10.0, min(10.0, rel_err))
        self._agent_sq_errors[agent_id].append(float(rel_err**2))

    def fit(self):
        if len(self._train_confs) < 10:
            return
        X = np.array(self._train_confs)
        y = np.array(self._train_labels, dtype=float)
        try:
            self.ir.fit(X, y)
            self._fitted = True
        except Exception:
            pass
            
        for aid, errs in self._agent_sq_errors.items():
            if len(errs) > 0:
                self._agent_vars[aid] = float(np.mean(errs))
                
        for aid, corrects in self._agent_corrects.items():
            if len(corrects) > 0:
                self._agent_historical_acc[aid] = float(np.mean(corrects))
                
    def get_variance(self, agent_id: int) -> float:
        """Return empirical variance \sigma_i^2, fallback to 1.0"""
        return self._agent_vars.get(agent_id, 1.0)
        
    def get_historical_accuracy(self, agent_id: int) -> Optional[float]:
        """Return empirical P(correct) for agent, or None if unknown"""
        return self._agent_historical_acc.get(agent_id, None)

    def calibrate(self, conf: float) -> float:
        if not self._fitted:
            return conf
        try:
            return float(self.ir.predict([conf])[0])
        except Exception:
            return conf


def should_halt_by_entropy(responses: List[AgentResponse], n_agents: int, threshold: float = 0.25) -> bool:
    """Voting improvement VII: entropy-based independent halt gate.
    Returns True if answer distribution is sufficiently concentrated.
    threshold=0.0 requires unanimity; threshold=0.5 accepts 2-of-3 majority.
    """
    answers = [r.answer for r in responses]
    counts = Counter(answers)
    n = len(answers)
    probs = np.array([c / n for c in counts.values()])
    H = -np.sum(probs * np.log2(probs + 1e-10))
    H_normalized = H / max(np.log2(n_agents), 1e-8)
    return bool(H_normalized < threshold)


def spectral_vote_weights(W: np.ndarray, base_confidences: List[float]) -> List[float]:
    """Voting improvement V: multiply agent confidence by in-degree centrality.
    Agents that other agents 'send to most' are trusted more.
    Zero extra API cost — uses already-computed W matrix.
    """
    W_off = W.copy()
    np.fill_diagonal(W_off, 0.0)
    in_degree = W_off.sum(axis=0)
    if in_degree.max() > in_degree.min() + 1e-8:
        centrality = 0.5 + (in_degree - in_degree.min()) / (in_degree.max() - in_degree.min())
    else:
        centrality = np.ones(len(base_confidences))
    return [float(c * centrality[i]) for i, c in enumerate(base_confidences)]


def compute_stability_weights(agents, n_agents: int) -> Dict[int, float]:
    """Voting improvement IV: penalize agents that flip-flopped between rounds.
    k answer changes → multiplier exp(-0.3k). Stable agents get weight 1.0.
    """
    stability = {}
    for agent in agents:
        if len(agent.history) <= 1:
            stability[agent.agent_id] = 1.0
        else:
            answers = [h.answer for h in agent.history]
            changes = sum(1 for i in range(1, len(answers)) if answers[i] != answers[i-1])
            stability[agent.agent_id] = float(np.exp(-0.3 * changes))
    # Fill any missing agent ids
    for aid in range(n_agents):
        stability.setdefault(aid, 1.0)
    return stability


def majority_vote(responses: List[AgentResponse], dataset_type: str = "gsm8k") -> str:
    """Simple majority vote with numeric normalization (Improvement I)."""
    answer_counts: Dict[str, int] = {}
    canonical_map: Dict[str, str] = {}
    for r in responses:
        key = _normalize_for_vote(r.answer, dataset_type)
        canonical_map.setdefault(key, r.answer)
        answer_counts[key] = answer_counts.get(key, 0) + 1
    best_key = max(answer_counts, key=answer_counts.get)
    return canonical_map[best_key]


def weighted_majority_vote(
    responses: List[AgentResponse],
    dataset_type: str = "gsm8k",
    W: Optional[np.ndarray] = None,
    agents=None,
    calibrator: Optional["PlattCalibrator"] = None,
) -> str:
    """Bayesian multiplicative vote (Improvement II) with numeric normalization (I),
    spectral centrality weighting (V), stability discounting (IV), and Platt calibration (VI).

    Falls back to additive weighting gracefully when optional inputs are absent.
    """
    if not responses:
        return ""

    # Step 1: Numeric normalization (Improvement I)
    normalized: List[str] = [_normalize_for_vote(r.answer, dataset_type) for r in responses]
    canonical_map: Dict[str, str] = {}
    for key, r in zip(normalized, responses):
        canonical_map.setdefault(key, r.answer)

    # Step 2: Build true confidence per agent (Improvement VI)
    # V5 Bugfix: Prioritize true historical P(correct) over LLM's self-reported confidence
    confs = []
    for r in responses:
        c = None
        if calibrator and hasattr(calibrator, 'get_historical_accuracy'):
            c = calibrator.get_historical_accuracy(r.agent_id)
            
        if c is None:
            if calibrator and calibrator._fitted:
                c = calibrator.calibrate(r.confidence)
            else:
                c = r.confidence
                
        confs.append(max(0.51, min(0.99, c)))  # Clamp for Bayesian log-odds

    # Step 3: Spectral centrality multiplier (Improvement V)
    if W is not None:
        confs = spectral_vote_weights(W, confs)
        confs = [max(0.51, min(0.99, c)) for c in confs]

    # Step 4: Stability discounting (Improvement IV)
    if agents is not None:
        stability = compute_stability_weights(agents, len(responses))
        confs = [c * stability.get(r.agent_id, 1.0) for c, r in zip(confs, responses)]
        confs = [max(0.51, min(0.99, c)) for c in confs]

    # Step 5: Bayesian multiplicative voting
    unique_answers = list(set(normalized))
    # V3-10: unanimous fast-path avoids artificial n_options=2 floor
    if len(unique_answers) == 1:
        return canonical_map[unique_answers[0]]
    n_options = max(len(unique_answers), 2)

    log_posterior: Dict[str, float] = {}
    for candidate in unique_answers:
        log_p = 0.0
        for ans, theta in zip(normalized, confs):
            if ans == candidate:
                log_p += np.log(theta)
            else:
                log_p += np.log((1.0 - theta) / (n_options - 1))
        log_posterior[candidate] = log_p

    # Numerically stable argmax
    best_key = max(log_posterior, key=log_posterior.get)

    # V5 Priority #5: Bayesian Aggregation (MAP estimate under Gaussian noise)
    # Replaces heuristic median. Optimal when \epsilon ~ N(0, \sigma^2).
    if dataset_type in ['gsm8k', 'aqua']:
        _num_entries = []
        for _key, _r, _c in zip(normalized, responses, confs):
            _nums = re.findall(r'-?\d+\.?\d*', _key)
            if _nums:
                try:
                    _num_entries.append((float(_nums[-1]), _r.agent_id, canonical_map[_key]))
                except (ValueError, IndexError):
                    pass
        if len(_num_entries) == len(responses) and len(_num_entries) > 0:
            _vals = np.array([v for v, _, _ in _num_entries])
            
            # V5 Bugfix: Empirical variance per agent, not a heuristic function of confidence
            if calibrator and hasattr(calibrator, 'get_variance'):
                _vars = np.array([calibrator.get_variance(aid) for _, aid, _ in _num_entries])
            else:
                _vars = np.ones(len(_vals))
                
            _precisions = 1.0 / (_vars + 1e-4)
            
            # MAP Estimate: weighted mean by precision
            _map_est = np.sum(_vals * _precisions) / np.sum(_precisions)
            
            if all(v.is_integer() for v in _vals):
                return str(int(round(_map_est)))
            return str(round(_map_est, 4))

    return canonical_map[best_key]


def numeric_median_vote(responses: List[AgentResponse], dataset_type: str = "gsm8k") -> str:
    """Voting improvement III: L1-optimal (weighted median) for numerical answers.
    Better than mode when agents produce close-but-not-identical numbers.
    Falls back to weighted_majority_vote for non-numeric or non-GSM8K.
    """
    if dataset_type != "gsm8k":
        return weighted_majority_vote(responses, dataset_type)

    numeric_answers = []
    for r in responses:
        nums = re.findall(r"-?\d+\.?\d*", r.answer.replace(",", ""))
        if nums:
            try:
                numeric_answers.append((float(nums[-1]), r.confidence, r.answer))
            except ValueError:
                pass

    if not numeric_answers:
        return weighted_majority_vote(responses, dataset_type)

    values = np.array([v for v, _, _ in numeric_answers])
    weights = np.array([c for _, c, _ in numeric_answers])
    weights /= weights.sum()

    idx = np.argsort(values)
    sorted_vals, sorted_w = values[idx], weights[idx]
    cumw = np.cumsum(sorted_w)
    median_idx = int(np.searchsorted(cumw, 0.5))
    median_val = sorted_vals[min(median_idx, len(sorted_vals) - 1)]
    best = min(numeric_answers, key=lambda x: abs(x[0] - median_val))
    return best[2]
