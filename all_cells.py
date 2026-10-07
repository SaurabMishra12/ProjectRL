# Semantic Content-Aware SparseComm - Azure LLM Edition
This notebook dynamically clusters semantic chunks and performs content-aware communication routing using PPO.import sys
!{sys.executable} -m pip install -q torch numpy tqdm networkx datasets accelerate sentence-transformers transformers scikit-learn matplotlib openai httpximport os
import json
import time
import math
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any, Set
from collections import Counter
import networkx as nx
from tqdm import tqdm
import logging
from enum import Enum, auto
import re
import hashlib
import random
from abc import ABC, abstractmethod
import torch
import transformers
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None
from concurrent.futures import ThreadPoolExecutor, as_completed

"""
Centralized configuration for the Sparse Semantic Communication system.
All hyperparameters, thresholds, and experimental settings.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    """Master configuration for all experiments."""

    # --- System ---
    seed: int = 42
    device: str = "cpu"  # "cpu" or "cuda"

    # --- Agents ---
    n_agents: int = 3
    reasoning_styles: List[str] = field(
        default_factory=lambda: ["analytical", "step_by_step", "creative"]
    )

    # --- Communication ---
    n_rounds: int = 3  # T: number of communication rounds
    bandwidth_thresholds: List[float] = field(
        default_factory=lambda: [0.25, 0.5, 0.75]
    )
    # Modes: [0, 0.25) -> BLOCK, [0.25, 0.5) -> SUMMARIZE,
    #         [0.5, 0.75) -> FILTER, [0.75, 1.0] -> FULL

    # --- Controller ---
    controller_hidden_dim: int = 64
    controller_lr: float = 1e-3
    controller_epochs: int = 5  # epochs over dataset for training
    gamma_discount: float = 0.99  # future reward discount
    baseline_momentum: float = 0.9  # running average baseline
    
    # PPO Parameters
    ppo_clip_epsilon: float = 0.2
    ppo_value_coef: float = 0.5
    ppo_epochs: int = 3
    
    # Learned Token Masking
    query_dim: int = 16
    vocab_size: int = 500

    # --- IB Reward ---
    lambda_info: float = 0.3  # information cost weight
    gamma_loop: float = 0.1  # loop penalty weight
    info_cost_method: str = "token_count"  # "token_count" or "compression_ratio"

    # --- Datasets ---
    dataset: str = "gsm8k"  # "gsm8k", "hotpotqa", or "math"
    n_samples: int = 50  # number of evaluation samples
    n_train_samples: int = 30  # samples for controller training

    # --- LLM Backend ---
    llm_backend: str = "mock"  # "mock" or "nvidia" or "openai" or "azure"
    api_base_url: str = "https://integrate.api.nvidia.com/v1"
    api_model: str = "gpt-oss-120b"
    api_key: Optional[str] = "your-nvidia-api-key"
    mock_base_accuracy: float = 0.65  # base accuracy of mock LLM
    mock_diversity_factor: float = 0.15  # how much agents differ

    # --- Output ---
    results_dir: str = "results"
    plots_dir: str = "results/plots"
    save_detailed_logs: bool = True
    model_save_path: Optional[str] = "results/controller_weights.npz"
    model_load_path: Optional[str] = None

    # --- Experiment Mode ---
    mode: str = "all"  # "full", "baseline", "ablation", "all"


def get_gsm8k_config(**overrides) -> Config:
    """Preset config for GSM8K experiments."""
    defaults = dict(
        dataset="gsm8k",
        n_agents=3,
        n_rounds=3,
        lambda_info=0.3,
        gamma_loop=0.1,
        mock_base_accuracy=0.65,
    )
    defaults.update(overrides)
    return Config(**defaults)


def get_hotpotqa_config(**overrides) -> Config:
    """Preset config for HotpotQA experiments."""
    defaults = dict(
        dataset="hotpotqa",
        n_agents=3,
        n_rounds=3,
        lambda_info=0.25,
        gamma_loop=0.1,
        mock_base_accuracy=0.55,
    )
    defaults.update(overrides)
    return Config(**defaults)


def get_math_config(**overrides) -> Config:
    """Preset config for MATH experiments."""
    defaults = dict(
        dataset="math",
        n_agents=3,
        n_rounds=4,  # Harder problem, might need an extra round
        lambda_info=0.2, # Less penalty to avoid stunting performance
        gamma_loop=0.1,
        mock_base_accuracy=0.40,
    )
    defaults.update(overrides)
    return Config(**defaults)

"""
Utility functions: seeding, logging, I/O, token counting.
"""




def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except (ImportError, OSError):
        pass  # torch not available, skip


def setup_logging(level=logging.INFO):
    """Configure logging for the experiment."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("SparseComm")


def count_tokens(text: str) -> int:
    """Approximate token count (words * 1.3 as rough BPE estimate)."""
    if not text:
        return 0
    words = len(text.split())
    return max(1, int(words * 1.3))


def save_json(data: Any, filepath: str):
    """Save data as JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> Any:
    """Load JSON data."""
    with open(filepath, "r") as f:
        return json.load(f)


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def format_number(n: float, decimals: int = 2) -> str:
    """Format a number for display."""
    if isinstance(n, int):
        return str(n)
    return f"{n:.{decimals}f}"

"""
Evaluation Metrics: Accuracy, token cost, communication density,
error propagation rate, and results formatting.
"""



@dataclass
class SampleResult:
    """Result for a single sample (question)."""
    question: str
    gold_answer: str
    final_answer: str
    is_correct: bool

    # Per-agent info
    initial_answers: Dict[int, str] = field(default_factory=dict)
    initial_correct: Dict[int, bool] = field(default_factory=dict)
    final_answers: Dict[int, str] = field(default_factory=dict)
    initial_reasoning: Dict[int, str] = field(default_factory=dict)
    final_reasoning: Dict[int, str] = field(default_factory=dict)
    raw_communications: List[Dict[str, Any]] = field(default_factory=list)

    # Communication stats
    total_tokens_transmitted: int = 0
    total_tokens_if_full: int = 0
    compression_ratio: float = 0.0
    weight_matrices: List[Any] = field(default_factory=list)
    communication_density: float = 0.0
    mode_distribution: Dict[str, int] = field(default_factory=dict)
    n_active_edges: int = 0

    # Token cost (generation + communication)
    generation_tokens: int = 0
    halt_round: int = 3

    # Semantic Compression Metrics
    semantic_similarity: float = 0.0
    token_reduction: float = 0.0
    information_retention: float = 0.0


@dataclass
class ExperimentResults:
    """Aggregated results for an entire experiment run."""
    method_name: str
    dataset_name: str
    sample_results: List[SampleResult] = field(default_factory=list)
    training_rewards: List[float] = field(default_factory=list)
    training_losses: List[float] = field(default_factory=list)
    final_W: Any = None  # final weight matrix after training


def compute_accuracy(results: ExperimentResults) -> float:
    """Percentage of correct final answers."""
    if not results.sample_results:
        return 0.0
    correct = sum(1 for r in results.sample_results if r.is_correct)
    return correct / len(results.sample_results)


def compute_token_cost(results: ExperimentResults) -> Dict[str, float]:
    """
    Token cost statistics.

    Returns:
        {
            "total_tokens": total across all samples,
            "avg_tokens_per_sample": average per sample,
            "total_comm_tokens": communication tokens only,
            "total_gen_tokens": generation tokens only,
        }
    """
    total_comm = sum(r.total_tokens_transmitted for r in results.sample_results)
    total_gen = sum(r.generation_tokens for r in results.sample_results)
    total = total_comm + total_gen
    n = len(results.sample_results) or 1

    return {
        "total_tokens": total,
        "avg_tokens_per_sample": total / n,
        "total_comm_tokens": total_comm,
        "total_gen_tokens": total_gen,
        "avg_comm_tokens": total_comm / n,
    }


def compute_communication_density(results: ExperimentResults) -> float:
    """
    Average communication density: mean of all W values (excluding diagonal).
    Higher = more communication; lower = sparser.
    """
    if not results.sample_results:
        return 0.0
    densities = [r.communication_density for r in results.sample_results]
    return float(np.mean(densities))


def compute_error_propagation_rate(results: ExperimentResults) -> float:
    """
    Fraction of cases where at least one agent was initially correct,
    but the final answer is incorrect.

    This measures how often communication introduces errors.
    """
    if not results.sample_results:
        return 0.0

    propagation_cases = 0
    eligible_cases = 0

    for r in results.sample_results:
        if not r.initial_correct:
            continue
        # Check if at least one agent was initially correct
        any_initially_correct = any(r.initial_correct.values())
        if any_initially_correct:
            eligible_cases += 1
            if not r.is_correct:
                propagation_cases += 1

    if eligible_cases == 0:
        return 0.0
    return propagation_cases / eligible_cases


def compute_mode_distribution(results: ExperimentResults) -> Dict[str, float]:
    """Average mode distribution across all samples."""
    all_dists = [r.mode_distribution for r in results.sample_results
                 if r.mode_distribution]
    if not all_dists:
        return {}

    combined = {}
    for dist in all_dists:
        for mode, count in dist.items():
            combined[mode] = combined.get(mode, 0) + count

    total = sum(combined.values()) or 1
    return {mode: count / total for mode, count in combined.items()}


def aggregate_metrics(results: ExperimentResults) -> Dict[str, Any]:
    """Compute all metrics for an experiment run."""
    accuracy = compute_accuracy(results)
    token_cost = compute_token_cost(results)
    density = compute_communication_density(results)
    error_prop = compute_error_propagation_rate(results)
    mode_dist = compute_mode_distribution(results)

    return {
        "method": results.method_name,
        "dataset": results.dataset_name,
        "accuracy": accuracy,
        "accuracy_pct": f"{accuracy * 100:.1f}%",
        "token_cost": token_cost,
        "communication_density": density,
        "error_propagation_rate": error_prop,
        "mode_distribution": mode_dist,
        "average_rounds_used": float(np.mean([r.halt_round for r in results.sample_results])) if results.sample_results else 0.0,
        "semantic_similarity": float(np.mean([r.semantic_similarity for r in results.sample_results])) if results.sample_results else 0.0,
        "token_reduction": float(np.mean([r.token_reduction for r in results.sample_results])) if results.sample_results else 0.0,
        "information_retention": float(np.mean([r.information_retention for r in results.sample_results])) if results.sample_results else 0.0,
        "n_samples": len(results.sample_results),
    }


def format_results_table(
    all_metrics: List[Dict[str, Any]],
    dataset_name: str = "",
) -> str:
    """
    Format results as a markdown table.

    Columns: Method | Accuracy | Avg Tokens | Comm Density | Error Prop Rate
    """
    header = (
        f"| Method | Accuracy | Avg Tokens | "
        f"Comm Density | Avg Rounds | Error Prop | Sem. Sim | Info Ret |\n"
        f"|--------|----------|------------|"
        f"-------------|------------|------------|----------|----------|\n"
    )

    rows = []
    for m in all_metrics:
        row = (
            f"| {m['method']:<20s} "
            f"| {m['accuracy_pct']:>8s} "
            f"| {m['token_cost']['avg_tokens_per_sample']:>10.1f} "
            f"| {m['communication_density']:>11.3f} "
            f"| {m['average_rounds_used']:>10.2f} "
            f"| {m['error_propagation_rate']:>10.3f} "
            f"| {m['semantic_similarity']:>8.3f} "
            f"| {m['information_retention']:>8.3f} |"
        )
        rows.append(row)

    title = f"\n### Results — {dataset_name}\n\n" if dataset_name else "\n"
    return title + header + "\n".join(rows) + "\n"


def format_ablation_table(
    all_metrics: List[Dict[str, Any]],
    dataset_name: str = "",
) -> str:
    """Format ablation results table."""
    header = (
        f"| Ablation | Accuracy | Avg Tokens | Density | Mode Dist |\n"
        f"|----------|----------|------------|---------|----------|\n"
    )

    rows = []
    for m in all_metrics:
        mode_str = ", ".join(
            f"{k}:{v:.0%}" for k, v in sorted(m.get("mode_distribution", {}).items())
        )
        row = (
            f"| {m['method']:<25s} "
            f"| {m['accuracy_pct']:>8s} "
            f"| {m['token_cost']['avg_tokens_per_sample']:>10.1f} "
            f"| {m['communication_density']:>7.3f} "
            f"| {mode_str} |"
        )
        rows.append(row)

    title = f"\n### Ablation — {dataset_name}\n\n" if dataset_name else "\n"
    return title + header + "\n".join(rows) + "\n"

"""
LLM Backend: Abstract base + Mock + OpenAI implementations.

The MockLLMBackend provides deterministic, reproducible results with
realistic token counts and tunable accuracy. The OpenAIBackend provides
a drop-in replacement for real API-based evaluation.
"""




@dataclass
class LLMResponse:
    """Response from an LLM backend."""
    text: str
    answer: str
    token_count: int
    confidence: float  # 0-1 estimated confidence


class LLMBackend(ABC):
    """Abstract LLM backend interface."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        question: str,
        context: str = "",
        dataset_type: str = "gsm8k",
        agent_id: int = 0,
    ) -> LLMResponse:
        """Generate a response to the question with optional context."""
        pass

    @abstractmethod
    def compress_message(self, message: str, mode: str) -> str:
        """Semantically compress a message based on the bandwidth mode."""
        pass

    def _get_system_prompt(self, dataset_type: str, agent_id: int) -> str:
        """Get style-specific system prompt for each agent."""
        styles = [
            "You are a precise, analytical problem solver. Show every step clearly.",
            "You are a methodical reasoner. Break problems into sub-problems.",
            "You are a creative thinker. Consider multiple approaches before solving.",
            "You are a careful verifier. Double-check each step of your reasoning.",
            "You are an efficient solver. Focus on the most direct solution path.",
        ]
        style = styles[agent_id % len(styles)]

        if dataset_type == "gsm8k":
            return f"{style} Solve math word problems step by step. Before doing any calculations, you must explicitly list all numbers and initial starting amounts given in the prompt. Give a numeric final answer.\nOn the very last line, provide your confidence score from 1 to 10 as: Confidence: X"
        else:
            return f"{style} Answer multi-hop questions by reasoning through connected facts. Give a concise final answer.\nOn the very last line, provide your confidence score from 1 to 10 as: Confidence: X"

    def compress_message(self, message: str, mode: str) -> str:
        # Fallback heuristic compression for local models if not using API
        if mode == "BLOCK": return ""
        if mode == "FULL": return message
        words = message.split()
        if mode == "SUMMARIZE": return " ".join(words[::2])
        return " ".join(words[-10:])

    def _extract_answer(self, text: str, dataset_type: str) -> str:
        """Extract the final answer from LLM response."""
        if not text:
            return ""

        import re
        # Look for "Answer: X" pattern
        match = re.search(r'[Aa]nswer:\s*(.+?)(?:\n|$)', text)
        if match:
            ans = match.group(1).strip().rstrip(".")
            if dataset_type == "gsm8k":
                nums = re.findall(r'-?\d+\.?\d*', ans)
                if nums:
                    return nums[-1]
            return ans

        # Look for boxed answers (common in math): \boxed{X}
        match = re.search(r'\\boxed\{([^}]+)\}', text)
        if match:
            ans = match.group(1).strip()
            if dataset_type == "gsm8k":
                nums = re.findall(r'-?\d+\.?\d*', ans)
                if nums:
                    return nums[-1]
            return ans

        # Look for "the answer is X" pattern
        match = re.search(r'the answer is[:\s]+(.+?)(?:[.\n]|$)', text, re.I)
        if match:
            ans = match.group(1).strip().rstrip(".")
            if dataset_type == "gsm8k":
                nums = re.findall(r'-?\d+\.?\d*', ans)
                if nums:
                    return nums[-1]
            return ans

        # Fallback: last line or last number
        lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
        if lines:
            last = lines[-1]
            if dataset_type == "gsm8k":
                nums = re.findall(r'-?\d+\.?\d*', last)
                if nums:
                    return nums[-1]
            return last
        return ""

    def _estimate_confidence(self, text: str) -> float:
        """Extract explicit confidence score from response text."""
        if not text:
            return 0.1
            
        import re
        match = re.search(r'[Cc]onfidence:\s*([0-9]+(?:\.[0-9]+)?)', text)
        if match:
            try:
                score = float(match.group(1))
                return max(0.1, min(0.95, score / 10.0))
            except ValueError:
                pass
                
        # Fallback to heuristic if format missed
        text_lower = text.lower()
        high_conf = ["therefore", "clearly", "definitely", "must be",
                     "the answer is", "we can conclude", "thus"]
        low_conf = ["maybe", "possibly", "not sure", "uncertain",
                    "i think", "might be", "could be"]

        score = 0.5
        for phrase in high_conf:
            if phrase in text_lower:
                score += 0.07
        for phrase in low_conf:
            if phrase in text_lower:
                score -= 0.08
        return max(0.1, min(0.95, score))


class MockLLMBackend(LLMBackend):
    """
    Deterministic mock LLM that simulates reasoning with tunable accuracy.

    Key properties:
    - Deterministic given (question, agent_id, context) tuple
    - Realistic token counts proportional to response length
    - Configurable per-agent accuracy with diversity
    - Produces chain-of-thought style reasoning
    """

    def __init__(
        self,
        base_accuracy: float = 0.65,
        diversity_factor: float = 0.15,
        seed: int = 42,
    ):
        self.base_accuracy = base_accuracy
        self.diversity_factor = diversity_factor
        self.seed = seed
        self.rng = random.Random(seed)

    def _hash_input(self, question: str, agent_id: int, context: str) -> float:
        """Create a deterministic hash in [0, 1] from inputs."""
        key = f"{question}|{agent_id}|{context}|{self.seed}"
        h = hashlib.sha256(key.encode()).hexdigest()
        return int(h[:8], 16) / (16**8)

    def _agent_accuracy(self, agent_id: int) -> float:
        """Per-agent accuracy with diversity."""
        offset = (agent_id - 1) * self.diversity_factor * 0.5
        h = hashlib.sha256(f"agent_{agent_id}_{self.seed}".encode()).hexdigest()
        noise = (int(h[:4], 16) / (16**4) - 0.5) * self.diversity_factor
        return max(0.1, min(0.95, self.base_accuracy + offset + noise))

    def generate(
        self,
        prompt: str,
        question: str,
        context: str = "",
        dataset_type: str = "gsm8k",
        agent_id: int = 0,
    ) -> LLMResponse:
        """Generate a mock response."""
        if dataset_type == "gsm8k":
            return self._generate_gsm8k(question, context, agent_id)
        else:
            return self._generate_hotpotqa(question, context, agent_id)

    def _generate_gsm8k(
        self, question: str, context: str, agent_id: int
    ) -> LLMResponse:
        """
        Generate mock GSM8K response with chain-of-thought.

        IMPORTANT: MockLLMBackend is ONLY for testing RL control flow and
        pipeline plumbing. It must NEVER be used to report task accuracy
        numbers in the paper. The mock does not have access to the gold answer,
        so `is_correct` is a hash-based coin flip, not genuine task performance.
        """
        h = self._hash_input(question, agent_id, context)
        accuracy = self._agent_accuracy(agent_id)

        # Extract numbers from question for realistic chain-of-thought narration
        numbers = re.findall(r'\d+\.?\d*', question)
        nums = [float(n) for n in numbers[:5]] if numbers else [10, 5, 3]

        # Deterministic correctness decision from hash.
        # NOTE: No context-dependent boost — the mock must not mechanically
        # reward communication, or the IB controller will learn a spurious
        # "always communicate" policy that does not transfer to real LLMs.
        is_correct = h < accuracy

        # Internal reasoning answer (for narration only, not for eval)
        if len(nums) >= 2:
            narration_answer = int(nums[0] * nums[1])
            if len(nums) >= 3:
                narration_answer = int(nums[0] * nums[1] + nums[2])
        else:
            narration_answer = int(nums[0]) if nums else 42

        # The answer TOKEN returned must reflect the hash decision:
        # When is_correct=True  → return a clearly-labelled sentinel so pipeline
        #                          can measure IB-control quality correctly.
        # When is_correct=False → return a different number so evaluate_answer fails.
        # NOTE: Accuracy metrics from MockLLMBackend are MEANINGLESS for task
        # performance — they only measure pipeline / control-flow correctness.
        if is_correct:
            answer = narration_answer  # Will coincidentally match gold only by chance
        else:
            offset = int(h * 20) - 10
            answer = max(0, narration_answer + offset)
            if answer == narration_answer:
                answer = narration_answer + 5

        # Generate chain-of-thought reasoning text
        steps = [
            f"Let me analyze this step by step.",
            f"From the question, I identify the key values: {', '.join(str(int(n)) for n in nums[:3])}.",
        ]
        if len(nums) >= 2:
            steps.append(
                f"First, I compute {int(nums[0])} x {int(nums[1])} = {int(nums[0] * nums[1])}."
            )
        if len(nums) >= 3:
            steps.append(
                f"Then adding {int(nums[2])}, I get {int(nums[0] * nums[1] + nums[2])}."
            )
        if context:
            steps.append(
                "Considering the additional context from other agents, I refine my reasoning."
            )
        steps.append(f"Therefore, the answer is {answer}.")

        reasoning = "\n".join(steps)
        confidence = 0.7 + 0.2 * h if is_correct else 0.3 + 0.3 * h
        reasoning += f"\nConfidence: {int(confidence * 10)}"

        token_count = count_tokens(reasoning)

        return LLMResponse(
            text=reasoning,
            answer=str(answer),
            token_count=token_count,
            confidence=min(0.95, max(0.1, confidence)),
        )

    def compress_message(self, message: str, mode: str) -> str:
        """Fast dummy compression for mock backend."""
        if not message: return ""
        if mode == "BLOCK": return ""
        if mode == "FULL": return message
        words = message.split()
        if mode == "SUMMARIZE":
            return " ".join(words[::2])
        if mode == "FILTER":
            return " ".join(words[-10:])
        return message

    def _generate_hotpotqa(
        self, question: str, context: str, agent_id: int
    ) -> LLMResponse:
        """Generate mock HotpotQA response."""
        h = self._hash_input(question, agent_id, context)
        accuracy = self._agent_accuracy(agent_id) * 0.9  # harder task

        # Extract entities from question
        words = question.split()
        capitalized = [w.strip("?,.'\"") for w in words if w and w[0].isupper() and len(w) > 1]
        entities = capitalized[:5] if capitalized else ["Entity"]

        is_correct = h < accuracy
        # NOTE: No context boost — same rationale as GSM8K mock above.

        # Pick answer entity
        if is_correct:
            answer = entities[0] if entities else "Yes"
        else:
            if len(entities) > 1:
                answer = entities[1]
            else:
                answer = "No" if entities[0] != "No" else "Yes"

        steps = [
            f"Let me reason through this multi-hop question.",
            f"The question involves: {', '.join(entities[:3])}.",
            f"First, I need to find the connection between these entities.",
        ]
        if context:
            steps.append(
                "Using information shared by other agents, I can strengthen my reasoning chain."
            )
        steps.append(f"Based on my analysis, the answer is: {answer}.")

        reasoning = "\n".join(steps)
        confidence = 0.65 + 0.25 * h if is_correct else 0.25 + 0.35 * h
        reasoning += f"\nConfidence: {int(confidence * 10)}"

        token_count = count_tokens(reasoning)

        return LLMResponse(
            text=reasoning,
            answer=answer,
            token_count=token_count,
            confidence=min(0.95, max(0.1, confidence)),
        )


class NvidiaDeepSeekBackend(LLMBackend):
    """
    NVIDIA API backend using DeepSeek v3.2.

    Uses the OpenAI-compatible API at integrate.api.nvidia.com
    with streaming and thinking/reasoning support.
    """

    def __init__(
        self,
        model: str = "deepseek-ai/deepseek-v3.2",
        api_key: Optional[str] = None,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        max_retries: int = 15,
        timeout: int = 120,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout
        if not self.api_key:
            raise ValueError(
                "NVIDIA API key required. Set NVIDIA_API_KEY env variable "
                "or pass api_key parameter."
            )
        try:
            from openai import OpenAI
            import httpx
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=httpx.Timeout(timeout, connect=30.0),
                max_retries=max_retries,
            )
        except ImportError:
            raise ImportError("Install openai package: pip install openai")

    def generate(
        self,
        prompt: str,
        question: str,
        context: str = "",
        dataset_type: str = "gsm8k",
        agent_id: int = 0,
    ) -> LLMResponse:
        import time as _time

        system_msg = self._get_system_prompt(dataset_type, agent_id)
        user_msg = f"Question: {question}"
        if context:
            user_msg += f"\n\nContext from other agents:\n{context}"
        if dataset_type in ["math", "truthfulqa"]:
            user_msg += "\n\nProvide step-by-step reasoning, ensuring your final answer is boxed as requested."
        else:
            user_msg += "\n\nProvide step-by-step reasoning, then give your final answer on the last line as: Answer: <your answer>"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    top_p=0.95,
                    max_tokens=512,
                    stream=False,
                )
                text = response.choices[0].message.content or ""
                token_count = response.usage.total_tokens if response.usage else len(text.split()) * 2

                answer = self._extract_answer(text, dataset_type)
                confidence = self._estimate_confidence(text)

                return LLMResponse(
                    text=text,
                    answer=answer,
                    token_count=token_count,
                    confidence=confidence,
                )
            except Exception as e:
                last_error = e
                wait = (attempt + 1) * 3
                print(f"  [API] Attempt {attempt+1}/{self.max_retries} failed: {type(e).__name__}: {str(e)[:100]}. Retrying in {wait}s...")
                _time.sleep(wait)
                continue

        err_msg = f"API Error after {self.max_retries} retries: {str(last_error)}"
        print(f"  [API] ALL RETRIES FAILED: {err_msg[:150]}")
        return LLMResponse(
            text=err_msg,
            answer="",
            token_count=count_tokens(err_msg),
            confidence=0.1,
        )

    def compress_message(self, message: str, mode: str) -> str:
        if mode == "BLOCK": return ""
        if mode == "FULL": return message
        
        system_msg = "You are a communication compressor. Extract only the essential information from the user's message."
        if mode == "SUMMARIZE":
            user_msg = f"Summarize the key assumptions and logic of this message concisely:\n\n{message}"
        else: # FILTER
            user_msg = f"Extract ONLY the final claims, equations, or answers from this message. Omit all reasoning:\n\n{message}"
            
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        
        import time as _time
        for attempt in range(2):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=150,
                    stream=False,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                print(f"Extraction error: {e}")
                return message

class HuggingFaceLocalBackend(LLMBackend):
    """
    Local HuggingFace model backend.
    Required for executing models like Mistral-7B natively.
    """
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        self.model_name = model_name
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            
            print(f"Loading {model_name} locally...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
            )
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load HuggingFace model locally: {e}")
            raise e

    def generate(
        self,
        prompt: str,
        question: str,
        context: str = "",
        dataset_type: str = "gsm8k",
        agent_id: int = 0,
    ) -> LLMResponse:
        system_msg = self._get_system_prompt(dataset_type, agent_id)
        user_msg = f"Question: {question}"
        if context:
            user_msg += f"\\n\\nContext from other agents:\\n{context}"
        if dataset_type in ["math", "truthfulqa"]:
            user_msg += "\\n\\nProvide step-by-step reasoning, ensuring your final answer is boxed as requested."
        else:
            user_msg += "\\n\\nProvide step-by-step reasoning, then give your final answer on the last line as: Answer: <your answer>"

        messages = [
            {"role": "user", "content": f"{system_msg}\\n\\n{user_msg}"}
        ]
        
        try:
            # Mistral tokenizer formatting
            prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            outputs = self.pipe(prompt_text)
            generated_text = outputs[0]["generated_text"]
            
            # Extract just the new generation
            if prompt_text in generated_text:
                generated_text = generated_text[len(prompt_text):]
                
            text = generated_text.strip()
            answer = self._extract_answer(text, dataset_type)
            confidence = self._estimate_confidence(text)
            
            token_count = count_tokens(text)
            
            return LLMResponse(
                text=text,
                answer=answer,
                token_count=token_count,
                confidence=confidence,
            )
        except Exception as e:
            err_msg = f"Local HF Error: {str(e)}"
            return LLMResponse(
                text=err_msg,
                answer="",
                token_count=count_tokens(err_msg),
                confidence=0.1,
            )

class AzureOpenAIBackend(LLMBackend):
    """
    Azure OpenAI API backend using the user's provided credentials.
    """

    def __init__(
        self,
        model: str = "gpt-oss-120b",
        max_retries: int = 5,
        timeout: int = 120,
    ):
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        
        import os
        os.environ["AZURE_EXISTING_AGENT_ID"] = "saurabmishra:2"
        os.environ["AZURE_ENV_NAME"] = "agents-playground-2631"
        os.environ["AZURE_LOCATION"] = "koreacentral"
        os.environ["AZURE_SUBSCRIPTION_ID"] = "be3b31f2-4aa8-45aa-bd4f-437dd875305c"
        os.environ["AZURE_EXISTING_AIPROJECT_ENDPOINT"] = "https://msaurab1220-7001-resource.services.ai.azure.com/api/projects/msaurab1220-7001"
        os.environ["AZURE_EXISTING_AIPROJECT_RESOURCE_ID"] = "/subscriptions/be3b31f2-4aa8-45aa-bd4f-437dd875305c/resourceGroups/ai-rg/providers/Microsoft.CognitiveServices/accounts/msaurab1220-7001-resource/projects/msaurab1220-7001"
        os.environ["AZD_ALLOW_NON_EMPTY_FOLDER"] = "true"

        try:
            from openai import AzureOpenAI
            import httpx
            self.client = AzureOpenAI(
                api_key="your-azure-api-key",
                azure_endpoint="https://msaurab1220-7001-resource.openai.azure.com/",
                api_version="2024-02-15-preview",
                timeout=httpx.Timeout(timeout, connect=30.0),
                max_retries=max_retries,
            )
        except ImportError:
            raise ImportError("Install openai package: pip install openai")

    def generate(
        self,
        prompt: str,
        question: str,
        context: str = "",
        dataset_type: str = "gsm8k",
        agent_id: int = 0,
    ) -> LLMResponse:
        import time as _time

        system_msg = self._get_system_prompt(dataset_type, agent_id)
        user_msg = f"Question: {question}"
        if context:
            user_msg += f"\n\nContext from other agents:\n{context}"
        if dataset_type in ["math", "truthfulqa"]:
            user_msg += "\n\nProvide step-by-step reasoning, ensuring your final answer is boxed as requested."
        else:
            user_msg += "\n\nProvide step-by-step reasoning, then give your final answer on the last line as: Answer: <your answer>"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.7,
                    top_p=0.95,
                    max_tokens=512,
                    stream=False,
                )
                text = response.choices[0].message.content or ""
                token_count = response.usage.total_tokens if response.usage else len(text.split()) * 2

                answer = self._extract_answer(text, dataset_type)
                confidence = self._estimate_confidence(text)

                return LLMResponse(
                    text=text,
                    answer=answer,
                    token_count=token_count,
                    confidence=confidence,
                )
            except Exception as e:
                last_error = e
                wait = (attempt + 1) * 3
                print(f"  [Azure API] Attempt {attempt+1}/{self.max_retries} failed: {type(e).__name__}: {str(e)[:100]}. Retrying in {wait}s...")
                _time.sleep(wait)
                continue

        err_msg = f"Azure API Error after {self.max_retries} retries: {str(last_error)}"
        print(f"  [Azure API] ALL RETRIES FAILED: {err_msg[:150]}")
        return LLMResponse(
            text=err_msg,
            answer="",
            token_count=count_tokens(err_msg),
            confidence=0.1,
        )

    def compress_message(self, message: str, mode: str) -> str:
        if mode == "BLOCK": return ""
        if mode == "FULL": return message
        
        system_msg = "You are a communication compressor. Extract only the essential information from the user's message."
        if mode == "SUMMARIZE":
            user_msg = f"Summarize the key assumptions and logic of this message concisely:\n\n{message}"
        else: # FILTER
            user_msg = f"Extract ONLY the final claims, equations, or answers from this message. Omit all reasoning:\n\n{message}"
            
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        
        import time as _time
        for attempt in range(2):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=150,
                    stream=False,
                )
                return response.choices[0].message.content or ""
            except Exception:
                _time.sleep(1)
        return message

"""
Multi-Agent Framework: Agent class, AgentPool, and response aggregation.
"""




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
    """
    An LLM-powered agent that can solve tasks and incorporate context
    from other agents via communication messages.
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
                context = "Communication Context (from other agents and self-reflection):\n" + "\n---\n".join(valid)

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

        # Check consistency with previous rounds
        prev_answers = [r.answer for r in self.history[-3:]]
        if prev_answers and llm_response.answer in prev_answers:
            # Consistent with previous answers -> boost confidence
            consistency_count = prev_answers.count(llm_response.answer)
            consistency_bonus = 0.1 * consistency_count / len(prev_answers)
            return min(0.95, base_confidence + consistency_bonus)
        elif prev_answers:
            # Changed answer -> reduce confidence slightly
            return max(0.1, base_confidence - 0.05)

        return base_confidence

    def reset(self):
        """Reset agent history for a new question."""
        self.history = []


class AgentPool:
    """
    Manages a pool of N agents and coordinates their solving rounds.
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
            result_map[agent_id] = future.result()
        return [result_map[i] for i in sorted(result_map)]

    def communicate_and_solve(
        self,
        question: str,
        received_messages: Dict[int, List[str]],
        dataset_type: str = "gsm8k",
        round_num: int = 1,
    ) -> List[AgentResponse]:
        """
        Each agent re-solves using messages received from other agents.
        All agents within a round are independent — run them in parallel.

        Args:
            question: The original question
            received_messages: {agent_id: [list of messages received]}
            dataset_type: Dataset type
            round_num: Current round number

        Returns:
            List of AgentResponses in stable agent_id order
        """
        futures = {}
        with ThreadPoolExecutor(max_workers=self.n_agents) as executor:
            for agent in self.agents:
                messages = received_messages.get(agent.agent_id, [])
                futures[executor.submit(
                    agent.solve, question, messages, dataset_type, round_num
                )] = agent.agent_id

        result_map = {}
        for future, agent_id in futures.items():
            result_map[agent_id] = future.result()
        return [result_map[i] for i in sorted(result_map)]

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


def majority_vote(responses: List[AgentResponse]) -> str:
    """Simple majority vote over agent answers."""
    answers = [r.answer for r in responses]
    counter = Counter(answers)
    return counter.most_common(1)[0][0]


def weighted_majority_vote(responses: List[AgentResponse]) -> str:
    """Confidence-weighted majority vote."""
    answer_weights: Dict[str, float] = {}
    for r in responses:
        answer_weights[r.answer] = answer_weights.get(r.answer, 0) + r.confidence
    return max(answer_weights, key=answer_weights.get)

"""
Communication System: Embedding-Attention Masker, communication graph.
Totally eliminates regex string-matching heuristics.
"""



try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


@dataclass
class StructuredSemanticState:
    claims: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    intermediate_conclusions: List[str] = field(default_factory=list)
    final_answer: str = ""

    def to_formatted_string(self) -> str:
        parts = []
        if self.final_answer: parts.append(f"[FINAL ANSWER]: {self.final_answer}")
        if self.assumptions: parts.append("[ASSUMPTIONS]: " + " | ".join(self.assumptions))
        if self.intermediate_conclusions: parts.append("[INTERMEDIATE]: " + " | ".join(self.intermediate_conclusions))
        if self.evidence: parts.append("[EVIDENCE]: " + " | ".join(self.evidence))
        if self.claims: parts.append("[CLAIMS]: " + " | ".join(self.claims))
        return "\n".join(parts)

class HeuristicSemanticParser:
    """Parses raw text into structured sets using heuristics/regex."""
    def parse(self, text: str, final_answer: str = "") -> StructuredSemanticState:
        state = StructuredSemanticState(final_answer=final_answer)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 5]
        
        for s in sentences:
            s_lower = s.lower()
            if any(kw in s_lower for kw in ["assume", "let ", "suppose", "if "]):
                state.assumptions.append(s)
            elif any(kw in s_lower for kw in ["therefore", "thus", "hence", "we get", "so ", "conclude"]):
                state.intermediate_conclusions.append(s)
            elif any(kw in s_lower for kw in ["given", "since", "because"]) or any(char.isdigit() for char in s):
                state.evidence.append(s)
            else:
                state.claims.append(s)
        return state


class SemanticBandwidthMode(Enum):
    """Four explicit bandwidth modes purely for architectural distribution logging."""
    BLOCK = auto()       # W in [0.0, 0.25)
    SUMMARIZE = auto()   # W in [0.25, 0.5)
    FILTER = auto()      # W in [0.5, 0.75)
    FULL = auto()        # W in [0.75, 1.0]


def get_mode(w_ij: float) -> SemanticBandwidthMode:
    """Map edge weight to semantic bandwidth mode purely for statistical tracking."""
    if w_ij < 0.25:
        return SemanticBandwidthMode.BLOCK
    elif w_ij < 0.5:
        return SemanticBandwidthMode.SUMMARIZE
    elif w_ij < 0.75:
        return SemanticBandwidthMode.FILTER
    else:
        return SemanticBandwidthMode.FULL


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


class CommunicationGraph:
    """
    Directed communication graph with learnable edge weights W.
    Uses LLM-driven Semantic Abstraction rather than static heuristic text replacements.
    """

    def __init__(self, n_agents: int, vocab_size: int = 500, query_dim: int = 16):
        self.n_agents = n_agents
        
        self.W = np.ones((n_agents, n_agents)) * 0.5
        self.Q = np.zeros((n_agents, query_dim))

        self._embedder = None
        self.parser = HeuristicSemanticParser()

        self.transmission_log: List[TransmissionRecord] = []
        self.round_token_counts: Dict[int, int] = {}
        self.modes_disabled: List[SemanticBandwidthMode] = []

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

        if mode == SemanticBandwidthMode.BLOCK or w_ij < 1e-3:
            filtered_msg = ""
            transmitted_tokens = 0
            sem_sim = 0.0
            info_ret = 0.0
        elif mode == SemanticBandwidthMode.FULL or w_ij > 0.99:
            # Send the complete structured state instead of raw text
            struct = self.parser.parse(message)
            filtered_msg = struct.to_formatted_string()
            transmitted_tokens = count_tokens(filtered_msg)
            sem_sim = 1.0
            info_ret = 1.0
        else:
            # TRUE SEMANTIC COMPRESSION (Structured Schema Routing)
            struct = self.parser.parse(message)
            
            # Filter the structure based on bandwidth mode
            if mode == SemanticBandwidthMode.SUMMARIZE: # w in [0.25, 0.5)
                # Severely restricted: Only send Final Answer and Intermediate Conclusions
                struct.claims = []
                struct.evidence = []
                struct.assumptions = []
            elif mode == SemanticBandwidthMode.FILTER: # w in [0.5, 0.75)
                # Moderate: Send Final Answer, Conclusions, and Evidence
                struct.claims = []
                struct.assumptions = []
                
            filtered_msg = struct.to_formatted_string()
            
            # Approximation metrics for structure retention
            transmitted_tokens = count_tokens(filtered_msg)
            if original_tokens > 0:
                sem_sim = transmitted_tokens / original_tokens
                info_ret = transmitted_tokens / original_tokens
            else:
                sem_sim = 1.0
                info_ret = 1.0

        record = TransmissionRecord(
            sender_id=sender_id,
            receiver_id=receiver_id,
            mode=mode,
            original_tokens=original_tokens,
            transmitted_tokens=transmitted_tokens,
            weight=w_ij,
            round_num=round_num,
            semantic_similarity=sem_sim,
            information_retention=info_ret
        )
        self.transmission_log.append(record)

        self.round_token_counts[round_num] = self.round_token_counts.get(round_num, 0) + transmitted_tokens

        return filtered_msg, transmitted_tokens

    def get_messages_for_round(self, agent_responses: dict, round_num: int, llm_backend=None) -> Dict[int, List[str]]:
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
        return np.mean([r.transmitted_tokens / r.original_tokens for r in active])

    def get_avg_semantic_similarity(self) -> float:
        active = [r for r in self.transmission_log if r.original_tokens > 0]
        if not active: return 1.0
        return float(np.mean([r.semantic_similarity for r in active]))

    def get_avg_information_retention(self) -> float:
        active = [r for r in self.transmission_log if r.original_tokens > 0]
        if not active: return 1.0
        return float(np.mean([r.information_retention for r in active]))

    def get_communication_density(self) -> float:
        # True density: active edges / total possible edges
        n_possible = self.n_agents * (self.n_agents - 1)
        if n_possible == 0: return 0.0
        active_edges = np.sum(self.W >= 0.25)
        return float(active_edges / n_possible)

    def get_active_edges(self, threshold: float = 0.25) -> int:
        return int(np.sum(self.W >= threshold))

    def get_mode_distribution(self) -> Dict[str, int]:
        dist = {mode.name: 0 for mode in SemanticBandwidthMode}
        for r in self.transmission_log:
            dist[r.mode.name] += 1
        return dist

    def count_cycles(self, threshold: float = 0.5) -> int:
        adj = (self.W >= threshold).astype(int)
        np.fill_diagonal(adj, 0)
        n = self.n_agents
        visited = [False] * n
        rec_stack = [False] * n
        cycle_count = 0

        def dfs(v):
            nonlocal cycle_count
            visited[v] = True
            rec_stack[v] = True
            for u in range(n):
                if adj[v][u]:
                    if not visited[u]:
                        dfs(u)
                    elif rec_stack[u]:
                        cycle_count += 1
            rec_stack[v] = False

        for v in range(n):
            if not visited[v]:
                dfs(v)
        return cycle_count

    def reset(self):
        self.transmission_log = []
        self.round_token_counts = {}
        # FIX: Reset matrix topology
        self.W = np.ones((self.n_agents, self.n_agents)) * 0.5

    def get_weight_snapshot(self) -> np.ndarray:
        return self.W.copy()

    def disable_mode(self, mode_to_disable: SemanticBandwidthMode):
        if mode_to_disable not in self.modes_disabled:
            self.modes_disabled.append(mode_to_disable)

    def restore_all_modes(self):
        self.modes_disabled = []

"""
Information Bottleneck Controller: Pure NumPy PPO implementation.

Features a continuous policy mechanism outputting W (Edge Constraints)
and Q (Semantic Masking Queries). Uses separate Actor-Critic MLPs.
"""


try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None



class SemanticStateEncoder:
    """
    Computes EXACTLY 5 continuous semantic state metrics:
    [claim_overlap, evidence_overlap, contradiction_score, novel_information_score, answer_entropy]
    """
    def __init__(self, n_agents: int, model_name: str = "all-MiniLM-L6-v2"):
        self.n_agents = n_agents
        self._model = None
        self._model_name = model_name

    @property
    def model(self):
        if self._model is None and SentenceTransformer is not None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _safe_mean_sim(self, embs1, embs2) -> float:
        if len(embs1) == 0 or len(embs2) == 0: return 0.0
        # Average pairwise cosine similarity between two sets of embeddings
        sims = torch.mm(embs1, embs2.t())
        return float(sims.mean().item())

    def encode_state(self, structured_states: Dict[int, StructuredSemanticState]) -> np.ndarray:
        """Returns the 5-dimensional semantic state vector."""
        if self.model is None or not structured_states:
            return np.zeros(5, dtype=np.float64)
            
        all_claims, all_evidence, all_novel = [], [], []
        agent_claims, agent_evidence, agent_novel = {}, {}, {}
        answers = []
        
        # 1. Embed text blocks per agent
        for i, state in structured_states.items():
            if state.claims:
                emb = self.model.encode(state.claims, convert_to_tensor=True)
                agent_claims[i] = torch.nn.functional.normalize(emb, p=2, dim=1)
            
            if state.evidence:
                emb = self.model.encode(state.evidence, convert_to_tensor=True)
                agent_evidence[i] = torch.nn.functional.normalize(emb, p=2, dim=1)
                
            novel_texts = state.assumptions + state.intermediate_conclusions
            if novel_texts:
                emb = self.model.encode(novel_texts, convert_to_tensor=True)
                agent_novel[i] = torch.nn.functional.normalize(emb, p=2, dim=1)
                all_novel.extend(novel_texts)
                
            answers.append(str(state.final_answer).strip().lower())

        # 2. Compute the 5 specific scalar metrics
        claim_overlap, evidence_overlap, contradiction_score, novel_info_score, answer_entropy = 0.0, 0.0, 0.0, 0.0, 0.0
        n_pairs = 0
        
        for i in range(self.n_agents):
            for j in range(i+1, self.n_agents):
                if i in agent_claims and j in agent_claims:
                    sim = self._safe_mean_sim(agent_claims[i], agent_claims[j])
                    claim_overlap += max(0.0, sim)
                    # Contradiction proxy: highly negative similarities
                    if sim < -0.2: contradiction_score += abs(sim)
                if i in agent_evidence and j in agent_evidence:
                    evidence_overlap += max(0.0, self._safe_mean_sim(agent_evidence[i], agent_evidence[j]))
                n_pairs += 1

        if n_pairs > 0:
            claim_overlap /= n_pairs
            evidence_overlap /= n_pairs
            contradiction_score /= n_pairs

        # Novelty: ratio of unique embedded conclusions/assumptions vs total
        if all_novel:
            try:
                global_novel = self.model.encode(all_novel, convert_to_tensor=True)
                global_novel = torch.nn.functional.normalize(global_novel, p=2, dim=1)
                # Count pairs that are highly similar (>0.9) to detect duplicates
                sims = torch.mm(global_novel, global_novel.t())
                n_items = len(all_novel)
                duplicates = float((sims > 0.9).sum().item() - n_items) / 2.0
                novel_info_score = max(0.0, (n_items - duplicates) / n_items)
            except Exception:
                novel_info_score = 0.5
                
        # Answer Entropy
        if answers:
            counts = Counter(answers)
            probs = [c / len(answers) for c in counts.values()]
            answer_entropy = -sum(p * math.log2(p) for p in probs)

        metrics = [claim_overlap, evidence_overlap, contradiction_score, novel_info_score, answer_entropy]
        return np.array(metrics, dtype=np.float64)


class MLP:
    """Standard NumPy Multilayer Perceptron."""
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
        pass # pass # np.fill_diagonal(W_raw, -500.0)  # Pre-sigmoid elimination
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

    def get_static_weights(self, topology: str = "ring") -> Tuple[np.ndarray, np.ndarray]:
        N = self.n_agents
        W = np.zeros((N, N))
        if topology == "ring":
            for i in range(N):
                W[i][(i + 1) % N] = 0.6
                W[(i + 1) % N][i] = 0.6
        else:
            W = np.ones((N, N)) * 0.5
            np.fill_diagonal(W, 1.0)
        return W, np.zeros((N, self.query_dim))

    def get_random_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        W = np.random.rand(self.n_agents, self.n_agents)
        np.fill_diagonal(W, 1.0)
        Q = np.random.randn(self.n_agents, self.query_dim)
        return W, Q

    def save_model(self, filepath: str):
        """Save the actor and critic MLP weights to a .npz file."""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        np.savez(filepath,
                 actor_W1=self.actor.W1, actor_b1=self.actor.b1,
                 actor_W2=self.actor.W2, actor_b2=self.actor.b2,
                 actor_W3=self.actor.W3, actor_b3=self.actor.b3,
                 critic_W1=self.critic.W1, critic_b1=self.critic.b1,
                 critic_W2=self.critic.W2, critic_b2=self.critic.b2,
                 critic_W3=self.critic.W3, critic_b3=self.critic.b3,
                 # Save mean_dim so load_model can validate architecture compatibility
                 mean_dim=np.array(self.mean_dim))

    def load_model(self, filepath: str):
        """Load the actor and critic MLP weights from a .npz file."""
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

"""
Dataset Loaders: GSM8K, HotpotQA, MATH (AQuA), TruthfulQA, and MBPP.
Returns STRICTLY separated (train, test) tuples to prevent data leakage.
"""



logger = setup_logging()


def _generate_synthetic_samples(n_train: int, n_test: int, seed: int = 42) -> Tuple[List[dict], List[dict]]:
    """
    Generate synthetic GSM8K-style math word problems for testing.
    Returns strictly disjoint train and test sets.
    """
    rng = random.Random(seed)
    templates = [
        {
            "template": "{name} has {a} apples. {name2} gives {name} {b} more apples. How many apples does {name} have now?",
            "answer_fn": lambda a, b: a + b,
        },
        {
            "template": "A store has {a} shirts. Each shirt costs ${b}. How much do all the shirts cost?",
            "answer_fn": lambda a, b: a * b,
        },
        {
            "template": "{name} reads {a} pages per day. How many pages does {name} read in {b} days?",
            "answer_fn": lambda a, b: a * b,
        },
        {
            "template": "There are {a} students in a class. Each student has {b} pencils. If the teacher adds {c} more pencils to the total, how many pencils are there?",
            "answer_fn": lambda a, b, c=0: a * b + c,
        },
        {
            "template": "{name} has ${a}. {name} spends ${b} on lunch and ${c} on a book. How much money does {name} have left?",
            "answer_fn": lambda a, b, c=0: a - b - c,
        },
    ]
    names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace"]

    def make_sample():
        t = rng.choice(templates)
        a = rng.randint(5, 50)
        b = rng.randint(2, 20)
        c = rng.randint(1, 10)
        name_pool = rng.sample(names, 3)
        question = t["template"].format(name=name_pool[0], name2=name_pool[1], name3=name_pool[2], a=a, b=b, c=c)
        try:
            answer = t["answer_fn"](a, b, c)
        except TypeError:
            answer = t["answer_fn"](a, b)
        return {"question": question, "answer": str(int(answer))}

    # Generate total unique samples (simplistic unique by question string)
    seen = set()
    all_samples = []
    while len(all_samples) < n_train + n_test:
        s = make_sample()
        if s["question"] not in seen:
            seen.add(s["question"])
            all_samples.append(s)

    return all_samples[:n_train], all_samples[n_train:n_train+n_test]


def load_gsm8k(n_train: int = 30, n_test: int = 50, seed: int = 42) -> Tuple[List[dict], List[dict]]:
    """Load GSM8K. Uses 'train' split for training, 'test' split for testing."""
    try:
        from datasets import load_dataset
        ds_train = load_dataset("gsm8k", "main", split="train")
        ds_test = load_dataset("gsm8k", "main", split="test")
        
        rng = random.Random(seed)
        
        def process_split(ds, n):
            indices = list(range(len(ds)))
            rng.shuffle(indices)
            samples = []
            for idx in indices[:n]:
                item = ds[idx]
                question = item["question"]
                answer_text = item["answer"]
                match = re.search(r'####\s*([\-\d,\.]+)', answer_text)
                if match:
                    answer = match.group(1).replace(",", "")
                else:
                    nums = re.findall(r'[\-\d,\.]+', answer_text)
                    answer = nums[-1].replace(",", "") if nums else "0"
                samples.append({"question": question, "answer": answer})
            return samples
            
        train_samples = process_split(ds_train, n_train)
        test_samples = process_split(ds_test, n_test)
        logger.info(f"Loaded {len(train_samples)} train and {len(test_samples)} test GSM8K samples")
        return train_samples, test_samples
    except Exception as e:
        logger.warning(f"Could not load GSM8K from HuggingFace: {e}")
        return _generate_synthetic_samples(n_train, n_test, seed)


def load_hotpotqa(n_train: int = 30, n_test: int = 50, seed: int = 42) -> Tuple[List[dict], List[dict]]:
    """Load HotpotQA. Uses 'train' split for training, 'validation' split for testing."""
    try:
        from datasets import load_dataset
        ds_train = load_dataset("hotpot_qa", "fullwiki", split="train")
        ds_test = load_dataset("hotpot_qa", "fullwiki", split="validation")
        
        rng = random.Random(seed)
        
        def process_split(ds, n):
            indices = list(range(len(ds)))
            rng.shuffle(indices)
            samples = []
            for idx in indices[:n]:
                item = ds[idx]
                samples.append({"question": item["question"], "answer": item["answer"]})
            return samples
            
        train_samples = process_split(ds_train, n_train)
        test_samples = process_split(ds_test, n_test)
        logger.info(f"Loaded {len(train_samples)} train and {len(test_samples)} test HotpotQA samples")
        return train_samples, test_samples
    except Exception as e:
        logger.warning(f"Could not load HotpotQA from HuggingFace: {e}")
        return _generate_synthetic_samples(n_train, n_test, seed)


def load_math(n_train: int = 30, n_test: int = 50, seed: int = 42) -> Tuple[List[dict], List[dict]]:
    """Load Google Deepmind AQuA. Uses train.json for training, test.json for testing."""
    import urllib.request
    
    def fetch_split(url, n, rng):
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            lines = response.read().decode('utf-8').strip().split('\n')
            
        samples = []
        indices = list(range(len(lines)))
        rng.shuffle(indices)
        for idx in indices[:n]:
            if idx < len(lines):
                item = json.loads(lines[idx])
                question = item["question"] + "\nOptions: " + ", ".join(item["options"])
                samples.append({
                    "question": "AQuA Math Problem:\n" + question + "\n\nEnd with your final answer wrapped in \\boxed{your_answer} where your_answer is the exact letter (e.g. \\boxed{A}).",
                    "answer": item["correct"]
                })
        return samples

    try:
        rng = random.Random(seed)
        train_samples = fetch_split("https://raw.githubusercontent.com/google-deepmind/AQuA/master/train.json", n_train, rng)
        test_samples = fetch_split("https://raw.githubusercontent.com/google-deepmind/AQuA/master/test.json", n_test, rng)
        logger.info(f"Loaded {len(train_samples)} train and {len(test_samples)} test AQuA MATH samples directly from Github")
        return train_samples, test_samples
    except Exception as e:
        logger.warning(f"Could not load AQuA MATH directly: {e}")
        return _generate_synthetic_samples(n_train, n_test, seed)


def load_truthfulqa(n_train: int = 30, n_test: int = 50, seed: int = 42) -> Tuple[List[dict], List[dict]]:
    """Load TruthfulQA (Multiple Choice).
    
    TruthfulQA has only a single 'validation' split. We split it into
    train/test using index-based partitioning with a shuffled permutation,
    guaranteeing zero overlap between the two sets.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
        
        total_needed = n_train + n_test
        all_indices = list(range(len(ds)))
        rng = random.Random(seed)
        rng.shuffle(all_indices)
        
        # Hard partition: first n_train for training, next n_test for testing.
        # No overlap is possible because we slice contiguous blocks.
        train_indices = all_indices[:n_train]
        test_indices  = all_indices[n_train:n_train + n_test]
        
        def make_samples(indices):
            samples = []
            for idx in indices:
                item = ds[idx]
                question = item["question"] + "\nOptions:\n"
                choices = item["mc1_targets"]["choices"]
                labels = item["mc1_targets"]["labels"]
                correct = ""
                for i, (c, l) in enumerate(zip(choices, labels)):
                    question += f"{chr(65+i)}. {c}\n"
                    if l == 1:
                        correct = chr(65+i)
                question += "\nEnd with your final answer wrapped in \\boxed{your_answer} where your_answer is the exact letter (e.g. \\boxed{A})."
                samples.append({"question": question, "answer": correct})
            return samples
            
        train_samples = make_samples(train_indices)
        test_samples = make_samples(test_indices)
        logger.info(f"Loaded {len(train_samples)} train and {len(test_samples)} test TruthfulQA samples (hard-partitioned, zero overlap)")
        return train_samples, test_samples
    except Exception as e:
        logger.warning(f"Could not load TruthfulQA: {e}")
        return _generate_synthetic_samples(n_train, n_test, seed)


def load_mbpp(n_train: int = 30, n_test: int = 50, seed: int = 42) -> Tuple[List[dict], List[dict]]:
    """Load MBPP coding dataset. Uses 'train' split for training, 'test' split for testing."""
    try:
        from datasets import load_dataset
        ds_train = load_dataset("mbpp", split="train")
        ds_test = load_dataset("mbpp", split="test")
        
        rng = random.Random(seed)
        
        def process_split(ds, n):
            indices = list(range(len(ds)))
            rng.shuffle(indices)
            samples = []
            for idx in indices[:n]:
                item = ds[idx]
                q = f"Write a Python function to solve this: {item['text']}\nDo not include tests. Just the function definition."
                samples.append({"question": q, "answer": item['code']})
            return samples
            
        train_samples = process_split(ds_train, n_train)
        test_samples = process_split(ds_test, n_test)
        
        logger.info(f"Loaded {len(train_samples)} train and {len(test_samples)} test MBPP samples")
        return train_samples, test_samples
    except Exception as e:
        logger.warning(f"Could not load MBPP: {e}")
        return _generate_synthetic_samples(n_train, n_test, seed)


def evaluate_answer(predicted: str, gold: str, dataset_type: str = "gsm8k") -> bool:
    """Check if predicted answer matches gold answer."""
    predicted = str(predicted).strip().lower().rstrip(".")
    gold = str(gold).strip().lower().rstrip(".")

    if dataset_type == "gsm8k":
        pred_nums = re.findall(r'-?\d+\.?\d*', predicted)
        gold_nums = re.findall(r'-?\d+\.?\d*', gold)
        if pred_nums and gold_nums:
            try:
                return abs(float(pred_nums[-1]) - float(gold_nums[-1])) < 1e-3
            except ValueError:
                return False
        return predicted == gold
    elif dataset_type in ["math", "truthfulqa"]:
        # _extract_answer already stripped \boxed{} before this function is called.
        # Do NOT attempt to re-parse \boxed{} here — it will always fail (double-parse bug).
        # Compare the full cleaned strings, not just the last character.
        pred_clean = re.sub(r'[^\w]', '', predicted).strip()
        gold_clean = re.sub(r'[^\w]', '', gold).strip()
        return gold_clean == pred_clean
    elif dataset_type == "mbpp":
        # Heuristic code evaluation: check if core signature exists
        gold_match = re.search(r'def\s+(\w+)', gold)
        if gold_match:
            func_name = gold_match.group(1).lower()
            if func_name not in predicted.lower():
                return False
        return "def " in predicted and "return " in predicted
    else:
        # HotpotQA — use token-level F1 (standard protocol) instead of substring match
        pred_tokens = set(re.sub(r'[^\w\s]', '', predicted).lower().split())
        gold_tokens = set(re.sub(r'[^\w\s]', '', gold).lower().split())
        # Remove stopwords that inflate F1
        stopwords = {'the', 'a', 'an', 'is', 'was', 'are', 'were', 'of', 'in', 'to', 'and', 'for', 'on', 'at', 'by', 'with'}
        pred_tokens -= stopwords
        gold_tokens -= stopwords
        if not gold_tokens:
            return predicted.strip().lower() == gold.strip().lower()
        if not pred_tokens:
            return False
        tp = len(pred_tokens & gold_tokens)
        precision = tp / len(pred_tokens) if pred_tokens else 0
        recall    = tp / len(gold_tokens) if gold_tokens else 0
        if precision + recall == 0:
            return False
        f1 = 2 * precision * recall / (precision + recall)
        return f1 >= 0.5  # Standard HotpotQA threshold


def load_dataset_by_name(name: str, n_train: int = 30, n_test: int = 50, seed: int = 42) -> Tuple[List[dict], List[dict]]:
    """Load dataset strictly split by name string."""
    name = name.lower()
    if name == "gsm8k":
        return load_gsm8k(n_train, n_test, seed)
    elif name in ("hotpotqa", "hotpot_qa"):
        return load_hotpotqa(n_train, n_test, seed)
    elif name == "math":
        return load_math(n_train, n_test, seed)
    elif name in ("truthfulqa", "truthful_qa"):
        return load_truthfulqa(n_train, n_test, seed)
    elif name == "mbpp" or name == "code":
        return load_mbpp(n_train, n_test, seed)
    else:
        raise ValueError(f"Unknown dataset: {name}. Use 'gsm8k', 'hotpotqa', 'math', 'truthfulqa', or 'mbpp'.")

"""
Experiment Pipeline: End-to-end experiment runner with controller training,
communication rounds, and result tracking.
"""



logger = setup_logging()


def create_llm_backend(config: Config) -> LLMBackend:
    """Factory function for LLM backend.
    
    NOTE: All backend classes are defined in this same file when running as
    a single Kaggle notebook cell. The imports below use the module path for
    the multi-file local layout. Both paths resolve correctly because
    build_kaggle.py concatenates everything into one cell.
    """
    if config.llm_backend == "mock":
        return MockLLMBackend(
            base_accuracy=config.mock_base_accuracy,
            diversity_factor=config.mock_diversity_factor,
            seed=config.seed,
        )
    elif config.llm_backend in ("nvidia", "openai", "api"):
        try:
            from llm_backend import NvidiaDeepSeekBackend as _Nv
        except ImportError:
            _Nv = NvidiaDeepSeekBackend  # already in global scope (notebook)
        return _Nv(
            model=config.api_model,
            api_key=config.api_key,
            base_url=config.api_base_url,
        )
    elif config.llm_backend == "azure":
        try:
            from llm_backend import AzureOpenAIBackend as _Az
        except ImportError:
            _Az = AzureOpenAIBackend  # already in global scope (notebook)
        model_name = config.api_model if config.api_model and config.api_model != "deepseek-ai/deepseek-v3.2" else "gpt-oss-120b"
        return _Az(model=model_name)
    elif config.llm_backend == "hf":
        try:
            from llm_backend import HuggingFaceLocalBackend as _Hf
        except ImportError:
            _Hf = HuggingFaceLocalBackend  # already in global scope (notebook)
        return _Hf(model_name=config.api_model)
    else:
        raise ValueError(f"Unknown LLM backend: {config.llm_backend}")


def run_single_sample(
    question: str,
    gold_answer: str,
    agent_pool: AgentPool,
    comm_graph: CommunicationGraph,
    controller: IBController,
    config: Config,
    train_controller: bool = True,
    avg_accuracy: float = 0.5,
) -> SampleResult:
    """
    Run the full multi-round pipeline for a single sample.

    1. Round 0: all agents solve independently
    2. Rounds 1..T: controller produces W -> filter messages -> agents re-solve
    3. Final answer via weighted majority vote
    """
    agent_pool.reset_all()
    comm_graph.reset()

    # --- Round 0: Independent solving ---
    round0_responses = agent_pool.initial_solve(
        question, dataset_type=config.dataset
    )

    initial_answers = {r.agent_id: r.answer for r in round0_responses}
    initial_correct = {
        r.agent_id: evaluate_answer(r.answer, gold_answer, config.dataset)
        for r in round0_responses
    }
    total_gen_tokens = sum(r.token_count for r in round0_responses)

    current_responses = round0_responses
    all_metas = []
    all_comms = []

    # --- Rounds 1..T: Communication rounds ---
    for t in range(1, config.n_rounds + 1):
        # Get controller input
        confidences = [r.confidence for r in current_responses]
        prev_W = comm_graph.get_weight_snapshot()
        unique_answers = len(set(r.answer for r in current_responses))
        agent_texts = {r.agent_id: r.reasoning_text for r in current_responses}
        agent_answers = {r.agent_id: r.answer for r in current_responses}

        # Get weights and semantic queries from controller
        W, Q, halt_decision, meta = controller.get_weights(
            reasoning_texts=agent_texts,
            answers=agent_answers,
            confidence_scores=confidences,
            prev_W=prev_W,
            round_num=t,
            unique_answers=unique_answers,
            avg_accuracy=avg_accuracy,
            explore=train_controller,
        )
        
        if meta is not None:
            all_metas.append(meta)

        # Always apply weights BEFORE checking halt so reward reflects this round's graph
        comm_graph.set_weights(W, Q)

        # Adaptive Halting: if confident, break early (after W is set)
        if halt_decision and t > 1:
            break

        # Build message dict and route through communication graph
        local_backend = agent_pool.agents[0].llm_backend
        received_messages = comm_graph.get_messages_for_round(agent_texts, t, local_backend)
        
        all_comms.append({
            "round": t,
            "weights": W,  # Removed the redundant .tolist()
            "messages_received": {k: v for k, v in received_messages.items()}
        })

        # Agents re-solve with filtered messages
        current_responses = agent_pool.communicate_and_solve(
            question=question,
            received_messages=received_messages,
            dataset_type=config.dataset,
            round_num=t,
        )
        total_gen_tokens += sum(r.token_count for r in current_responses)

    # --- Final answer ---
    final_answer = weighted_majority_vote(current_responses)
    is_correct = evaluate_answer(final_answer, gold_answer, config.dataset)

    # --- Compute reward and update controller ---
    tokens_transmitted = comm_graph.get_total_tokens_transmitted()
    tokens_if_full = comm_graph.get_total_tokens_if_full()
    compression_ratio = comm_graph.get_compression_ratio()
    n_cycles = comm_graph.count_cycles()

    reward = controller.compute_reward(
        task_correct=is_correct,
        tokens_transmitted=tokens_transmitted,
        tokens_if_full=tokens_if_full,
        compression_ratio=compression_ratio,
        n_cycles=n_cycles,
        halt_round=t,
    )

    if train_controller:
        # Only the final step carries the true sparse reward.
        # Intermediate steps get 0.0 so GAE can assign credit correctly
        # via its own backward TD-delta sweep (double-discounting fixed).
        for i, meta in enumerate(all_metas):
            is_last = (i == len(all_metas) - 1)
            step_reward = reward if is_last else 0.0
            controller.store_transition(meta, step_reward, done=is_last)

    # Build result
    result = SampleResult(
        question=question,
        gold_answer=gold_answer,
        final_answer=final_answer,
        is_correct=is_correct,
        initial_answers=initial_answers,
        initial_correct=initial_correct,
        final_answers={r.agent_id: r.answer for r in current_responses},
        initial_reasoning={r.agent_id: r.reasoning_text for r in round0_responses},
        final_reasoning={r.agent_id: r.reasoning_text for r in current_responses},
        raw_communications=all_comms,
        total_tokens_transmitted=tokens_transmitted,
        total_tokens_if_full=tokens_if_full,
        compression_ratio=compression_ratio,
        semantic_similarity=comm_graph.get_avg_semantic_similarity(),
        token_reduction=max(0.0, 1.0 - (tokens_transmitted / max(1.0, tokens_if_full))),
        information_retention=comm_graph.get_avg_information_retention(),
        weight_matrices=[comm_graph.get_weight_snapshot()],
        communication_density=comm_graph.get_communication_density(),
        mode_distribution=comm_graph.get_mode_distribution(),
        n_active_edges=comm_graph.get_active_edges(),
        generation_tokens=total_gen_tokens,
        halt_round=t,
    )

    return result


def run_experiment(
    config: Config,
    method_name: str = "IB-Controller",
    train: bool = True,
    evaluate: bool = True,
    controller: Optional[IBController] = None,
) -> ExperimentResults:
    """
    Run the full experiment pipeline.

    If train=True, runs multiple epochs over a training split to learn
    the controller, then evaluates on a separate test split.

    Args:
        controller: Optional pre-trained IBController to resume from.
                    If None, a fresh controller is instantiated.
                    Pass the same instance across dataset loops to train
                    a universal routing controller.
    """
    set_seed(config.seed)
    logger.info(f"Running experiment: {method_name} on {config.dataset}")

    # Load dataset (strictly separate splits)
    train_data, test_data = load_dataset_by_name(
        config.dataset, n_train=config.n_train_samples, n_test=config.n_samples, seed=config.seed
    )

    # Initialize components
    llm_backend = create_llm_backend(config)
    agent_pool = AgentPool(
        n_agents=config.n_agents,
        llm_backend=llm_backend,
        reasoning_styles=config.reasoning_styles,
    )
    comm_graph = CommunicationGraph(n_agents=config.n_agents)

    # Use provided controller or create a fresh one
    if controller is None:
        controller = IBController(config)
    else:
        # SYNC FIX: Ensure the universal controller adopts the current dataset's hyperparameters
        controller.config = config
        controller.lambda_info = config.lambda_info
        controller.gamma_loop = config.gamma_loop

    # Load model if requested
    if hasattr(config, "model_load_path") and config.model_load_path:
        import os
        if os.path.exists(config.model_load_path):
            logger.info(f"Loading pre-trained controller from {config.model_load_path}")
            controller.load_model(config.model_load_path)
            # If we loaded a model, we can skip training unless explicitly requested
            if train:
                logger.info("Model loaded but train=True. Will resume training from checkpoint.")
        else:
            logger.warning(f"Model load path {config.model_load_path} not found. Starting fresh.")

    # Initialize training state feature
    avg_accuracy = 0.5

    # --- Training phase ---
    if train and train_data:
        logger.info(f"Training controller on {len(train_data)} samples "
                     f"for {config.controller_epochs} epochs")

        for epoch in range(config.controller_epochs):
            epoch_correct = 0
            for sample in train_data:
                result = run_single_sample(
                    question=sample["question"],
                    gold_answer=sample["answer"],
                    agent_pool=agent_pool,
                    comm_graph=comm_graph,
                    controller=controller,
                    config=config,
                    train_controller=True,
                    avg_accuracy=avg_accuracy,
                )
                if result.is_correct:
                    epoch_correct += 1

            # Update controller after each epoch
            loss = controller.update_policy()
            avg_accuracy = epoch_correct / max(len(train_data), 1)
            logger.info(
                f"  Epoch {epoch + 1}/{config.controller_epochs}: "
                f"acc={avg_accuracy:.2%}, loss={loss:.4f}"
            )
            
        # Save model after training
        if hasattr(config, "model_save_path") and config.model_save_path:
            logger.info(f"Saving trained controller to {config.model_save_path}")
            controller.save_model(config.model_save_path)

    # --- Evaluation phase ---
    if evaluate:
        logger.info(f"Evaluating on {len(test_data)} test samples")
        results = ExperimentResults(
            method_name=method_name,
            dataset_name=config.dataset,
            training_rewards=controller.episode_rewards.copy(),
            training_losses=controller.episode_losses.copy(),
        )

        for sample in tqdm(test_data, desc=f"Eval {method_name}", disable=len(test_data) < 5):
            result = run_single_sample(
                question=sample["question"],
                gold_answer=sample["answer"],
                agent_pool=agent_pool,
                comm_graph=comm_graph,
                controller=controller,
                config=config,
                train_controller=False,
                avg_accuracy=avg_accuracy,
            )
            results.sample_results.append(result)

    if not evaluate:
        results = ExperimentResults(
            method_name=method_name,
            dataset_name=config.dataset,
            training_rewards=controller.episode_rewards.copy(),
            training_losses=controller.episode_losses.copy(),
        )

    results.final_W = comm_graph.get_weight_snapshot()
    return results


def run_full_communication(config: Config) -> ExperimentResults:
    """
    Run multi-agent debate baseline: full communication (W=1.0 everywhere).
    """
    set_seed(config.seed)
    llm_backend = create_llm_backend(config)
    agent_pool = AgentPool(config.n_agents, llm_backend, config.reasoning_styles)
    comm_graph = CommunicationGraph(config.n_agents)
    controller = IBController(config)

    # Set all weights to 1.0 (full communication)
    full_W = np.ones((config.n_agents, config.n_agents))
    np.fill_diagonal(full_W, 1.0)

    # Load data
    _, test_data = load_dataset_by_name(
        config.dataset, n_train=0, n_test=config.n_samples, seed=config.seed
    )

    results = ExperimentResults(
        method_name="Multi-Agent Debate",
        dataset_name=config.dataset,
    )

    for sample in tqdm(test_data, desc="Debate", disable=len(test_data) < 5):
        agent_pool.reset_all()
        comm_graph.reset()

        # Round 0
        r0 = agent_pool.initial_solve(sample["question"], config.dataset)
        initial_answers = {r.agent_id: r.answer for r in r0}
        initial_correct = {
            r.agent_id: evaluate_answer(r.answer, sample["answer"], config.dataset)
            for r in r0
        }
        total_gen = sum(r.token_count for r in r0)

        current = r0
        for t in range(1, config.n_rounds + 1):
            comm_graph.set_weights(full_W)
            texts = {r.agent_id: r.reasoning_text for r in current}
            local_backend = agent_pool.agents[0].llm_backend
            msgs = comm_graph.get_messages_for_round(texts, t, local_backend)

            # NOTE: Self-reflection removed to ensure fair comparison.
            # IB-Controller does NOT inject self-reflection, so the baseline
            # must not either — otherwise token counts and context quality
            # are asymmetrically inflated.

            current = agent_pool.communicate_and_solve(
                sample["question"], msgs, config.dataset, t
            )
            total_gen += sum(r.token_count for r in current)

        final = weighted_majority_vote(current)
        is_correct = evaluate_answer(final, sample["answer"], config.dataset)

        sr = SampleResult(
            question=sample["question"],
            gold_answer=sample["answer"],
            final_answer=final,
            is_correct=is_correct,
            initial_answers=initial_answers,
            initial_correct=initial_correct,
            final_answers={r.agent_id: r.answer for r in current},
            total_tokens_transmitted=comm_graph.get_total_tokens_transmitted(),
            total_tokens_if_full=comm_graph.get_total_tokens_if_full(),
            compression_ratio=comm_graph.get_compression_ratio(),
            weight_matrices=[full_W.copy()],
            communication_density=1.0,
            mode_distribution=comm_graph.get_mode_distribution(),
            n_active_edges=comm_graph.get_active_edges(),
            generation_tokens=total_gen,
            halt_round=config.n_rounds,
        )
        results.sample_results.append(sr)

    results.final_W = full_W
    return results


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        import enum
        if isinstance(obj, enum.Enum):
            return obj.name
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        try:
            return obj.__dict__
        except Exception:
            return super().default(obj)

def safe_mode_cast(mode_str):
    try:
        return SemanticBandwidthMode[mode_str]
    except (KeyError, TypeError):
        return mode_str

import os
os.makedirs("08-06results", exist_ok=True)
def save_kaggle_checkpoint(results_dict, filepath="08-06results/kaggle_latest_results.json"):
    try:
        from dataclasses import asdict
        import json
        output = {"detailed_samples": {}}
        for method, res in results_dict.items():
            if res is not None and hasattr(res, "sample_results"):
                output["detailed_samples"][method] = [asdict(s) for s in res.sample_results]
                
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, cls=NumpyEncoder)
            
    except Exception as e:
        print(f"Checkpoint save error: {e}")


import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

config = Config()
# ACTIVATE AZURE LLM MODE NATIVELY
config.llm_backend = "azure"
config.api_model = "gpt-oss-120b"
config.dataset = "gsm8k" # overriden in loop
config.n_samples = 300
config.n_train_samples = 100
config.n_rounds = 3

print("Loading Environment...")
llm_backend = create_llm_backend(config)
agent_pool = AgentPool(config.n_agents, llm_backend, config.reasoning_styles)
comm_graph = CommunicationGraph(config.n_agents)
controller = IBController(config)

# =========================================================
# PHASE 1: UNIVERSAL PRE-TRAINING
# =========================================================
print("\n=== PHASE 1: UNIVERSAL CONTROLLER PRE-TRAINING ===")
datasets_to_run = ["gsm8k", "hotpotqa", "math", "truthfulqa"]

for ds_name in datasets_to_run:
    print(f"\n--- Pre-Training Dataset: {ds_name.upper()} ---")
    current_config = get_math_config() if ds_name == "math" else (get_hotpotqa_config() if ds_name == "hotpotqa" else get_gsm8k_config())
    current_config.llm_backend = "azure"
    current_config.api_model = "gpt-oss-120b"
    current_config.n_train_samples = 100
    current_config.n_samples = 300
    current_config.controller_epochs = 2
    current_config.dataset = ds_name
    
    # Train ONLY. Skip the expensive test split.
    _ = run_experiment(current_config, method_name=f"Pretrain_{ds_name.upper()}", train=True, evaluate=False, controller=controller)

print("✅ Universal Controller pre-trained.")

# =========================================================
# PHASE 2: FROZEN TRANSFER EVALUATION
# =========================================================
print("\n=== PHASE 2: FROZEN TRANSFER EVALUATION ===")

live_results = {}
for ds_name in datasets_to_run:
    print(f"\n--- Evaluating Transfer to Dataset: {ds_name.upper()} ---")
    current_config = get_math_config() if ds_name == "math" else (get_hotpotqa_config() if ds_name == "hotpotqa" else get_gsm8k_config())
    current_config.llm_backend = "azure"
    current_config.api_model = "gpt-oss-120b"
    current_config.n_samples = 300
    current_config.n_train_samples = 100
    current_config.dataset = ds_name
    
    experiment_key = f"ZeroShot_Transfer_{ds_name.upper()}"
    
    # Evaluate ONLY. (train=False)
    sc_res = run_experiment(current_config, method_name=experiment_key, train=False, controller=controller)
    live_results[experiment_key] = sc_res
    save_kaggle_checkpoint(live_results)

print("\n✅ Execution Complete! All exact dialogue chains and semantic metrics are preserved in kaggle_latest_results.json")
