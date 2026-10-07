import os
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
    llm_backend: str = "mock"  # "mock" or "nvidia" or "openai"
    api_base_url: str = "https://integrate.api.nvidia.com/v1"
    api_model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    api_key: Optional[str] = "your-nvidia-api-key"
    mock_base_accuracy: float = 0.65  # base accuracy of mock LLM
    mock_diversity_factor: float = 0.15  # how much agents differ

    # --- Output ---
    results_dir: str = "results"
    plots_dir: str = "results/plots"
    save_detailed_logs: bool = True

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
    densities = [r.communication_density for r in results.sample_results
                 if r.communication_density > 0]
    return float(np.mean(densities)) if densities else 0.0


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
        f"| Method | Accuracy | Avg Tokens/Sample | "
        f"Comm Density | Error Prop Rate |\n"
        f"|--------|----------|-------------------|"
        f"-------------|----------------|\n"
    )

    rows = []
    for m in all_metrics:
        row = (
            f"| {m['method']:<20s} "
            f"| {m['accuracy_pct']:>8s} "
            f"| {m['token_cost']['avg_tokens_per_sample']:>17.1f} "
            f"| {m['communication_density']:>11.3f} "
            f"| {m['error_propagation_rate']:>14.3f} |"
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
        """Generate mock GSM8K response with chain-of-thought."""
        h = self._hash_input(question, agent_id, context)
        accuracy = self._agent_accuracy(agent_id)

        # Extract numbers from question for realistic reasoning
        numbers = re.findall(r'\d+\.?\d*', question)
        nums = [float(n) for n in numbers[:5]] if numbers else [10, 5, 3]

        # Determine if answer is correct
        is_correct = h < accuracy

        # Context can improve accuracy (simulates learning from peers)
        if context:
            context_boost = 0.1 * min(len(context) / 200, 1.0)
            h2 = self._hash_input(question, agent_id, context[:50])
            is_correct = is_correct or (h2 < accuracy + context_boost)

        # Generate plausible answer
        if len(nums) >= 2:
            correct_answer = int(nums[0] * nums[1])
            if len(nums) >= 3:
                correct_answer = int(nums[0] * nums[1] + nums[2])
        else:
            correct_answer = int(nums[0]) if nums else 42

        if is_correct:
            answer = correct_answer
        else:
            # Plausible but wrong
            offset = int(h * 20) - 10
            answer = max(0, correct_answer + offset)
            if answer == correct_answer:
                answer = correct_answer + 5

        # Generate chain-of-thought
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

        from utils import count_tokens
        token_count = count_tokens(reasoning)

        return LLMResponse(
            text=reasoning,
            answer=str(answer),
            token_count=token_count,
            confidence=min(0.95, max(0.1, confidence)),
        )

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
        if context:
            context_boost = 0.12 * min(len(context) / 200, 1.0)
            h2 = self._hash_input(question, agent_id, context[:50])
            is_correct = is_correct or (h2 < accuracy + context_boost)

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

        from utils import count_tokens
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
        """Generate response using NVIDIA DeepSeek API with hard timeout per call."""
        import time as _time
        import threading

        system_msg = self._get_system_prompt(dataset_type, agent_id)
        user_msg = f"Question: {question}"
        if context:
            user_msg += f"\n\nContext from other agents:\n{context}"
        user_msg += "\n\nProvide step-by-step reasoning, then give your final answer on the last line as: Answer: <your answer>"

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        last_error = None
        for attempt in range(self.max_retries):
            # Use thread-based hard timeout to prevent indefinite hangs
            result_holder = [None]
            error_holder = [None]

            def _api_call():
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.7,
                        top_p=0.95,
                        max_tokens=512,
                        stream=False,
                    )
                    result_holder[0] = resp
                except Exception as e:
                    error_holder[0] = e

            thread = threading.Thread(target=_api_call, daemon=True)
            thread.start()
            thread.join(timeout=self.timeout)

            if thread.is_alive():
                # Thread is still running = hung API call
                last_error = TimeoutError(f"API call timed out after {self.timeout}s")
                print(f"  [API] Attempt {attempt+1}/{self.max_retries} TIMED OUT after {self.timeout}s. Retrying...")
                # Recreate client to reset any stuck connections
                try:
                    from openai import OpenAI
                    import httpx
                    self.client = OpenAI(
                        base_url=self.base_url,
                        api_key=self.api_key,
                        timeout=httpx.Timeout(self.timeout, connect=15.0),
                        max_retries=0,  # We handle retries ourselves
                    )
                except Exception:
                    pass
                _time.sleep(3)
                continue

            if error_holder[0] is not None:
                last_error = error_holder[0]
                wait = (attempt + 1) * 3
                print(f"  [API] Attempt {attempt+1}/{self.max_retries} failed: {type(last_error).__name__}: {str(last_error)[:100]}. Retrying in {wait}s...")
                _time.sleep(wait)
                continue

            response = result_holder[0]
            if response is None:
                last_error = RuntimeError("No response received")
                _time.sleep(3)
                continue

            try:
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
                print(f"  [API] Attempt {attempt+1}/{self.max_retries} parse error: {e}")
                _time.sleep(3)
                continue

        # All retries failed
        err_msg = f"API Error after {self.max_retries} retries: {str(last_error)}"
        print(f"  [API] ALL RETRIES FAILED: {err_msg[:150]}")
        from utils import count_tokens
        return LLMResponse(
            text=err_msg,
            answer="",
            token_count=count_tokens(err_msg),
            confidence=0.1,
        )

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
            return f"{style} Solve math word problems step by step. Give a numeric final answer."
        else:
            return f"{style} Answer multi-hop questions by reasoning through connected facts. Give a concise final answer."

    def _extract_answer(self, text: str, dataset_type: str) -> str:
        """Extract the final answer from LLM response."""
        if not text:
            return ""

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
        """Heuristic confidence from response text."""
        if not text:
            return 0.1
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
            
            from utils import count_tokens
            token_count = count_tokens(text)
            
            return LLMResponse(
                text=text,
                answer=answer,
                token_count=token_count,
                confidence=confidence,
            )
        except Exception as e:
            from utils import count_tokens
            err_msg = f"Local HF Error: {str(e)}"
            return LLMResponse(
                text=err_msg,
                answer="",
                token_count=count_tokens(err_msg),
                confidence=0.1,
            )
\n
class KaggleHuggingFaceBackend(LLMBackend):
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        self.model_name = model_name
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        
        print(f"Loading {model_name} on Kaggle GPU in natively pure bfloat16 precision...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        print("Kaggle GPU Model loaded successfully.")

    def generate(self, prompt: str, question: str, context: str = "", dataset_type: str = "gsm8k", agent_id: int = 0) -> LLMResponse:
        system_msg = self._get_system_prompt(dataset_type, agent_id)
        user_msg = f"Question: {question}"
        if context:
            user_msg += f"\n\nContext from other agents:\n{context}"
        user_msg += "\n\nProvide step-by-step reasoning, then give your final answer on the last line as: Answer: <your answer>"

        messages = [{"role": "user", "content": f"{system_msg}\n\n{user_msg}"}]
        
        try:
            prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            outputs = self.pipe(prompt_text)
            generated_text = outputs[0]["generated_text"]
            
            if prompt_text in generated_text:
                generated_text = generated_text[len(prompt_text):]
                
            text = generated_text.strip()
            answer = self._extract_answer(text, dataset_type)
            confidence = self._estimate_confidence(text)
            
            # Use raw length tracking
            token_count = len(self.tokenizer.tokenize(text))
            
            return LLMResponse(text=text, answer=answer, token_count=token_count, confidence=confidence)
            
        except Exception as e:
            return LLMResponse(text=f"Error: {str(e)}", answer="", token_count=0, confidence=0.1)

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
                context = "Other agents' reasoning:\n" + "\n---\n".join(valid)

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
        """All agents solve the question independently (Round 0)."""
        responses = []
        for agent in self.agents:
            resp = agent.solve(question, dataset_type=dataset_type, round_num=0)
            responses.append(resp)
        return responses

    def communicate_and_solve(
        self,
        question: str,
        received_messages: Dict[int, List[str]],
        dataset_type: str = "gsm8k",
        round_num: int = 1,
    ) -> List[AgentResponse]:
        """
        Each agent re-solves using messages received from other agents.

        Args:
            question: The original question
            received_messages: {agent_id: [list of messages received]}
            dataset_type: Dataset type
            round_num: Current round number

        Returns:
            List of AgentResponses
        """
        responses = []
        for agent in self.agents:
            messages = received_messages.get(agent.agent_id, [])
            resp = agent.solve(
                question,
                context_messages=messages,
                dataset_type=dataset_type,
                round_num=round_num,
            )
            responses.append(resp)
        return responses

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


class CommunicationGraph:
    """
    Directed communication graph with learnable edge weights W and semantic queries Q.
    Uses Learned Token Masking rather than static heuristic text replacements.
    """

    def __init__(self, n_agents: int, vocab_size: int = 500, query_dim: int = 16):
        self.n_agents = n_agents
        self.vocab_size = vocab_size
        self.query_dim = query_dim
        
        self.W = np.ones((n_agents, n_agents)) * 0.5
        np.fill_diagonal(self.W, 1.0)
        self.Q = np.zeros((n_agents, query_dim))
        
        # Static hash-based embedding map initialized consistently
        rng = np.random.RandomState(42)
        self.E = rng.randn(vocab_size, query_dim)

        self.transmission_log: List[TransmissionRecord] = []
        self.round_token_counts: Dict[int, int] = {}
        self.modes_disabled: List[SemanticBandwidthMode] = []

    def set_weights(self, W: np.ndarray, Q: Optional[np.ndarray] = None):
        """Set the full weight matrix and optional query semantics from policy."""
        self.W = W.copy()
        np.fill_diagonal(self.W, 1.0)
        if Q is not None:
            self.Q = Q.copy()

    def transmit(self, sender_id: int, receiver_id: int, message: str, round_num: int) -> Tuple[str, int]:
        """
        Transmit a message applying Learned Token Masking via Query-Embedding dot product.
        """
        if sender_id == receiver_id:
            return "", 0

        w_ij = self.W[sender_id][receiver_id]
        mode = get_mode(w_ij)
        
        # Ablation fallback check
        if mode in self.modes_disabled:
            mode = SemanticBandwidthMode.BLOCK
            w_ij = 0.0

        q_j = self.Q[receiver_id] if self.Q is not None else np.zeros(self.query_dim)
        
        original_tokens = count_tokens(message)

        if mode == SemanticBandwidthMode.BLOCK or w_ij < 1e-3:
            filtered_msg = ""
            transmitted_tokens = 0
        elif mode == SemanticBandwidthMode.FULL or w_ij > 0.99:
            filtered_msg = message
            transmitted_tokens = original_tokens
        else:
            words = message.split()
            filtered_words = []
            
            for w in words:
                clean_w = re.sub(r'[^a-zA-Z0-9]', '', w.lower())
                if not clean_w:
                    filtered_words.append(w)
                    continue
                
                # Simple deterministic string hash to vocab index
                h = hash(clean_w) % self.vocab_size
                
                # Learned score: dot product between agent Query and Word Embedding
                score = np.dot(q_j, self.E[h])
                
                # Map score into [0, 1] probability
                alignment_prob = 1.0 / (1.0 + np.exp(-np.clip(score, -10, 10)))
                
                # Threshold drops progressively as W_ij increases. 
                # If W_ij = 1.0 -> Threshold is 0.0 (Accepts all words)
                # If W_ij = 0.0 -> Threshold is 1.0 (Blocks all words)
                if alignment_prob > (1.0 - w_ij):
                    filtered_words.append(w)
                    
            filtered_msg = " ".join(filtered_words)
            transmitted_tokens = count_tokens(filtered_msg)

        record = TransmissionRecord(
            sender_id=sender_id,
            receiver_id=receiver_id,
            mode=mode,
            original_tokens=original_tokens,
            transmitted_tokens=transmitted_tokens,
            weight=w_ij,
            round_num=round_num,
        )
        self.transmission_log.append(record)

        self.round_token_counts[round_num] = self.round_token_counts.get(round_num, 0) + transmitted_tokens

        return filtered_msg, transmitted_tokens

    def get_messages_for_round(self, agent_responses: dict, round_num: int) -> Dict[int, List[str]]:
        received: Dict[int, List[str]] = {i: [] for i in range(self.n_agents)}
        for sender_id in range(self.n_agents):
            for receiver_id in range(self.n_agents):
                if sender_id == receiver_id:
                    continue
                msg = agent_responses.get(sender_id, "")
                filtered_msg, _ = self.transmit(sender_id, receiver_id, msg, round_num)
                if filtered_msg.strip():
                    received[receiver_id].append(filtered_msg)
        return received

    def get_total_tokens_transmitted(self) -> int:
        return sum(r.transmitted_tokens for r in self.transmission_log)

    def get_total_tokens_if_full(self) -> int:
        return sum(r.original_tokens for r in self.transmission_log)

    def get_compression_ratio(self) -> float:
        active = [r for r in self.transmission_log if r.original_tokens > 0]
        if not active: return 0.0
        return np.mean([r.transmitted_tokens / r.original_tokens for r in active])

    def get_communication_density(self) -> float:
        mask = np.ones_like(self.W, dtype=bool)
        np.fill_diagonal(mask, False)
        return float(np.mean(self.W[mask]))

    def get_active_edges(self, threshold: float = 0.25) -> int:
        mask = np.ones_like(self.W, dtype=bool)
        np.fill_diagonal(mask, False)
        return int(np.sum(self.W[mask] >= threshold))

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
                    if not visited[u]: dfs(u)
                    elif rec_stack[u]: cycle_count += 1
            rec_stack[v] = False

        for v in range(n):
            if not visited[v]: dfs(v)
        return cycle_count

    def reset(self):
        self.transmission_log = []
        self.round_token_counts = {}

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
    """PPO Controller with native Actor-Critic MLPs."""

    def __init__(self, config: Config):
        self.config = config
        self.n_agents = config.n_agents
        self.query_dim = getattr(config, 'query_dim', 16)
        self.input_dim = self.n_agents + self.n_agents * self.n_agents + 2
        
        # Actor outputs W (N*N) and Q (N * query_dim)
        self.action_dim = self.n_agents * self.n_agents + self.n_agents * self.query_dim

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

    def build_features(self, confs: List[float], prev_W: np.ndarray, round_num: int, avg_acc: float = 0.5) -> np.ndarray:
        feats = list(confs) + prev_W.flatten().tolist() + [round_num / max(self.config.n_rounds, 1), avg_acc]
        return np.array(feats, dtype=np.float64)

    def _split_action(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        N = self.n_agents
        w_flat = action[:N*N]
        q_flat = action[N*N:]
        
        # W must be [0,1], Q can float freely
        W = 1 / (1 + np.exp(-np.clip(w_flat, -500, 500)))
        W = W.reshape(N, N)
        np.fill_diagonal(W, 1.0)
        
        Q = q_flat.reshape(N, self.query_dim)
        return W, Q

    def get_weights(self, confs: List[float], prev_W: np.ndarray, round_num: int, avg_acc: float = 0.5, explore: bool = True):
        features = self.build_features(confs, prev_W, round_num, avg_acc)
        
        mean_action, actor_cache = self.actor.forward(features)
        value, critic_cache = self.critic.forward(features)
        
        sigma = 0.2
        if explore:
            action = mean_action + np.random.randn(*mean_action.shape) * sigma
        else:
            action = mean_action
            
        # Log probability of Gaussian action
        log_prob = -0.5 * np.sum(((action - mean_action)/sigma)**2 + np.log(2 * np.pi * sigma**2))
        
        W, Q = self._split_action(action)

        meta = {
            'features': features,
            'action': action,
            'log_prob': log_prob,
            'value': value[0],
            'actor_cache': actor_cache,
            'mean_action': mean_action
        }
        return W, Q, meta

    def compute_reward(self, task_correct: bool, tokens_transmitted: int, tokens_if_full: int, compression_ratio: float, n_cycles: int) -> float:
        task_rew = 1.0 if task_correct else -0.5
        info_cost = tokens_transmitted / max(tokens_if_full, 1) if self.config.info_cost_method == "token_count" else compression_ratio
        loop_penalty = n_cycles / max(self.n_agents * (self.n_agents - 1) // 2, 1)
        return task_rew - self.lambda_info * info_cost - self.gamma_loop * loop_penalty

    def store_transition(self, meta, reward: float):
        if meta:
            meta['reward'] = reward
            self.saved_transitions.append(meta)

    def update_policy(self) -> float:
        if not self.saved_transitions: return 0.0

        n_steps = len(self.saved_transitions)
        
        # Trajectories contain independent queries with sparse episodic rewards at step T.
        # Compute Monte Carlo Returns strictly
        returns = np.zeros(n_steps)
        advantages = np.zeros(n_steps)
        
        for i in range(n_steps):
            r = self.saved_transitions[i]['reward']
            v = self.saved_transitions[i]['value']
            returns[i] = r # MC sparse rewards identically applied to entire path context
            advantages[i] = returns[i] - v
            
        # Normalize advantages
        if np.std(advantages) > 1e-8:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        total_actor_loss, total_critic_loss = 0, 0
        sigma = 0.2

        for epoch in range(self.ppo_epochs):
            actor_grads = {n: np.zeros_like(getattr(self.actor, n)) for n in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']}
            critic_grads = {n: np.zeros_like(getattr(self.critic, n)) for n in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']}
            
            for i in range(n_steps):
                feat = self.saved_transitions[i]['features']
                action = self.saved_transitions[i]['action']
                old_log_prob = self.saved_transitions[i]['log_prob']
                adv = advantages[i]
                ret = returns[i]
                
                # Forward re-evaluation
                mean_a, acache = self.actor.forward(feat)
                v_pred, vcache = self.critic.forward(feat)
                
                new_log_prob = -0.5 * np.sum(((action - mean_a)/sigma)**2 + np.log(2 * np.pi * sigma**2))
                ratio = np.exp(new_log_prob - old_log_prob)
                
                surr1 = ratio * adv
                surr2 = np.clip(ratio, 1.0 - self.ppo_epsilon, 1.0 + self.ppo_epsilon) * adv
                actor_loss = -min(surr1, surr2)
                
                # Actor backward
                if surr1 < surr2:
                    d_ratio = adv
                else:
                    d_ratio = 0.0 if np.clip(ratio, 1.0-self.ppo_epsilon, 1.0+self.ppo_epsilon) != ratio else adv
                
                d_log_prob = d_ratio * ratio
                d_mean = d_log_prob * (action - mean_a) / (sigma ** 2)
                agrad = self.actor.backward(acache, -d_mean / n_steps) # Negative because minimizing loss
                for k in actor_grads: actor_grads[k] += agrad[k]
                
                # Critic backward
                critic_loss = 0.5 * (ret - v_pred[0])**2
                vgrad = self.critic.backward(vcache, np.array([(v_pred[0] - ret) / n_steps]))
                for k in critic_grads: critic_grads[k] += vgrad[k]
                
                total_actor_loss += abs(actor_loss)
                total_critic_loss += critic_loss
            
            self.actor.update(actor_grads)
            self.critic.update(critic_grads)

        avg_loss = (total_actor_loss + self.ppo_v_coef * total_critic_loss) / (n_steps * self.ppo_epochs)
        if hasattr(self, 'episode_losses'):
            self.episode_losses.append(avg_loss)
            self.episode_rewards.extend([t['reward'] for t in self.saved_transitions])

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

"""
Dataset Loaders: GSM8K and HotpotQA with fallback bundled subsets.
"""



logger = setup_logging()


def _generate_gsm8k_samples(n: int, seed: int = 42) -> List[dict]:
    """
    Generate synthetic GSM8K-style math word problems for testing.
    Used as fallback when HuggingFace datasets is unavailable.
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
        {
            "template": "A bakery makes {a} muffins per hour. How many muffins does it make in {b} hours?",
            "answer_fn": lambda a, b: a * b,
        },
        {
            "template": "{name} has {a} marbles. {name} gives {b} to {name2} and receives {c} from {name3}. How many marbles does {name} have?",
            "answer_fn": lambda a, b, c=0: a - b + c,
        },
        {
            "template": "A train travels at {a} km/h for {b} hours. How far does it travel?",
            "answer_fn": lambda a, b: a * b,
        },
    ]
    names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace"]

    samples = []
    for i in range(n):
        t = rng.choice(templates)
        a = rng.randint(5, 50)
        b = rng.randint(2, 20)
        c = rng.randint(1, 10)
        name_pool = rng.sample(names, 3)

        question = t["template"].format(
            name=name_pool[0], name2=name_pool[1], name3=name_pool[2],
            a=a, b=b, c=c
        )

        try:
            answer = t["answer_fn"](a, b, c)
        except TypeError:
            answer = t["answer_fn"](a, b)

        samples.append({
            "question": question,
            "answer": str(int(answer)),
        })

    return samples


def _generate_hotpotqa_samples(n: int, seed: int = 42) -> List[dict]:
    """
    Generate synthetic HotpotQA-style multi-hop questions for testing.
    """
    rng = random.Random(seed)

    entities = [
        ("Paris", "France", "Eiffel Tower"),
        ("London", "United Kingdom", "Big Ben"),
        ("Tokyo", "Japan", "Mount Fuji"),
        ("Berlin", "Germany", "Brandenburg Gate"),
        ("Rome", "Italy", "Colosseum"),
        ("Beijing", "China", "Great Wall"),
        ("Moscow", "Russia", "Red Square"),
        ("Cairo", "Egypt", "Pyramids"),
        ("Sydney", "Australia", "Opera House"),
        ("Mumbai", "India", "Gateway of India"),
    ]

    templates = [
        {
            "template": "What country is the city of {city} located in, which is famous for {landmark}?",
            "answer_idx": 1,
        },
        {
            "template": "What landmark is located in {city}, the capital of {country}?",
            "answer_idx": 2,
        },
        {
            "template": "In which city can you find {landmark}, a famous site in {country}?",
            "answer_idx": 0,
        },
    ]

    samples = []
    for i in range(n):
        entity = entities[i % len(entities)]
        t = rng.choice(templates)
        question = t["template"].format(
            city=entity[0], country=entity[1], landmark=entity[2]
        )
        answer = entity[t["answer_idx"]]
        samples.append({"question": question, "answer": answer})

    return samples


def load_gsm8k(n_samples: int = 50, seed: int = 42) -> List[dict]:
    """
    Load GSM8K samples. Tries HuggingFace datasets first, falls back to
    synthetic generation.

    Returns:
        List of {"question": str, "answer": str}
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split="test")
        samples = []
        indices = list(range(len(ds)))
        rng = random.Random(seed)
        rng.shuffle(indices)

        for idx in indices[:n_samples]:
            item = ds[idx]
            question = item["question"]
            # Extract final numeric answer from GSM8K format
            answer_text = item["answer"]
            # GSM8K answers end with #### <number>
            match = re.search(r'####\s*([\-\d,\.]+)', answer_text)
            if match:
                answer = match.group(1).replace(",", "")
            else:
                nums = re.findall(r'[\-\d,\.]+', answer_text)
                answer = nums[-1].replace(",", "") if nums else "0"

            samples.append({"question": question, "answer": answer})

        logger.info(f"Loaded {len(samples)} GSM8K samples from HuggingFace")
        return samples

    except Exception as e:
        logger.warning(f"Could not load GSM8K from HuggingFace: {e}")
        logger.info(f"Generating {n_samples} synthetic GSM8K samples")
        return _generate_gsm8k_samples(n_samples, seed)


def load_hotpotqa(n_samples: int = 50, seed: int = 42) -> List[dict]:
    """
    Load HotpotQA samples. Tries HuggingFace datasets first, falls back
    to synthetic generation.

    Returns:
        List of {"question": str, "answer": str}
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("hotpot_qa", "fullwiki", split="validation")
        samples = []
        indices = list(range(len(ds)))
        rng = random.Random(seed)
        rng.shuffle(indices)

        for idx in indices[:n_samples]:
            item = ds[idx]
            samples.append({
                "question": item["question"],
                "answer": item["answer"],
            })

        logger.info(f"Loaded {len(samples)} HotpotQA samples from HuggingFace")
        return samples

    except Exception as e:
        logger.warning(f"Could not load HotpotQA from HuggingFace: {e}")
        logger.info(f"Generating {n_samples} synthetic HotpotQA samples")
        return _generate_hotpotqa_samples(n_samples, seed)


def load_math(n_samples: int = 50, seed: int = 42) -> List[dict]:
    """
    Load Hendrycks MATH (competition math) samples. Harder than GSM8K.
    """
def load_math(n_samples: int = 50, seed: int = 42) -> List[dict]:
    """
    Load Google Deepmind AQuA dataset directly. Harder than GSM8K.
    """
    import urllib.request
    try:
        url = "https://raw.githubusercontent.com/google-deepmind/AQuA/master/test.json"
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            lines = response.read().decode('utf-8').strip().split('\n')
        
        samples = []
        indices = list(range(len(lines)))
        rng = random.Random(seed)
        rng.shuffle(indices)

        for idx in indices[:n_samples]:
            if idx < len(lines):
                item = json.loads(lines[idx])
                question = item["question"] + "\nOptions: " + ", ".join(item["options"])
                samples.append({
                    "question": "AQuA Math Problem:\n" + question + "\n\nEnd with your final answer wrapped in \\boxed{your_answer} where your_answer is the exact letter (e.g. \\boxed{A}).",
                    "answer": item["correct"]
                })

        logger.info(f"Loaded {len(samples)} AQuA MATH samples directly from Github")
        return samples

    except Exception as e:
        logger.warning(f"Could not load AQuA MATH directly: {e}")
        logger.info(f"Generating synthetic MATH samples (fallback to GSM8K logic)")
        return _generate_gsm8k_samples(n_samples, seed)


def evaluate_answer(predicted: str, gold: str, dataset_type: str = "gsm8k") -> bool:
    """
    Check if predicted answer matches gold answer.
    """
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
    elif dataset_type == "math":
        # Check if the exact gold string matches inside the predicted boxed or last text
        # DeepSeek is asked to use \boxed{}
        pred_box = re.search(r'\\boxed{(.*?)}', predicted)
        if pred_box:
            pred_clean = re.sub(r'[^\w]', '', pred_box.group(1)).strip()
        else:
            pred_clean = re.sub(r'[^\w]', '', predicted).strip()
            
        gold_clean = re.sub(r'[^\w]', '', gold).strip()
        return gold_clean in pred_clean
    else:
        # HotpotQA
        pred_clean = re.sub(r'[^\w\s]', '', predicted).strip()
        gold_clean = re.sub(r'[^\w\s]', '', gold).strip()
        return pred_clean == gold_clean or gold_clean in pred_clean or pred_clean in gold_clean


def load_dataset_by_name(name: str, n_samples: int = 50, seed: int = 42) -> List[dict]:
    """Load dataset by name string."""
    if name.lower() == "gsm8k":
        return load_gsm8k(n_samples, seed)
    elif name.lower() in ("hotpotqa", "hotpot_qa"):
        return load_hotpotqa(n_samples, seed)
    elif name.lower() == "math":
        return load_math(n_samples, seed)
    else:
        raise ValueError(f"Unknown dataset: {name}. Use 'gsm8k', 'hotpotqa', or 'math'.")

"""
Experiment Pipeline: End-to-end experiment runner with controller training,
communication rounds, and result tracking.
"""



logger = setup_logging()


def create_llm_backend(config: Config) -> LLMBackend:
    """Factory function for LLM backend."""
    if config.llm_backend == "mock":
        return MockLLMBackend(
            base_accuracy=config.mock_base_accuracy,
            diversity_factor=config.mock_diversity_factor,
            seed=config.seed,
        )
    elif config.llm_backend in ("nvidia", "openai", "api"):
        from llm_backend import NvidiaDeepSeekBackend
        return NvidiaDeepSeekBackend(
            model=config.api_model,
            api_key=config.api_key,
            base_url=config.api_base_url,
        )
    elif config.llm_backend == "kaggle":
        return KaggleHuggingFaceBackend(model_name=config.api_model)
    elif config.llm_backend == "hf":
        from llm_backend import HuggingFaceLocalBackend
        return HuggingFaceLocalBackend(model_name=config.api_model)
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

        # Get weights and semantic queries from controller
        W, Q, meta = controller.get_weights(
            confidence_scores=confidences,
            prev_W=prev_W,
            round_num=t,
            avg_accuracy=avg_accuracy,
            explore=train_controller,
        )
        comm_graph.set_weights(W, Q)

        if meta is not None:
            all_metas.append(meta)

        # Build messages through communication graph
        agent_texts = {r.agent_id: r.reasoning_text for r in current_responses}
        received_messages = comm_graph.get_messages_for_round(agent_texts, t)
        
        all_comms.append({
            "round": t,
            "weights": W.tolist() if hasattr(W, "tolist") else W,
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
    )

    if train_controller:
        # Store each round's transition individually
        for meta in all_metas:
            controller.store_transition(meta, reward)

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
        weight_matrices=[comm_graph.get_weight_snapshot()],
        communication_density=comm_graph.get_communication_density(),
        mode_distribution=comm_graph.get_mode_distribution(),
        n_active_edges=comm_graph.get_active_edges(),
        generation_tokens=total_gen_tokens,
    )

    return result


def run_experiment(
    config: Config,
    method_name: str = "IB-Controller",
    train: bool = True,
) -> ExperimentResults:
    """
    Run the full experiment pipeline.

    If train=True, runs multiple epochs over a training split to learn
    the controller, then evaluates on a separate test split.
    """
    set_seed(config.seed)
    logger.info(f"Running experiment: {method_name} on {config.dataset}")

    # Load dataset
    total_samples = config.n_train_samples + config.n_samples
    all_data = load_dataset_by_name(
        config.dataset, n_samples=total_samples, seed=config.seed
    )

    # Split into train and test
    train_data = all_data[:config.n_train_samples]
    test_data = all_data[config.n_train_samples:config.n_train_samples + config.n_samples]

    # Initialize components
    llm_backend = create_llm_backend(config)
    agent_pool = AgentPool(
        n_agents=config.n_agents,
        llm_backend=llm_backend,
        reasoning_styles=config.reasoning_styles,
    )
    comm_graph = CommunicationGraph(n_agents=config.n_agents)
    controller = IBController(config)

    # --- Training phase ---
    if train and train_data:
        logger.info(f"Training controller on {len(train_data)} samples "
                     f"for {config.controller_epochs} epochs")
        avg_accuracy = 0.5

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

    # --- Evaluation phase ---
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
            avg_accuracy=0.5,
        )
        results.sample_results.append(result)

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

    data = load_dataset_by_name(config.dataset, config.n_samples, config.seed)

    results = ExperimentResults(
        method_name="Multi-Agent Debate",
        dataset_name=config.dataset,
    )

    for sample in tqdm(data, desc="Debate", disable=len(data) < 5):
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
            msgs = comm_graph.get_messages_for_round(texts, t)
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
        )
        results.sample_results.append(sr)

    results.final_W = full_W
    return results


def save_kaggle_checkpoint(results_dict, filepath="kaggle_latest_results.json"):
    try:
        from dataclasses import asdict
        import json
        output = {"detailed_samples": {}}
        for method, res in results_dict.items():
            if res is not None and hasattr(res, "sample_results"):
                output["detailed_samples"][method] = [asdict(s) for s in res.sample_results]
                
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
            
    except Exception as e:
        print(f"Checkpoint save error: {e}")


import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

config = Config()
# ACTIVATE KAGGLE HF MODE NATIVELY
config.llm_backend = "kaggle"
config.api_model = "mistralai/Mistral-7B-Instruct-v0.2"
config.dataset = "gsm8k"
config.n_samples = 100
config.n_train_samples = 50
config.n_rounds = 3

print("Loading Kaggle Environment...")
llm_backend = create_llm_backend(config)
agent_pool = AgentPool(config.n_agents, llm_backend, config.reasoning_styles)
comm_graph = CommunicationGraph(config.n_agents)
controller = IBController(config)

# 1. Dataset Load
total_samples = config.n_train_samples + config.n_samples
all_data = load_dataset_by_name(config.dataset, n_samples=total_samples, seed=config.seed)
train_data = all_data[:config.n_train_samples]
test_data = all_data[config.n_train_samples:]

live_results = {}

print("=== EXECUTING CONTINUOUS CHECKPOINT EVALUATION ===")
# Example: Running Self-Consistency (Independent baseline)
sc_res = ExperimentResults(method_name="Self-Consistency", dataset_name=config.dataset)
for i, sample in enumerate(test_data):
    q = sample["question"]
    a = sample["answer"]
    print(f"\nProcessing Sample {i+1}/{len(test_data)}")
    res = run_single_sample(q, a, agent_pool, comm_graph, controller, config, train_controller=False)
    sc_res.sample_results.append(res)
    
    # Save checkpoint actively during processing step
    live_results["Self-Consistency"] = sc_res
    save_kaggle_checkpoint(live_results)

print("\n✅ Execution Complete! All exact dialogue chains are preserved in kaggle_latest_results.json")
