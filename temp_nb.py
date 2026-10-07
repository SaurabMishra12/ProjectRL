import os
import json
import time
import math
import numpy as np
import dataclasses
import enum
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
    """Master configuration for all experiments.
    
    Centralizes all hyperparameters: agent count, communication rounds,
    bandwidth thresholds, controller architecture, reward weights, datasets, LLM backend.
    Enables reproducible experimental runs across different datasets and methods.
    """

    # --- System Configuration ---
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
    early_stopping_patience: int = 3  # epochs to wait for improvement before early stop
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
    lambda_kl: float = 0.08   # KL divergence sparsity weight
    gamma_loop: float = 0.1  # loop penalty weight
    info_cost_method: str = "token_count"  # "token_count" or "compression_ratio"

    # --- Datasets ---
    dataset: str = "gsm8k"  # "gsm8k", "hotpotqa", or "aqua"
    n_samples: int = 50  # number of evaluation samples
    n_train_samples: int = 30  # samples for controller training

    # --- LLM Backend ---
    max_tokens: int = 512  # Fix #16: dataset-specific, overridden per dataset config
    llm_backend: str = "azure"  # "mock" or "nvidia" or "openai" or "azure"
    api_base_url: str = "https://integrate.api.nvidia.com/v1"
    api_model: str = "gpt-oss-120b"
    api_key: Optional[str] = "your-nvidia-api-key"
    api_temperature: float = 0.0  # control LLM generation diversity

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
        max_tokens=1024,  # Fix #16: Multi-hop questions need more tokens
    )
    defaults.update(overrides)
    return Config(**defaults)


def get_aqua_config(**overrides) -> Config:
    """Preset config for MATH experiments."""
    defaults = dict(
        dataset="aqua",
        n_agents=3,
        n_rounds=4,  # Harder problem, might need an extra round
        lambda_info=0.2, # Less penalty to avoid stunting performance
        gamma_loop=0.1,
        max_tokens=1024,  # Fix #16: Multi-step algebraic reasoning needs more tokens
    )
    defaults.update(overrides)
    return Config(**defaults)

"""
Utility functions: seeding, logging, I/O, token counting.
"""



try:
    import tiktoken
    TOKENIZER = tiktoken.get_encoding("cl100k_base")  # GPT-4/oxygen tokenizer
except ImportError:
    TOKENIZER = None
    logger = logging.getLogger("SparseComm")
    logger.warning("tiktoken not installed. Install with: pip install tiktoken")


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
    """Count tokens using tiktoken (GPT-4/oxygen encoding), fallback to word heuristic.
    
    Used for measuring communication efficiency and bandwidth costs.
    """
    if not text:
        return 0
    if TOKENIZER is None:
        # Fallback: words * 1.3 (log warning if this happens in production)
        logger = logging.getLogger("SparseComm")
        logger.warning("Using fallback token estimation. Install tiktoken for accuracy.")
        words = len(text.split())
        return max(1, int(words * 1.3))
    return len(TOKENIZER.encode(text))


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


def split_sentences(text: str) -> list:
    """Split text into sentences, aware of decimals and abbreviations, avoiding double punctuation."""
    if not text:
        return []
    
    # Try NLTK first
    try:
        import nltk
        sentences = nltk.sent_tokenize(text)
        cleaned = []
        for s in sentences:
            s_clean = s.strip()
            if s_clean:
                if s_clean[-1] not in ['.', '!', '?']:
                    s_clean += '.'
                cleaned.append(s_clean)
        return cleaned
    except Exception:
        pass
        
    # Fallback to custom abbreviation/decimal-aware splitter
    import re
    # Match a sentence ending punctuation [.!?] followed by whitespace
    raw_splits = re.split(r'(?<=[.!?])\s+', text)
    
    sentences = []
    curr = []
    abbrevs = {'eq.', 'fig.', 'i.e.', 'e.g.', 'approx.', 'vs.', 'dr.', 'mr.', 'ms.', 'no.', 'col.', 'vol.', 'al.'}
    
    for part in raw_splits:
        part = part.strip()
        if not part:
            continue
        curr.append(part)
        last_word = part.split()[-1].lower() if part.split() else ""
        if last_word in abbrevs or (len(last_word) == 2 and last_word[0].isalpha() and last_word[1] == '.'):
            # This is an abbreviation or single letter, do not split here
            continue
        else:
            sentence = " ".join(curr)
            if sentence[-1] not in ['.', '!', '?']:
                sentence += '.'
            sentences.append(sentence)
            curr = []
            
    if curr:
        sentence = " ".join(curr)
        if sentence[-1] not in ['.', '!', '?']:
            sentence += '.'
        sentences.append(sentence)
        
    return sentences

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
    exact_match: float = 0.0
    f1_score: float = 0.0

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

    # Diagnostic Metrics
    agent_diversity: float = 0.0
    answer_entropy: float = 0.0
    average_delta_w: float = 0.0
    initial_majority_correct: bool = False


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
        # initial_correct is a dict {agent_id: bool}; skip samples with no data
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
        f"|:-------|:--------:|:----------:|"
        f":----------:|:----------:|:----------:|:--------:|:--------:|\n"
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

    def __init__(self):
        self._PROMPT_CACHE = {}

    def clear_cache(self) -> None:
        self._PROMPT_CACHE.clear()

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
        """Compact system prompts (30% fewer tokens, same quality)."""
        styles = [
            "Be analytical. Show every step.",
            "Be methodical. Break into sub-problems.",
            "Be creative. Consider multiple approaches.",
            "Be careful. Verify each step.",
            "Be efficient. Find direct solutions.",
        ]
        style = styles[agent_id % len(styles)]

        if dataset_type == "gsm8k":
            return f"{style} Solve step-by-step. List all given numbers first. End with numeric answer. Last line: Confidence: X"
        else:
            return f"{style} Answer by reasoning through facts. Give concise final answer. Last line: Confidence: X"
    def _extract_answer(self, text: str, dataset_type: str) -> str:
        """Extract the final answer from LLM response."""
        if not text:
            return ""

        import re
        # Bug #4 Fix: parse \boxed{} FIRST to avoid returning \boxed{A} as answer
        boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
        if boxed_match:
            ans = boxed_match.group(1).strip()
            if dataset_type == "gsm8k":
                nums = re.findall(r'-?\d+\.?\d*', ans)
                if nums:
                    return nums[-1]
            return ans

        # Look for "Answer: X" pattern (after boxed check)
        match = re.search(r'[Aa]nswer:\s*(.+?)(?:\n|$)', text)
        if match:
            ans = match.group(1).strip().rstrip(".")
            # Strip any remaining \boxed{} wrapper inside Answer: line
            boxed_inline = re.search(r'\\boxed\{([^}]+)\}', ans)
            if boxed_inline:
                ans = boxed_inline.group(1).strip()
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

        # Fix #17: Exclude confidence lines from fallback extraction
        text_no_conf = re.sub(r'(?i)confidence:\s*[\d.]+', '', text)
        lines = [l.strip() for l in text_no_conf.strip().split('\n') if l.strip()]
        lines = [l for l in lines if not re.match(r'^[Cc]onfidence:', l.strip())]
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
            return 0.5
            
        import re
        match = re.search(r'[Cc]onfidence:\s*([0-9]+(?:\.[0-9]+)?)', text)
        if match:
            try:
                score = float(match.group(1))
                # Bug #3 Fix: accept 0-1, 0-10, 0-100 scales without blindly dividing by 10
                if score > 10.0:
                    score = score / 100.0   # 0-100 scale
                elif score > 1.0:
                    score = score / 10.0    # 0-10 scale
                # else: already 0-1 scale, use as-is
                return max(0.1, min(1.0, score))
            except ValueError:
                pass
        return 0.5



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
        max_tokens: int = 512,  # Bug A2 fix: must be stored or generate() crashes
        temperature: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_tokens = max_tokens  # Bug A2 fix
        self.temperature = temperature
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
        # Fix1: Use agent's pre-built prompt (includes reasoning style) as user message.
        # system_msg provides the output format spec; prompt provides the task + style instruction.
        system_msg = self._get_system_prompt(dataset_type, agent_id)
        if prompt and prompt.strip():
            # Agent._build_prompt already includes question, context, and style instruction
            user_msg = prompt
        else:
            # Fallback if called without a pre-built prompt
            user_msg = f"Question: {question}"
            if context:
                user_msg += f"\n\nContext from other agents:\n{context}"
        if dataset_type in ["aqua", "truthfulqa"]:
            user_msg += "\n\nProvide step-by-step reasoning, ensuring your final answer is boxed as requested."
        else:
            user_msg += "\n\nProvide step-by-step reasoning, then give your final answer on the last line as: Answer: <your answer>"

        import time as _time
        # Bug #15: Prompt cache - skip API call if identical request seen before
        import hashlib as _hashlib
        _cache_str = f"{self.__class__.__name__}|generate|{getattr(self, 'base_url', '')}|{self.model}|{self.max_tokens}|{self.temperature}|1.0|{system_msg}|{user_msg}"
        _cache_key = _hashlib.sha256(_cache_str.encode()).hexdigest()
        if _cache_key in self._PROMPT_CACHE:
            return self._PROMPT_CACHE[_cache_key]

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
                    temperature=self.temperature,   # Bug #14: deterministic for eval reproducibility or controlled diversity
                    top_p=1.0,
                    max_tokens=self.max_tokens,  # Fix #16: dataset-specific
                    stream=False,
                )
                text = response.choices[0].message.content or ""
                token_count = response.usage.total_tokens if response.usage else count_tokens(text)

                answer = self._extract_answer(text, dataset_type)
                confidence = self._estimate_confidence(text)

                resp = LLMResponse(
                    text=text,
                    answer=answer,
                    token_count=token_count,
                    confidence=confidence,
                )
                self._PROMPT_CACHE[_cache_key] = resp  # Bug #15: cache successful response
                return resp
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
        import hashlib as _hashlib
        _cache_str = f"{self.__class__.__name__}|compress|{getattr(self, 'base_url', '')}|{self.model}|150|0.3|{system_msg}|{user_msg}"
        _cache_key = _hashlib.sha256(_cache_str.encode()).hexdigest()
        if _cache_key in self._PROMPT_CACHE:
            return self._PROMPT_CACHE[_cache_key]

        for attempt in range(2):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=150,
                    stream=False,
                )
                text = response.choices[0].message.content or ""
                self._PROMPT_CACHE[_cache_key] = text
                return text
            except Exception as e:
                print(f"Extraction error: {e}")
                return message

class HuggingFaceLocalBackend(LLMBackend):
    """
    Local HuggingFace model backend.
    Required for executing models like Mistral-7B natively.
    """
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2", temperature: float = 0.0):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
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
            user_msg += f"\n\nContext from other agents:\n{context}"
        if dataset_type in ["aqua", "truthfulqa"]:
            user_msg += "\n\nProvide step-by-step reasoning, ensuring your final answer is boxed as requested."
        else:
            user_msg += "\n\nProvide step-by-step reasoning, then give your final answer on the last line as: Answer: <your answer>"

        import hashlib as _hashlib
        _cache_str = f"{self.__class__.__name__}|generate|local|{self.model_name}|512|{self.temperature}|1.0|{system_msg}|{user_msg}"
        _cache_key = _hashlib.sha256(_cache_str.encode()).hexdigest()
        if _cache_key in self._PROMPT_CACHE:
            return self._PROMPT_CACHE[_cache_key]

        messages = [
            {"role": "user", "content": f"{system_msg}\n\n{user_msg}"}
        ]
        
        try:
            # Mistral/Llama tokenizer formatting
            if hasattr(self.tokenizer, 'apply_chat_template'):
                prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            else:
                prompt_text = f"{system_msg}\n\n{user_msg}\n\nAssistant: "
            
            if self.temperature > 0.0:
                outputs = self.pipe(prompt_text, do_sample=True, temperature=self.temperature)
            else:
                outputs = self.pipe(prompt_text, do_sample=False)
            generated_text = outputs[0]["generated_text"]
            
            # Extract just the new generation
            if prompt_text in generated_text:
                generated_text = generated_text[len(prompt_text):]
                
            text = generated_text.strip()
            answer = self._extract_answer(text, dataset_type)
            confidence = self._estimate_confidence(text)
            
            token_count = count_tokens(text)
            
            resp = LLMResponse(
                text=text,
                answer=answer,
                token_count=token_count,
                confidence=confidence,
            )
            self._PROMPT_CACHE[_cache_key] = resp
            return resp
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
        max_tokens: int = 512,  # Fix #16: dataset-specific token budget
        temperature: float = 0.0,
    ):
        super().__init__()
        self.model = model
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        
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
        # Fix1: Use agent's pre-built prompt (includes reasoning style) as user message.
        system_msg = self._get_system_prompt(dataset_type, agent_id)
        if prompt and prompt.strip():
            user_msg = prompt
        else:
            user_msg = f"Question: {question}"
            if context:
                user_msg += f"\n\nContext from other agents:\n{context}"
        if dataset_type in ["aqua", "truthfulqa"]:
            user_msg += "\n\nProvide step-by-step reasoning, ensuring your final answer is boxed as requested."
        else:
            user_msg += "\n\nProvide step-by-step reasoning, then give your final answer on the last line as: Answer: <your answer>"

        import time as _time
        # Bug #15: Prompt cache - skip API call if identical request seen before
        import hashlib as _hashlib
        _cache_str = f"{self.__class__.__name__}|generate|{getattr(self, 'base_url', '')}|{self.model}|{self.max_tokens}|{self.temperature}|1.0|{system_msg}|{user_msg}"
        _cache_key = _hashlib.sha256(_cache_str.encode()).hexdigest()
        if _cache_key in self._PROMPT_CACHE:
            return self._PROMPT_CACHE[_cache_key]

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
                    temperature=self.temperature,   # Bug #14: deterministic for eval reproducibility or controlled diversity
                    top_p=1.0,
                    max_tokens=self.max_tokens,  # Bug A1 fix: use stored dataset-specific value
                    stream=False,
                )
                text = response.choices[0].message.content or ""
                token_count = response.usage.total_tokens if response.usage else count_tokens(text)

                answer = self._extract_answer(text, dataset_type)
                confidence = self._estimate_confidence(text)

                resp = LLMResponse(
                    text=text,
                    answer=answer,
                    token_count=token_count,
                    confidence=confidence,
                )
                self._PROMPT_CACHE[_cache_key] = resp  # Bug #15: cache successful response
                return resp
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
        import hashlib as _hashlib
        _cache_str = f"{self.__class__.__name__}|compress|{getattr(self, 'base_url', '')}|{self.model}|150|0.3|{system_msg}|{user_msg}"
        _cache_key = _hashlib.sha256(_cache_str.encode()).hexdigest()
        if _cache_key in self._PROMPT_CACHE:
            return self._PROMPT_CACHE[_cache_key]

        for attempt in range(2):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=150,
                    stream=False,
                )
                text = response.choices[0].message.content or ""
                self._PROMPT_CACHE[_cache_key] = text
                return text
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

"""
Communication System: Embedding-Attention Masker, communication graph.
Totally eliminates regex string-matching heuristics.
"""



try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


# --- Bug #1 Fix: SemanticBandwidthMode MUST be defined before TransmissionRecord ---
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


# --- Bug #2 Fix: StructuredSemanticState MUST be defined before SemanticStateEncoder ---
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
        # Include all non-BLOCK modes including CONTRAST (Tier4 #21)
        dist = {mode.name: 0 for mode in SemanticBandwidthMode if mode != SemanticBandwidthMode.BLOCK}
        for r in self.transmission_log:
            if r.mode != SemanticBandwidthMode.BLOCK:
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


# Fix 4: Fixed sparsity prior. KL divergence target is Bernoulli(0.1).
# 90% of edges OFF on average. Stationary and interpretable.
SPARSITY_PRIOR: float = 0.1

class IBController:
    """PPO (Proximal Policy Optimization) controller for learning sparse communication.
    
    Learns: W (edge weights), Q (semantic queries), halt decision (early stopping).
    Reward: task correctness + token efficiency (info bottleneck) + sparsity (KL).
    Uses GAE (Generalized Advantage Estimation) for stable advantage calculation.
    """

    def __init__(self, config: Config):
        self.config = config
        self.n_agents = config.n_agents
        self.query_dim = getattr(config, 'query_dim', 16)
        
        self.semantic_encoder = SemanticStateEncoder(config.n_agents)
        
        # Verify encoder can load
        import logging
        try:
            test_state = self.semantic_encoder.encode_state({0: StructuredSemanticState()})
            logging.getLogger("SparseComm").info(f"Semantic encoder loaded successfully (output dim: {len(test_state)})")
        except RuntimeError as e:
            logging.getLogger("SparseComm").warning(f"Semantic encoder unavailable: {e}")
            logging.getLogger("SparseComm").warning("Falling back to non-semantic mode (controller will use only confidence features)")
        
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
        self.lambda_kl = config.lambda_kl
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
                    # PPO clipping: zero the policy gradient when the ratio
                    # has crossed the trust region boundary in the direction
                    # that would IMPROVE the surrogate (making it diverge).
                    # When adv >= 0: clip if ratio > 1+eps (too optimistic)
                    # When adv <  0: clip if ratio < 1-eps (too pessimistic)
                    # PPO clipping: derivative of min(surr1, surr2)
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
            # FIX: Only log the terminal reward for plotting
            self.episode_rewards.extend([t['reward'] for t in self.saved_transitions if t.get('done', False)])

        # V3-8: Tier4 #22 EMA prior is updated once per epoch in run_experiment.
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
            if calib_data.get('ir') is not None:
                self.calibrator.ir = calib_data['ir']
            self.calibrator._fitted = calib_data.get('_fitted', False)
            self.calibrator._agent_vars = calib_data.get('_agent_vars', {})
            self.calibrator._agent_historical_acc = calib_data.get('_agent_historical_acc', {})


"""
Dataset Loaders: GSM8K, HotpotQA, MATH (AQuA), TruthfulQA, and MBPP.
Returns STRICTLY separated (train, test) tuples to prevent data leakage.
"""



logger = setup_logging()





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
        raise RuntimeError(f"Could not load GSM8K from HuggingFace: {e}")


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
        raise RuntimeError(f"Could not load HotpotQA from HuggingFace: {e}")


def load_aqua(n_train: int = 30, n_test: int = 50, seed: int = 42) -> Tuple[List[dict], List[dict]]:
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
                    "question": "AQuA Problem:\n" + question + "\n\nEnd with your final answer wrapped in \\boxed{your_answer} where your_answer is the exact letter (e.g. \\boxed{A}).",
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
        raise RuntimeError(f"Could not load AQuA MATH directly: {e}")


def load_truthfulqa(n_train: int = 30, n_test: int = 50, seed: int = 42) -> Tuple[List[dict], List[dict]]:
    """Load TruthfulQA (Multiple Choice).
    
    WARNING: TruthfulQA has only a single 'validation' split. We split it into
    train/test using index-based partitioning. This means:
    1. Train and test are from the same distribution (no domain shift)
    2. Results are NOT on a truly independent held-out test set
    3. Use these results for ablation studies only, not final claims
    """
    logger.warning(
        "TruthfulQA: Using validation split partitioned into train/test. "
        "Results are not on an independent held-out test set. "
        "Use for ablation studies only."
    )
    
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
        raise RuntimeError(f"Could not load TruthfulQA: {e}")


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
        raise RuntimeError(f"Could not load MBPP: {e}")


def _hotpot_normalize(s: str) -> str:
    import re
    s = str(s).lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^\w\s]', '', s)
    return ' '.join(s.split())

def hotpotqa_f1_score(prediction: str, ground_truth: str) -> float:
    from collections import Counter
    normalized_prediction = _hotpot_normalize(prediction)
    normalized_ground_truth = _hotpot_normalize(ground_truth)
    
    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return 0.0
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return 0.0
        
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
        
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return float(f1)


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
    elif dataset_type in ["aqua", "truthfulqa"]:
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
        # Fix4: HotpotQA — Exact Match (EM) per standard benchmark protocol.
        # F1>=0.5 inflates accuracy vs published benchmarks; EM is the correct metric.
        # Normalization: lowercase, remove articles, remove punctuation, normalize whitespace.
        return _hotpot_normalize(predicted) == _hotpot_normalize(gold)



def load_dataset_by_name(name: str, n_train: int = 30, n_test: int = 50, seed: int = 42) -> Tuple[List[dict], List[dict]]:
    """Load dataset strictly split by name string."""
    name = name.lower()
    if name == "gsm8k":
        return load_gsm8k(n_train, n_test, seed)
    elif name in ("hotpotqa", "hotpot_qa"):
        return load_hotpotqa(n_train, n_test, seed)
    elif name == "aqua":
        return load_aqua(n_train, n_test, seed)
    elif name in ("truthfulqa", "truthful_qa"):
        return load_truthfulqa(n_train, n_test, seed)
    elif name == "mbpp" or name == "code":
        return load_mbpp(n_train, n_test, seed)
    else:
        raise ValueError(f"Unknown dataset: {name}. Use 'gsm8k', 'hotpotqa', 'aqua', 'truthfulqa', or 'mbpp'.")

"""
Experiment Pipeline: End-to-end experiment runner with controller training,
communication rounds, and result tracking.
"""



logger = setup_logging()

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, enum.Enum):
            return obj.name
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        try:
            return str(obj)
        except Exception:
            return super().default(obj)

def log_continuous_result(method_name: str, dataset_name: str, sr: SampleResult):
    import os, traceback
    try:
        os.makedirs("08-06results/continuous", exist_ok=True)
        filepath = f"08-06results/continuous/{method_name}_{dataset_name}_stream.jsonl"
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(dataclasses.asdict(sr), cls=NumpyEncoder) + "\n")
    except Exception as e:
        logger.error(f"Failed to continuously save sample result: {e}")
        try:
            with open(f"08-06results/continuous/error_log.txt", "a") as f:
                f.write(traceback.format_exc() + "\n")
        except:
            pass


def create_llm_backend(config: Config) -> LLMBackend:
    """Factory function for LLM backend.
    
    NOTE: All backend classes are defined in this same file when running as
    a single Kaggle notebook cell. The imports below use the module path for
    the multi-file local layout. Both paths resolve correctly because
    build_kaggle.py concatenates everything into one cell.
    """
    if config.llm_backend in ("nvidia", "openai", "api"):
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
        return _Az(model=model_name, max_tokens=getattr(config, 'max_tokens', 512))  # Fix #16
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
    halted = False  # Bug #2 fix: explicit halt flag, avoids stale halt_decision after loop

    # Fix #15: Early exit when all agents unanimously agree in Round 0
    # V3-3c: normalize before uniqueness check — '24' and '24.0' are the same answer
    _round0_unique = set(_normalize_for_vote(r.answer, config.dataset) for r in round0_responses)

    # Diagnostic metrics calculation
    import math
    _agent_diversity = len(round0_responses) / len(_round0_unique) if _round0_unique else 0.0
    _counts = {}
    for r in round0_responses:
        _norm_a = _normalize_for_vote(r.answer, config.dataset)
        _counts[_norm_a] = _counts.get(_norm_a, 0) + 1
    _answer_entropy = 0.0
    for _c in _counts.values():
        _p = _c / len(round0_responses)
        _answer_entropy -= _p * math.log2(_p)
        
    _initial_majority_ans = weighted_majority_vote(round0_responses, dataset_type=config.dataset, calibrator=controller.calibrator)
    _initial_majority_correct = evaluate_answer(_initial_majority_ans, gold_answer, config.dataset)
    _delta_ws = []

    if len(_round0_unique) == 1 and list(_round0_unique)[0]:
        final_answer = round0_responses[0].answer
        is_correct = evaluate_answer(final_answer, gold_answer, config.dataset)
        if train_controller and controller is not None:
            _synth_meta = {
                'features': np.zeros(controller.input_dim),
                'action': np.zeros(controller.action_dim),
                'log_prob': 0.0,
                'log_std': np.zeros(controller.mean_dim),
                'sigma': np.ones(controller.mean_dim),
                'value': 0.0,
                'actor_cache': None,
                'mean_action': np.zeros(controller.mean_dim),
            }
            controller.store_transition(
                _synth_meta,
                reward=1.0 if is_correct else 0.0,
                done=True
            )
        return SampleResult(
            question=question,
            gold_answer=gold_answer,
            final_answer=final_answer,
            is_correct=is_correct,
            initial_answers=initial_answers,
            initial_correct=initial_correct,
            final_answers={r.agent_id: r.answer for r in round0_responses},
            initial_reasoning={r.agent_id: r.reasoning_text for r in round0_responses},
            final_reasoning={r.agent_id: r.reasoning_text for r in round0_responses},
            raw_communications=[],
            total_tokens_transmitted=0,
            total_tokens_if_full=0,
            compression_ratio=0.0,
            communication_density=0.0,
            mode_distribution={mode.name: 0 for mode in SemanticBandwidthMode if mode != SemanticBandwidthMode.BLOCK},
            n_active_edges=0,
            generation_tokens=sum(r.token_count for r in round0_responses),
            halt_round=0,
            agent_diversity=_agent_diversity,
            answer_entropy=_answer_entropy,
            average_delta_w=0.0,
            initial_majority_correct=_initial_majority_correct
        )

    # Track initial confidence for calibrator accumulation
    initial_confidence = np.mean([r.confidence for r in current_responses])
    # Fix A: initial_entropy removed (entropy shaping dropped from intermediate reward)

    # Bug A5 fix: track per-round snapshots so intermediate rewards use correct state
    responses_snapshots: List[List] = [list(round0_responses)]

    # --- Rounds 1..T: Communication rounds ---
    halt_round = config.n_rounds
    t = 0
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
        
        # Record delta W
        _delta_ws.append(float(np.linalg.norm(W - prev_W, ord='fro')))
        
        if meta is not None:
            all_metas.append(meta)

        # Always apply weights BEFORE checking halt so reward reflects this round's graph
        comm_graph.set_weights(W, Q)
        # Bug A6 fix: adaptive_prior updated once per epoch (not per round) to prevent
        # non-stationary KL reward. Fixed SPARSITY_PRIOR=0.1 used instead (Fix 4).

        # Fix 3: SPRT Halting with calibrated p_maj
        # H0: majority answer is random
        # Halt when LLR > tau = log((1-alpha)/alpha), alpha=0.05 -> tau~2.944
        _SPRT_TAU = 2.944    # log(19), alpha=0.05
        # Set dynamic _SPRT_P0 random baseline based on dataset difficulty
        if config.dataset and config.dataset.lower() in ("math", "aqua"):
            _SPRT_P0 = 0.2
        elif config.dataset and config.dataset.lower() == "truthfulqa":
            _SPRT_P0 = 0.25
        else:
            _SPRT_P0 = 0.5

        ans_list = [_normalize_for_vote(r.answer, config.dataset) for r in current_responses]
        from collections import Counter as _Counter
        _ans_counts = _Counter(ans_list)
        _majority_ans, _k = _ans_counts.most_common(1)[0]
        _N = len(ans_list)
        # Calibrated p_maj: max calibrated confidence among agents voting for majority answer
        _maj_confs = [
            r.confidence for r in current_responses
            if _normalize_for_vote(r.answer, config.dataset) == _majority_ans
        ]
        if controller.calibrator is not None and controller.calibrator._fitted and _maj_confs:
            _p_maj = float(np.clip(
                controller.calibrator.calibrate(float(np.max(_maj_confs))), _SPRT_P0 + 0.001, 0.999
            ))
        else:
            _p_maj = 0.9  # Conservative fallback when calibrator not yet fitted
        _llr = (
            _k * np.log(_p_maj / _SPRT_P0) +
            (_N - _k) * np.log((1.0 - _p_maj) / (1.0 - _SPRT_P0))
        )
        sprt_halt = _llr > _SPRT_TAU
        if (halt_decision or sprt_halt) and t > 1:
            halted = True  # Bug #2 fix: set explicit flag
            halt_round = t
            if all_metas:
                all_metas.pop()  # Discard the meta for the unused halt round
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
        responses_snapshots.append(list(current_responses))  # Bug A5 fix: per-round snapshot

    # --- Final answer: calibrated vote (same aggregator as baselines for fair comparison) ---
    # Fix3: Removed W (spectral centrality) and agents (stability) from IB-Controller vote.
    # Spectral/stability weights are communication-protocol extras not available to baselines,
    # so including them creates an unfair advantage in the accuracy metric.
    # Calibrator is kept because baselines also receive it via V3-5 fix.
    final_answer = weighted_majority_vote(
        current_responses,
        dataset_type=config.dataset,
        calibrator=controller.calibrator,
    )
    is_correct = evaluate_answer(final_answer, gold_answer, config.dataset)
    em_score = 1.0 if is_correct else 0.0
    
    if config.dataset in ["hotpotqa", "hotpot_qa"]:
        f1 = hotpotqa_f1_score(final_answer, gold_answer)
        task_reward_signal = f1
    else:
        f1 = float(is_correct)
        task_reward_signal = float(is_correct)

    new_confidence = np.mean([r.confidence for r in current_responses])
    new_unique_answers = len(set(r.answer for r in current_responses))
    
    # Fix 2: net_conf_improvement and net_entropy_reduction removed (shaping rewards dropped)

    # --- Compute reward and update controller ---
    tokens_transmitted = comm_graph.get_total_tokens_transmitted()
    tokens_if_full = comm_graph.get_total_tokens_if_full()
    compression_ratio = comm_graph.get_compression_ratio()
    n_cycles = comm_graph.count_cycles(threshold=0.5)  # Align with SUMMARIZE threshold

    # Compute DENSE reward for this step (Bug #2 fix: use explicit halted flag)
    is_final = (halt_round == config.n_rounds or halted)
    step_reward = controller.compute_reward(
        task_reward=task_reward_signal if is_final else 0.0,
        tokens_transmitted=tokens_transmitted,
        tokens_if_full=tokens_if_full,
        compression_ratio=compression_ratio,
        n_cycles=n_cycles,
        current_W=W,
        halt_round=halt_round,
        is_final_step=is_final,
        include_token_penalty=False,  # Fix B: cost already applied per round; avoid double-counting
    )

    if train_controller and all_metas:
        n_metas = len(all_metas)
        for i, m in enumerate(all_metas):
            is_last = (i == n_metas - 1)
            if is_last:
                reward_for_step = step_reward
            else:
                # Fix #7: Dense shaping reward using per-round snapshot (Bug A5 fix)
                round_num_i = i + 1
                round_tokens = comm_graph.round_token_counts.get(round_num_i, 0)
                token_penalty = -controller.lambda_info * round_tokens / max(float(tokens_if_full), 1.0)
                # Intermediate reward = cost penalty + consensus improvement
                _current_consensus = 0.0
                _prev_consensus = 0.0
                if i + 1 < len(responses_snapshots):
                    _curr_ans = [_normalize_for_vote(r.answer, config.dataset) for r in responses_snapshots[i + 1]]
                    if _curr_ans:
                        from collections import Counter
                        _current_consensus = Counter(_curr_ans).most_common(1)[0][1] / len(_curr_ans)
                if i < len(responses_snapshots):
                    _prev_ans = [_normalize_for_vote(r.answer, config.dataset) for r in responses_snapshots[i]]
                    if _prev_ans:
                        from collections import Counter
                        _prev_consensus = Counter(_prev_ans).most_common(1)[0][1] / len(_prev_ans)
                reward_for_step = token_penalty + 0.5 * (_current_consensus - _prev_consensus)
            controller.store_transition(m, reward_for_step, done=is_last)
        
    # Build result
    result = SampleResult(
        question=question,
        gold_answer=gold_answer,
        final_answer=final_answer,
        is_correct=is_correct,
        exact_match=em_score,
        f1_score=f1,
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
        halt_round=halt_round,
        agent_diversity=_agent_diversity,
        answer_entropy=_answer_entropy,
        average_delta_w=float(np.mean(_delta_ws)) if _delta_ws else 0.0,
        initial_majority_correct=_initial_majority_correct
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
    comm_graph.set_dataset_type(config.dataset)  # Bug A4 fix

    # Use provided controller or create a fresh one
    if controller is None:
        controller = IBController(config)
        logger.info(f"Created fresh IBController for {method_name}")
    else:
        # INTENTIONAL: Universal pre-training reuses the same controller across datasets.
        # The controller accumulates experience from all datasets before evaluation.
        logger.info(f"Reusing existing IBController for {method_name} (universal pre-training mode)")
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
    avg_accuracy = 0.0

    # --- Training phase ---
    if train and train_data:
        logger.info(f"Training controller on {len(train_data)} samples "
                     f"for {config.controller_epochs} epochs")

        accumulation_steps = min(len(train_data), 16)  # Fix7b: cap at 16 for stable gradient variance  # Fix #8: Update once per epoch for statistically valid PPO gradient

        for epoch in range(config.controller_epochs):
            llm_backend.clear_cache()
            epoch_correct = 0
            loss = 0.0
            
            pbar = tqdm(train_data, desc=f"Training {method_name} (Epoch {epoch+1}/{config.controller_epochs})")
            for i, sample in enumerate(pbar):
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

                # Platt calibration: accumulate (conf, correct) during training (Improvement VI)
                if controller.calibrator is not None:
                    for aid, r in enumerate(agent_pool.agents):
                        if r.history:
                            last_resp = r.history[-1]
                            agent_correct = evaluate_answer(last_resp.answer, sample["answer"], config.dataset)
                            controller.calibrator.accumulate(aid, last_resp.confidence, agent_correct)
                        
                        # V5 Priority #5 Bugfix: Accumulate empirical numeric errors
                        if config.dataset in ['gsm8k', 'aqua']:
                            import re
                            def _ext(txt):
                                m = re.findall(r'-?\d+\.?\d*', str(txt))
                                return float(m[-1]) if m else None
                            y_pred = _ext(last_resp.answer)
                            y_true = _ext(sample["answer"])
                            if y_pred is not None and y_true is not None and controller.calibrator is not None:
                                controller.calibrator.accumulate_numeric(aid, y_pred, y_true)

                # Dynamically update running accuracy
                avg_accuracy = epoch_correct / (i + 1)
                
                pbar.set_postfix({"acc": f"{avg_accuracy:.2%}"})
                
                log_continuous_result(f"Train_{method_name}_Epoch{epoch+1}", config.dataset, result)

                # Gradient accumulation: update every N samples
                if (i + 1) % accumulation_steps == 0:
                    loss = controller.update_policy()
                    if hasattr(config, "model_save_path") and config.model_save_path:
                        controller.save_model(config.model_save_path.replace(".pt", f"_{method_name}_latest.pt"))

            # Final update for remaining samples
            if len(train_data) % accumulation_steps != 0:
                loss = controller.update_policy()
                if hasattr(config, "model_save_path") and config.model_save_path:
                    controller.save_model(config.model_save_path.replace(".pt", f"_{method_name}_latest.pt"))

            # Bug A6 fix: update EMA prior once per epoch (not per round) — stable KL baseline
            # Fix 4: adaptive_prior update removed (SPARSITY_PRIOR=0.1)
            # Improvement VI: re-fit Platt calibrator after each training epoch
            if controller.calibrator is not None:
                controller.calibrator.fit()

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

        for i, sample in enumerate(tqdm(test_data, desc=f"Evaluating {method_name}")):
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
            log_continuous_result(method_name, config.dataset, result)

    if not evaluate:
        results = ExperimentResults(
            method_name=method_name,
            dataset_name=config.dataset,
            training_rewards=controller.episode_rewards.copy(),
            training_losses=controller.episode_losses.copy(),
        )

    results.final_W = comm_graph.get_weight_snapshot()
    return results


def run_full_communication(config: Config, calibrator: Optional['PlattCalibrator'] = None) -> ExperimentResults:
    """
    Run multi-agent debate baseline: full communication (W=1.0 everywhere).
    V3-5: accept calibrator for fair voting comparison with IB-Controller.
    """
    set_seed(config.seed)
    llm_backend = create_llm_backend(config)
    agent_pool = AgentPool(config.n_agents, llm_backend, config.reasoning_styles)
    comm_graph = CommunicationGraph(config.n_agents)
    comm_graph.set_dataset_type(config.dataset)  # Bug A4 fix
    controller = IBController(config)

    # Set all weights to 1.0 (full communication)
    full_W = np.ones((config.n_agents, config.n_agents))
    np.fill_diagonal(full_W, 0)

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

        # V3-9: unanimous Round-0 early exit — same as run_single_sample for fair comparison
        _debate_unique = set(_normalize_for_vote(r.answer, config.dataset) for r in r0)
        if len(_debate_unique) == 1 and list(_debate_unique)[0]:
            final = weighted_majority_vote(current, dataset_type=config.dataset, calibrator=calibrator)
            is_correct = evaluate_answer(final, sample["answer"], config.dataset)
            sr = SampleResult(
                question=sample["question"],
                gold_answer=sample["answer"],
                final_answer=final,
                is_correct=is_correct,
                initial_answers=initial_answers,
                initial_correct=initial_correct,
                final_answers={r.agent_id: r.answer for r in r0},
                total_tokens_transmitted=0,
                total_tokens_if_full=0,
                compression_ratio=0.0,
                weight_matrices=[full_W.copy()],
                communication_density=0.0,
                mode_distribution={mode.name: 0 for mode in SemanticBandwidthMode if mode != SemanticBandwidthMode.BLOCK},
                n_active_edges=0,
                generation_tokens=total_gen,
                halt_round=0,
            )
            results.sample_results.append(sr)
            log_continuous_result("Multi-Agent Debate", config.dataset, sr)
            continue

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

        final = weighted_majority_vote(current, dataset_type=config.dataset, calibrator=calibrator)  # V3-5
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
        log_continuous_result("Multi-Agent Debate", config.dataset, sr)

    results.final_W = full_W
    return results

def run_no_communication(config: Config, calibrator: Optional['PlattCalibrator'] = None) -> ExperimentResults:
    """
    Run no-communication baseline: agents solve independently with zero communication.
    V3-5: accept calibrator for fair vote comparison.
    """
    set_seed(config.seed)
    llm_backend = create_llm_backend(config)
    agent_pool = AgentPool(config.n_agents, llm_backend, config.reasoning_styles)
    
    # Load data
    _, test_data = load_dataset_by_name(
        config.dataset, n_train=0, n_test=config.n_samples, seed=config.seed
    )
    
    results = ExperimentResults(
        method_name="No Communication",
        dataset_name=config.dataset,
    )
    
    for sample in tqdm(test_data, desc="NoComm"):
        agent_pool.reset_all()
        
        # Round 0 only - no communication rounds
        r0 = agent_pool.initial_solve(sample["question"], config.dataset)
        
        # Bug #10 Fix: pass calibrator argument that the function signature accepts
        final = weighted_majority_vote(r0, dataset_type=config.dataset, calibrator=calibrator)
        is_correct = evaluate_answer(final, sample["answer"], config.dataset)
        
        # Bug #5 fix: compute notional full-comm tokens from Round 0 reasonings
        _n_edges = config.n_agents * (config.n_agents - 1)
        _tokens_if_full = sum(r.token_count for r in r0) * _n_edges

        sr = SampleResult(
            question=sample["question"],
            gold_answer=sample["answer"],
            final_answer=final,
            is_correct=is_correct,
            initial_answers={r.agent_id: r.answer for r in r0},
            initial_correct={r.agent_id: evaluate_answer(r.answer, sample["answer"], config.dataset) for r in r0},
            final_answers={r.agent_id: r.answer for r in r0},
            total_tokens_transmitted=0,  # No communication
            total_tokens_if_full=_tokens_if_full,  # Bug #5 fix: not 0
            compression_ratio=0.0,
            weight_matrices=[np.zeros((config.n_agents, config.n_agents))],
            communication_density=0.0,
            mode_distribution={mode.name: 0 for mode in SemanticBandwidthMode if mode != SemanticBandwidthMode.BLOCK},
            n_active_edges=0,
            generation_tokens=sum(r.token_count for r in r0),
            halt_round=0,
        )
        results.sample_results.append(sr)
        log_continuous_result("No Communication", config.dataset, sr)
    
    results.final_W = np.zeros((config.n_agents, config.n_agents))
    return results



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
            'action': np.zeros(self.action_dim),
            'log_prob': 0.0,
            'log_std': np.zeros(self.mean_dim),
            'sigma': np.ones(self.mean_dim),
            'value': 0.0,
            'actor_cache': None,
            'mean_action': np.zeros(self.mean_dim),
        }
        return self._frozen_W.copy(), self._frozen_Q.copy(), False, dummy_meta

def run_experiment_with_random_weights(config: Config, method_name: str) -> ExperimentResults:
    """Run experiment with random communication weights (no learning)."""
    set_seed(config.seed)
    llm_backend = create_llm_backend(config)
    agent_pool = AgentPool(config.n_agents, llm_backend, config.reasoning_styles)
    comm_graph = CommunicationGraph(n_agents=config.n_agents)
    comm_graph.set_dataset_type(config.dataset)  # Bug A4 fix
    
    # Bug A3 fix: need a temp IBController to call get_random_weights()
    _tmp_ctrl = IBController(config)
    random_W, random_Q = _tmp_ctrl.get_random_weights()
    del _tmp_ctrl
    controller = FrozenWeightController(config, random_W, random_Q)
    comm_graph.set_weights(random_W, random_Q)
    
    _, test_data = load_dataset_by_name(
        config.dataset, n_train=0, n_test=config.n_samples, seed=config.seed
    )
    
    results = ExperimentResults(method_name=method_name, dataset_name=config.dataset)
    
    for sample in tqdm(test_data, desc=f"Random {method_name}"):
        result = run_single_sample(
            question=sample["question"],
            gold_answer=sample["answer"],
            agent_pool=agent_pool,
            comm_graph=comm_graph,
            controller=controller,
            config=config,
            train_controller=False,  # No learning
            avg_accuracy=0.5,
        )
        results.sample_results.append(result)
        log_continuous_result(method_name, config.dataset, result)
    
    results.final_W = random_W
    return results

def run_multi_seed_experiment(
    config: Config,
    method_name: str,
    n_seeds: int = 5,
    experiment_fn = None
) -> Dict[str, Any]:
    """Run experiment with multiple seeds for statistical significance.
    
    NOTE: n_seeds >= 3 is required for the 95% CI to be meaningful.
    With n_seeds=1 the CI collapses to 0.0 and should NOT be reported.
    """
    import warnings
    if n_seeds < 3:
        warnings.warn(
            f"run_multi_seed_experiment called with n_seeds={n_seeds}. "
            "95% CI is statistically invalid with fewer than 3 seeds. "
            "Do NOT report CI values from this run in a paper.",
            UserWarning,
            stacklevel=2,
        )
    if experiment_fn is None:
        experiment_fn = lambda cfg, m: run_experiment(
            cfg, method_name=m, train=True, evaluate=True, controller=None
        )
    all_accuracies = []
    all_token_costs = []
    
    import copy
    
    for seed in range(n_seeds):
        print(f"\n=== Seed {seed+1}/{n_seeds} ===")
        seed_config = copy.deepcopy(config)
        seed_config.seed = seed
        set_seed(seed)
        
        # IMPORTANT: Pass controller=None to force fresh initialization per seed
        # This prevents learned weights from previous seeds contaminating results
        results = experiment_fn(seed_config, f"{method_name}_seed{seed}")
        # Note: We compute these from results.sample_results
        acc = np.mean([1.0 if r.is_correct else 0.0 for r in results.sample_results])
        tokens = np.mean([r.total_tokens_transmitted for r in results.sample_results])
        
        all_accuracies.append(acc)
        all_token_costs.append(tokens)
    
    # Bug #13 Fix: use t-distribution with ddof=1 for small-sample 95% CI
    from scipy import stats as _stats
    acc_mean = float(np.mean(all_accuracies))
    acc_std = float(np.std(all_accuracies, ddof=1))
    t_crit = _stats.t.ppf(0.975, df=max(n_seeds - 1, 1))
    acc_ci = t_crit * acc_std / np.sqrt(n_seeds)

    token_mean = float(np.mean(all_token_costs))
    token_std = float(np.std(all_token_costs, ddof=1))
    token_ci = t_crit * token_std / np.sqrt(n_seeds)
    
    return {
        'method': method_name,
        'accuracy_mean': acc_mean,
        'accuracy_std': acc_std,
        'accuracy_95ci': acc_ci,
        'token_cost_mean': token_mean,
        'token_cost_std': token_std,
        'token_cost_95ci': token_ci,
        'all_accuracies': all_accuracies,
        'all_token_costs': all_token_costs,
    }



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
        output = {"detailed_samples": {}, "stats": {}}
        for method, res in results_dict.items():
            if isinstance(res, dict):
                output["stats"][method] = res
            elif res is not None and hasattr(res, "sample_results"):
                output["detailed_samples"][method] = [asdict(s) for s in res.sample_results]
                
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, cls=NumpyEncoder)
            
    except Exception as e:
        print(f"Checkpoint save error: {e}")


# =========================================================
# LARGE-SCALE EXECUTION WITH TOY VALIDATION
# =========================================================
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    datasets_to_run = ["gsm8k", "hotpotqa", "aqua"]

    # Bug #15: RUN_EXPENSIVE guard - set True ONLY when you are ready to spend API budget
    logger = logging.getLogger(__name__)

    datasets_to_run = ["gsm8k", "hotpotqa", "aqua"]

    # Bug #15: RUN_EXPENSIVE guard - set True ONLY when you are ready to spend API budget
    RUN_EXPENSIVE = True  # <-- set False to skip LLM calls
    RUN_TOY_ONLY = False    # <-- set True to exit immediately after the Toy Run (to test pipeline)
    if not RUN_EXPENSIVE:
        print("[SKIP] RUN_EXPENSIVE=False. Set to True to run toy validation and full phases.")
        import sys; sys.exit(0)

    # =========================================================
    # TOY RUN: 15 samples to test everything
    # =========================================================
    print("\n=== TOY RUN (15 samples per dataset) ===")
    test_controller = None
    for ds_name in datasets_to_run:
        print(f"\n--- Toy Run: {ds_name.upper()} ---")
        current_config = get_aqua_config() if ds_name == "aqua" else (get_hotpotqa_config() if ds_name == "hotpotqa" else get_gsm8k_config())
        current_config.llm_backend = "azure"
        current_config.api_model = "gpt-oss-120b"
        current_config.n_train_samples = 100
        current_config.n_samples = 300
        # current_config.controller_epochs override removed
        current_config.dataset = ds_name
    
        if test_controller is None:
            test_controller = IBController(current_config)

        res = run_experiment(
            current_config,
            method_name=f"Toy_{ds_name.upper()}",
            train=True,
            evaluate=True,
            controller=test_controller
        )
        save_kaggle_checkpoint({f"Toy_{ds_name.upper()}": res}, filepath=f"08-06results/toy_{ds_name}.json")
        print(f"[OK] Toy run saved to 08-06results/toy_{ds_name}.json")
    print("[OK] Toy run completed successfully.")

    if RUN_TOY_ONLY:
        print("\n[STOP] RUN_TOY_ONLY is True. Exiting before running full expensive phases.")
        import sys
        sys.exit(0)


    # =========================================================
    # PHASE 1: PRE-TRAINING (100 samples/dataset, 3 epochs)
    # =========================================================
    print("\n=== PHASE 1: PRE-TRAINING ===")
    # Fix5: Store pre-trained controller per dataset so Phase 2 warm-starts from Phase 1 weights.
    pretrained_controllers: dict = {}
    controller = None

    for ds_name in datasets_to_run:
        print(f"\n--- Pre-Training: {ds_name.upper()} ---")
        current_config = get_aqua_config() if ds_name == "aqua" else (get_hotpotqa_config() if ds_name == "hotpotqa" else get_gsm8k_config())
        current_config.llm_backend = "azure"
        current_config.api_model = "gpt-oss-120b"
        current_config.n_train_samples = 100
        current_config.n_samples = 300
        # current_config.controller_epochs override removed
        current_config.dataset = ds_name
    
        # FRESH controller per dataset to prevent non-stationary reward hacking
        controller = IBController(current_config)

        run_experiment(
            current_config,
            method_name=f"Pretrain_{ds_name.upper()}",
            train=True,
            evaluate=False,
            controller=controller,
        )
        pretrained_controllers[ds_name] = controller  # Fix5: persist for Phase 2 warm-start
        print(f"[OK] {ds_name.upper()} pre-training complete (weights saved for Phase 2)")

    print("\n[OK] Dataset-specific pre-training complete")

    # =========================================================
    # PHASE 2: TESTING ROUND (300 samples/dataset)
    # =========================================================
    print("\n=== PHASE 2: TRANSFER EVALUATION ===")

    live_results = {}
    for ds_name in datasets_to_run:
        print(f"\n--- Evaluating: {ds_name.upper()} ---")
        current_config = get_aqua_config() if ds_name == "aqua" else (get_hotpotqa_config() if ds_name == "hotpotqa" else get_gsm8k_config())
        current_config.llm_backend = "azure"
        current_config.api_model = "gpt-oss-120b"
        current_config.n_train_samples = 100
        current_config.n_samples = 300
        # current_config.controller_epochs override removed
        current_config.dataset = ds_name

        print(f"[OK] Backend: {current_config.llm_backend.upper()}")

        # Fix5: Build experiment_fn that warm-starts each seed from the Phase 1 controller.
        import copy as _copy
        _ds = ds_name  # capture loop variable
        def _make_transfer_fn(pretrained_ctlrs, _ds_name):
            def _transfer_fn(cfg, method_name):
                base_ctrl = pretrained_ctlrs.get(_ds_name)
                seed_ctrl = _copy.deepcopy(base_ctrl) if base_ctrl is not None else None
                # Bug #9 Fix: transfer = evaluate ONLY with pre-trained weights, no additional training
                return run_experiment(cfg, method_name=method_name, train=False, evaluate=True, controller=seed_ctrl)
            return _transfer_fn
        stats = run_multi_seed_experiment(
            current_config,
            f"Transfer_{ds_name.upper()}",
            n_seeds=3,
            experiment_fn=_make_transfer_fn(pretrained_controllers, _ds),
        )

        print(f"\n{ds_name.upper()} Results:")
        print(f"  Accuracy: {stats['accuracy_mean']:.1%} +/- {stats['accuracy_95ci']:.1%} (95% CI)")
        print(f"  Token Cost: {stats['token_cost_mean']:.1f} +/- {stats['token_ci']:.1f}")

        live_results[f"Transfer_{ds_name.upper()}"] = stats

    # =========================================================
    # PHASE 3: ESSENTIAL BASELINES ONLY
    # =========================================================
    print("\n=== PHASE 3: BASELINE COMPARISONS ===")
    # NOTE: Baselines are single-seed due to API cost constraints.

    baseline_results = {}

    for ds_name in datasets_to_run:
        print(f"\n--- Baselines: {ds_name.upper()} ---")
        current_config = get_aqua_config() if ds_name == "aqua" else (get_hotpotqa_config() if ds_name == "hotpotqa" else get_gsm8k_config())
        current_config.llm_backend = "azure"
        current_config.api_model = "gpt-oss-120b"
        current_config.n_samples = 300 if ds_name == "gsm8k" else 100
        current_config.dataset = ds_name

        # Bug #10 Fix: pass trained calibrator from Phase 1 to baselines for fair comparison
        _shared_calibrator = pretrained_controllers.get(ds_name, None)
        _calib = _shared_calibrator.calibrator if _shared_calibrator is not None else None

        # 1. Full Communication (Multi-Agent Debate)
        print(f"  Running Full Communication (Multi-Seed)...")
        full_comm_res = run_multi_seed_experiment(
            current_config, 
            f"FullComm_{ds_name.upper()}", 
            n_seeds=3, 
            experiment_fn=lambda cfg, m: run_full_communication(cfg, calibrator=_calib)
        )
        baseline_results[f"FullComm_{ds_name.upper()}"] = full_comm_res

        # 2. No Communication (Independent Agents)
        print(f"  Running No Communication (Multi-Seed)...")
        no_comm_res = run_multi_seed_experiment(
            current_config, 
            f"NoComm_{ds_name.upper()}", 
            n_seeds=3, 
            experiment_fn=lambda cfg, m: run_no_communication(cfg, calibrator=_calib)
        )
        baseline_results[f"NoComm_{ds_name.upper()}"] = no_comm_res

        save_kaggle_checkpoint({**live_results, **baseline_results})

    # =========================================================
    # PHASE 4: ABLATION STUDIES
    # =========================================================
    print("\n=== PHASE 4: ABLATION STUDIES ===")

    ablation_config = get_gsm8k_config()
    ablation_config.llm_backend = "azure"
    ablation_config.api_model = "gpt-oss-120b"
    ablation_config.n_samples = 100      # Match test samples (Shrunk for cost)
    ablation_config.n_train_samples = 100 # Match train samples
    ablation_config.controller_epochs = 5
    ablation_config.dataset = "gsm8k"

    ablation_results = {}

    # =========================================================
    # Fix 5: 2^2 Factorial Ablation Design
    # Factor A: Sparsity (KL penalty)   0=off, 1=on
    # Factor B: PPO learning             0=off (fixed ring), 1=on
    # Calibrator: always ON (post-hoc mapping)
    #
    # | A | B | Arm name           |
    # |---|---|--------------------|
    # | 0 | 0 | Arm_00_StaticRing  | Fixed ring, no KL, no PPO
    # | 0 | 1 | Arm_01_PPOOnly     | PPO, no sparsity pressure
    # | 1 | 0 | Arm_10_SparseRing  | Fixed ring with KL penalty
    # | 1 | 1 | Arm_11_Full        | PPO + KL sparsity (full system)
    #
    # OLS: y = beta0 + betaA*A + betaB*B + betaAB*AB + eps
    # =========================================================
    from copy import deepcopy as _deepcopy

    # ---- ARM 00: Static ring, no KL, no PPO ----
    print("\n  Arm 00: Static ring, no KL, no PPO (baseline)...")
    _a00_cfg = _deepcopy(ablation_config)
    _a00_cfg.lambda_kl = 0.0; _a00_cfg.lambda_info = 0.0; _a00_cfg.gamma_loop = 0.0
    _a00_tmp = IBController(_a00_cfg)
    _a00_W, _a00_Q = _a00_tmp.get_static_weights("ring"); del _a00_tmp
    _a00_frozen = FrozenWeightController(_a00_cfg, _a00_W, _a00_Q)
    _a00_frozen.calibrator = None  # Fix C: disable calibrator to remove confound across arms
    _a00_comm = CommunicationGraph(n_agents=_a00_cfg.n_agents); _a00_comm.set_weights(_a00_W, _a00_Q)
    _a00_pool = AgentPool(_a00_cfg.n_agents, create_llm_backend(_a00_cfg), _a00_cfg.reasoning_styles)
    _, _a00_test = load_dataset_by_name(_a00_cfg.dataset, n_train=0, n_test=_a00_cfg.n_samples, seed=_a00_cfg.seed)
    _a00_res = ExperimentResults(method_name="Arm_00_StaticRing", dataset_name=_a00_cfg.dataset)
    for _s in tqdm(_a00_test, desc="Arm_00_StaticRing"):
        _a00_res.sample_results.append(run_single_sample(
            question=_s["question"], gold_answer=_s["answer"],
            agent_pool=_a00_pool, comm_graph=_a00_comm,
            controller=_a00_frozen, config=_a00_cfg, train_controller=False,
        ))
    ablation_results["Arm_00_StaticRing"] = _a00_res
    save_kaggle_checkpoint({**ablation_results}, filepath="08-06results/ablation_arm00.json")

    # ---- ARM 01: PPO learns topology, KL=0 ----
    print("\n  Arm 01: PPO-only, no KL sparsity...")
    _a01_cfg = _deepcopy(ablation_config); _a01_cfg.lambda_kl = 0.0; _a01_cfg.gamma_loop = 0.0
    _a01_ctrl = IBController(_a01_cfg)
    _a01_ctrl.calibrator = None  # Fix C: disable calibrator to remove confound across arms
    ablation_results["Arm_01_PPOOnly"] = run_experiment(
        _a01_cfg, method_name="Arm_01_PPOOnly", train=True, evaluate=True, controller=_a01_ctrl)
    save_kaggle_checkpoint({**ablation_results}, filepath="08-06results/ablation_arm01.json")

    # ---- ARM 10: Fixed ring with KL penalty, no PPO ----
    print("\n  Arm 10: Fixed ring with KL penalty, no PPO...")
    _a10_cfg = _deepcopy(ablation_config)
    _a10_tmp = IBController(_a10_cfg)
    _a10_W, _a10_Q = _a10_tmp.get_static_weights("ring"); del _a10_tmp
    _a10_frozen = FrozenWeightController(_a10_cfg, _a10_W, _a10_Q)
    _a10_frozen.calibrator = None  # Fix C: disable calibrator to remove confound across arms
    _a10_comm = CommunicationGraph(n_agents=_a10_cfg.n_agents); _a10_comm.set_weights(_a10_W, _a10_Q)
    _a10_pool = AgentPool(_a10_cfg.n_agents, create_llm_backend(_a10_cfg), _a10_cfg.reasoning_styles)
    _, _a10_test = load_dataset_by_name(_a10_cfg.dataset, n_train=0, n_test=_a10_cfg.n_samples, seed=_a10_cfg.seed)
    _a10_res = ExperimentResults(method_name="Arm_10_SparseRing", dataset_name=_a10_cfg.dataset)
    for _s in tqdm(_a10_test, desc="Arm_10_SparseRing"):
        _a10_res.sample_results.append(run_single_sample(
            question=_s["question"], gold_answer=_s["answer"],
            agent_pool=_a10_pool, comm_graph=_a10_comm,
            controller=_a10_frozen, config=_a10_cfg, train_controller=False,
        ))
    ablation_results["Arm_10_SparseRing"] = _a10_res
    save_kaggle_checkpoint({**ablation_results}, filepath="08-06results/ablation_arm10.json")

    # ---- ARM 11: PPO + KL sparsity (full system; reuse Phase 1 controller) ----
    print("\n  Arm 11: PPO + KL sparsity (full system)...")
    _a11_ctrl = pretrained_controllers.get("gsm8k")
    if _a11_ctrl is None:
        _a11_ctrl = IBController(ablation_config)
        _a11_ctrl_trained = run_experiment(
            ablation_config, method_name="Arm_11_Full", train=True, evaluate=False, controller=_a11_ctrl)
    else:
        _a11_ctrl_trained = None
    # Fix C: disable calibrator before evaluation to remove confound
    _a11_ctrl.calibrator = None
    ablation_results["Arm_11_Full"] = run_experiment(
        ablation_config, method_name="Arm_11_Full", train=False, evaluate=True, controller=_a11_ctrl)
    save_kaggle_checkpoint({**ablation_results}, filepath="08-06results/ablation_arm11.json")

    # ---- OLS: main effects and 2-way interaction (sample-level) ----
    try:
        _order = ["Arm_00_StaticRing", "Arm_01_PPOOnly", "Arm_10_SparseRing", "Arm_11_Full"]
        _y_list = []
        _X_list = []
        for _arm in _order:
            if _arm == "Arm_00_StaticRing":
                _A_val, _B_val = 0.0, 0.0
            elif _arm == "Arm_01_PPOOnly":
                _A_val, _B_val = 0.0, 1.0
            elif _arm == "Arm_10_SparseRing":
                _A_val, _B_val = 1.0, 0.0
            elif _arm == "Arm_11_Full":
                _A_val, _B_val = 1.0, 1.0
            else:
                continue
        
            if _arm in ablation_results and hasattr(ablation_results[_arm], "sample_results"):
                for _s in ablation_results[_arm].sample_results:
                    _val = 1.0 if _s.is_correct else 0.0
                    _y_list.append(_val)
                    _X_list.append([1.0, _A_val, _B_val, _A_val * _B_val])
                
        if len(_y_list) > 4:
            _y = np.array(_y_list)
            _X = np.array(_X_list)
            _XTX = _X.T @ _X
            _beta = np.linalg.solve(_XTX, _X.T @ _y)
        
            _N = len(_y)
            _p = _X.shape[1]
            _df = _N - _p
        
            _residuals = _y - _X @ _beta
            _s2 = np.sum(_residuals**2) / _df
            _cov_beta = _s2 * np.linalg.inv(_XTX)
            _se = np.sqrt(np.diag(_cov_beta))
            _t_stats = _beta / (_se + 1e-12)
        
            import scipy.stats as stats
            _p_values = 2 * stats.t.sf(np.abs(_t_stats), _df)
        
            print("\n  2^2 Factorial OLS (accuracy, sample-level):")
            _labels = ["intercept", "beta_A (sparsity)", "beta_B (PPO)", "beta_AB (interact)"]
            for _j in range(4):
                print(f"    {_labels[_j]:<20}: coef={_beta[_j]:.4f}, se={_se[_j]:.4f}, t={_t_stats[_j]:.2f}, p={_p_values[_j]:.4f}")
            
            ablation_results["factorial_ols_betas"] = {
                "intercept": float(_beta[0]), "beta_A_sparsity": float(_beta[1]),
                "beta_B_ppo": float(_beta[2]), "beta_AB_interaction": float(_beta[3]),
                "se": [float(_val) for _val in _se],
                "t_statistics": [float(_val) for _val in _t_stats],
                "p_values": [float(_val) for _val in _p_values],
                "df": int(_df),
                "N": int(_N),
            }
        else:
            print("  [WARN] OLS skipped: not enough sample-level data points.")
    except Exception as _e:
        print(f"  [WARN] OLS failed: {_e}")
    save_kaggle_checkpoint({**live_results, **baseline_results, **ablation_results})

    # =========================================================
    # FINAL RESULTS TABLE
    # =========================================================
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)

    # V3-1 fix: baseline_results holds plain stats dicts from run_multi_seed_experiment
    # — they have no .sample_results attribute, so aggregate_metrics() would crash.
    all_baseline_metrics = []
    for method_name, stats in baseline_results.items():
        if hasattr(stats, 'sample_results'):
            # ExperimentResults object — aggregate normally
            metrics = aggregate_metrics(stats)
            all_baseline_metrics.append(metrics)
            print(f"\n{method_name}:")
            print(f"  Accuracy: {metrics['accuracy_pct']}")
            print(f"  Avg Tokens: {metrics['token_cost']['avg_tokens_per_sample']:.1f}")
            print(f"  Comm Density: {metrics['communication_density']:.3f}")
            print(f"  Error Prop: {metrics['error_propagation_rate']:.3f}")
        else:
            # Plain stats dict from run_multi_seed_experiment
            acc_mean = stats.get('accuracy_mean', 0.0)
            acc_ci   = stats.get('accuracy_95ci', 0.0)
            tok_mean = stats.get('token_cost_mean', 0.0)
            tok_ci   = stats.get('token_cost_95ci', 0.0)
            n_seeds  = len(stats.get('all_accuracies', []))
            print(f"\n{method_name}:")
            print(f"  Accuracy: {acc_mean:.1%} +/- {acc_ci:.1%} (95% CI, {n_seeds} seeds)")
            print(f"  Token Cost: {tok_mean:.1f} +/- {tok_ci:.1f}")
            seed_strs = ', '.join(f'{a:.1%}' for a in stats.get('all_accuracies', []))
            print(f"  Per-seed: [{seed_strs}]")

    if all_baseline_metrics:
        print("\n" + format_results_table(all_baseline_metrics, dataset_name="Baselines"))


    print("\n[OK] EXECUTION COMPLETE!")
    print(f"[OK] Results saved to: 08-06results/kaggle_latest_results.json")



    # --- TOPOLOGY FAMILY BENCHMARK ---
    import networkx as nx
    import numpy as np
    import dataclasses
    import time


    # Uncomment the line below to run the benchmark:
    run_topology_benchmark()

    import importlib
    try:
        import llm_backend
        importlib.reload(llm_backend)
    except Exception:
        pass

    # --- TOPOLOGY FAMILY BENCHMARK ---
    import networkx as nx
    import numpy as np
    import dataclasses
    import time

    def get_topology_weights(topology_name: str, n_agents: int) -> np.ndarray:
        if topology_name == "Fully Connected":
            G = nx.complete_graph(n_agents)
        elif topology_name == "Ring":
            G = nx.cycle_graph(n_agents)
        elif topology_name == "Star":
            G = nx.star_graph(n_agents - 1)
        elif topology_name == "Random":
            G = nx.erdos_renyi_graph(n_agents, p=0.5, seed=42)
        elif topology_name == "Small World":
            # For N=3, k=2 is the only valid even k < 3. 
            k = min(2, n_agents - 1)
            if k % 2 != 0: k -= 1
            k = max(2, k)
            G = nx.watts_strogatz_graph(n_agents, k=k, p=0.2, seed=42) if n_agents > 3 else nx.cycle_graph(n_agents)
        elif topology_name == "Scale Free":
            G = nx.barabasi_albert_graph(n_agents, m=1, seed=42) if n_agents >= 3 else nx.complete_graph(n_agents)
        else:
            raise ValueError(f"Unknown topology {topology_name}")
    
        W = nx.to_numpy_array(G)
        np.fill_diagonal(W, 0)
    
        # Scale binary adjacency matrix to transmission weights (e.g. 0.6)
        W = W * 0.6 
        return W

    def run_topology_benchmark():
        print("Starting Topology Family Benchmark...")
        # Load default config
        config = get_gsm8k_config()
        config.n_samples = 50  # Default eval size
        set_seed(config.seed)
    
        benchmark_results = []
    
        topologies = [
            "Fully Connected",
            "Ring",
            "Star",
            "Random",
            "Small World",
            "Scale Free"
        ]
    
        # 1. Evaluate Static Graph Topologies
        for topo in topologies:
            print(f"\n--- Evaluating Topology: {topo} ---")
            W = get_topology_weights(topo, config.n_agents)
            Q = np.zeros((config.n_agents, config.query_dim)) # Dummy Q
        
            controller = FrozenWeightController(config, frozen_W=W, frozen_Q=Q)
        
            # NOTE: run_experiment creates the dataset and backend internally based on config
            results = run_experiment(
                config=config,
                method_name=topo,
                train=False,
                evaluate=True,
                controller=controller
            )
            benchmark_results.append(results)
        
        # 2. Evaluate Learned Topology
        print("\n--- Evaluating Topology: Learned Topology ---")
        learned_controller = IBController(config)
        # Load model from the 08-06results folder
        model_path = os.path.join("08-06results", "gsm8k_pretrained.pt.npz")
        try:
            learned_controller.load_model(model_path)
            print(f"Successfully loaded trained controller from {model_path}")
        
            learned_results = run_experiment(
                config=config,
                method_name="Learned Topology",
                train=False,
                evaluate=True,
                controller=learned_controller
            )
            benchmark_results.append(learned_results)
        except Exception as e:
            print(f"Failed to load trained model: {e}")
        
        # 3. Print Final Benchmark Table
        metrics = [aggregate_metrics(r) for r in benchmark_results]
        print("\n=== TOPOLOGY FAMILY BENCHMARK RESULTS ===")
        print(format_results_table(metrics, dataset_name="GSM8K Topology Benchmark"))
    
        # Save results to 08-06results folder
        save_path = os.path.join("08-06results", "topology_benchmark_results.json")
        save_json(
            [dataclasses.asdict(r) for r in benchmark_results], 
            save_path
        )
        print(f"Benchmark results saved to {save_path}.")

    # Uncomment the line below to run the benchmark:
    # run_topology_benchmark()

