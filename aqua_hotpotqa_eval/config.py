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
    n_train_start: int = 0  # start offset for training samples

    # --- LLM Backend ---
    llm_backend: str = "azure"  # "mock" or "nvidia" or "openai" or "azure"
    api_base_url: str = "https://integrate.api.nvidia.com/v1"
    api_model: str = "gpt-oss-120b"
    api_key: Optional[str] = "your-azure-api-key"
    api_temperature: float = 0.0  # control LLM generation diversity
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


def get_aqua_config(**overrides) -> Config:
    """Preset config for AQuA experiments."""
    defaults = dict(
        dataset="aqua",
        n_agents=3,
        n_rounds=4,  # Harder problem, might need an extra round
        lambda_info=0.2, # Less penalty to avoid stunting performance
        gamma_loop=0.1,
        mock_base_accuracy=0.40,
    )
    defaults.update(overrides)
    return Config(**defaults)
