"""
Run Our Method (SparseComm IB-Controller trained on 400 GSM8K samples)
for 150 GSM8K samples using the second NVIDIA API key concurrently.
"""

import os
import sys
import json
import time
from dataclasses import asdict
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_run"))

from config import get_gsm8k_config
from utils import set_seed, NumpyEncoder
from evaluation import SampleResult, ExperimentResults
from controller import IBController
from dataset_loaders import load_dataset_by_name
from pipeline import run_experiment

OUTPUT_DIR = "08-06results/nemotron_gsm8k_150"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "nvidia/nemotron-3-ultra-550b-a55b"
API_BASE_URL = "https://integrate.api.nvidia.com/v1"
API_KEY = "your-nvidia-api-key"
N_SAMPLES = 150
MODEL_400_PATH = "08-06results/gsm8k_pretrained_400.pt.npz"


def save_our_results(results_list):
    filepath = os.path.join(OUTPUT_DIR, "our_results.json")
    serialized = [asdict(r) for r in results_list]
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2, cls=NumpyEncoder)
    print(f"[SAVED] Our SparseComm results saved to {filepath}")


def load_our_results():
    filepath = os.path.join(OUTPUT_DIR, "our_results.json")
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        results = []
        for s in data:
            sr = SampleResult(
                question=s.get("question", ""),
                gold_answer=s.get("gold_answer", ""),
                final_answer=s.get("final_answer", ""),
                is_correct=s.get("is_correct", False),
                exact_match=s.get("exact_match", 0.0),
                f1_score=s.get("f1_score", 0.0),
                initial_answers=s.get("initial_answers", {}),
                initial_correct=s.get("initial_correct", {}),
                final_answers=s.get("final_answers", {}),
                initial_reasoning=s.get("initial_reasoning", {}),
                final_reasoning=s.get("final_reasoning", {}),
                raw_communications=s.get("raw_communications", []),
                total_tokens_transmitted=s.get("total_tokens_transmitted", 0),
                total_tokens_if_full=s.get("total_tokens_if_full", 0),
                compression_ratio=s.get("compression_ratio", 0.0),
                communication_density=s.get("communication_density", 0.0),
                mode_distribution=s.get("mode_distribution", {}),
                n_active_edges=s.get("n_active_edges", 0),
                generation_tokens=s.get("generation_tokens", 0),
                halt_round=s.get("halt_round", 3),
            )
            results.append(sr)
        print(f"[LOADED] Loaded {len(results)} saved Our SparseComm results from {filepath}")
        return results
    except Exception as e:
        print(f"[WARN] Error loading {filepath}: {e}")
        return []


def main():
    print(f"===========================================================")
    print(f" Starting Our SparseComm GSM8K Evaluation ({N_SAMPLES} Samples)")
    print(f" Model: {MODEL_NAME}")
    print(f" API Key: {API_KEY[:10]}...")
    print(f" Controller Checkpoint: {MODEL_400_PATH}")
    print(f"===========================================================\n")

    cfg = get_gsm8k_config()
    cfg.llm_backend = "nvidia"
    cfg.api_model = MODEL_NAME
    cfg.api_base_url = API_BASE_URL
    cfg.api_key = API_KEY
    cfg.api_temperature = 1.0
    cfg.n_samples = N_SAMPLES

    base_ctrl = IBController(cfg)
    if os.path.exists(MODEL_400_PATH):
        base_ctrl.load_model(MODEL_400_PATH)
        print(f"[OK] Loaded 400-sample trained IBController from {MODEL_400_PATH}")
    else:
        print(f"[WARN] Checkpoint {MODEL_400_PATH} not found.")

    our_saved = load_our_results()
    if len(our_saved) >= N_SAMPLES:
        print(f"[CHECKPOINT] Our SparseComm evaluation already complete with {len(our_saved)} samples.")
        our_exp = ExperimentResults(method_name="Our SparseComm", dataset_name="gsm8k", sample_results=our_saved)
    else:
        our_exp = run_experiment(
            cfg,
            method_name="Our SparseComm",
            train=False,
            evaluate=True,
            controller=base_ctrl,
        )
        save_our_results(our_exp.sample_results)

    our_acc = np.mean([r.is_correct for r in our_exp.sample_results])
    correct_count = sum(r.is_correct for r in our_exp.sample_results)
    total_count = len(our_exp.sample_results)

    print(f"\n===========================================================")
    print(f" OUR SPARSECOMM GSM8K RESULTS")
    print(f" Accuracy: {our_acc:.2%} ({correct_count}/{total_count})")
    print(f" Avg Compression Ratio: {np.mean([r.compression_ratio for r in our_exp.sample_results]):.2%}")
    print(f" Total Transmitted Tokens: {sum(r.total_tokens_transmitted for r in our_exp.sample_results):,}")
    print(f"===========================================================\n")


if __name__ == "__main__":
    main()
