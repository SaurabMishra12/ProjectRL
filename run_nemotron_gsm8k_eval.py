"""
Run GSM8K 150-sample evaluation using nvidia/nemotron-3-ultra-550b-a55b via NVIDIA NIM API.
Evaluates 3 configurations:
  1. NoComm (No Communication Baseline)
  2. MAD (Multi-Agent Debate / Full Communication Baseline)
  3. Our Method (SparseComm IB-Controller trained on 400 GSM8K samples)
Saves detailed per-sample results and overall summary table.
"""

import os
import sys
import json
import time
from dataclasses import asdict
import numpy as np
from tqdm import tqdm

# Ensure local imports work cleanly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_run"))

from config import get_gsm8k_config
from utils import set_seed, NumpyEncoder
from evaluation import SampleResult, ExperimentResults
from controller import IBController
from llm_backend import NvidiaDeepSeekBackend
from dataset_loaders import load_dataset_by_name, evaluate_answer
from pipeline import (
    create_llm_backend,
    run_single_sample,
    run_full_communication,
    run_no_communication,
    run_experiment,
)

OUTPUT_DIR = "08-06results/nemotron_gsm8k_150"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "nvidia/nemotron-3-ultra-550b-a55b"
API_BASE_URL = "https://integrate.api.nvidia.com/v1"
API_KEY = os.environ.get(
    "NVIDIA_API_KEY",
    "your-nvidia-api-key"
)
N_SAMPLES = 150
MODEL_400_PATH = "08-06results/gsm8k_pretrained_400.pt.npz"


def create_nemotron_config():
    cfg = get_gsm8k_config()
    cfg.llm_backend = "nvidia"
    cfg.api_model = MODEL_NAME
    cfg.api_base_url = API_BASE_URL
    cfg.api_key = API_KEY
    cfg.api_temperature = 1.0
    cfg.n_samples = N_SAMPLES
    return cfg


def save_method_results(results_list, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    serialized = [asdict(r) for r in results_list]
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2, cls=NumpyEncoder)
    print(f"[SAVED] Results saved to {filepath}")


def load_method_results(filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
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
        print(f"[LOADED] Loaded {len(results)} saved results from {filepath}")
        return results
    except Exception as e:
        print(f"[WARN] Error loading {filepath}: {e}")
        return []


def main():
    print(f"===========================================================")
    print(f" Starting GSM8K Nemotron-3 Ultra 550B Evaluation ({N_SAMPLES} Samples)")
    print(f" Model: {MODEL_NAME}")
    print(f" Output Directory: {OUTPUT_DIR}")
    print(f" Controller Checkpoint: {MODEL_400_PATH}")
    print(f"===========================================================\n")

    cfg = create_nemotron_config()

    # Load 400-sample trained IBController to access calibrator
    base_ctrl = IBController(cfg)
    if os.path.exists(MODEL_400_PATH):
        base_ctrl.load_model(MODEL_400_PATH)
        print(f"[OK] Loaded 400-sample trained IBController from {MODEL_400_PATH}")
    else:
        print(f"[WARN] Checkpoint {MODEL_400_PATH} not found. Running Our method without pre-trained weights.")

    calibrator = base_ctrl.calibrator

    # -------------------------------------------------------------
    # 1. No Communication Baseline (NoComm)
    # -------------------------------------------------------------
    print("\n>>> 1/3 Running No Communication Baseline (NoComm)...")
    nocomm_saved = load_method_results("nocomm_results.json")
    if len(nocomm_saved) >= N_SAMPLES:
        print(f"[CHECKPOINT] NoComm complete with {len(nocomm_saved)} samples.")
        nocomm_exp = ExperimentResults(method_name="No Communication", dataset_name="gsm8k", sample_results=nocomm_saved)
    else:
        cfg_no = create_nemotron_config()
        nocomm_exp = run_no_communication(cfg_no, calibrator=calibrator)
        save_method_results(nocomm_exp.sample_results, "nocomm_results.json")

    nocomm_acc = np.mean([r.is_correct for r in nocomm_exp.sample_results])
    print(f"--> NoComm Accuracy: {nocomm_acc:.2%} ({sum(r.is_correct for r in nocomm_exp.sample_results)}/{len(nocomm_exp.sample_results)})")

    # -------------------------------------------------------------
    # 2. Multi-Agent Debate Baseline (MAD)
    # -------------------------------------------------------------
    print("\n>>> 2/3 Running Multi-Agent Debate Baseline (MAD)...")
    mad_saved = load_method_results("mad_results.json")
    if len(mad_saved) >= N_SAMPLES:
        print(f"[CHECKPOINT] MAD complete with {len(mad_saved)} samples.")
        mad_exp = ExperimentResults(method_name="Multi-Agent Debate", dataset_name="gsm8k", sample_results=mad_saved)
    else:
        cfg_mad = create_nemotron_config()
        mad_exp = run_full_communication(cfg_mad, calibrator=calibrator)
        save_method_results(mad_exp.sample_results, "mad_results.json")

    mad_acc = np.mean([r.is_correct for r in mad_exp.sample_results])
    print(f"--> MAD Accuracy: {mad_acc:.2%} ({sum(r.is_correct for r in mad_exp.sample_results)}/{len(mad_exp.sample_results)})")

    # -------------------------------------------------------------
    # 3. Our Method (SparseComm IB-Controller 400-Sample Model)
    # -------------------------------------------------------------
    print("\n>>> 3/3 Running Our Method (SparseComm 400-Sample Model)...")
    our_saved = load_method_results("our_results.json")
    if len(our_saved) >= N_SAMPLES:
        print(f"[CHECKPOINT] Our method complete with {len(our_saved)} samples.")
        our_exp = ExperimentResults(method_name="Our SparseComm", dataset_name="gsm8k", sample_results=our_saved)
    else:
        cfg_our = create_nemotron_config()
        our_exp = run_experiment(
            cfg_our,
            method_name="Our SparseComm",
            train=False,
            evaluate=True,
            controller=base_ctrl,
        )
        save_method_results(our_exp.sample_results, "our_results.json")

    our_acc = np.mean([r.is_correct for r in our_exp.sample_results])
    print(f"--> Our SparseComm Accuracy: {our_acc:.2%} ({sum(r.is_correct for r in our_exp.sample_results)}/{len(our_exp.sample_results)})")

    # -------------------------------------------------------------
    # Summary & Reporting
    # -------------------------------------------------------------
    summary_data = {
        "model": MODEL_NAME,
        "dataset": "gsm8k",
        "n_samples": N_SAMPLES,
        "methods": {
            "No Communication": {
                "accuracy": float(nocomm_acc),
                "correct_count": int(sum(r.is_correct for r in nocomm_exp.sample_results)),
                "total_count": len(nocomm_exp.sample_results),
                "total_generation_tokens": int(sum(r.generation_tokens for r in nocomm_exp.sample_results)),
                "total_transmitted_tokens": int(sum(r.total_tokens_transmitted for r in nocomm_exp.sample_results)),
            },
            "Multi-Agent Debate (MAD)": {
                "accuracy": float(mad_acc),
                "correct_count": int(sum(r.is_correct for r in mad_exp.sample_results)),
                "total_count": len(mad_exp.sample_results),
                "total_generation_tokens": int(sum(r.generation_tokens for r in mad_exp.sample_results)),
                "total_transmitted_tokens": int(sum(r.total_tokens_transmitted for r in mad_exp.sample_results)),
            },
            "Our SparseComm (400-sample Model)": {
                "accuracy": float(our_acc),
                "correct_count": int(sum(r.is_correct for r in our_exp.sample_results)),
                "total_count": len(our_exp.sample_results),
                "total_generation_tokens": int(sum(r.generation_tokens for r in our_exp.sample_results)),
                "total_transmitted_tokens": int(sum(r.total_tokens_transmitted for r in our_exp.sample_results)),
                "avg_compression_ratio": float(np.mean([r.compression_ratio for r in our_exp.sample_results])),
            }
        }
    }

    summary_path = os.path.join(OUTPUT_DIR, "summary_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, cls=NumpyEncoder)
    print(f"\n[SAVED] Comprehensive summary saved to {summary_path}")

    print("\n===========================================================")
    print(f" FINAL GSM8K EXPERIMENT RESULTS ({MODEL_NAME})")
    print("===========================================================")
    print(f"{'Method':<35} | {'Accuracy':<10} | {'Correct/Total':<15} | {'Transmitted Tokens':<20}")
    print("-" * 88)
    for m_name, m_info in summary_data["methods"].items():
        acc_str = f"{m_info['accuracy']:.2%}"
        ct_str = f"{m_info['correct_count']}/{m_info['total_count']}"
        tok_str = f"{m_info['total_transmitted_tokens']:,}"
        print(f"{m_name:<35} | {acc_str:<10} | {ct_str:<15} | {tok_str:<20}")
    print("===========================================================\n")


if __name__ == "__main__":
    main()
