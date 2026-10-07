import os
import json
import time
import math
import numpy as np
import dataclasses
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

# Local Module Imports
from config import *
from utils import *
from evaluation import *
from llm_backend import *
from agents import *
from communication import *
from controller import *
from dataset_loaders import *
from pipeline import *

if __name__ == "__main__":
    import logging
    import os
    import json
    import copy
    from copy import deepcopy as _deepcopy
    from tqdm import tqdm
    import numpy as np

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    datasets_to_run = ["gsm8k", "hotpotqa", "aqua"]
    os.makedirs("08-06results", exist_ok=True)

    # -------------------------------------------------------------
    # Helpers for Checkpoint Saving & Loading
    # -------------------------------------------------------------
    def get_model_path(ds_name, samples):
        return f"08-06results/{ds_name}_pretrained_{samples}.pt.npz" if samples > 100 else f"08-06results/{ds_name}_pretrained.pt.npz"

    def run_ablation_for_model(model_samples, model_paths):
        """Runs the 2^2 Factorial Ablation Study using the specified model checkpoints."""
        result_path = f"08-06results/ablation_{model_samples}.json"
        
        # Check if the file already exists
        if os.path.exists(result_path):
            print(f"[CHECKPOINT] Ablation results for {model_samples} samples already exist at {result_path}. Skipping.")
            with open(result_path, "r") as f:
                return json.load(f)

        # For 100 samples, try to reconstruct from individual arm files to save API calls
        if model_samples == 100:
            reconstructed = {}
            for arm_name, file_suffix in [
                ("Arm_00_StaticRing", "arm00"),
                ("Arm_01_PPOOnly", "arm01"),
                ("Arm_10_SparseRing", "arm10"),
                ("Arm_11_Full", "arm11")
            ]:
                path = f"08-06results/ablation_{file_suffix}.json"
                if os.path.exists(path):
                    try:
                        with open(path, "r") as f:
                            data = json.load(f)
                        if arm_name in data:
                            reconstructed[arm_name] = data[arm_name]
                        elif "detailed_samples" in data and arm_name in data["detailed_samples"]:
                            reconstructed[arm_name] = data["detailed_samples"][arm_name]
                        else:
                            reconstructed[arm_name] = data.get("detailed_samples", {}).get(arm_name, data)
                    except Exception as e:
                        print(f"Error loading {path}: {e}")
            if len(reconstructed) == 4:
                print("Successfully reconstructed 100-sample ablation results from arm files. Running OLS...")
                try:
                    # Parse into standard structures
                    ablation_results = {}
                    for arm, val in reconstructed.items():
                        # Wrap as ExperimentResults-like object for OLS
                        class DummyRes:
                            pass
                        d = DummyRes()
                        d.sample_results = []
                        samples_list = val if isinstance(val, list) else []
                        for s in samples_list:
                            class DummySample:
                                pass
                            ds = DummySample()
                            ds.is_correct = s.get("is_correct", False)
                            d.sample_results.append(ds)
                        ablation_results[arm] = d

                    # Fit OLS
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
                        if _arm in ablation_results:
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
                        print(f"\n  2^2 Factorial OLS (accuracy, sample-level) at 100 samples:")
                        _labels = ["intercept", "beta_A (sparsity)", "beta_B (PPO)", "beta_AB (interact)"]
                        for _j in range(4):
                            print(f"    {_labels[_j]:<20}: coef={_beta[_j]:.4f}, se={_se[_j]:.4f}, t={_t_stats[_j]:.2f}, p={_p_values[_j]:.4f}")
                        reconstructed["factorial_ols_betas"] = {
                            "intercept": float(_beta[0]), "beta_A_sparsity": float(_beta[1]),
                            "beta_B_ppo": float(_beta[2]), "beta_AB_interaction": float(_beta[3]),
                            "se": [float(_val) for _val in _se],
                            "t_statistics": [float(_val) for _val in _t_stats],
                            "p_values": [float(_val) for _val in _p_values],
                            "df": int(_df),
                            "N": int(_N),
                        }
                    with open(result_path, "w", encoding="utf-8") as f:
                        json.dump(reconstructed, f, indent=2, cls=NumpyEncoder)
                    print(f"[OK] Reconstructed and saved 100-sample ablation results to {result_path}")
                    return reconstructed
                except Exception as ex:
                    print(f"[WARN] Reconstruction OLS failed: {ex}. Running 100-sample ablation from scratch.")

        print(f"\n=== RUNNING ABLATION STUDY AT {model_samples} SAMPLES ===")
        from dataclasses import asdict
        ablation_config = get_gsm8k_config()
        ablation_config.llm_backend = "azure"
        ablation_config.api_model = "gpt-oss-120b"
        ablation_config.n_samples = 150
        ablation_config.n_train_samples = 100
        ablation_config.controller_epochs = 5
        ablation_config.dataset = "gsm8k"

        ablation_results = {}

        def load_arm_checkpoint(arm_name, suffix, model_samples):
            import os
            import json
            path = f"08-06results/ablation_{model_samples}_{suffix}.json"
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    res = ExperimentResults(method_name=arm_name, dataset_name="gsm8k")
                    samples_list = data if isinstance(data, list) else []
                    for s in samples_list:
                        sr = SampleResult(
                            question=s.get("question", ""),
                            gold_answer=s.get("gold_answer", ""),
                            final_answer=s.get("final_answer", ""),
                            is_correct=s.get("is_correct", False),
                            exact_match=s.get("exact_match", 0.0),
                            f1_score=s.get("f1_score", 0.0),
                            total_tokens_transmitted=s.get("total_tokens_transmitted", 0),
                            total_tokens_if_full=s.get("total_tokens_if_full", 0),
                            compression_ratio=s.get("compression_ratio", 0.0),
                            communication_density=s.get("communication_density", 0.0),
                            n_active_edges=s.get("n_active_edges", 0),
                            generation_tokens=s.get("generation_tokens", 0),
                            halt_round=s.get("halt_round", 3),
                        )
                        res.sample_results.append(sr)
                    print(f"[CHECKPOINT] Loaded ablation arm {arm_name} from {path}")
                    return res
                except Exception as e:
                    print(f"Error loading arm checkpoint {path}: {e}")
            return None

        def save_arm_checkpoint(res, suffix, model_samples):
            import os
            import json
            path = f"08-06results/ablation_{model_samples}_{suffix}.json"
            try:
                os.makedirs("08-06results", exist_ok=True)
                serialized = [asdict(s) for s in res.sample_results]
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(serialized, f, indent=2, cls=NumpyEncoder)
                print(f"Saved ablation arm checkpoint to {path}")
            except Exception as e:
                print(f"Error saving arm checkpoint {path}: {e}")

        # Arm 00: Static ring, no KL, no PPO
        _a00_res = load_arm_checkpoint("Arm_00_StaticRing", "arm00", model_samples)
        if _a00_res is not None:
            ablation_results["Arm_00_StaticRing"] = _a00_res
        else:
            print("\n  Arm 00: Static ring, no KL, no PPO...")
            _a00_cfg = _deepcopy(ablation_config)
            _a00_cfg.lambda_kl = 0.0; _a00_cfg.lambda_info = 0.0; _a00_cfg.gamma_loop = 0.0
            _a00_tmp = IBController(_a00_cfg)
            _a00_W, _a00_Q = _a00_tmp.get_static_weights("ring"); del _a00_tmp
            _a00_frozen = FrozenWeightController(_a00_cfg, _a00_W, _a00_Q)
            _a00_frozen.calibrator = None
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
            save_arm_checkpoint(_a00_res, "arm00", model_samples)
            ablation_results["Arm_00_StaticRing"] = _a00_res

        # Arm 01: PPO only, no KL
        _a01_res = load_arm_checkpoint("Arm_01_PPOOnly", "arm01", model_samples)
        if _a01_res is not None:
            ablation_results["Arm_01_PPOOnly"] = _a01_res
        else:
            print("\n  Arm 01: PPO-only, no KL sparsity...")
            _a01_cfg = _deepcopy(ablation_config); _a01_cfg.lambda_kl = 0.0; _a01_cfg.gamma_loop = 0.0
            _a01_ctrl = IBController(_a01_cfg)
            _a01_ctrl.calibrator = None
            _a01_res = run_experiment(
                _a01_cfg, method_name="Arm_01_PPOOnly", train=True, evaluate=True, controller=_a01_ctrl)
            save_arm_checkpoint(_a01_res, "arm01", model_samples)
            ablation_results["Arm_01_PPOOnly"] = _a01_res

        # Arm 10: Fixed ring with KL, no PPO
        _a10_res = load_arm_checkpoint("Arm_10_SparseRing", "arm10", model_samples)
        if _a10_res is not None:
            ablation_results["Arm_10_SparseRing"] = _a10_res
        else:
            print("\n  Arm 10: Fixed ring with KL penalty, no PPO...")
            _a10_cfg = _deepcopy(ablation_config)
            _a10_tmp = IBController(_a10_cfg)
            _a10_W, _a10_Q = _a10_tmp.get_static_weights("ring"); del _a10_tmp
            _a10_frozen = FrozenWeightController(_a10_cfg, _a10_W, _a10_Q)
            _a10_frozen.calibrator = None
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
            save_arm_checkpoint(_a10_res, "arm10", model_samples)
            ablation_results["Arm_10_SparseRing"] = _a10_res

        # Arm 11: Full system
        _a11_res = load_arm_checkpoint("Arm_11_Full", "arm11", model_samples)
        if _a11_res is not None:
            ablation_results["Arm_11_Full"] = _a11_res
        else:
            print(f"\n  Arm 11: PPO + KL sparsity (using model trained on {model_samples} samples)...")
            _a11_ctrl = IBController(ablation_config)
            model_path = model_paths.get("gsm8k")
            if model_path and os.path.exists(model_path):
                _a11_ctrl.load_model(model_path)
                print(f"Loaded controller weights from {model_path}")
            else:
                print(f"[WARN] Controller weights not found at {model_path}. Running Arm 11 without loaded weights.")
            _a11_ctrl.calibrator = None
            _a11_res = run_experiment(
                ablation_config, method_name="Arm_11_Full", train=False, evaluate=True, controller=_a11_ctrl)
            save_arm_checkpoint(_a11_res, "arm11", model_samples)
            ablation_results["Arm_11_Full"] = _a11_res

        # OLS
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
                print(f"\n  2^2 Factorial OLS (accuracy, sample-level) at {model_samples} samples:")
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
        except Exception as _e:
            print(f"  [WARN] OLS failed: {_e}")

        # Serialize results to json
        serialized = {}
        for k, v in ablation_results.items():
            if hasattr(v, "sample_results"):
                serialized[k] = [asdict(s) for s in v.sample_results]
            else:
                serialized[k] = v
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2, cls=NumpyEncoder)
        print(f"[OK] Saved ablation results for {model_samples} samples to {result_path}")
        return serialized

    # -------------------------------------------------------------
    # 1. Ablation at 100 samples
    # -------------------------------------------------------------
    model_paths_100 = {ds: get_model_path(ds, 100) for ds in datasets_to_run}
    run_ablation_for_model(100, model_paths_100)

    # -------------------------------------------------------------
    # 2. Incremental Training 100 -> 200 & Ablation at 200
    # -------------------------------------------------------------
    print("\n=== INCREMENTAL TRAINING: 100 -> 200 SAMPLES ===")
    for ds_name in datasets_to_run:
        prev_model = get_model_path(ds_name, 100)
        target_model = get_model_path(ds_name, 200)
        if os.path.exists(target_model):
            print(f"[CHECKPOINT] Model trained on 200 samples for {ds_name} already exists at {target_model}. Skipping training.")
            continue
        print(f"\n--- Training {ds_name.upper()} on samples 100-200 ---")
        current_config = get_aqua_config() if ds_name == "aqua" else (get_hotpotqa_config() if ds_name == "hotpotqa" else get_gsm8k_config())
        current_config.llm_backend = "azure"
        current_config.api_model = "gpt-oss-120b"
        current_config.n_train_start = 100
        current_config.n_train_samples = 100
        current_config.n_samples = 300
        current_config.dataset = ds_name
        current_config.model_save_path = target_model
        
        controller = IBController(current_config)
        if os.path.exists(prev_model):
            print(f"Loading base weights from {prev_model}")
            controller.load_model(prev_model)
        else:
            print(f"[WARN] Base model {prev_model} not found. Training from scratch.")
        run_experiment(
            current_config,
            method_name=f"Pretrain_{ds_name.upper()}_200",
            train=True,
            evaluate=False,
            controller=controller
        )
        controller.save_model(target_model)
        print(f"[OK] Saved model to {target_model}")

    model_paths_200 = {ds: get_model_path(ds, 200) for ds in datasets_to_run}
    run_ablation_for_model(200, model_paths_200)

    # -------------------------------------------------------------
    # 3. Incremental Training 200 -> 300 & Ablation at 300
    # -------------------------------------------------------------
    print("\n=== INCREMENTAL TRAINING: 200 -> 300 SAMPLES ===")
    for ds_name in datasets_to_run:
        prev_model = get_model_path(ds_name, 200)
        target_model = get_model_path(ds_name, 300)
        if os.path.exists(target_model):
            print(f"[CHECKPOINT] Model trained on 300 samples for {ds_name} already exists at {target_model}. Skipping training.")
            continue
        print(f"\n--- Training {ds_name.upper()} on samples 200-300 ---")
        current_config = get_aqua_config() if ds_name == "aqua" else (get_hotpotqa_config() if ds_name == "hotpotqa" else get_gsm8k_config())
        current_config.llm_backend = "azure"
        current_config.api_model = "gpt-oss-120b"
        current_config.n_train_start = 200
        current_config.n_train_samples = 100
        current_config.n_samples = 300
        current_config.dataset = ds_name
        current_config.model_save_path = target_model
        
        controller = IBController(current_config)
        if os.path.exists(prev_model):
            print(f"Loading base weights from {prev_model}")
            controller.load_model(prev_model)
        else:
            print(f"[WARN] Base model {prev_model} not found. Training from scratch.")
        run_experiment(
            current_config,
            method_name=f"Pretrain_{ds_name.upper()}_300",
            train=True,
            evaluate=False,
            controller=controller
        )
        controller.save_model(target_model)
        print(f"[OK] Saved model to {target_model}")

    model_paths_300 = {ds: get_model_path(ds, 300) for ds in datasets_to_run}
    run_ablation_for_model(300, model_paths_300)

    # -------------------------------------------------------------
    # 4. Incremental Training 300 -> 400
    # -------------------------------------------------------------
    print("\n=== INCREMENTAL TRAINING: 300 -> 400 SAMPLES ===")
    for ds_name in datasets_to_run:
        prev_model = get_model_path(ds_name, 300)
        target_model = get_model_path(ds_name, 400)
        if os.path.exists(target_model):
            print(f"[CHECKPOINT] Model trained on 400 samples for {ds_name} already exists at {target_model}. Skipping training.")
            continue
        print(f"\n--- Training {ds_name.upper()} on samples 300-400 ---")
        current_config = get_aqua_config() if ds_name == "aqua" else (get_hotpotqa_config() if ds_name == "hotpotqa" else get_gsm8k_config())
        current_config.llm_backend = "azure"
        current_config.api_model = "gpt-oss-120b"
        current_config.n_train_start = 300
        current_config.n_train_samples = 100
        current_config.n_samples = 300
        current_config.dataset = ds_name
        current_config.model_save_path = target_model
        
        controller = IBController(current_config)
        if os.path.exists(prev_model):
            print(f"Loading base weights from {prev_model}")
            controller.load_model(prev_model)
        else:
            print(f"[WARN] Base model {prev_model} not found. Training from scratch.")
        run_experiment(
            current_config,
            method_name=f"Pretrain_{ds_name.upper()}_400",
            train=True,
            evaluate=False,
            controller=controller
        )
        controller.save_model(target_model)
        print(f"[OK] Saved model to {target_model}")

    # Ablation at 400 samples
    model_paths_400 = {ds: get_model_path(ds, 400) for ds in datasets_to_run}
    run_ablation_for_model(400, model_paths_400)

    # -------------------------------------------------------------
    # 5. Final Multi-Seed Evaluation at 400 Samples
    # -------------------------------------------------------------
    print("\n=== FINAL EVALUATIONS AT 400 SAMPLES (3 SEEDS, 300 TEST SAMPLES) ===")
    final_results_path = "08-06results/final_results.json"
    final_results = {}
    if os.path.exists(final_results_path):
        try:
            with open(final_results_path, "r") as f:
                final_results = json.load(f)
            print(f"Loaded existing final results from {final_results_path}")
        except Exception as e:
            print(f"Error loading {final_results_path}: {e}")

    for ds_name in datasets_to_run:
        current_config = get_aqua_config() if ds_name == "aqua" else (get_hotpotqa_config() if ds_name == "hotpotqa" else get_gsm8k_config())
        current_config.llm_backend = "azure"
        current_config.api_model = "gpt-oss-120b"
        current_config.n_train_samples = 100
        current_config.n_samples = 300
        current_config.dataset = ds_name

        model_400 = get_model_path(ds_name, 400)

        # 1. Transfer (Our Method)
        method_key = f"Transfer_{ds_name.upper()}"
        if method_key in final_results:
            print(f"[CHECKPOINT] {method_key} already evaluated. Skipping.")
        else:
            print(f"\n--- Evaluating Transfer (Our Method): {ds_name.upper()} ---")
            import copy as _copy
            base_ctrl = IBController(current_config)
            if os.path.exists(model_400):
                base_ctrl.load_model(model_400)
                print(f"Loaded 400-sample controller from {model_400}")
            else:
                print(f"[WARN] 400-sample controller not found at {model_400}. Evaluated without loading.")

            def _make_transfer_fn(ctrl_instance):
                def _transfer_fn(cfg, method_name):
                    seed_ctrl = _copy.deepcopy(ctrl_instance)
                    return run_experiment(cfg, method_name=method_name, train=False, evaluate=True, controller=seed_ctrl)
                return _transfer_fn

            stats = run_multi_seed_experiment(
                current_config,
                method_key,
                n_seeds=3,
                experiment_fn=_make_transfer_fn(base_ctrl),
            )
            final_results[method_key] = stats
            with open(final_results_path, "w") as f:
                json.dump(final_results, f, indent=2, cls=NumpyEncoder)

        # 2. Full Communication
        method_key_full = f"FullComm_{ds_name.upper()}"
        if method_key_full in final_results:
            print(f"[CHECKPOINT] {method_key_full} already evaluated. Skipping.")
        else:
            print(f"\n--- Evaluating FullComm: {ds_name.upper()} ---")
            base_ctrl = IBController(current_config)
            if os.path.exists(model_400):
                base_ctrl.load_model(model_400)
            _calib = base_ctrl.calibrator

            full_comm_res = run_multi_seed_experiment(
                current_config,
                method_key_full,
                n_seeds=3,
                experiment_fn=lambda cfg, m: run_full_communication(cfg, calibrator=_calib)
            )
            final_results[method_key_full] = full_comm_res
            with open(final_results_path, "w") as f:
                json.dump(final_results, f, indent=2, cls=NumpyEncoder)

        # 3. No Communication
        method_key_no = f"NoComm_{ds_name.upper()}"
        if method_key_no in final_results:
            print(f"[CHECKPOINT] {method_key_no} already evaluated. Skipping.")
        else:
            print(f"\n--- Evaluating NoComm: {ds_name.upper()} ---")
            base_ctrl = IBController(current_config)
            if os.path.exists(model_400):
                base_ctrl.load_model(model_400)
            _calib = base_ctrl.calibrator

            no_comm_res = run_multi_seed_experiment(
                current_config,
                method_key_no,
                n_seeds=3,
                experiment_fn=lambda cfg, m: run_no_communication(cfg, calibrator=_calib)
            )
            final_results[method_key_no] = no_comm_res
            with open(final_results_path, "w") as f:
                json.dump(final_results, f, indent=2, cls=NumpyEncoder)

    save_kaggle_checkpoint(final_results, filepath="08-06results/kaggle_latest_results.json")
    print("\n[OK] Sequential loop completed successfully!")




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
        config.llm_backend = "azure"
        config.api_model = "gpt-oss-120b"
        config.n_samples = 150  # Default eval size
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
        model_path = os.path.join("08-06results", "gsm8k_pretrained_400.pt.npz")
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

    import importlib
    try:
        import llm_backend
        importlib.reload(llm_backend)
    except Exception:
        pass

    # Run the benchmark:
    run_topology_benchmark()

