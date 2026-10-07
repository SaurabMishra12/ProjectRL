"""
Experiment Pipeline: End-to-end experiment runner with controller training,
communication rounds, and result tracking.
"""

import copy
import numpy as np
from typing import List, Dict, Optional, Any
from tqdm import tqdm

from config import Config
from llm_backend import MockLLMBackend, LLMBackend
from agents import AgentPool, AgentResponse, weighted_majority_vote, majority_vote, _normalize_for_vote
from communication import CommunicationGraph, SemanticBandwidthMode
from controller import IBController
from dataset_loaders import load_dataset_by_name, evaluate_answer, hotpotqa_f1_score
from evaluation import SampleResult, ExperimentResults
from utils import set_seed, setup_logging, log_continuous_result

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
                'action': np.zeros(controller.mean_dim),
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


def load_continuous_results(method_logged_name: str, dataset_name: str, seed: int, n_samples: int) -> List[SampleResult]:
    import os
    import json
    
    is_shared = method_logged_name in ["Multi-Agent Debate", "No Communication"]
    if is_shared:
        filepath = f"08-06results/continuous/{method_logged_name}_{dataset_name}_stream.jsonl"
    else:
        filepath = f"08-06results/continuous/{method_logged_name}_{dataset_name}_stream.jsonl"
        
    if not os.path.exists(filepath):
        return []
        
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"[WARN] Error reading stream file {filepath}: {e}")
        return []
        
    if is_shared:
        start_idx = seed * n_samples
        end_idx = start_idx + n_samples
        if len(lines) <= start_idx:
            return []
        seed_lines = lines[start_idx:min(len(lines), end_idx)]
    else:
        seed_lines = lines
        
    completed = []
    for line in seed_lines:
        try:
            s = json.loads(line)
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
                weight_matrices=s.get("weight_matrices", []),
                communication_density=s.get("communication_density", 0.0),
                mode_distribution=s.get("mode_distribution", {}),
                n_active_edges=s.get("n_active_edges", 0),
                generation_tokens=s.get("generation_tokens", 0),
                halt_round=s.get("halt_round", 3),
                semantic_similarity=s.get("semantic_similarity", 0.0),
                token_reduction=s.get("token_reduction", 0.0),
                information_retention=s.get("information_retention", 0.0),
                agent_diversity=s.get("agent_diversity", 0.0),
                answer_entropy=s.get("answer_entropy", 0.0),
                average_delta_w=s.get("average_delta_w", 0.0),
                initial_majority_correct=s.get("initial_majority_correct", False),
            )
            completed.append(sr)
        except Exception as e:
            print(f"[WARN] Failed to parse JSON line: {e}")
            
    return completed


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
    if config.llm_backend == "mock" and evaluate and not train:
        raise ValueError("CRITICAL ERROR: Cannot evaluate with llm_backend='mock'")
    set_seed(config.seed)
    logger.info(f"Running experiment: {method_name} on {config.dataset}")

    # Load dataset (strictly separate splits)
    start_offset = getattr(config, 'n_train_start', 0)
    total_train = config.n_train_samples + start_offset
    train_data, test_data = load_dataset_by_name(
        config.dataset, n_train=total_train, n_test=config.n_samples, seed=config.seed
    )
    if start_offset > 0:
        train_data = train_data[start_offset:]

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

        # Check for intermediate epoch checkpoint
        start_epoch = 0
        if hasattr(config, "model_save_path") and config.model_save_path:
            import os
            for e in reversed(range(config.controller_epochs)):
                base, ext = os.path.splitext(config.model_save_path)
                ckpt_path = f"{base}_{method_name}_epoch_{e+1}{ext}"
                if os.path.exists(ckpt_path):
                    logger.info(f"Resuming training from epoch checkpoint: {ckpt_path}")
                    controller.load_model(ckpt_path)
                    start_epoch = e + 1
                    break
        
        for epoch in range(start_epoch, config.controller_epochs):
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
                        base, ext = os.path.splitext(config.model_save_path)
                        controller.save_model(f"{base}_{method_name}_latest{ext}")

            # Final update for remaining samples
            if len(train_data) % accumulation_steps != 0:
                loss = controller.update_policy()
                if hasattr(config, "model_save_path") and config.model_save_path:
                    base, ext = os.path.splitext(config.model_save_path)
                    controller.save_model(f"{base}_{method_name}_latest{ext}")

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
            # Save intermediate epoch checkpoint
            if hasattr(config, "model_save_path") and config.model_save_path:
                base, ext = os.path.splitext(config.model_save_path)
                ckpt_path = f"{base}_{method_name}_epoch_{epoch+1}{ext}"
                controller.save_model(ckpt_path)
                logger.info(f"Saved epoch {epoch+1} checkpoint to {ckpt_path}")
                
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

        completed = load_continuous_results(method_name, config.dataset, config.seed, config.n_samples)
        start_idx = len(completed)
        if completed:
            results.sample_results.extend(completed)
            logger.info(f"[CHECKPOINT] Resumed {method_name} evaluation from sample {start_idx}/{len(test_data)}")

        for i, sample in enumerate(tqdm(test_data, desc=f"Evaluating {method_name}")):
            if i < start_idx:
                continue
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

    completed = load_continuous_results("Multi-Agent Debate", config.dataset, config.seed, config.n_samples)
    start_idx = len(completed)
    if completed:
        results.sample_results.extend(completed)
        print(f"[CHECKPOINT] Resumed Multi-Agent Debate from sample {start_idx}/{len(test_data)}")

    for i, sample in enumerate(tqdm(test_data, desc="Debate", disable=len(test_data) < 5)):
        if i < start_idx:
            continue
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
    
    completed = load_continuous_results("No Communication", config.dataset, config.seed, config.n_samples)
    start_idx = len(completed)
    if completed:
        results.sample_results.extend(completed)
        print(f"[CHECKPOINT] Resumed No Communication from sample {start_idx}/{len(test_data)}")
        
    for i, sample in enumerate(tqdm(test_data, desc="NoComm")):
        if i < start_idx:
            continue
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
            'action': np.zeros(self.mean_dim),
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
        
        import os
        import json
        seed_ckpt_path = f"08-06results/seed_ckpt_{method_name}_seed{seed}.json"
        if os.path.exists(seed_ckpt_path):
            try:
                with open(seed_ckpt_path, "r", encoding="utf-8") as f:
                    seed_data = json.load(f)
                acc = seed_data["accuracy"]
                tokens = seed_data["token_cost"]
                print(f"[CHECKPOINT] Loaded seed {seed+1} results for {method_name} from {seed_ckpt_path}")
            except Exception as e:
                print(f"[WARN] Failed to load seed checkpoint {seed_ckpt_path}: {e}. Re-running seed.")
                results = experiment_fn(seed_config, f"{method_name}_seed{seed}")
                acc = np.mean([1.0 if r.is_correct else 0.0 for r in results.sample_results])
                tokens = np.mean([r.total_tokens_transmitted for r in results.sample_results])
                try:
                    os.makedirs("08-06results", exist_ok=True)
                    with open(seed_ckpt_path, "w", encoding="utf-8") as f:
                        json.dump({"accuracy": float(acc), "token_cost": float(tokens)}, f)
                except Exception as ex:
                    print(f"[WARN] Failed to save seed checkpoint: {ex}")
        else:
            results = experiment_fn(seed_config, f"{method_name}_seed{seed}")
            acc = np.mean([1.0 if r.is_correct else 0.0 for r in results.sample_results])
            tokens = np.mean([r.total_tokens_transmitted for r in results.sample_results])
            try:
                os.makedirs("08-06results", exist_ok=True)
                with open(seed_ckpt_path, "w", encoding="utf-8") as f:
                    json.dump({"accuracy": float(acc), "token_cost": float(tokens)}, f)
                print(f"Saved seed {seed+1} checkpoint to {seed_ckpt_path}")
            except Exception as ex:
                print(f"[WARN] Failed to save seed checkpoint: {ex}")
        
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




