def run_single_sample(question: str, gold_answer: str, agent_pool: AgentPool, comm_graph: CommunicationGraph, controller: IBController, config: Config, train_controller: bool=True, avg_accuracy: float=0.5) -> SampleResult:
    """
    Run the full multi-round pipeline for a single sample.

    1. Round 0: all agents solve independently
    2. Rounds 1..T: controller produces W -> filter messages -> agents re-solve
    3. Final answer via weighted majority vote
    """
    agent_pool.reset_all()
    comm_graph.reset()
    round0_responses = agent_pool.initial_solve(question, dataset_type=config.dataset)
    initial_answers = {r.agent_id: r.answer for r in round0_responses}
    initial_correct = {r.agent_id: evaluate_answer(r.answer, gold_answer, config.dataset) for r in round0_responses}
    total_gen_tokens = sum((r.token_count for r in round0_responses))
    current_responses = round0_responses
    all_metas = []
    all_comms = []
    halted = False
    _round0_unique = set((_normalize_for_vote(r.answer, config.dataset) for r in round0_responses))
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
        return SampleResult(question=question, gold_answer=gold_answer, final_answer=final_answer, is_correct=is_correct, initial_answers=initial_answers, initial_correct=initial_correct, final_answers={r.agent_id: r.answer for r in round0_responses}, initial_reasoning={r.agent_id: r.reasoning_text for r in round0_responses}, final_reasoning={r.agent_id: r.reasoning_text for r in round0_responses}, raw_communications=[], total_tokens_transmitted=0, total_tokens_if_full=0, compression_ratio=0.0, communication_density=0.0, mode_distribution={mode.name: 0 for mode in SemanticBandwidthMode if mode != SemanticBandwidthMode.BLOCK}, n_active_edges=0, generation_tokens=sum((r.token_count for r in round0_responses)), halt_round=0, agent_diversity=_agent_diversity, answer_entropy=_answer_entropy, average_delta_w=0.0, initial_majority_correct=_initial_majority_correct)
    initial_confidence = np.mean([r.confidence for r in current_responses])
    responses_snapshots: List[List] = [list(round0_responses)]
    for t in range(1, config.n_rounds + 1):
        confidences = [r.confidence for r in current_responses]
        prev_W = comm_graph.get_weight_snapshot()
        unique_answers = len(set((r.answer for r in current_responses)))
        agent_texts = {r.agent_id: r.reasoning_text for r in current_responses}
        agent_answers = {r.agent_id: r.answer for r in current_responses}
        W, Q, halt_decision, meta = controller.get_weights(reasoning_texts=agent_texts, answers=agent_answers, confidence_scores=confidences, prev_W=prev_W, round_num=t, unique_answers=unique_answers, avg_accuracy=avg_accuracy, explore=train_controller)
        _delta_ws.append(float(np.linalg.norm(W - prev_W, ord='fro')))
        if meta is not None:
            all_metas.append(meta)
        comm_graph.set_weights(W, Q)
        _round_tokens = comm_graph.round_token_counts.get(t - 1, 0)
        _total_budget = sum((r.token_count for r in round0_responses)) * max(config.n_agents, 1)
        controller._prev_comm_cost_norm = _round_tokens / max(float(_total_budget), 1.0)
        _SPRT_TAU = 2.944
        _SPRT_P0 = 0.5
        ans_list = [_normalize_for_vote(r.answer, config.dataset) for r in current_responses]
        from collections import Counter as _Counter
        _ans_counts = _Counter(ans_list)
        _majority_ans, _k = _ans_counts.most_common(1)[0]
        _N = len(ans_list)
        _maj_confs = [r.confidence for r in current_responses if _normalize_for_vote(r.answer, config.dataset) == _majority_ans]
        if controller.calibrator is not None and controller.calibrator._fitted and _maj_confs:
            _p_maj = float(np.clip(controller.calibrator.calibrate(float(np.max(_maj_confs))), 0.501, 0.999))
        else:
            _p_maj = 0.7
        _n_agree = sum((1 for _i in range(_N) for _j in range(_i + 1, _N) if ans_list[_i] == ans_list[_j]))
        _rho = _n_agree / max(_N * (_N - 1) // 2, 1)
        _ess = max(1.0, _N / (1.0 + (_N - 1) * _rho))
        _k_eff = _k * (_ess / _N)
        _llr = _k_eff * np.log(_p_maj / _SPRT_P0) + (_ess - _k_eff) * np.log((1.0 - _p_maj) / (1.0 - _SPRT_P0))
        sprt_halt = _llr > _SPRT_TAU
        if (halt_decision or sprt_halt) and t > 1:
            halted = True
            if all_metas:
                all_metas.pop()
            break
        local_backend = agent_pool.agents[0].llm_backend
        received_messages = comm_graph.get_messages_for_round(agent_texts, t, local_backend)
        all_comms.append({'round': t, 'weights': W, 'messages_received': {k: v for k, v in received_messages.items()}})
        current_responses = agent_pool.communicate_and_solve(question=question, received_messages=received_messages, dataset_type=config.dataset, round_num=t)
        total_gen_tokens += sum((r.token_count for r in current_responses))
        responses_snapshots.append(list(current_responses))
    final_answer = weighted_majority_vote(current_responses, dataset_type=config.dataset, calibrator=controller.calibrator)
    is_correct = evaluate_answer(final_answer, gold_answer, config.dataset)
    em_score = 1.0 if is_correct else 0.0
    if config.dataset in ['hotpotqa', 'hotpot_qa']:
        f1 = hotpotqa_f1_score(final_answer, gold_answer)
        task_reward_signal = f1
    else:
        f1 = float(is_correct)
        task_reward_signal = float(is_correct)
    new_confidence = np.mean([r.confidence for r in current_responses])
    new_unique_answers = len(set((r.answer for r in current_responses)))
    tokens_transmitted = comm_graph.get_total_tokens_transmitted()
    tokens_if_full = comm_graph.get_total_tokens_if_full()
    compression_ratio = comm_graph.get_compression_ratio()
    n_cycles = comm_graph.count_cycles(threshold=0.5)
    is_final = t == config.n_rounds or halted
    step_reward = controller.compute_reward(task_reward=task_reward_signal if is_final else 0.0, tokens_transmitted=tokens_transmitted, tokens_if_full=tokens_if_full, compression_ratio=compression_ratio, n_cycles=n_cycles, current_W=W, halt_round=t, is_final_step=is_final, include_token_penalty=False)
    if train_controller and all_metas:
        n_metas = len(all_metas)
        for i, m in enumerate(all_metas):
            is_last = i == n_metas - 1
            if is_last:
                reward_for_step = step_reward
            else:
                round_num_i = i + 1
                round_tokens = comm_graph.round_token_counts.get(round_num_i, 0)
                token_penalty = -controller.lambda_info * round_tokens / max(float(tokens_if_full), 1.0)
                reward_for_step = token_penalty
            controller.store_transition(m, reward_for_step, done=is_last)
    result = SampleResult(question=question, gold_answer=gold_answer, final_answer=final_answer, is_correct=is_correct, exact_match=em_score, f1_score=f1, initial_answers=initial_answers, initial_correct=initial_correct, final_answers={r.agent_id: r.answer for r in current_responses}, initial_reasoning={r.agent_id: r.reasoning_text for r in round0_responses}, final_reasoning={r.agent_id: r.reasoning_text for r in current_responses}, raw_communications=all_comms, total_tokens_transmitted=tokens_transmitted, total_tokens_if_full=tokens_if_full, compression_ratio=compression_ratio, semantic_similarity=comm_graph.get_avg_semantic_similarity(), token_reduction=max(0.0, 1.0 - tokens_transmitted / max(1.0, tokens_if_full)), information_retention=comm_graph.get_avg_information_retention(), weight_matrices=[comm_graph.get_weight_snapshot()], communication_density=comm_graph.get_communication_density(), mode_distribution=comm_graph.get_mode_distribution(), n_active_edges=comm_graph.get_active_edges(), generation_tokens=total_gen_tokens, halt_round=t, agent_diversity=_agent_diversity, answer_entropy=_answer_entropy, average_delta_w=float(np.mean(_delta_ws)) if _delta_ws else 0.0, initial_majority_correct=_initial_majority_correct)
    return result
