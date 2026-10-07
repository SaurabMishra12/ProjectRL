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
        