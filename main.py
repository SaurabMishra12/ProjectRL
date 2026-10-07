"""
Main Entry Point: CLI interface for running experiments.

Usage:
    python main.py --dataset gsm8k --mode all --n-samples 20
    python main.py --dataset gsm8k --mode all --n-samples 20 --llm nvidia
    python main.py --dataset hotpotqa --mode baseline --n-samples 50
"""

import argparse
import os
import sys
import json
import time

from config import Config, get_gsm8k_config, get_hotpotqa_config, get_math_config
from pipeline import run_experiment
from baselines import run_all_baselines
from ablation import run_all_ablations
from evaluation import (
    aggregate_metrics,
    format_results_table,
    format_ablation_table,
)
from visualization import generate_all_plots
from analysis import run_full_analysis, format_analysis_report
from utils import set_seed, setup_logging, save_json, ensure_dir

logger = setup_logging()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sparse Semantic Communication via IB Control"
    )
    parser.add_argument(
        "--dataset", type=str, default="gsm8k",
        choices=["gsm8k", "hotpotqa", "math"],
        help="Dataset to use"
    )
    parser.add_argument(
        "--mode", type=str, default="all",
        choices=["full", "baseline", "ablation", "all"],
        help="Experiment mode"
    )
    parser.add_argument(
        "--n-samples", type=int, default=20,
        help="Number of test samples"
    )
    parser.add_argument(
        "--n-train", type=int, default=15,
        help="Number of training samples for controller"
    )
    parser.add_argument(
        "--n-agents", type=int, default=3,
        help="Number of agents"
    )
    parser.add_argument(
        "--n-rounds", type=int, default=3,
        help="Number of communication rounds"
    )
    parser.add_argument(
        "--llm", type=str, default="mock",
        choices=["mock", "nvidia", "api", "hf", "azure"],
        help="LLM backend to use"
    )
    parser.add_argument(
        "--api-model", type=str, default="deepseek-ai/deepseek-v3.2",
        help="Model string to use for API backends"
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="API Key override"
    )
    parser.add_argument(
        "--api-base-url", type=str, default=None,
        help="API Base URL override"
    )
    parser.add_argument(
        "--lambda-info", type=float, default=0.3,
        help="Information cost weight lambda"
    )
    parser.add_argument(
        "--gamma-loop", type=float, default=0.1,
        help="Loop penalty weight gamma"
    )
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="Controller training epochs"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--info-cost", type=str, default="token_count",
        choices=["token_count", "compression_ratio"],
        help="Information cost method"
    )
    parser.add_argument(
        "--results-dir", type=str, default="results",
        help="Directory to save results"
    )
    return parser.parse_args()


def build_config(args) -> Config:
    """Build config from CLI arguments."""
    if args.dataset == "gsm8k":
        config = get_gsm8k_config()
    elif args.dataset == "hotpotqa":
        config = get_hotpotqa_config()
    else:
        config = get_math_config()

    # Override with CLI args
    config.n_samples = args.n_samples
    config.n_train_samples = args.n_train
    config.n_agents = args.n_agents
    config.n_rounds = args.n_rounds
    config.llm_backend = args.llm
    if args.api_model:
        config.api_model = args.api_model
    if args.api_key:
        config.api_key = args.api_key
    if getattr(args, 'api_base_url', None):
        config.api_base_url = args.api_base_url
    if getattr(args, 'seed', None):
        config.seed = args.seed
        
    config.lambda_info = args.lambda_info
    config.gamma_loop = args.gamma_loop
    config.controller_epochs = args.epochs
    config.seed = args.seed
    config.info_cost_method = args.info_cost
    config.mode = args.mode

    if getattr(args, 'results_dir', None):
        config.results_dir = args.results_dir
        config.plots_dir = os.path.join(args.results_dir, "plots")

    # Adjust reasoning styles for n_agents
    all_styles = ["analytical", "step_by_step", "creative", "careful", "efficient"]
    config.reasoning_styles = all_styles[:config.n_agents]

    return config


def run_full_experiment(config: Config):
    """Run the complete experiment pipeline."""
    start_time = time.time()

    ensure_dir(config.results_dir)
    ensure_dir(config.plots_dir)

    all_results = {}
    all_metrics = []

    # ============================================
    # 1. IB-Controller (main method)
    # ============================================
    if config.mode in ("full", "all"):
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 1: IB-Controller Experiment")
        logger.info("=" * 60)

        main_results = run_experiment(
            config, method_name="IB-Controller", train=True
        )
        main_metrics = aggregate_metrics(main_results)
        all_results["ib_controller"] = main_results
        all_metrics.append(main_metrics)

        logger.info(f"IB-Controller Accuracy: {main_metrics['accuracy_pct']}")
        logger.info(f"Avg Tokens: {main_metrics['token_cost']['avg_tokens_per_sample']:.1f}")
    else:
        main_results = None

    # ============================================
    # 2. Baselines
    # ============================================
    if config.mode in ("baseline", "all"):
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2: Baseline Experiments")
        logger.info("=" * 60)

        baseline_results = run_all_baselines(config)
        baseline_metrics = [aggregate_metrics(r) for r in baseline_results]
        all_results["baselines"] = baseline_results
        all_metrics.extend(baseline_metrics)

        for bm in baseline_metrics:
            logger.info(f"  {bm['method']}: {bm['accuracy_pct']} acc, "
                        f"{bm['token_cost']['avg_tokens_per_sample']:.1f} tokens")
    else:
        baseline_results = []
        baseline_metrics = []

    # ============================================
    # 3. Ablation Studies
    # ============================================
    if config.mode in ("ablation", "all"):
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 3: Ablation Studies")
        logger.info("=" * 60)

        ablation_results = run_all_ablations(config)
        ablation_metrics = [aggregate_metrics(r) for r in ablation_results]
        all_results["ablations"] = ablation_results

        for am in ablation_metrics:
            logger.info(f"  {am['method']}: {am['accuracy_pct']} acc")
    else:
        ablation_results = []
        ablation_metrics = []

    # ============================================
    # 4. Results Tables
    # ============================================
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)

    if all_metrics:
        table = format_results_table(all_metrics, config.dataset)
        print(table)

        # Save table
        with open(os.path.join(config.results_dir, f"results_table_{config.dataset}.md"), "w") as f:
            f.write(table)

    if ablation_metrics:
        abl_table = format_ablation_table(ablation_metrics, config.dataset)
        print(abl_table)

        with open(os.path.join(config.results_dir, f"ablation_table_{config.dataset}.md"), "w") as f:
            f.write(abl_table)

    # ============================================
    # 5. Visualizations
    # ============================================
    if main_results and config.mode == "all":
        logger.info("\nGenerating plots...")
        plot_data = generate_all_plots(
            main_results=main_results,
            baseline_results=baseline_results,
            ablation_results=ablation_results,
            plots_dir=config.plots_dir,
            dataset_name=config.dataset,
        )

    # ============================================
    # 6. Analysis
    # ============================================
    if main_results and baseline_results:
        logger.info("\nRunning analysis...")
        analysis = run_full_analysis(
            main_results, baseline_results, ablation_results
        )
        report = format_analysis_report(analysis, config.dataset)
        print(report)

        save_json(analysis, os.path.join(config.results_dir, f"analysis_{config.dataset}.json"))
        with open(os.path.join(config.results_dir, f"analysis_report_{config.dataset}.txt"), "w") as f:
            f.write(report)

    # ============================================
    # 7. Save Raw Results
    # ============================================
    raw_results = {}
    if all_metrics:
        raw_results["metrics"] = all_metrics
    if ablation_metrics:
        raw_results["ablation_metrics"] = ablation_metrics

    from dataclasses import asdict
    raw_results["detailed_samples"] = {}

    if "ib_controller" in all_results and all_results["ib_controller"]:
        raw_results["detailed_samples"]["ib_controller"] = [asdict(s) for s in all_results["ib_controller"].sample_results]
        
    if "baselines" in all_results:
        raw_results["detailed_samples"]["baselines"] = {
            res.method_name: [asdict(s) for s in res.sample_results] for res in all_results["baselines"]
        }
        
    if "ablations" in all_results:
        raw_results["detailed_samples"]["ablations"] = {
            res.method_name: [asdict(s) for s in res.sample_results] for res in all_results["ablations"]
        }

    # Save config
    raw_results["config"] = {
        "dataset": config.dataset,
        "n_agents": config.n_agents,
        "n_rounds": config.n_rounds,
        "n_samples": config.n_samples,
        "n_train_samples": config.n_train_samples,
        "llm_backend": config.llm_backend,
        "lambda_info": config.lambda_info,
        "gamma_loop": config.gamma_loop,
        "info_cost_method": config.info_cost_method,
        "seed": config.seed,
    }

    save_json(raw_results, os.path.join(config.results_dir, f"full_results_{config.dataset}.json"))

    elapsed = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"Experiment complete in {elapsed:.1f}s")
    logger.info(f"Results saved to {config.results_dir}/")
    logger.info(f"Plots saved to {config.plots_dir}/")
    logger.info(f"{'='*60}")

    return all_results


if __name__ == "__main__":
    args = parse_args()
    config = build_config(args)

    logger.info(f"Configuration:")
    logger.info(f"  Dataset: {config.dataset}")
    logger.info(f"  Mode: {config.mode}")
    logger.info(f"  LLM Backend: {config.llm_backend}")
    logger.info(f"  N Agents: {config.n_agents}")
    logger.info(f"  N Rounds: {config.n_rounds}")
    logger.info(f"  N Samples: {config.n_samples}")
    logger.info(f"  N Train: {config.n_train_samples}")
    logger.info(f"  Lambda: {config.lambda_info}")
    logger.info(f"  Gamma: {config.gamma_loop}")
    logger.info(f"  Info Cost: {config.info_cost_method}")
    logger.info(f"  Seed: {config.seed}")

    run_full_experiment(config)
