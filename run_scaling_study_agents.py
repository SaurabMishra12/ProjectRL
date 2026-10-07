import os
import sys
import json
import time
import threading
import numpy as np
from copy import deepcopy

# Add current directory to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from config import get_gsm8k_config
from controller import IBController
from pipeline import run_experiment, run_full_communication, run_no_communication
from agents import AgentPool
from communication import CommunicationGraph
from evaluation import ExperimentResults, SampleResult, aggregate_metrics

# =====================================================================
# PERSISTENT DISK CACHE FOR AZURE OPENAI (THREAD-SAFE)
# =====================================================================
CACHE_PATH = "08-06results/azure_prompt_cache.json"
disk_cache = {}
cache_lock = threading.Lock()

if os.path.exists(CACHE_PATH):
    try:
        with open(CACHE_PATH, "r") as f:
            disk_cache = json.load(f)
        print(f"Loaded {len(disk_cache)} cached prompt responses from disk.")
    except Exception as e:
        print("Error loading disk cache:", e)

from llm_backend import AzureOpenAIBackend, LLMResponse

# Monkeypatch generate
original_generate = AzureOpenAIBackend.generate

def cached_generate(self, prompt, question, context="", dataset_type="gsm8k", agent_id=0):
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

    cache_key = f"generate|{self.model}|{self.max_tokens}|{self.temperature}|{system_msg}|{user_msg}"
    
    with cache_lock:
        if cache_key in disk_cache:
            cached_data = disk_cache[cache_key]
            return LLMResponse(
                text=cached_data["text"],
                answer=cached_data["answer"],
                token_count=cached_data["token_count"],
                confidence=cached_data["confidence"]
            )
        
    resp = original_generate(self, prompt, question, context, dataset_type, agent_id)
    
    # Only cache valid responses
    if resp.answer or resp.text:
        with cache_lock:
            disk_cache[cache_key] = {
                "text": resp.text,
                "answer": resp.answer,
                "token_count": resp.token_count,
                "confidence": resp.confidence
            }
            try:
                with open(CACHE_PATH, "w") as f:
                    json.dump(disk_cache, f, indent=2)
            except Exception as e:
                print("Failed to save disk cache:", e)
            
    return resp

AzureOpenAIBackend.generate = cached_generate

# Monkeypatch compress_message
original_compress = AzureOpenAIBackend.compress_message

def cached_compress_message(self, message, mode):
    cache_key = f"compress|{self.model}|{mode}|{message}"
    
    with cache_lock:
        if cache_key in disk_cache:
            return disk_cache[cache_key]
        
    compressed = original_compress(self, message, mode)
    
    if compressed:
        with cache_lock:
            disk_cache[cache_key] = compressed
            try:
                with open(CACHE_PATH, "w") as f:
                    json.dump(disk_cache, f, indent=2)
            except Exception as e:
                print("Failed to save disk cache:", e)
            
    return compressed

AzureOpenAIBackend.compress_message = cached_compress_message
# =====================================================================

class MultiAgentScaleController(IBController):
    """
    Wrapper controller that maps state features and action vectors
    between N_eval agents (2, 4, or 5) and the trained N=3 controller.
    """
    def __init__(self, config, base_controller):
        super().__init__(config)
        self.base_controller = base_controller
        self.n_eval_agents = config.n_agents
        # Share the fitted calibrator
        self.calibrator = base_controller.calibrator

    def get_weights(self, reasoning_texts, answers, confidence_scores, prev_W, round_num, unique_answers=1, avg_accuracy=0.5, explore=False):
        N_eval = self.n_eval_agents
        
        # 1. Map input features from N_eval to N=3
        if N_eval == 2:
            # Pad inputs
            padded_confs = confidence_scores + [0.5]
            padded_W = np.zeros((3, 3))
            padded_W[:2, :2] = prev_W
            padded_texts = {0: reasoning_texts.get(0, ""), 1: reasoning_texts.get(1, ""), 2: ""}
            padded_answers = {0: answers.get(0, ""), 1: answers.get(1, ""), 2: ""}
            
            W_3, Q_3, halt_decision, meta = self.base_controller.get_weights(
                reasoning_texts=padded_texts,
                answers=padded_answers,
                confidence_scores=padded_confs,
                prev_W=padded_W,
                round_num=round_num,
                unique_answers=unique_answers,
                avg_accuracy=avg_accuracy,
                explore=False
            )
            
            # Slice output back to N_eval = 2
            W_eval = W_3[:2, :2].copy()
            np.fill_diagonal(W_eval, 0.0)
            Q_eval = Q_3[:2].copy()
            
        elif N_eval == 3:
            # Direct pass-through
            W_eval, Q_eval, halt_decision, meta = self.base_controller.get_weights(
                reasoning_texts=reasoning_texts,
                answers=answers,
                confidence_scores=confidence_scores,
                prev_W=prev_W,
                round_num=round_num,
                unique_answers=unique_answers,
                avg_accuracy=avg_accuracy,
                explore=False
            )
            
        else: # N_eval = 4 or 5
            # Slice inputs
            sliced_confs = confidence_scores[:3]
            sliced_W = prev_W[:3, :3]
            sliced_texts = {i: reasoning_texts.get(i, "") for i in range(3)}
            sliced_answers = {i: answers.get(i, "") for i in range(3)}
            
            W_3, Q_3, halt_decision, meta = self.base_controller.get_weights(
                reasoning_texts=sliced_texts,
                answers=sliced_answers,
                confidence_scores=sliced_confs,
                prev_W=sliced_W,
                round_num=round_num,
                unique_answers=unique_answers,
                avg_accuracy=avg_accuracy,
                explore=False
            )
            
            # Expand output to N_eval
            Q_eval = np.zeros((N_eval, 16))
            Q_eval[:3] = Q_3
            for i in range(3, N_eval):
                Q_eval[i] = Q_3[i % 3]
                
            W_eval = np.zeros((N_eval, N_eval))
            for i in range(N_eval):
                for j in range(N_eval):
                    if i != j:
                        W_eval[i, j] = W_3[i % 3, j % 3]
                        if i % 3 == j % 3:
                            W_eval[i, j] = 0.5  # default fallback connection
            np.fill_diagonal(W_eval, 0.0)
            
        return W_eval, Q_eval, halt_decision, meta

def main():
    print("=" * 60)
    print("Agent Count Scaling Study - SparseComm Only (N = 2, 4, 5)")
    print("Dataset: GSM8K (150 samples)")
    print("Trained Model: 400-trained GSM8K Controller (Epoch 5)")
    print("=" * 60)

    # 1. Load the 400-sample trained controller
    base_config = get_gsm8k_config()
    base_config.n_agents = 3
    base_controller = IBController(base_config)
    
    model_path = "08-06results/gsm8k_pretrained_400.pt_Pretrain_GSM8K_400_epoch_5.npz"
    print(f"Loading base model from {model_path}...")
    base_controller.load_model(model_path)
    if base_controller.calibrator and base_controller.calibrator._fitted:
        print("Calibrator loaded and fitted successfully.")
    else:
        print("Calibrator not fitted yet or missing. Evaluation will continue with default confidences.")

    results_dir = "08-06results"
    os.makedirs(results_dir, exist_ok=True)
    
    agent_counts = [2, 4, 5]
    n_samples = 150
    seed = 42

    scaling_results = {}

    for N in agent_counts:
        print(f"\n" + "-"*50)
        print(f"Evaluating N = {N} agents (SparseComm)...")
        print("-"*50)
        
        # Configure for N agents
        config = get_gsm8k_config()
        config.n_agents = N
        config.n_samples = n_samples
        config.seed = seed
        config.llm_backend = "azure"
        config.api_model = "gpt-oss-120b"
        
        # Setup reasoning styles
        styles = ["analytical", "step_by_step", "creative", "careful", "efficient"]
        config.reasoning_styles = [styles[i % len(styles)] for i in range(N)]
        
        # Setup wrapper controller
        wrapper_controller = MultiAgentScaleController(config, base_controller)
        
        # SparseComm (Our Method)
        print(f"Running SparseComm (N={N})...")
        sparse_comm_res = run_experiment(
            config=config,
            method_name=f"IB-Controller_N{N}",
            train=False,
            evaluate=True,
            controller=wrapper_controller
        )
        sparse_comm_metrics = aggregate_metrics(sparse_comm_res)
        print(f"  Accuracy: {sparse_comm_metrics['accuracy_pct']}")
        print(f"  Avg Tokens: {sparse_comm_metrics['token_cost']['avg_tokens_per_sample']:.1f}")
        print(f"  Comm Density: {sparse_comm_metrics['communication_density']:.3f}")
        
        # Store metrics
        scaling_results[str(N)] = {
            "SparseComm": sparse_comm_metrics
        }
        
        # Intermediate save
        with open(os.path.join(results_dir, "scaling_study_agents_temp.json"), "w") as f:
            json.dump(scaling_results, f, indent=2, default=str)

    # Save final results
    final_path = os.path.join(results_dir, "scaling_study_agents.json")
    with open(final_path, "w") as f:
        json.dump(scaling_results, f, indent=2, default=str)
    print(f"\nStudy complete! Results saved to {final_path}")

    # Generate Markdown Table
    print("\n" + "="*40)
    print("SUMMARY RESULTS TABLE (SparseComm Only)")
    print("="*40)
    
    md_table = "| Agents (N) | Method | Accuracy | Avg Tokens | Comm Density | Avg Rounds | Error Prop | Sem. Sim | Info Ret |\n"
    md_table += "|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|\n"
    
    for N in agent_counts:
        n_str = str(N)
        metrics = scaling_results[n_str]["SparseComm"]
        md_table += f"| {N} | SparseComm (Ours) | {metrics['accuracy_pct']} | {metrics['token_cost']['avg_tokens_per_sample']:.1f} | {metrics['communication_density']:.3f} | {metrics['average_rounds_used']:.2f} | {metrics['error_propagation_rate']:.3f} | {metrics['semantic_similarity']:.3f} | {metrics['information_retention']:.3f} |\n"
            
    print(md_table)
    
    # Save md table to results
    with open(os.path.join(results_dir, "scaling_study_agents_table.md"), "w") as f:
        f.write(md_table)

if __name__ == "__main__":
    main()
