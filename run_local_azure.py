import os
import json
import logging
from dataclasses import asdict
import numpy as np

# Import from local project
from config import Config
from llm_backend import create_llm_backend
from agents import AgentPool
from communication import CommunicationGraph
from controller import IBController
from pipeline import run_experiment

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        try:
            return obj.__dict__
        except Exception:
            return super().default(obj)

def save_checkpoint(results_dict, filepath):
    try:
        output = {"detailed_samples": {}}
        for method, res in results_dict.items():
            if res is not None and hasattr(res, "sample_results"):
                output["detailed_samples"][method] = [asdict(s) for s in res.sample_results]
                
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, cls=NumpyEncoder)
    except Exception as e:
        print(f"Checkpoint save error: {e}")

def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    # Set up config for the user's specific request
    config = Config()
    config.llm_backend = "azure"
    config.api_model = "gpt-oss-120b"
    config.n_samples = 10  # 10 test samples
    config.n_train_samples = 5  # 5 train samples (total 15 samples per dataset)
    config.n_rounds = 3
    
    os.makedirs("local_azure_outputs", exist_ok=True)
    checkpoint_file = "local_azure_outputs/live_results_checkpoint.json"

    datasets_to_run = ["gsm8k", "hotpotqa", "math", "truthfulqa"]
    live_results = {}

    print("=== EXECUTING LOCAL AZURE RUN ===")
    print(f"Total samples per dataset: {config.n_samples + config.n_train_samples} (Train: {config.n_train_samples}, Test: {config.n_samples})")

    for ds_name in datasets_to_run:
        config.dataset = ds_name
        print(f"\n--- Starting Dataset: {ds_name.upper()} ---")
        experiment_key = f"Self-Consistency_{ds_name.upper()}"
        
        # Run the full pipeline
        sc_res = run_experiment(config, method_name=experiment_key, train=True)
        
        # Save checkpoint continuously
        live_results[experiment_key] = sc_res
        save_checkpoint(live_results, checkpoint_file)
        print(f"Checkpoint saved to {checkpoint_file}")

    print("\n✅ Execution Complete! All exact dialogue chains and outputs are preserved locally.")
    
    # Save a massive final dump
    output_dump = {
        "config": asdict(config),
        "results": {k: v.__dict__ for k, v in live_results.items()}
    }
    
    final_dump_file = "local_azure_outputs/final_full_dump.json"
    with open(final_dump_file, "w", encoding="utf-8") as f:
        json.dump(output_dump, f, cls=NumpyEncoder, indent=2)
    print(f"Full dump saved to {final_dump_file}")

if __name__ == "__main__":
    main()
