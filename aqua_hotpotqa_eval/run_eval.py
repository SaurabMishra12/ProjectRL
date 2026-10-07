import os
import json
import copy
import logging
import numpy as np
from tqdm import tqdm

# Monkeypatch huggingface_hub URI validator to support root-level datasets like hotpot_qa
import huggingface_hub.utils._hf_uris as hf_uris
from huggingface_hub.utils._hf_uris import HfUri, HfUriError, unquote
from huggingface_hub.utils._validators import validate_repo_id, HFValidationError

def patched_post_init(self) -> None:
    uri = self._raw or ""
    if self.type not in hf_uris._VALID_URI_TYPES:
        raise HfUriError(uri=uri, msg=f"Invalid type '{self.type}'. Must be one of {sorted(hf_uris._VALID_URI_TYPES)}.")
    if not self.id:
        raise HfUriError(uri=uri, msg=f"Id must be 'namespace/name', got '{self.id}'.")
    if self.type != "bucket":
        try:
            validate_repo_id(self.id)
        except HFValidationError as e:
            raise HfUriError(uri=uri, msg=str(e)) from e
    if self.revision is not None and not self.revision:
        raise HfUriError(uri=uri, msg="Revision must not be an empty string.")
    if self.type == "bucket" and self.revision is not None:
        raise HfUriError(uri=uri, msg="Bucket URIs do not support a revision.")
    if self.path_in_repo:
        if self.path_in_repo.startswith("/") or "//" in self.path_in_repo:
            raise HfUriError(uri=uri, msg=f"Path must not contain empty segments (got '{self.path_in_repo}').")

HfUri.__post_init__ = patched_post_init

orig_parse_repo_body = hf_uris._parse_repo_body
def patched_parse_repo_body(location: str, type_, *, raw: str) -> HfUri:
    try:
        return orig_parse_repo_body(location, type_, raw=raw)
    except HfUriError as e:
        if "Repository id must be" in str(e) and "hotpot_qa" in location:
            location = location.strip("/")
            at_idx = location.find("@")
            if at_idx != -1:
                repo_id = location[:at_idx]
                rev_and_path = location[at_idx + 1 :]
                slash_idx = rev_and_path.find("/")
                if slash_idx == -1:
                    revision = rev_and_path
                    path_in_repo = ""
                else:
                    revision = rev_and_path[:slash_idx]
                    path_in_repo = rev_and_path[slash_idx + 1 :]
                revision = unquote(revision)
                return HfUri(
                    type=type_,
                    id=repo_id,
                    revision=revision,
                    path_in_repo=path_in_repo,
                    _raw=raw,
                )
        raise e

hf_uris._parse_repo_body = patched_parse_repo_body


# Import local modules
from config import get_aqua_config, get_hotpotqa_config
from controller import IBController
from pipeline import run_experiment

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def run_evaluation():
    print("=== STARTING ISOLATED TRANSFER-ONLY EVALUATION (400-TRAINED MODEL) ===")
    
    datasets = ["aqua", "hotpotqa"]
    seeds = [0, 1, 2]
    n_samples = 300
    
    api_key = "your-azure-api-key"
    llm_backend = "azure"
    api_model = "gpt-oss-120b"
    
    # Store aggregated results
    final_results = {}
    
    # Checkpoints mapped to absolute paths
    model_checkpoints = {
        "aqua": "/home/saurab/Documents/projectRLMAIN/08-06results/aqua_pretrained_400.pt.npz",
        "hotpotqa": "/home/saurab/Documents/projectRLMAIN/08-06results/hotpotqa_pretrained_400.pt.npz"
    }

    for ds in datasets:
        print(f"\n--------------------------------------")
        print(f"Evaluating Dataset: {ds.upper()} (3 Seeds × 300 samples)")
        print(f"--------------------------------------")
        
        # Load preset configuration
        if ds == "aqua":
            config = get_aqua_config(
                llm_backend=llm_backend,
                api_model=api_model,
                api_key=api_key,
                n_samples=n_samples
            )
        else:
            config = get_hotpotqa_config(
                llm_backend=llm_backend,
                api_model=api_model,
                api_key=api_key,
                n_samples=n_samples
            )
            
        checkpoint_path = model_checkpoints[ds]
        if not os.path.exists(checkpoint_path):
            print(f"ERROR: Checkpoint for {ds} not found at {checkpoint_path}!")
            continue
            
        print(f"Found checkpoint at {checkpoint_path}")
        
        all_accuracies = []
        all_token_costs = []
        
        method_name = f"Transfer_{ds.upper()}"
        
        for seed in seeds:
            print(f"\n--- Running Seed {seed} for {method_name} ---")
            
            # Clone config and set seed
            seed_config = copy.deepcopy(config)
            seed_config.seed = seed
            
            # Setup fresh controller and load checkpoints
            controller = IBController(seed_config)
            controller.load_model(checkpoint_path)
            
            # Run experiment
            results = run_experiment(
                config=seed_config,
                method_name=f"{method_name}_seed{seed}",
                train=False,
                evaluate=True,
                controller=controller
            )
            
            # Calculate accuracy and token cost from sample results
            sample_results = results.sample_results
            acc = np.mean([1.0 if r.is_correct else 0.0 for r in sample_results])
            tokens = np.mean([r.total_tokens_transmitted for r in sample_results])
            
            print(f"Seed {seed} finished | Accuracy: {acc*100:.2f}% | Avg Tokens: {tokens:.2f}")
            
            all_accuracies.append(acc)
            all_token_costs.append(tokens)
            
            # Save individual seed checkpoint results
            seed_ckpt_path = f"08-06results/seed_ckpt_{method_name}_seed{seed}.json"
            with open(seed_ckpt_path, "w", encoding="utf-8") as sf:
                json.dump({"accuracy": float(acc), "token_cost": float(tokens)}, sf, indent=2)
            print(f"Saved seed checkpoint to {seed_ckpt_path}")
            
        # Calculate stats with 95% Confidence Interval using t-distribution
        from scipy import stats as _stats
        acc_mean = float(np.mean(all_accuracies))
        acc_std = float(np.std(all_accuracies, ddof=1))
        t_crit = _stats.t.ppf(0.975, df=max(len(seeds) - 1, 1))
        acc_ci = t_crit * acc_std / np.sqrt(len(seeds))

        token_mean = float(np.mean(all_token_costs))
        token_std = float(np.std(all_token_costs, ddof=1))
        token_ci = t_crit * token_std / np.sqrt(len(seeds))
        
        final_results[method_name] = {
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
        
        print(f"\n--- Aggregated Results for {method_name} ---")
        print(f"  Accuracy: {acc_mean*100:.2f}% ± {acc_ci*100:.2f}% (std: {acc_std*100:.2f}%)")
        print(f"  Token Cost: {token_mean:.2f} ± {token_ci:.2f} (std: {token_std:.2f})")
        
    # Write aggregated final results to JSON file
    final_results_path = "08-06results/final_results.json"
    with open(final_results_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
        
    print(f"\nEvaluation completed. Results saved to {final_results_path}.")

if __name__ == "__main__":
    run_evaluation()
