import os
import json

abstranet_dir = r"c:\research work 25-05-26\LLM evolve\abstranet_mvp"
projectrl_dir = r"c:\Users\msaur\OneDrive\Desktop\projectRL"

files_to_read = [
    r"src\reasoning_graph.py",
    r"src\cognitive_state.py",
    r"src\semantic_extractor.py",
    r"src\ppo_controller.py",
    r"src\reward_engine.py",
    r"src\self_critic.py"
]

refactored_code = ""
for f in files_to_read:
    path = os.path.join(abstranet_dir, f)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as file:
            content = file.read()
            # Clean imports
            lines = content.split('\n')
            cleaned = []
            for line in lines:
                if line.startswith("import ") or line.startswith("from ") or line.startswith('"""'):
                    continue
                cleaned.append(line)
            refactored_code += "\n" + "\n".join(cleaned) + "\n"

# The original build_kaggle.py content:
original_build = r'''import os
import json

def construct_kaggle_notebook():
    cells = []
    
    def add_markdown(text):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": [text]})
        
    def add_code(text):
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [text]})

    add_markdown("# Sparse Semantic Communication via Information Bottleneck - Azure LLM Edition\nThis notebook natively uses the Azure OpenAI LLM backend, continuous execution checkpointing, and raw JSON dialogue extraction.")
    add_code("import sys\n!{sys.executable} -m pip install -q torch numpy tqdm networkx datasets accelerate transformers scikit-learn matplotlib openai httpx")
    
    # Imports
    imports_block = "import os\nimport json\nimport time\nimport math\nimport numpy as np\nfrom dataclasses import dataclass, field, asdict\nfrom typing import List, Dict, Tuple, Optional, Any, Set\nfrom collections import Counter\nimport networkx as nx\nfrom tqdm import tqdm\nimport logging\nfrom enum import Enum, auto\nimport re\nimport hashlib\nimport random\nfrom abc import ABC, abstractmethod\nimport torch\nimport transformers\nfrom concurrent.futures import ThreadPoolExecutor, as_completed\n"
    
    giant_code = imports_block + "\n"
    
    # Read files sequentially
    def read_clean(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            return "".join([line for line in lines if not line.startswith("import ") and not line.startswith("from ")])
            
    with open("config.py", "r", encoding="utf-8") as f:
        config_text = f.read()
    config_text = config_text.replace('api_model: str = "deepseek-ai/deepseek-v3.2"', 'api_model: str = "gpt-oss-120b"')
    
    giant_code += config_text + "\n"
    giant_code += read_clean("utils.py") + "\n"
    giant_code += read_clean("evaluation.py") + "\n"
    
    # LLM Backend modifications
    llm_text = read_clean("llm_backend.py")
    giant_code += llm_text + "\n"
    
    giant_code += read_clean("agents.py") + "\n"
    giant_code += read_clean("communication.py") + "\n"
    giant_code += read_clean("controller.py") + "\n"
    giant_code += read_clean("dataset_loaders.py") + "\n"
    
    pipe_text = read_clean("pipeline.py")
    giant_code += pipe_text + "\n"
    
    REFACTORED_INJECTION
    
    checkpoint_code = """
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        import enum
        # IMPORTANT: Enums are serialized as their .name string (e.g. "SUMMARIZE").
        # When loading this JSON for analysis, use safe_mode_cast() to rehydrate them.
        if isinstance(obj, enum.Enum):
            return obj.name
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

def safe_mode_cast(mode_str):
    \"\"\"
    Rehydrate a serialized SemanticBandwidthMode string back to its Enum member.
    Use this in analysis scripts when reading kaggle_latest_results.json.
    Example: record['mode'] = safe_mode_cast(record['mode'])
    \"\"\"
    try:
        return SemanticBandwidthMode[mode_str]
    except (KeyError, TypeError):
        return mode_str  # return raw string if cast fails (graceful degradation)

import os
os.makedirs("08-06results", exist_ok=True)
def save_kaggle_checkpoint(results_dict, filepath="08-06results/kaggle_latest_results.json"):
    try:
        from dataclasses import asdict
        import json
        output = {"detailed_samples": {}}
        for method, res in results_dict.items():
            if res is not None and hasattr(res, "sample_results"):
                output["detailed_samples"][method] = [asdict(s) for s in res.sample_results]
                
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, cls=NumpyEncoder)
            
    except Exception as e:
        print(f"Checkpoint save error: {e}")
"""
    giant_code += checkpoint_code + "\n"
    
    execution_block = """
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

config = Config()
# ACTIVATE AZURE LLM MODE NATIVELY
config.llm_backend = "azure"
config.api_model = "gpt-oss-120b"
config.dataset = "gsm8k" # overriden in loop
config.n_samples = 300
config.n_train_samples = 100
config.n_rounds = 3

print("Loading Environment...")
llm_backend = create_llm_backend(config)
agent_pool = AgentPool(config.n_agents, llm_backend, config.reasoning_styles)
comm_graph = CommunicationGraph(config.n_agents)
controller = PPOMetaController(num_agents=config.n_agents)

# =========================================================
# PHASE 1: UNIVERSAL PRE-TRAINING
# =========================================================
print("\\n=== PHASE 1: UNIVERSAL CONTROLLER PRE-TRAINING ===")
datasets_to_run = ["gsm8k", "hotpotqa", "math", "truthfulqa"]

for ds_name in datasets_to_run:
    print(f"\\n--- Pre-Training Dataset: {ds_name.upper()} ---")
    current_config = get_math_config() if ds_name == "math" else (get_hotpotqa_config() if ds_name == "hotpotqa" else get_gsm8k_config())
    current_config.llm_backend = "azure"
    current_config.api_model = "gpt-oss-120b"
    current_config.n_train_samples = 100
    current_config.n_samples = 300
    current_config.controller_epochs = 2
    current_config.dataset = ds_name
    
    # Train ONLY. Skip the expensive test split.
    _ = run_experiment(current_config, method_name=f"Pretrain_{ds_name.upper()}", train=True, evaluate=False, controller=controller)

# Save the universal master brain
import os
os.makedirs("08-06results", exist_ok=True)
import torch
torch.save(controller.model.state_dict(), "08-06results/universal_controller.pt")
print("✅ Universal Controller pre-trained and saved.")

# =========================================================
# PHASE 2: FROZEN TRANSFER EVALUATION
# =========================================================
print("\\n=== PHASE 2: FROZEN TRANSFER EVALUATION ===")
# Freeze the controller (disable exploration)
frozen_controller = PPOMetaController(num_agents=config.n_agents)
frozen_controller.model.load_state_dict(torch.load("08-06results/universal_controller.pt"))
frozen_controller.model.eval()

live_results = {}
for ds_name in datasets_to_run:
    print(f"\\n--- Evaluating Transfer to Dataset: {ds_name.upper()} ---")
    current_config = get_math_config() if ds_name == "math" else (get_hotpotqa_config() if ds_name == "hotpotqa" else get_gsm8k_config())
    current_config.llm_backend = "azure"
    current_config.api_model = "gpt-oss-120b"
    current_config.n_samples = 300
    current_config.n_train_samples = 100
    current_config.dataset = ds_name
    
    experiment_key = f"ZeroShot_Transfer_{ds_name.upper()}"
    
    # Evaluate ONLY. (train=False)
    sc_res = run_experiment(current_config, method_name=experiment_key, train=False, controller=frozen_controller)
    live_results[experiment_key] = sc_res
    save_kaggle_checkpoint(live_results)

print("\\n✅ Execution Complete! All exact dialogue chains are preserved in kaggle_latest_results.json")
import json
print("\\n--- FULL RESULT DUMP ---")
output_dump = {
    "config": asdict(config),
    "results": {k: v.__dict__ for k, v in live_results.items()}
}
print(json.dumps(output_dump, cls=NumpyEncoder, indent=2))
"""
    giant_code += execution_block
    
    add_code(giant_code)
    
    notebook = {
        "cells": cells,
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    with open("SparseComm_Kaggle.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
        
    print("Notebook compiled exactly to specifications.")

if __name__ == "__main__":
    construct_kaggle_notebook()
'''

safe_code_repr = repr(refactored_code)
injection = f"    giant_code += {safe_code_repr}\n"

final_build_script = original_build.replace("    REFACTORED_INJECTION\n", injection)

build_script_path = os.path.join(projectrl_dir, "build_kaggle.py")
with open(build_script_path, "w", encoding="utf-8") as f:
    f.write(final_build_script)
    
print("build_kaggle.py successfully restored and updated.")
