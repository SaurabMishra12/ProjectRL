
toy_code = """
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("SparseComm")

datasets_to_run = ["gsm8k", "hotpotqa", "math"]

print("\\n=== TOY RUN (15 samples per dataset) ===")
test_controller = None
for ds_name in datasets_to_run:
    print(f"\\n--- Toy Run: {ds_name.upper()} ---")
    current_config = get_math_config() if ds_name == "math" else (get_hotpotqa_config() if ds_name == "hotpotqa" else get_gsm8k_config())
    current_config.llm_backend = "azure"
    current_config.api_model = "gpt-oss-120b"
    current_config.n_train_samples = 15
    current_config.n_samples = 15
    current_config.controller_epochs = 1
    current_config.dataset = ds_name
    
    if test_controller is None:
        test_controller = IBController(current_config)

    run_experiment(
        current_config,
        method_name=f"Toy_{ds_name.upper()}",
        train=True,
        evaluate=True,
        controller=test_controller
    )
print("[OK] Toy run completed successfully.")
"""

with open("run_toy.py", "r", encoding="utf-8") as f:
    code = f.read()

# Add necessary imports at the top
imports = """
import sys
import os
import json
import time
import math
import numpy as np
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
"""

with open("run_toy.py", "w", encoding="utf-8") as f:
    f.write(imports + "\n" + code + "\n" + toy_code)

