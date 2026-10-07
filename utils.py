"""
Utility functions: seeding, logging, I/O, token counting.
"""

import json
import os
import random
import logging
from typing import Any, Dict

import numpy as np
import dataclasses


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except (ImportError, OSError):
        pass  # torch not available, skip


def setup_logging(level=logging.INFO):
    """Configure logging for the experiment."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("SparseComm")


def count_tokens(text: str) -> int:
    """Approximate token count (words * 1.3 as rough BPE estimate)."""
    if not text:
        return 0
    words = len(text.split())
    return max(1, int(words * 1.3))


def save_json(data: Any, filepath: str):
    """Save data as JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> Any:
    """Load JSON data."""
    with open(filepath, "r") as f:
        return json.load(f)


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def format_number(n: float, decimals: int = 2) -> str:
    """Format a number for display."""
    if isinstance(n, int):
        return str(n)
    return f"{n:.{decimals}f}"


def split_sentences(text: str) -> list:
    """Split text into sentences, aware of decimals and abbreviations, avoiding double punctuation."""
    if not text:
        return []
    
    # Try NLTK first
    try:
        import nltk
        sentences = nltk.sent_tokenize(text)
        cleaned = []
        for s in sentences:
            s_clean = s.strip()
            if s_clean:
                if s_clean[-1] not in ['.', '!', '?']:
                    s_clean += '.'
                cleaned.append(s_clean)
        return cleaned
    except Exception:
        pass
        
    # Fallback to custom abbreviation/decimal-aware splitter
    import re
    # Match a sentence ending punctuation [.!?] followed by whitespace
    raw_splits = re.split(r'(?<=[.!?])\s+', text)
    
    sentences = []
    curr = []
    abbrevs = {'eq.', 'fig.', 'i.e.', 'e.g.', 'approx.', 'vs.', 'dr.', 'mr.', 'ms.', 'no.', 'col.', 'vol.', 'al.'}
    
    for part in raw_splits:
        part = part.strip()
        if not part:
            continue
        curr.append(part)
        last_word = part.split()[-1].lower() if part.split() else ""
        if last_word in abbrevs or (len(last_word) == 2 and last_word[0].isalpha() and last_word[1] == '.'):
            # This is an abbreviation or single letter, do not split here
            continue
        else:
            sentence = " ".join(curr)
            if sentence[-1] not in ['.', '!', '?']:
                sentence += '.'
            sentences.append(sentence)
            curr = []
            
    if curr:
        sentence = " ".join(curr)
        if sentence[-1] not in ['.', '!', '?']:
            sentence += '.'
        sentences.append(sentence)
        
    return sentences


import enum
import dataclasses

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, enum.Enum):
            return obj.name
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        try:
            return str(obj)
        except Exception:
            return super().default(obj)


def log_continuous_result(method_name: str, dataset_name: str, sr: Any):
    import traceback
    try:
        os.makedirs("08-06results/continuous", exist_ok=True)
        filepath = f"08-06results/continuous/{method_name}_{dataset_name}_stream.jsonl"
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(dataclasses.asdict(sr), cls=NumpyEncoder) + "\n")
    except Exception as e:
        logger = logging.getLogger("SparseComm")
        logger.error(f"Failed to continuously save sample result: {e}")
        try:
            with open(f"08-06results/continuous/error_log.txt", "a") as f:
                f.write(traceback.format_exc() + "\n")
        except:
            pass

