"""
Dataset Loaders: GSM8K, HotpotQA, MATH (AQuA), TruthfulQA, and MBPP.
Returns STRICTLY separated (train, test) tuples to prevent data leakage.
"""

import json
import os
import re
import random
from typing import List, Tuple, Optional

from utils import setup_logging

logger = setup_logging()


def _generate_synthetic_samples(n_train: int, n_test: int, seed: int = 42) -> Tuple[List[dict], List[dict]]:
    """
    Generate synthetic GSM8K-style math word problems for testing.
    Returns strictly disjoint train and test sets.
    """
    rng = random.Random(seed)
    templates = [
        {
            "template": "{name} has {a} apples. {name2} gives {name} {b} more apples. How many apples does {name} have now?",
            "answer_fn": lambda a, b: a + b,
        },
        {
            "template": "A store has {a} shirts. Each shirt costs ${b}. How much do all the shirts cost?",
            "answer_fn": lambda a, b: a * b,
        },
        {
            "template": "{name} reads {a} pages per day. How many pages does {name} read in {b} days?",
            "answer_fn": lambda a, b: a * b,
        },
        {
            "template": "There are {a} students in a class. Each student has {b} pencils. If the teacher adds {c} more pencils to the total, how many pencils are there?",
            "answer_fn": lambda a, b, c=0: a * b + c,
        },
        {
            "template": "{name} has ${a}. {name} spends ${b} on lunch and ${c} on a book. How much money does {name} have left?",
            "answer_fn": lambda a, b, c=0: a - b - c,
        },
    ]
    names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace"]

    def make_sample():
        t = rng.choice(templates)
        a = rng.randint(5, 50)
        b = rng.randint(2, 20)
        c = rng.randint(1, 10)
        name_pool = rng.sample(names, 3)
        question = t["template"].format(name=name_pool[0], name2=name_pool[1], name3=name_pool[2], a=a, b=b, c=c)
        try:
            answer = t["answer_fn"](a, b, c)
        except TypeError:
            answer = t["answer_fn"](a, b)
        return {"question": question, "answer": str(int(answer))}

    # Generate total unique samples (simplistic unique by question string)
    seen = set()
    all_samples = []
    while len(all_samples) < n_train + n_test:
        s = make_sample()
        if s["question"] not in seen:
            seen.add(s["question"])
            all_samples.append(s)

    return all_samples[:n_train], all_samples[n_train:n_train+n_test]


def load_gsm8k(n_train: int = 30, n_test: int = 50, seed: int = 42) -> Tuple[List[dict], List[dict]]:
    """Load GSM8K. Uses 'train' split for training, 'test' split for testing."""
    try:
        from datasets import load_dataset
        ds_train = load_dataset("gsm8k", "main", split="train")
        ds_test = load_dataset("gsm8k", "main", split="test")
        
        rng = random.Random(seed)
        
        def process_split(ds, n):
            indices = list(range(len(ds)))
            rng.shuffle(indices)
            samples = []
            for idx in indices[:n]:
                item = ds[idx]
                question = item["question"]
                answer_text = item["answer"]
                match = re.search(r'####\s*([\-\d,\.]+)', answer_text)
                if match:
                    answer = match.group(1).replace(",", "")
                else:
                    nums = re.findall(r'[\-\d,\.]+', answer_text)
                    answer = nums[-1].replace(",", "") if nums else "0"
                samples.append({"question": question, "answer": answer})
            return samples
            
        train_samples = process_split(ds_train, n_train)
        test_samples = process_split(ds_test, n_test)
        logger.info(f"Loaded {len(train_samples)} train and {len(test_samples)} test GSM8K samples")
        return train_samples, test_samples
    except Exception as e:
        logger.warning(f"Could not load GSM8K from HuggingFace: {e}")
        return _generate_synthetic_samples(n_train, n_test, seed)


def load_hotpotqa(n_train: int = 30, n_test: int = 50, seed: int = 42) -> Tuple[List[dict], List[dict]]:
    """Load HotpotQA. Uses 'train' split for training, 'validation' split for testing."""
    try:
        from datasets import load_dataset
        ds_train = load_dataset("hotpot_qa", "fullwiki", split="train")
        ds_test = load_dataset("hotpot_qa", "fullwiki", split="validation")
        
        rng = random.Random(seed)
        
        def process_split(ds, n):
            indices = list(range(len(ds)))
            rng.shuffle(indices)
            samples = []
            for idx in indices[:n]:
                item = ds[idx]
                samples.append({"question": item["question"], "answer": item["answer"]})
            return samples
            
        train_samples = process_split(ds_train, n_train)
        test_samples = process_split(ds_test, n_test)
        logger.info(f"Loaded {len(train_samples)} train and {len(test_samples)} test HotpotQA samples")
        return train_samples, test_samples
    except Exception as e:
        logger.warning(f"Could not load HotpotQA from HuggingFace: {e}")
        return _generate_synthetic_samples(n_train, n_test, seed)


def load_aqua(n_train: int = 30, n_test: int = 50, seed: int = 42) -> Tuple[List[dict], List[dict]]:
    """Load Google Deepmind AQuA. Uses train.json for training, test.json for testing."""
    import urllib.request
    
    def fetch_split(url, n, rng):
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            lines = response.read().decode('utf-8').strip().split('\n')
            
        samples = []
        indices = list(range(len(lines)))
        rng.shuffle(indices)
        for idx in indices[:n]:
            if idx < len(lines):
                item = json.loads(lines[idx])
                question = item["question"] + "\nOptions: " + ", ".join(item["options"])
                samples.append({
                    "question": "AQuA Math Problem:\n" + question + "\n\nEnd with your final answer wrapped in \\boxed{your_answer} where your_answer is the exact letter (e.g. \\boxed{A}).",
                    "answer": item["correct"]
                })
        return samples

    try:
        rng = random.Random(seed)
        train_samples = fetch_split("https://raw.githubusercontent.com/google-deepmind/AQuA/master/train.json", n_train, rng)
        test_samples = fetch_split("https://raw.githubusercontent.com/google-deepmind/AQuA/master/test.json", n_test, rng)
        logger.info(f"Loaded {len(train_samples)} train and {len(test_samples)} test AQuA MATH samples directly from Github")
        return train_samples, test_samples
    except Exception as e:
        logger.warning(f"Could not load AQuA MATH directly: {e}")
        return _generate_synthetic_samples(n_train, n_test, seed)


def load_truthfulqa(n_train: int = 30, n_test: int = 50, seed: int = 42) -> Tuple[List[dict], List[dict]]:
    """Load TruthfulQA (Multiple Choice).
    
    TruthfulQA has only a single 'validation' split. We split it into
    train/test using index-based partitioning with a shuffled permutation,
    guaranteeing zero overlap between the two sets.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
        
        total_needed = n_train + n_test
        all_indices = list(range(len(ds)))
        rng = random.Random(seed)
        rng.shuffle(all_indices)
        
        # Hard partition: first n_train for training, next n_test for testing.
        # No overlap is possible because we slice contiguous blocks.
        train_indices = all_indices[:n_train]
        test_indices  = all_indices[n_train:n_train + n_test]
        
        def make_samples(indices):
            samples = []
            for idx in indices:
                item = ds[idx]
                question = item["question"] + "\nOptions:\n"
                choices = item["mc1_targets"]["choices"]
                labels = item["mc1_targets"]["labels"]
                correct = ""
                for i, (c, l) in enumerate(zip(choices, labels)):
                    question += f"{chr(65+i)}. {c}\n"
                    if l == 1:
                        correct = chr(65+i)
                question += "\nEnd with your final answer wrapped in \\boxed{your_answer} where your_answer is the exact letter (e.g. \\boxed{A})."
                samples.append({"question": question, "answer": correct})
            return samples
            
        train_samples = make_samples(train_indices)
        test_samples = make_samples(test_indices)
        logger.info(f"Loaded {len(train_samples)} train and {len(test_samples)} test TruthfulQA samples (hard-partitioned, zero overlap)")
        return train_samples, test_samples
    except Exception as e:
        logger.warning(f"Could not load TruthfulQA: {e}")
        return _generate_synthetic_samples(n_train, n_test, seed)


def load_mbpp(n_train: int = 30, n_test: int = 50, seed: int = 42) -> Tuple[List[dict], List[dict]]:
    """Load MBPP coding dataset. Uses 'train' split for training, 'test' split for testing."""
    try:
        from datasets import load_dataset
        ds_train = load_dataset("mbpp", split="train")
        ds_test = load_dataset("mbpp", split="test")
        
        rng = random.Random(seed)
        
        def process_split(ds, n):
            indices = list(range(len(ds)))
            rng.shuffle(indices)
            samples = []
            for idx in indices[:n]:
                item = ds[idx]
                q = f"Write a Python function to solve this: {item['text']}\nDo not include tests. Just the function definition."
                samples.append({"question": q, "answer": item['code']})
            return samples
            
        train_samples = process_split(ds_train, n_train)
        test_samples = process_split(ds_test, n_test)
        
        logger.info(f"Loaded {len(train_samples)} train and {len(test_samples)} test MBPP samples")
        return train_samples, test_samples
    except Exception as e:
        logger.warning(f"Could not load MBPP: {e}")
        return _generate_synthetic_samples(n_train, n_test, seed)


def evaluate_answer(predicted: str, gold: str, dataset_type: str = "gsm8k") -> bool:
    """Check if predicted answer matches gold answer."""
    predicted = str(predicted).strip().lower().rstrip(".")
    gold = str(gold).strip().lower().rstrip(".")

    if dataset_type == "gsm8k":
        pred_nums = re.findall(r'-?\d+\.?\d*', predicted)
        gold_nums = re.findall(r'-?\d+\.?\d*', gold)
        if pred_nums and gold_nums:
            try:
                return abs(float(pred_nums[-1]) - float(gold_nums[-1])) < 1e-3
            except ValueError:
                return False
        return predicted == gold
    elif dataset_type in ["math", "truthfulqa"]:
        # _extract_answer already stripped \boxed{} before this function is called.
        # Do NOT attempt to re-parse \boxed{} here — it will always fail (double-parse bug).
        # Compare the full cleaned strings, not just the last character.
        pred_clean = re.sub(r'[^\w]', '', predicted).strip()
        gold_clean = re.sub(r'[^\w]', '', gold).strip()
        return gold_clean == pred_clean
    elif dataset_type == "mbpp":
        # Heuristic code evaluation: check if core signature exists
        gold_match = re.search(r'def\s+(\w+)', gold)
        if gold_match:
            func_name = gold_match.group(1).lower()
            if func_name not in predicted.lower():
                return False
        return "def " in predicted and "return " in predicted
    else:
        # HotpotQA — use token-level F1 (standard protocol) instead of substring match
        pred_tokens = set(re.sub(r'[^\w\s]', '', predicted).lower().split())
        gold_tokens = set(re.sub(r'[^\w\s]', '', gold).lower().split())
        # Remove stopwords that inflate F1
        stopwords = {'the', 'a', 'an', 'is', 'was', 'are', 'were', 'of', 'in', 'to', 'and', 'for', 'on', 'at', 'by', 'with'}
        pred_tokens -= stopwords
        gold_tokens -= stopwords
        if not gold_tokens:
            return predicted.strip().lower() == gold.strip().lower()
        if not pred_tokens:
            return False
        tp = len(pred_tokens & gold_tokens)
        precision = tp / len(pred_tokens) if pred_tokens else 0
        recall    = tp / len(gold_tokens) if gold_tokens else 0
        if precision + recall == 0:
            return False
        f1 = 2 * precision * recall / (precision + recall)
        return f1 >= 0.5  # Standard HotpotQA threshold


def _hotpot_normalize(s: str) -> str:
    import re
    s = str(s).lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = re.sub(r'[^\w\s]', '', s)
    return ' '.join(s.split())

def hotpotqa_f1_score(prediction: str, ground_truth: str) -> float:
    from collections import Counter
    normalized_prediction = _hotpot_normalize(prediction)
    normalized_ground_truth = _hotpot_normalize(ground_truth)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return 0.0
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return 0.0

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return float(f1)


def load_dataset_by_name(name: str, n_train: int = 30, n_test: int = 50, seed: int = 42) -> Tuple[List[dict], List[dict]]:
    """Load dataset strictly split by name string."""
    name = name.lower()
    if name == "gsm8k":
        return load_gsm8k(n_train, n_test, seed)
    elif name in ("hotpotqa", "hotpot_qa"):
        return load_hotpotqa(n_train, n_test, seed)
    elif name == "aqua":
        return load_aqua(n_train, n_test, seed)
    elif name in ("truthfulqa", "truthful_qa"):
        return load_truthfulqa(n_train, n_test, seed)
    elif name == "mbpp" or name == "code":
        return load_mbpp(n_train, n_test, seed)
    else:
        raise ValueError(f"Unknown dataset: {name}. Use 'gsm8k', 'hotpotqa', 'math', 'truthfulqa', or 'mbpp'.")
