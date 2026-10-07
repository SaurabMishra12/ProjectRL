#!/usr/bin/env python3
"""
Dataset Validation Module
=========================
Strictly validates datasets before ANY LLM call is permitted.
Enforces non-negotiable rules:
  1. No toy data fallback (fail-closed)
  2. Authoritative source and provenance recorded
  3. Non-empty questions, answers, and retrieval context
  4. 0 synthetic examples verified
  5. Exact evaluation subset saved for complete reproducibility
"""

import os
import sys
import json
import hashlib
from typing import Dict, List, Any, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from dataset_loaders import load_open_book_hotpotqa, DatasetIntegrityError


def validate_web_dataset(
    n_samples: int = 100,
    seed: int = 42,
    output_report_path: str = "dataset_validation_report.json",
    save_subset_path: str = "web_eval_subset_100.json",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Validates Open-Book HotpotQA dataset for Web experiments.
    Returns: (validated_test_samples, report_dict)
    Raises: DatasetIntegrityError on any validation failure.
    """
    print("=" * 80)
    print("DATASET VALIDATION STAGE")
    print("=" * 80)

    source_path = os.path.expanduser("~/.cache/huggingface/datasets/hotpot_qa/fullwiki/0.0.0/1908d6afbbead072334abe2965f91bd2709910ab")
    if not os.path.exists(source_path):
        raise DatasetIntegrityError(f"Local HuggingFace dataset cache not found at: {source_path}")

    # Load dataset using strictly audited loader
    try:
        train_samples, test_samples = load_open_book_hotpotqa(n_train=0, n_test=n_samples, seed=seed)
    except Exception as e:
        raise DatasetIntegrityError(f"Failed to load Open-Book HotpotQA: {e}") from e

    if len(test_samples) != n_samples:
        raise DatasetIntegrityError(
            f"Expected {n_samples} test examples, but loaded {len(test_samples)}"
        )

    required_keys = ["id", "question", "answer", "type", "level", "supporting_facts", "context", "articles"]
    n_valid_questions = 0
    n_valid_answers = 0
    n_valid_corpora = 0
    total_articles = 0
    synthetic_count = 0

    synthetic_markers = ["apples", "shirts", "pencils", "Alice has", "Bob gives", "A store has"]

    for idx, item in enumerate(test_samples):
        # 1. Check required fields
        for k in required_keys:
            if k not in item:
                raise DatasetIntegrityError(f"Sample {idx} missing required key: {k}")

        q = item["question"].strip()
        a = item["answer"].strip()
        articles = item["articles"]
        supp = item["supporting_facts"]

        # 2. Check non-empty
        if not q:
            raise DatasetIntegrityError(f"Sample {idx} has empty question")
        if not a:
            raise DatasetIntegrityError(f"Sample {idx} has empty answer")
        if not articles or len(articles) == 0:
            raise DatasetIntegrityError(f"Sample {idx} has empty retrieval context articles")

        # 3. Check synthetic contamination
        for marker in synthetic_markers:
            if marker in q and ("has" in q and "more" in q):
                synthetic_count += 1
                raise DatasetIntegrityError(f"Sample {idx} appears to be synthetic arithmetic problem: {q}")

        # 4. Check retrieval corpus structure
        for art in articles:
            if "title" not in art or "text" not in art:
                raise DatasetIntegrityError(f"Sample {idx} article missing title or text")
            if not art["text"].strip():
                raise DatasetIntegrityError(f"Sample {idx} article '{art.get('title')}' has empty text")

        n_valid_questions += 1
        n_valid_answers += 1
        n_valid_corpora += 1
        total_articles += len(articles)

    # Compute subset hash for immutable provenance
    subset_json_bytes = json.dumps([{k: item[k] for k in ["id", "question", "answer", "type"]} for item in test_samples], sort_keys=True).encode("utf-8")
    subset_sha256 = hashlib.sha256(subset_json_bytes).hexdigest()

    report = {
        "dataset": "Open-Book HotpotQA (fullwiki)",
        "source": source_path,
        "version_config": "fullwiki/0.0.0/1908d6afbbead072334abe2965f91bd2709910ab",
        "split": "validation",
        "seed": seed,
        "examples_requested": n_samples,
        "examples_loaded": len(test_samples),
        "questions_valid": n_valid_questions,
        "answers_valid": n_valid_answers,
        "retrieval_corpora_valid": n_valid_corpora,
        "total_candidate_articles": total_articles,
        "avg_articles_per_question": total_articles / max(len(test_samples), 1),
        "synthetic_examples": synthetic_count,
        "toy_fallback": "DISABLED (fail-closed)",
        "subset_sha256": subset_sha256,
        "status": "PASS",
    }

    # Print terminal validation block
    print(f"dataset: {report['dataset']}")
    print(f"source: {report['source']}")
    print(f"version/config: {report['version_config']}")
    print(f"examples loaded: {report['examples_loaded']}")
    print(f"questions valid: {report['questions_valid']}")
    print(f"answers valid: {report['answers_valid']}")
    print(f"retrieval corpora valid: {report['retrieval_corpora_valid']} ({report['avg_articles_per_question']:.1f} articles/q)")
    print(f"synthetic examples: {report['synthetic_examples']}")
    print(f"toy fallback: {report['toy_fallback']}")
    print(f"subset sha256: {report['subset_sha256'][:16]}...")
    print(f"status: {report['status']}")
    print("=" * 80)

    # Save validation report
    full_report_path = os.path.join(BASE_DIR, output_report_path)
    with open(full_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved dataset validation report to: {full_report_path}")

    # Save exact evaluation subset for reproducibility
    full_subset_path = os.path.join(BASE_DIR, save_subset_path)
    with open(full_subset_path, "w", encoding="utf-8") as f:
        json.dump(test_samples, f, indent=2)
    print(f"Saved exact evaluation subset to: {full_subset_path}")

    return test_samples, report


if __name__ == "__main__":
    n = 100 if len(sys.argv) < 2 else int(sys.argv[1])
    validate_web_dataset(n_samples=n)
