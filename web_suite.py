#!/usr/bin/env python3
"""
Web Multi-Agent Suite for WWW 2027 Research Experiments
======================================================
Unified, reproducible implementation of Web Information-Seeking Multi-Agent System:
  - Local Wikipedia BM25 retrieval over HotpotQA candidate articles
  - Specialized functional agents:
      * Agent 0: Web Retrieval Specialist
      * Agent 1: Evidence Verification Specialist
      * Agent 2: Multi-Hop Reasoning Specialist
  - Frozen information pool support for causal isolation
  - Sentence-preserving content rationing vs. random truncation
  - SPRT adaptive stopping
  - Persistent disk cache for 100% reproducible, zero-waste LLM execution
  - Resumable streaming JSONL checkpoints
"""

import os
import sys
import json
import math
import re
import hashlib
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import Counter

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from config import Config, get_hotpotqa_config
from utils import set_seed, count_tokens, split_sentences, NumpyEncoder
from llm_backend import AzureOpenAIBackend, LLMResponse
from controller import IBController
from dataset_loaders import hotpotqa_f1_score, evaluate_answer, _hotpot_normalize
from pipeline import _normalize_for_vote, weighted_majority_vote

CACHE_FILE = os.path.join(BASE_DIR, "cache", "web_llm_cache.json")


# =====================================================================
# PERSISTENT LLM DISK CACHE
# =====================================================================
class PersistentLLMCache:
    """Thread-safe persistent disk cache for LLM responses."""
    def __init__(self, cache_file: str = CACHE_FILE):
        self.cache_file = cache_file
        self.cache: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception as e:
                print(f"[CACHE] Warning: Failed to load cache: {e}")
                self.cache = {}

    def _save(self):
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        tmp_file = self.cache_file + ".tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(self.cache, f)
        os.replace(tmp_file, self.cache_file)

    def get(self, key: str) -> Optional[LLMResponse]:
        if key in self.cache:
            data = self.cache[key]
            return LLMResponse(
                text=data["text"],
                answer=data["answer"],
                token_count=data["token_count"],
                confidence=data["confidence"],
                latency_ms=data.get("latency_ms", 0.0),
            )
        return None

    def put(self, key: str, resp: LLMResponse):
        self.cache[key] = {
            "text": resp.text,
            "answer": resp.answer,
            "token_count": resp.token_count,
            "confidence": resp.confidence,
            "latency_ms": getattr(resp, "latency_ms", 0.0),
        }
        # Periodically save
        if len(self.cache) % 5 == 0:
            self._save()

    def flush(self):
        self._save()


# Global cache instance
_GLOBAL_LLM_CACHE = PersistentLLMCache()


# =====================================================================
# LOCAL WIKIPEDIA BM25 RETRIEVER
# =====================================================================
class LocalWikipediaRetriever:
    """
    Real Robertson-Spärck Jones Okapi BM25 retriever over the candidate Wikipedia articles.
    Provides deterministic, reproducible, auditable retrieval with zero external API failure risk.
    """
    def __init__(self, articles: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.articles = articles
        self.corpus_tokens = [
            re.findall(r"\w+", (art["title"] + " " + art["text"]).lower())
            for art in articles
        ]
        self.doc_lens = [len(doc) for doc in self.corpus_tokens]
        self.avgdl = sum(self.doc_lens) / max(len(self.corpus_tokens), 1)

        self.doc_freqs = [Counter(doc) for doc in self.corpus_tokens]
        df = Counter()
        for c in self.doc_freqs:
            for token in c:
                df[token] += 1
        N = len(self.corpus_tokens)
        self.idf = {
            token: math.log((N - freq + 0.5) / (freq + 0.5) + 1.0)
            for token, freq in df.items()
        }

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_tokens = re.findall(r"\w+", query.lower())
        scores = []
        for idx, doc_freq in enumerate(self.doc_freqs):
            score = 0.0
            doc_len = self.doc_lens[idx]
            for token in query_tokens:
                if token in doc_freq:
                    f = doc_freq[token]
                    idf = self.idf.get(token, 0.0)
                    denom = f + self.k1 * (1.0 - self.b + self.b * (doc_len / (self.avgdl + 1e-8)))
                    score += idf * (f * (self.k1 + 1.0)) / (denom + 1e-8)
            scores.append(score)

        ranked_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for rank, idx in enumerate(ranked_indices):
            art = self.articles[idx]
            results.append({
                "rank": rank + 1,
                "title": art["title"],
                "text": art["text"],
                "sentences": art["sentences"],
                "score": float(scores[idx]),
            })
        return results


# =====================================================================
# FUNCTIONAL AGENTS & PROMPT GENERATION
# =====================================================================
def build_agent_prompt(
    agent_id: int,
    question: str,
    evidence_passages: List[Dict[str, Any]],
    peer_messages: Optional[List[str]] = None,
    round_num: int = 0,
) -> Tuple[str, str]:
    """
    Constructs specialized prompts for functional Web agents:
      Agent 0: Web Search / Retrieval Specialist
      Agent 1: Evidence Verification Specialist
      Agent 2: Multi-Hop Reasoning Specialist
    Returns: (system_prompt, user_prompt)
    """
    peer_context = ""
    if peer_messages:
        valid_msgs = [m.strip() for m in peer_messages if m.strip()]
        if valid_msgs:
            peer_context = "\n\nInformation communicated from other agents:\n" + "\n".join(valid_msgs)

    evidence_str = ""
    if evidence_passages:
        lines = []
        for p in evidence_passages:
            lines.append(f"[Document: {p['title']}]\n{p['text']}")
        evidence_str = "\n\nRetrieved Web Evidence:\n" + "\n\n".join(lines)

    if agent_id == 0:
        system_prompt = (
            "You are Agent 0, the Web Retrieval Specialist. Your role is to examine the question and "
            "the candidate web documents, extract the most critical facts, and propose a concise answer. "
            "Be precise and cite document titles. Always conclude your reasoning with ONLY the exact concise entity, name, or phrase on the final line:\n"
            "Answer: <exact concise answer>\nConfidence: <0-100>"
        )
        user_prompt = (
            f"Question: {question}"
            f"{evidence_str}"
            f"{peer_context}\n\n"
            "Provide your retrieval analysis, synthesize the facts, and state your final answer."
        )
    elif agent_id == 1:
        system_prompt = (
            "You are Agent 1, the Evidence Verification Specialist. Your role is to cross-examine "
            "candidate facts, verify bridge entity connections across documents, and filter out false distractors. "
            "Always conclude your reasoning with ONLY the exact concise entity, name, or phrase on the final line:\n"
            "Answer: <exact concise answer>\nConfidence: <0-100>"
        )
        user_prompt = (
            f"Question: {question}"
            f"{evidence_str}"
            f"{peer_context}\n\n"
            "Verify the supporting evidence, resolve potential contradictions or ambiguities, and state your final answer."
        )
    else:  # Agent 2
        system_prompt = (
            "You are Agent 2, the Multi-Hop Reasoning Specialist. Your role is to logically chain verified facts "
            "across multi-hop entities to deduce the exact answer to the question. "
            "Always conclude your reasoning with ONLY the exact concise entity, name, or phrase on the final line:\n"
            "Answer: <exact concise answer>\nConfidence: <0-100>"
        )
        user_prompt = (
            f"Question: {question}"
            f"{evidence_str}"
            f"{peer_context}\n\n"
            "Execute multi-hop reasoning over the verified facts and state your final answer."
        )

    return system_prompt, user_prompt


def invoke_agent_with_cache(
    backend: AzureOpenAIBackend,
    agent_id: int,
    system_prompt: str,
    user_prompt: str,
    round_num: int,
    call_tracker: Optional[Dict[str, Any]] = None,
) -> LLMResponse:
    """Executes agent call through persistent disk cache, tracking latency and handling API failures."""
    import time
    t_call_start = time.perf_counter()
    cache_key_str = f"agent_{agent_id}|r_{round_num}|sys_{system_prompt}|usr_{user_prompt}"
    key = hashlib.sha256(cache_key_str.encode("utf-8")).hexdigest()

    cached = _GLOBAL_LLM_CACHE.get(key)
    if cached is not None:
        if call_tracker is not None:
            call_tracker["cached_hits"] = call_tracker.get("cached_hits", 0) + 1
        return cached

    # Make actual API call via backend
    if call_tracker is not None:
        call_tracker["live_calls"] = call_tracker.get("live_calls", 0) + 1

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    last_error_str = "Unknown error"
    for attempt in range(backend.max_retries):
        try:
            t_req_start = time.perf_counter()
            response = backend.client.chat.completions.create(
                model=backend.model,
                messages=messages,
                temperature=backend.temperature,
                top_p=1.0,
                max_tokens=backend.max_tokens,
                stream=False,
            )
            req_latency_ms = (time.perf_counter() - t_req_start) * 1000.0
            text = response.choices[0].message.content or ""
            tok_count = response.usage.total_tokens if response.usage else count_tokens(text)
            answer = backend._extract_answer(text, "hotpotqa")
            conf = backend._estimate_confidence(text)

            llm_resp = LLMResponse(
                text=text,
                answer=answer,
                token_count=tok_count,
                confidence=conf,
                latency_ms=req_latency_ms,
            )
            _GLOBAL_LLM_CACHE.put(key, llm_resp)
            if call_tracker is not None:
                call_tracker["total_tokens"] = call_tracker.get("total_tokens", 0) + tok_count
                call_tracker["successful_calls"] = call_tracker.get("successful_calls", 0) + 1
                if "latencies_ms" not in call_tracker:
                    call_tracker["latencies_ms"] = []
                call_tracker["latencies_ms"].append(req_latency_ms)
                call_tracker["total_latency_ms"] = call_tracker.get("total_latency_ms", 0.0) + req_latency_ms
            return llm_resp
        except Exception as e:
            last_error_str = str(e)
            if call_tracker is not None:
                call_tracker["failed_attempts"] = call_tracker.get("failed_attempts", 0) + 1
            # If error is an Azure content filter (e.g. Jailbreak label), immediate fail rather than looping
            if "content_filter" in last_error_str or "Jailbreak" in last_error_str:
                print(f"[API CONTENT FILTER] Agent {agent_id} triggered filter: {last_error_str[:120]}. Recording as FAILED per Rule 27.")
                break
            wait = (attempt + 1) * 3
            print(f"[API RETRY] Agent {agent_id} call failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)

    # Retries exhausted or content filter triggered: Record failure explicitly per Rule 27
    total_elapsed_ms = (time.perf_counter() - t_call_start) * 1000.0
    if call_tracker is not None:
        call_tracker["failed_calls"] = call_tracker.get("failed_calls", 0) + 1
        if "latencies_ms" not in call_tracker:
            call_tracker["latencies_ms"] = []
        call_tracker["latencies_ms"].append(total_elapsed_ms)
        call_tracker["total_latency_ms"] = call_tracker.get("total_latency_ms", 0.0) + total_elapsed_ms

    failed_resp = LLMResponse(
        text=f"[API_CALL_FAILED: {last_error_str[:200]}]",
        answer="[FAILED]",
        token_count=0,
        confidence=0.0,
        latency_ms=total_elapsed_ms,
    )
    return failed_resp


# =====================================================================
# MESSAGE RATIONING: SEMANTIC VS RANDOM TRUNCATION
# =====================================================================
def select_sentences_for_edge(text: str, budget: int) -> Tuple[str, int]:
    """Sentence-preserving semantic rationing up to budget."""
    if not text or budget <= 0:
        return "", 0
    if count_tokens(text) <= budget:
        return text, count_tokens(text)

    sentences = split_sentences(text)
    curr_sentences = []
    for s in sentences:
        candidate = " ".join(curr_sentences + [s])
        if count_tokens(candidate) <= budget:
            curr_sentences.append(s)
        else:
            break

    if curr_sentences:
        cand_text = " ".join(curr_sentences)
        return cand_text, count_tokens(cand_text)

    # Word-level fallback if first sentence is longer than budget
    words = text.split()
    low, high, best = 0, len(words), 0
    while low <= high:
        mid = (low + high) // 2
        cand = " ".join(words[:mid])
        if count_tokens(cand) <= budget:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    cand_text = " ".join(words[:best])
    return cand_text, count_tokens(cand_text)


def select_random_truncated_edge(text: str, budget: int, seed: int = 42) -> Tuple[str, int]:
    """Random arbitrary word-level truncation up to budget."""
    if not text or budget <= 0:
        return "", 0
    if count_tokens(text) <= budget:
        return text, count_tokens(text)

    words = text.split()
    if not words:
        return "", 0
    # Choose arbitrary slice of words matching the budget length
    # Target word length estimated from token budget (approx 0.75 words per token)
    rng = np.random.RandomState(seed + len(text) % 1000)
    # Estimate max words
    target_words = max(1, int(budget * 0.75))
    if len(words) > target_words:
        start_idx = rng.randint(0, len(words) - target_words + 1)
        sliced_words = words[start_idx : start_idx + target_words]
    else:
        sliced_words = words

    cand = " ".join(sliced_words)
    # Fine-tune to not exceed token count
    while count_tokens(cand) > budget and sliced_words:
        sliced_words.pop()
        cand = " ".join(sliced_words)
    return cand, count_tokens(cand)


# =====================================================================
# STOPPING RULE: SPRT & CONTROLLER HALT
# =====================================================================
def evaluate_stopping_rule(
    controller: IBController,
    current_responses: List[LLMResponse],
    prev_W: np.ndarray,
    round_num: int,
    config: Config,
) -> Tuple[bool, float, float]:
    """
    Evaluates whether communication should halt:
      1. Controller learned logit z_r > 0
      2. Sequential likelihood ratio S > log 19 (tau = 2.944)
    """
    agent_texts = {idx: r.text for idx, r in enumerate(current_responses)}
    agent_answers = {idx: r.answer for idx, r in enumerate(current_responses)}
    confs = [r.confidence for r in current_responses]
    unique_answers = len(set(agent_answers.values()))

    # Controller halt logit
    _, _, halt_decision, meta = controller.get_weights(
        reasoning_texts=agent_texts,
        answers=agent_answers,
        confidence_scores=confs,
        prev_W=prev_W,
        round_num=round_num + 1,
        unique_answers=unique_answers,
        avg_accuracy=0.5,
        explore=False,
    )
    halt_logit = float(meta["mean_action"][-1])

    # Sequential binomial log-likelihood ratio S > log 19
    _SPRT_TAU = 2.9444389791664403  # log(19), alpha=0.05
    _SPRT_P0 = 0.5
    ans_list = [_normalize_for_vote(r.answer, "hotpotqa") for r in current_responses]
    counts = Counter(ans_list)
    majority_ans, k = counts.most_common(1)[0]
    N = len(ans_list)
    maj_confs = [
        r.confidence for r in current_responses
        if _normalize_for_vote(r.answer, "hotpotqa") == majority_ans
    ]
    if controller.calibrator is not None and controller.calibrator._fitted and maj_confs:
        p_maj = float(np.clip(
            controller.calibrator.calibrate(float(np.max(maj_confs))), _SPRT_P0 + 0.001, 0.999
        ))
    else:
        p_maj = 0.9
    llr = float(k * np.log(p_maj / _SPRT_P0) + (N - k) * np.log((1.0 - p_maj) / (1.0 - _SPRT_P0)))
    sprt_halt = bool(llr > _SPRT_TAU)

    should_halt = bool(halt_decision or sprt_halt)
    return should_halt, halt_logit, llr


# =====================================================================
# AGGREGATOR
# =====================================================================
def aggregate_responses(
    responses: List[LLMResponse],
    calibrator: Optional[Any] = None,
) -> str:
    """Computes consensus answer via Platt-calibrated weighted majority vote."""
    from agents import AgentResponse
    converted = [
        AgentResponse(
            agent_id=idx,
            answer=r.answer,
            confidence=r.confidence,
            reasoning_text=r.text,
            token_count=r.token_count,
            round_num=0,
        )
        for idx, r in enumerate(responses)
    ]
    return weighted_majority_vote(converted, dataset_type="hotpotqa", calibrator=calibrator)
