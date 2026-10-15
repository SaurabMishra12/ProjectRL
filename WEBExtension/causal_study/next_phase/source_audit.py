"""Outcome-independent source provenance and blinded annotation preparation.

This module uses only the standard library and local saved files. It does not
import an inference backend, score responses, or use correctness to select cases.
Human adjudication is deliberately not replaced by lexical matching or by the
bounded provisional agent review stored separately in source_artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
VERSION = "source-audit-v1"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(text):
    text = "".join(c for c in text.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def literal_occurrence(answer, text):
    """Whole normalized token sequence, never a semantic sufficiency judgment."""
    answer = normalize(answer)
    return bool(answer) and (" " + answer + " ") in (" " + normalize(text) + " ")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def by_id(rows):
    index = {r["id"]: r for r in rows}
    if len(index) != len(rows):
        raise ValueError("Duplicate question identities")
    return index


def source_index(documents):
    titles = [d["title"] for d in documents]
    if len(titles) != len(set(titles)):
        raise ValueError("Ambiguous duplicate source titles")
    return {(d["title"], i): sentence for d in documents
            for i, sentence in enumerate(d["sentences"])}


def source_record(qid, question, question_type, documents, gold, supports,
                  cohort, provenance, disagreement=None):
    refs = source_index(documents)
    support_titles = {title for title, _ in supports}
    pool_titles = {d["title"] for d in documents}
    documents = [{**d, "document_sha256": digest(d)} for d in documents]
    lexical = [{"title": title, "sentence_id": i,
                "is_annotated_support_title": title in support_titles,
                "sentence_sha256": digest(text)}
               for (title, i), text in refs.items() if literal_occurrence(gold, text)]
    title_matches = [d["title"] for d in documents if literal_occurrence(gold, d["title"])]
    return {
        "id": qid, "cohort": cohort, "question": question, "type": question_type,
        "review_id": digest([VERSION, qid])[:20], "documents": documents,
        "source_union_sha256": digest(documents), "source_provenance": provenance,
        "private_reference_answer": gold, "annotated_support_facts": supports,
        "annotated_support_titles_present": sorted(support_titles & pool_titles),
        "annotated_support_titles_absent": sorted(support_titles - pool_titles),
        "annotated_support_sentences_present": [list(r) for r in map(tuple, supports) if r in refs],
        "annotated_support_sentences_absent": [list(r) for r in map(tuple, supports) if r not in refs],
        "all_annotated_support_present": all(tuple(r) in refs for r in supports),
        "literal_reference_sentence_occurrences": lexical,
        "literal_reference_title_occurrences": title_matches,
        "lexical_special_answer_warning": normalize(gold) in {"yes", "no", "noanswer"},
        "baseline_initial_disagreement": disagreement,
        "both_articles_necessary_for_answer": "UNRESOLVED_HUMAN_NEEDED",
        "one_article_semantically_sufficient": "UNRESOLVED_HUMAN_NEEDED",
        "distractor_answer_bearing": "UNRESOLVED_HUMAN_NEEDED",
        "union_semantic_sufficiency": "UNRESOLVED_HUMAN_NEEDED",
        "reliable_human_annotation_status": "NOT_ANNOTATED",
        "future_complementarity_eligibility": "NOT_QUALIFIED",
    }


def select_old_review(records, per_stratum=2):
    """Freeze a bounded source review without consulting correctness or gains."""
    strata = defaultdict(list)
    for row in records:
        if row["cohort"] == "E8_OLD_300":
            strata[(row["all_annotated_support_present"], row["baseline_initial_disagreement"])].append(row)
    result = []
    for (support, disagreement), rows in sorted(strata.items()):
        ranked = sorted(rows, key=lambda r: hashlib.sha256((VERSION + "|" + r["id"]).encode()).hexdigest())
        result.append({"all_support_present": support, "baseline_disagreement": disagreement,
                       "stratum_n": len(rows), "selected_ids": [r["id"] for r in ranked[:per_stratum]]})
    return result


def blinded_source_packet(record):
    """No gold, support annotation, proposal, arm, performance, or eligibility."""
    return {"review_id": record["review_id"], "question": record["question"],
            "documents": [{"title": d["title"], "sentences": d["sentences"],
                           "document_sha256": d["document_sha256"]} for d in record["documents"]],
            "annotation_status": "AWAITING_TWO_INDEPENDENT_HUMANS"}


def classify_citation(record, answer, citations):
    """Mechanical provenance is observed; entailment remains an annotation task."""
    refs = source_index(record["documents"])
    cited = []
    for citation in citations:
        valid_shape = (isinstance(citation, (list, tuple)) and len(citation) == 2
                       and isinstance(citation[0], str) and type(citation[1]) is int)
        key = tuple(citation) if valid_shape else None
        text = refs.get(key) if key else None
        cited.append({"citation": citation, "source_exists": text is not None,
                      "text": text, "sentence_sha256": digest(text) if text is not None else None})
    abstention = normalize(answer) == "noanswer"
    if not normalize(answer):
        status = "EMPTY_OR_MALFORMED_ANSWER_NO_ASSERTION"
    elif abstention:
        status = "ABSTENTION_NO_ANSWER_ASSERTION"
    elif not cited:
        status = "NO_CITATION_SUPPLIED"
    elif not all(c["source_exists"] for c in cited):
        status = "UNRESOLVED_WITH_MISSING_OR_INVALID_SOURCE"
    else:
        status = "UNRESOLVED_HUMAN_NEEDED"
    return {"answer": answer, "citations": cited, "entailment": status,
            "source_provenance_valid": bool(cited) and all(c["source_exists"] for c in cited),
            "reliable_human_annotation_status": "NOT_ANNOTATED"}


def qualified_by_independent_annotations(annotation):
    """Do not let agent judgments or answer occurrences qualify a new cohort."""
    reviewers = annotation.get("reviewers", [])
    return (len({r for r in reviewers if isinstance(r, str) and r.strip()}) >= 2
            and annotation.get("adjudicated") is True
            and annotation.get("reviewer_type") == "independent_human"
            and annotation.get("outcome_blinded") is True
            and annotation.get("question_valid") is True
            and annotation.get("reference_valid") is True
            and annotation.get("union_sufficient") is True
            and annotation.get("both_articles_necessary") is True
            and annotation.get("single_article_sufficient") is False
            and annotation.get("alternative_source_answer_bearing") is False)


def load_sources(repo=REPO):
    web = Path(repo) / "WEBExtension"
    active_path = web / "artifacts/active_machine_runs.json"
    active = json.loads(active_path.read_text())
    old_dir = Path(active["controlled_run"])
    e5_dir = Path(active["controller_study"])
    gold_path = old_dir / "sample_gold.json"
    e1_path, e5_path = old_dir / "E1.jsonl", e5_dir / "E5.jsonl"
    gold, e1, e5 = (by_id(json.loads(gold_path.read_text())),
                    by_id(read_jsonl(e1_path)), by_id(read_jsonl(e5_path)))
    if not (set(gold) == set(e1) == set(e5)) or len(gold) != 300:
        raise ValueError("Expected aligned 300-question old artifacts")
    sources = [active_path, gold_path, e1_path, e5_path]
    records = []
    for qid, question in sorted(gold.items()):
        partitioned, shared = e1[qid]["states"]["P"], e1[qid]["states"]["R"]
        if canonical(partitioned["pool"]) != canonical(shared["pool"]):
            raise ValueError("Shared and partitioned source unions differ")
        original = {d["title"]: d for d in question["documents"]}
        if any(canonical(d) != canonical(original[d["title"]]) for d in partitioned["pool"]):
            raise ValueError("Retrieved source differs from saved candidate text")
        baseline = e5[qid]["conditions"]["P_NoComm"]["responses"]
        disagreement = len({normalize(r["answer"]) for r in baseline}) > 1
        records.append(source_record(qid, question["question"], question["type"], partitioned["pool"],
            question["answer"], question["supporting_facts"], "E8_OLD_300",
            {"question_file": str(gold_path.relative_to(repo)),
             "source_file": str(e1_path.relative_to(repo)),
             "json_pointer": "/states/P/pool", "retrieval": "SAVED_QUESTION_ONLY_TOP_6"}, disagreement))
    manifest_path = web / "causal_study/frozen/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    labels_path = web / "causal_study/frozen/private_labels.json"
    labels = json.loads(labels_path.read_text())["labels"]
    sources += [manifest_path, labels_path]
    for question in manifest["questions"]:
        qid = question["id"]
        if qid in gold:
            raise ValueError("E6 overlaps historical sample")
        label = labels[qid]
        records.append(source_record(qid, question["question"], question["type"], question["documents"],
            label["answer"], label["supporting_facts"], "E6_PILOT_8",
            {"source_file": str(manifest_path.relative_to(repo)), "json_pointer": "/questions/documents",
             "retrieval": "CONTROLLED_GOLD_CONSTRUCTED_6_DOCUMENT_UNION"}))
    paths = {f"E{i}": old_dir / f"E{i}.jsonl" for i in range(1, 5)}
    paths["E5"] = e5_path
    paths["E3-v2"] = Path(active["e3_corrected"]["run_dir"]) / "E3.jsonl"
    sources += list(paths.values())
    sources += [web / "causal_study/paid_pilot/questions" / (r["id"] + ".json")
                for r in records if r["cohort"] == "E6_PILOT_8"]
    return records, paths, list(dict.fromkeys(sources))


def build_citation_register(records, paths, repo=REPO):
    record_index = {r["id"]: r for r in records}
    assertions = {}

    def add(qid, response, location):
        citations = response.get("citations", [])
        key = digest([qid, normalize(response["answer"]), sorted(citations, key=canonical)])
        if key not in assertions:
            assertions[key] = {"assertion_id": key[:24], "id": qid,
                "review_id": record_index[qid]["review_id"],
                **classify_citation(record_index[qid], response["answer"], citations), "observations": []}
        observation = {**location, "response_key": response.get("call_key", response.get("request_hash"))}
        if observation not in assertions[key]["observations"]:
            assertions[key]["observations"].append(observation)

    for experiment, path in paths.items():
        for row in read_jsonl(path):
            for state, payload in row.get("states", {}).items():
                for node, response in enumerate(payload.get("responses", [])):
                    add(row["id"], response, {"file": str(path.relative_to(repo)),
                        "experiment": experiment, "state": state, "node": node})
            for condition, payload in row["conditions"].items():
                for node, response in enumerate(payload.get("responses", [])):
                    add(row["id"], response, {"file": str(path.relative_to(repo)),
                        "experiment": experiment, "condition": condition, "node": node})
    for record in records:
        if record["cohort"] != "E6_PILOT_8":
            continue
        path = Path(repo) / "WEBExtension/causal_study/paid_pilot/questions" / (record["id"] + ".json")
        row = json.loads(path.read_text())
        for placement, responses in row["initial"].items():
            for node, response in enumerate(responses):
                add(record["id"], response, {"file": str(path.relative_to(repo)),
                    "experiment": "E6", "state": placement, "node": node})
        for condition, cell in row["cells"].items():
            for node, response in enumerate(cell["responses"]):
                add(record["id"], response, {"file": str(path.relative_to(repo)),
                    "experiment": "E6", "condition": condition, "node": node})
    return sorted(assertions.values(), key=lambda r: (r["id"], r["assertion_id"]))


def summary(records, assertions, selection):
    counts = {}
    for cohort in sorted({r["cohort"] for r in records}):
        rows = [r for r in records if r["cohort"] == cohort]
        counts[cohort] = {
            "n_questions": len(rows),
            "all_annotated_support_present": sum(r["all_annotated_support_present"] for r in rows),
            "support_title_count_distribution": dict(Counter(len(r["annotated_support_titles_present"]) for r in rows)),
            "any_literal_reference_sentence": sum(bool(r["literal_reference_sentence_occurrences"]) for r in rows),
            "literal_reference_in_nonsupport_sentence": sum(any(not x["is_annotated_support_title"]
                for x in r["literal_reference_sentence_occurrences"]) for r in rows),
            "literal_reference_title": sum(bool(r["literal_reference_title_occurrences"]) for r in rows),
            "independent_human_semantic_annotations": 0, "complementarity_qualified": 0,
            "bounded_provisional_agent_source_reviews": sum("provisional_agent_assessment" in r for r in rows),
        }
    return {"protocol": VERSION, "cohorts": counts, "bounded_old_review_selection": selection,
            "citation_unique_answer_source_assertions": len(assertions),
            "citation_mechanical_status_counts": dict(Counter(r["entailment"] for r in assertions)),
            "semantic_evidence_tier": "PROVISIONAL_BOUNDED_AGENT_REVIEW_PLUS_UNRESOLVED",
            "bounded_provisional_agent_citation_reviews": sum("provisional_agent_assessment" in a for a in assertions),
            "new_inference_requests": 0, "paid_inference_requests": 0,
            "warning": "Lexical occurrence and annotated support presence do not establish sufficiency or entailment."}


def generate(repo=REPO, output=None):
    repo = Path(repo).resolve()
    output = Path(output or HERE / "source_artifacts").resolve()
    # A new output directory cannot overwrite or rename any historical artifact.
    allowed = (repo / "WEBExtension/causal_study/next_phase/source_artifacts").resolve()
    if output != allowed and allowed not in output.parents:
        raise ValueError("Source audit output must remain inside new source_artifacts")
    output.mkdir(parents=True, exist_ok=True)
    records, paths, sources = load_sources(repo)
    selection = select_old_review(records)
    assertions = build_citation_register(records, paths, repo)
    # Bounded qualitative judgments are optional private inputs. They cannot
    # change selection, independent-label fields, blinded packets, or eligibility.
    records_by_id = {r["id"]: r for r in records}
    assertions_by_id = {a["assertion_id"]: a for a in assertions}
    for name in ["agent_source_reviews.json", "agent_citation_reviews.json"]:
        path = output / name
        if not path.exists():
            continue
        sources.append(path)
        reviews = json.loads(path.read_text())["reviews"]
        for review in reviews:
            if review.get("independent_human_ground_truth") is not False or review.get("reviewer_type") != "agent_provisional":
                raise ValueError("Agent review cannot be labeled independent ground truth")
            if name == "agent_source_reviews.json":
                target = records_by_id[review["id"]]
                refs = source_index(target["documents"])
                for fact in review["material_source_sentences"] + review["alternative_source_observations"]:
                    if refs.get((fact["title"], fact["sentence_id"])) != fact["text"]:
                        raise ValueError("Provisional source review quote/provenance mismatch")
            else:
                target = assertions_by_id[review["assertion_id"]]
                if review["id"] != target["id"] or review["cited_sources"] != target["citations"]:
                    raise ValueError("Provisional citation review has changed sources")
            target["provisional_agent_assessment"] = review
    protocol_path = output / "BLINDED_ANNOTATION_PROTOCOL.md"
    if protocol_path.exists():
        sources.append(protocol_path)
    result = summary(records, assertions, selection)
    files = {
        "source_observations.jsonl": records,
        "blinded_source_review.jsonl": [blinded_source_packet(r) for r in records],
        "citation_assertions_private.jsonl": assertions,
        "blinded_citation_review.jsonl": [{"assertion_id": a["assertion_id"],
            "review_id": a["review_id"], "question": next(r["question"] for r in records if r["id"] == a["id"]),
            "proposed_answer": a["answer"], "cited_sources": a["citations"]}
            for a in assertions if a["citations"] and normalize(a["answer"]) != "noanswer"],
    }
    for filename, rows in files.items():
        (output / filename).write_text("".join(canonical(row) + "\n" for row in rows))
    (output / "source_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    (output / "bounded_review_selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    with (output / "human_source_annotation_template.csv").open("w", newline="") as stream:
        fields = ["review_id", "reviewer_id", "independently_resolved_answer", "union_sufficient",
                  "minimal_sufficient_source_sets", "single_article_sufficient", "both_articles_necessary",
                  "alternative_source_answer_bearing", "support_sentence_ids", "ambiguity_reason", "confidence"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({"review_id": r["review_id"]} for r in records)
    generated = list(files) + ["source_summary.json", "bounded_review_selection.json", "human_source_annotation_template.csv"]
    manifest = {"protocol": VERSION, "source_hashes": {str(p.relative_to(repo)): file_hash(p) for p in sources},
                "analysis_code_sha256": file_hash(__file__),
                "generated_hashes": {name: file_hash(output / name) for name in generated},
                "selection_sha256": digest(selection), "physical_api_attempts": 0,
                "source_labels_are_not_independent_ground_truth": True}
    (output / "source_audit_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    args = parser.parse_args()
    print(json.dumps(generate(args.repo), indent=2))
