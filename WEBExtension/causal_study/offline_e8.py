"""Offline proposal/revision/voting diagnostics. Never imports a provider client.

Run from the repository root: python3 WEBExtension/causal_study/offline_e8.py
Saved responses are observations, not interventions on their internal mechanisms.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
import re
import string
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def normalize(answer):
    answer = "".join(c for c in answer.lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", answer).split())


def answer_score(answer, gold):
    """Official HotpotQA answer-only EM/F1 in percentage points."""
    prediction, reference = normalize(answer), normalize(gold)
    common = sum((Counter(prediction.split()) & Counter(reference.split())).values())
    f1 = 0.0
    special = {"yes", "no", "noanswer"}
    if common and not ((prediction in special or reference in special) and prediction != reference):
        precision = common / len(prediction.split())
        recall = common / len(reference.split())
        f1 = 200 * precision * recall / (precision + recall)
    return {"em": 100.0 * (prediction == reference), "f1": f1}


def majority(responses):
    if not responses:
        raise ValueError("Cannot vote over absent candidates")
    counts = Counter(normalize(response["answer"]) for response in responses)
    winner = max(counts, key=counts.get)
    return next(response["answer"] for response in responses if normalize(response["answer"]) == winner)


def keys(responses):
    return [response["call_key"] for response in responses]


def validate_initial_keys(initial, condition):
    """Do not borrow an initial state merely because its question ID matches."""
    recorded = condition.get("cost", {}).get("call_keys", [])
    if recorded[:len(initial)] != keys(initial):
        raise ValueError("Saved initial call keys do not identify the proposed baseline")


def classify(initial_correct, ever_correct, revised_correct, final_correct, communication, endpoint):
    if not ever_correct:
        return "correct_proposal_never_produced"
    if not revised_correct:
        return "correct_proposal_produced_but_lost_during_revision"
    if not final_correct:
        return ("correct_proposal_available_but_rejected_by_majority" if endpoint == "majority"
                else "correct_proposal_available_elsewhere_but_receiver_incorrect")
    if not initial_correct:
        return ("correct_answer_recovered_through_communication" if communication
                else "correct_answer_recovered_without_communication")
    return "correct_proposal_preserved_and_selected"


def condition_record(experiment, row, label, condition, initial, gold, history=None,
                     initial_source="saved_in_record", endpoint="majority"):
    revised = condition.get("responses")
    if not revised or len(revised) != len(initial):
        raise ValueError("Absent or inconsistent candidate identities")
    validate_initial_keys(initial, condition)
    history = history or [initial, revised]
    scores = [[answer_score(response["answer"], gold) for response in state] for state in history]
    first, last = scores[0], scores[-1]
    correct = lambda candidate: candidate["em"] == 100.0
    initial_any, revised_any = any(map(correct, first)), any(map(correct, last))
    ever_any = any(any(map(correct, state)) for state in scores)
    aggregate = answer_score(condition["answer"], gold)
    for metric in ["f1", "em"]:
        if abs(aggregate[metric] - condition[metric]) > 1e-8:
            raise ValueError("Stored endpoint score disagrees with recomputed answer score")
    final_majority = majority(revised)
    if endpoint == "majority" and normalize(condition["answer"]) != normalize(final_majority):
        raise ValueError("Stored answer is not the implemented deterministic majority")
    if endpoint == "receiver_1" and normalize(condition["answer"]) != normalize(revised[1]["answer"]):
        raise ValueError("Stored E3 endpoint is not receiver 1")
    baseline_answer = majority(initial) if endpoint == "majority" else initial[1]["answer"]
    baseline = answer_score(baseline_answer, gold)
    initial_majority = answer_score(majority(initial), gold)
    final_vote = answer_score(final_majority, gold)
    transition_pairs = [(before[i], after[i]) for before, after in zip(scores, scores[1:])
                        for i in range(len(initial))]
    harmed_nodes = [i for i, (before, after) in enumerate(zip(first, last))
                    if correct(before) and not correct(after)]
    recovered_nodes = [i for i, (before, after) in enumerate(zip(first, last))
                       if not correct(before) and correct(after)]
    communication = bool(condition.get("deliveries"))
    initial_answers = {normalize(response["answer"]) for response in initial}
    novel_correct = any(correct(score) and normalize(response["answer"]) not in initial_answers
                        for score, response in zip(last, revised))
    best_initial, best_final = max(x["f1"] for x in first), max(x["f1"] for x in last)
    mean_final = sum(x["f1"] for x in last)/len(last)
    return {
        "experiment": experiment, "id": row["id"], "type": row.get("type"), "condition": label,
        "endpoint": endpoint, "initial_source": initial_source, "observability": "OBSERVED_SAVED_TRACE",
        "initial_call_keys": keys(initial), "final_call_keys": keys(revised),
        "initial_answers": [response["answer"] for response in initial],
        "final_answers": [response["answer"] for response in revised], "final_answer": condition["answer"],
        "initial_candidate_em": [x["em"] for x in first], "final_candidate_em": [x["em"] for x in last],
        "initial_candidate_f1": [x["f1"] for x in first], "final_candidate_f1": [x["f1"] for x in last],
        "initial_any_em": initial_any, "initial_majority_em": initial_majority["em"],
        "initial_majority_f1": initial_majority["f1"], "initial_endpoint_em": baseline["em"],
        "initial_endpoint_f1": baseline["f1"], "final_any_em": revised_any,
        "ever_any_em": ever_any, "communication_delivered": communication,
        "new_correct_proposal_after_communication": novel_correct if communication else False,
        "new_correct_proposal_after_any_revision": novel_correct,
        "any_initial_correct_agent_became_incorrect": bool(harmed_nodes),
        "initial_correct_to_incorrect_nodes": harmed_nodes, "initial_incorrect_to_correct_nodes": recovered_nodes,
        "correct_to_incorrect_step_events": sum(correct(a) and not correct(b) for a, b in transition_pairs),
        "incorrect_to_correct_step_events": sum(not correct(a) and correct(b) for a, b in transition_pairs),
        "aggregator_rejected_correct_candidate": (revised_any and aggregate["em"] != 100.0)
        if endpoint == "majority" else None,
        "aggregation_observability": "OBSERVED_DETERMINISTIC_MAJORITY" if endpoint == "majority"
        else "NOT_APPLICABLE_RECEIVER_ENDPOINT",
        "final_em": aggregate["em"], "final_f1": aggregate["f1"],
        "final_minus_initial_endpoint_em": aggregate["em"] - baseline["em"],
        "final_minus_initial_endpoint_f1": aggregate["f1"] - baseline["f1"],
        "diagnostic_final_majority_em": final_vote["em"], "diagnostic_final_majority_f1": final_vote["f1"],
        "oracle_initial_best_f1": best_initial, "oracle_final_best_f1": best_final,
        "oracle_best_f1_change": best_final - best_initial,
        "oracle_final_f1_minus_endpoint_f1": best_final - aggregate["f1"],
        "final_candidate_mean_f1": mean_final,
        "final_f1_minus_candidate_mean_f1": aggregate["f1"] - mean_final,
        "initial_vote_rejected_correct_candidate": initial_any and initial_majority["em"] != 100.0,
        "final_correctness_category": classify(initial_any, ever_any, revised_any, aggregate["em"] == 100.0,
                                                communication, endpoint),
        "causal_evidence_use": "UNOBSERVABLE_FROM_ANSWER_AND_CITATION_TRACE",
    }


def wilson(k, n):
    if not n:
        return None
    z = 1.959963984540054
    p, denominator = k / n, 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1-p) / n + z*z / (4*n*n)) / denominator
    return [100 * (center-half), 100 * (center+half)]


def mean_ci(values, seed=20271015, draws=2000):
    rng = random.Random(seed)
    n = len(values)
    samples = sorted(sum(rng.choices(values, k=n))/n for _ in range(draws))
    return [samples[round(.025*(draws-1))], samples[round(.975*(draws-1))]]


def summarize(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["experiment"], record["condition"]].append(record)
    summaries = []
    for (experiment, condition), group in sorted(grouped.items()):
        if len({record["id"] for record in group}) != len(group):
            raise ValueError("Summary observations must be unique questions")
        n = len(group)
        binary = {}
        for name in ["initial_any_em", "final_any_em", "initial_vote_rejected_correct_candidate",
                     "new_correct_proposal_after_communication", "any_initial_correct_agent_became_incorrect",
                     "aggregator_rejected_correct_candidate"]:
            observed = [record[name] for record in group if record[name] is not None]
            count = sum(observed)
            binary[name] = {"count": count if observed else None, "observed_n": len(observed),
                            "percent": 100*count/len(observed) if observed else None,
                            "wilson_ci95_percent": wilson(count, len(observed))}
        mean_fields = ["initial_majority_f1", "initial_majority_em", "final_f1", "final_em",
                       "oracle_initial_best_f1", "oracle_final_best_f1", "oracle_final_f1_minus_endpoint_f1",
                       "final_minus_initial_endpoint_f1", "final_minus_initial_endpoint_em",
                       "final_f1_minus_candidate_mean_f1"]
        means = {name: sum(record[name] for record in group)/len(group) for name in mean_fields}
        delta = [record["final_minus_initial_endpoint_f1"] for record in group]
        summaries.append({"experiment": experiment, "condition": condition, "endpoint": group[0]["endpoint"],
                          "n_unique_questions": n, "binary": binary, "means": means,
                          "final_minus_initial_endpoint_f1_ci95": mean_ci(delta),
                          "f1_helped": sum(x > 1e-9 for x in delta),
                          "f1_harmed": sum(x < -1e-9 for x in delta),
                          "f1_unchanged": sum(abs(x) <= 1e-9 for x in delta),
                          "categories": dict(Counter(record["final_correctness_category"] for record in group))})
    return summaries


def read_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate question checkpoint: " + str(path))
    return rows


def load_default(root=ROOT):
    active_path = root / "artifacts/active_machine_runs.json"
    active = json.loads(active_path.read_text())
    controlled = Path(active["controlled_run"])
    controller = Path(active["controller_study"])
    corrected = Path(active["e3_corrected"]["run_dir"])
    paths = {experiment: (controller if experiment == "E5" else controlled) / (experiment + ".jsonl")
             for experiment in ["E1", "E2", "E3", "E4", "E5"]}
    paths["E3-v2"] = corrected / "E3.jsonl"
    gold_path = controlled / "sample_gold.json"
    gold = {row["id"]: row for row in json.loads(gold_path.read_text())}
    data, evidence = {}, [active_path, gold_path]
    for experiment, path in paths.items():
        rows = read_rows(path)
        status_path = path.parent / ("status.json" if experiment == "E3-v2" else experiment + "_status.json")
        # Corrected E3 stores its authoritative completion gate separately.
        if experiment == "E3-v2":
            status_path = path.parent / "completion_gate_verified.json"
        status = json.loads(status_path.read_text())
        if experiment == "E3-v2" and (status.get("completion_gate") != "PASS"
                                      or status.get("final_output_sha256") != sha256(path)):
            raise ValueError("Incomplete or changed corrected E3 experiment")
        if experiment != "E3-v2" and (status.get("status") != "COMPLETE" or status.get("sha256") != sha256(path)):
            raise ValueError("Incomplete or changed saved experiment: " + experiment)
        if len(rows) != 300 or {row["id"] for row in rows} != set(gold):
            raise ValueError("Expected identical completed 300-question sets")
        data[experiment] = rows
        evidence += [path, status_path]
    evidence += [controlled/"frozen_manifest.json", controller/"manifest.json", corrected/"frozen_manifest.json",
                 corrected/"payload_plan.json", root/"protocol.py", root/"experiments.py", root/"backend.py",
                 root/"controller_study.py", root/"learned_controller.py", root/"e3_corrected.py",
                 root/"controller_splits.py", root/"references/hotpot_evaluate_v1.py"]
    evidence += [root/"analysis/final_findings_20261014"/name for name in [
        "audit_metrics.json", "paired_effects.json", "physical_usage_lower_bounds.json",
        "E3_v2_updated_analysis.json", "split_overlap_audit.json", "final_manifest.json",
        "01_COMPLETE_RESULTS_AND_METRICS.md", "02_DEEP_RAW_OUTPUT_FINDINGS.md",
        "03_PAPER_NARRATIVE_AND_REVISION_PLAN.md", "04_TECHNICAL_METHODS_AND_PSEUDOCODE.md",
        "sources/current_paper_extracted.txt"]]
    evidence += [controlled/"calls.jsonl", controller/"inference/calls.jsonl", corrected/"calls.jsonl"]
    return data, gold, evidence


def analyze(data, gold):
    records, missing = [], []
    e1 = {row["id"]: row for row in data["E1"]}
    for experiment, rows in data.items():
        for row in rows:
            reference = gold[row["id"]]["answer"]
            if experiment == "E3-v2" and not row.get("eligible"):
                for label in ["R_prefix", "R_crop"]:
                    missing.append({"experiment": experiment, "id": row["id"], "condition": label,
                                    "status": "NOT_RUN_INELIGIBLE_NO_SENDER_EVIDENCE"})
            for label, condition in row["conditions"].items():
                regime = label.split("_")[0]
                history = None
                if experiment in ["E1", "E2"]:
                    initial = row["states"][regime]["responses"]
                    source = "same_record.states." + regime
                elif experiment in ["E3", "E3-v2"]:
                    initial = e1[row["id"]]["states"]["R"]["responses"]
                    source = "E1.states.R; initial_call_keys_verified_against_condition_cost"
                elif experiment == "E4":
                    initial = row["trajectory"][0]["responses"]
                    depth = row["stop_depth"] if label == "P_stopped" else len(row["trajectory"])-1
                    history = [snapshot["responses"] for snapshot in row["trajectory"][:depth+1]]
                    source = "same_record.trajectory[0]"
                else:
                    initial = row["conditions"][regime+"_NoComm"]["responses"]
                    source = "same_record.conditions." + regime + "_NoComm"
                if experiment == "E3-v2" and not row.get("eligible") and label in ["R_prefix", "R_crop"]:
                    missing.append({"experiment": experiment, "id": row["id"], "condition": label,
                                    "status": "NOT_RUN_INELIGIBLE_NO_SENDER_EVIDENCE"})
                    continue
                endpoint = "receiver_1" if experiment in ["E3", "E3-v2"] else "majority"
                records.append(condition_record(experiment, row, label, condition, initial, reference,
                                                history, source, endpoint))
            if experiment == "E4":
                initial = row["trajectory"][0]["responses"]
                for depth, snapshot in enumerate(row["trajectory"]):
                    records.append(condition_record("E4_round", row, "P_depth_"+str(depth), snapshot,
                                                    initial, reference,
                                                    [s["responses"] for s in row["trajectory"][:depth+1]],
                                                    "same_record.trajectory[0]"))
    return records, missing


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def repository_inventory(data, gold, evidence):
    """Recompute artifact facts, rather than transcribing earlier report assertions."""
    experiments = {}
    for experiment, rows in data.items():
        request_keys = {key for row in rows for condition in row["conditions"].values()
                        for key in condition["cost"]["call_keys"]}
        experiments[experiment] = {"n_unique_questions": len(rows), "distinct_referenced_request_keys": len(request_keys),
            "sum_arm_logical_calls_not_physical_spend": sum(condition["cost"]["logical_calls"] for row in rows
                                                           for condition in row["conditions"].values())}
    candidate_counts, retained_counts, strata = Counter(), Counter(), Counter()
    for row in data["E1"]:
        supporting = {title for title, index in gold[row["id"]]["supporting_facts"]}
        candidate = {doc["title"] for doc in gold[row["id"]]["documents"]}
        retained = {doc["title"] for doc in row["states"]["P"]["pool"]}
        candidate_counts[len(supporting & candidate)] += 1
        retained_counts[len(supporting & retained)] += 1
        colocated = any(supporting <= {doc["title"] for doc in assigned}
                        for assigned in row["states"]["P"]["assignment"])
        strata["missing_gold" if not supporting <= retained else "colocated_gold" if colocated else "distributed_gold"] += 1
    matching = Counter()
    for row in data["E5"]:
        for regime in ["R", "P"]:
            for seed in [7, 19, 31]:
                learned = row["conditions"][f"{regime}_Learned_seed{seed}"]
                random_arm = row["conditions"][f"{regime}_MatchedRandom_seed{seed}"]
                matching["pairs"] += 1
                matching["edge_count_equal"] += len(learned["action"]["edges"]) == len(random_arm["action"]["edges"])
                matching["quota_equal"] += learned["action"]["quota"] == random_arm["action"]["quota"]
                matching["logical_call_mismatch"] += learned["cost"]["logical_calls"] != random_arm["cost"]["logical_calls"]
                matching["payload_mismatch"] += learned["payload_tokens"] != random_arm["payload_tokens"]
    ledgers = {}
    for path in evidence:
        if path.name != "calls.jsonl":
            continue
        events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        success = [event for event in events if event["event"] == "success"]
        ledgers[str(path)] = {"event_counts": dict(Counter(event["event"] for event in events)),
            "reported_input_tokens_lower_bound": sum((event.get("usage") or {}).get("prompt_tokens") or 0 for event in success),
            "reported_output_tokens_lower_bound": sum((event.get("usage") or {}).get("completion_tokens") or 0 for event in success),
            "success_input_usage_missing": sum((event.get("usage") or {}).get("prompt_tokens") is None for event in success),
            "success_output_usage_missing": sum((event.get("usage") or {}).get("completion_tokens") is None for event in success),
            "note": "Raw event counts; uncertain retries do not erase original billing uncertainty"}
    manifest_path = next(path for path in evidence if path.name == "frozen_manifest.json"
                         and path.parent.name.startswith("run_"))
    manifest = json.loads(manifest_path.read_text())
    source_checks = {name: {"recorded_sha256": checksum, "current_sha256": sha256(ROOT/name),
                            "matches_frozen": checksum == sha256(ROOT/name)}
                     for name, checksum in manifest["plan"]["sources"].items()}
    return {"experiments": experiments, "candidate_gold_article_counts": dict(candidate_counts),
            "retained_gold_article_counts": dict(retained_counts), "retained_gold_location_strata": dict(strata),
            "E5_resource_matching": dict(matching), "ledger_inventory": ledgers,
            "controlled_frozen_source_checks": source_checks,
            "source_graph_status": "Parent Verify preflight: index generation 2026-10-13T17:28:45Z has no tracked WEBExtension coverage; exact-source fallback used"}


def verify_official_scores(records, gold):
    sys.modules.setdefault("ujson", json)
    spec = importlib.util.spec_from_file_location("e8_official_hotpot", ROOT/"references/hotpot_evaluate_v1.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    checked = 0
    for record in records:
        reference = gold[record["id"]]["answer"]
        for answer in record["initial_answers"] + record["final_answers"] + [record["final_answer"]]:
            observed = answer_score(answer, reference)
            expected = {"em": 100*evaluator.exact_match_score(answer, reference),
                        "f1": 100*evaluator.f1_score(answer, reference)[0]}
            if any(abs(observed[metric]-expected[metric]) > 1e-8 for metric in ["em", "f1"]):
                raise ValueError("Answer metric disagrees with preserved official evaluator")
            checked += 1
    return {"checked_answer_references_including_repeats": checked, "mismatches": 0,
            "official_evaluator_sha256": sha256(ROOT/"references/hotpot_evaluate_v1.py")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "offline")
    args = parser.parse_args(argv)
    data, gold, evidence = load_default()
    records, missing = analyze(data, gold)
    summary = summarize(records)
    args.output.mkdir(parents=True, exist_ok=True)
    inventory = repository_inventory(data, gold, evidence)
    inventory["official_score_crosscheck"] = verify_official_scores(records, gold)
    save_json(args.output/"repository_inventory.json", inventory)
    output = args.output / "e8_question_conditions.jsonl"
    output.write_text("".join(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n" for record in records))
    definitions = {
        "scope": "OFFLINE_EXPLORATORY_OBSERVATIONAL_SAVED_TRACES; no Azure or network inference",
        "unit": "One unique question per experiment-condition; never pool seeds/arms/suites as independent questions",
        "oracle": "Any-agent EM and best-agent F1 are diagnostic upper bounds requiring gold selection, not deployable performance",
        "aggregation": "Deterministic majority of normalized answers, ties follow fixed node order; E3 is receiver 1, not majority",
        "f1": "Continuous official answer F1; no arbitrary F1 correctness threshold; EM analyzed separately",
        "delta": "Final endpoint minus initial endpoint includes revision and selection, not a causal effect of aggregation alone",
        "mean_candidate_reference": "Final F1 minus candidate-mean F1 describes vote selection relative to uniform candidate sampling; not a measured randomized arm",
        "intervals": "Descriptive marginal 95% Wilson intervals for question-level counts and 2000-resample question bootstrap for mean F1 change; not multiplicity-adjusted",
        "unobservable": ["Whether model internally used delivered evidence", "Citation entailment", "Counterfactual outcome without a particular correct proposal", "Azure monetary cost"],
    }
    save_json(args.output / "e8_summary.json", {"definitions": definitions, "summary": summary,
                                               "not_observed": missing, "physical_inference_attempts": 0})
    flat = []
    for item in summary:
        row = {key: item[key] for key in ["experiment", "condition", "endpoint", "n_unique_questions", "f1_helped", "f1_harmed", "f1_unchanged"]}
        row.update(item["means"])
        row.update({name+"_count": value["count"] for name, value in item["binary"].items()})
        flat.append(row)
    with (args.output / "e8_summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    save_json(args.output / "provenance.json", {"analysis_source": str(Path(__file__).resolve()),
              "analysis_source_sha256": sha256(__file__), "source_artifacts": {str(path.resolve()): sha256(path) for path in evidence},
              "generated_artifacts": {str(path.name): sha256(path) for path in args.output.iterdir()
                                      if path.is_file() and path.name != "provenance.json"},
              "physical_inference_attempts": 0, "no_provider_client_imported": True})
    lines = ["# E8 offline proposal and aggregation audit", "", *[f"- **{key}:** {value}" for key, value in definitions.items()], "",
             "## Results by saved protocol", "", "All values are descriptive. Detailed per-question answers, EM/F1, transitions and call keys are in `e8_question_conditions.jsonl`; uncertainty intervals are in `e8_summary.json`.", ""]
    for item in summary:
        binary = item["binary"]
        lines.append(f"- {item['experiment']} {item['condition']} (n={item['n_unique_questions']}): initial any-correct {binary['initial_any_em']['count']}; revised any-correct {binary['final_any_em']['count']}; vote rejects correct candidate {binary['aggregator_rejected_correct_candidate']['count']}; final F1 {item['means']['final_f1']:.4f}; final EM {item['means']['final_em']:.4f}; categories {item['categories']}.")
    lines += ["", "## Interpretation limits", "", "E2 reuses E1 shared-evidence calls and is not independent replication. E3-v2 follows prior inspection of the same 300 identities and is exploratory. A new correct answer after delivery is temporal evidence, not proof that message information caused it; self-update can also recover answers. A previously correct candidate becoming wrong does not prove harmful message influence without an adequate controlled contrast. In E3, majority voting is only a hypothetical diagnostic, and aggregator rejection is not applicable. Final-vote F1 cannot exceed the best available candidate F1: the oracle gap quantifies selection headroom without claiming oracle performance.", "", f"Not-run records: {len(missing)}. Physical API attempts for this audit: 0.", ""]
    (args.output / "e8_report.md").write_text("\n".join(lines))
    # Include the generated narrative in the final provenance inventory.
    provenance = json.loads((args.output / "provenance.json").read_text())
    provenance["generated_artifacts"]["e8_report.md"] = sha256(args.output/"e8_report.md")
    save_json(args.output/"provenance.json", provenance)
    print(json.dumps({"status": "COMPLETE_OFFLINE", "condition_question_records": len(records),
                      "summary_groups": len(summary), "not_run_records": len(missing), "inference_attempts": 0,
                      "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
