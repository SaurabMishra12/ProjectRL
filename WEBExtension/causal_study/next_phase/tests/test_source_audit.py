import ast
import copy
import importlib.util
import json
from pathlib import Path
import unittest

MODULE = Path(__file__).resolve().parents[1] / "source_audit.py"
spec = importlib.util.spec_from_file_location("source_audit", MODULE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class SourceAuditTests(unittest.TestCase):
    def record(self):
        return audit.source_record("q", "Who taught Ada?", "bridge",
            [{"title": "Byron", "sentences": ["Byron was a poet."]},
             {"title": "Ada", "sentences": ["Ada was taught by Mary."]}],
            "Byron", [["Ada", 0]], "E8_OLD_300", {}, True)

    def test_gold_occurrence_is_never_sufficiency(self):
        row = self.record()
        self.assertTrue(row["literal_reference_sentence_occurrences"])
        self.assertEqual(row["one_article_semantically_sufficient"], "UNRESOLVED_HUMAN_NEEDED")
        self.assertEqual(row["future_complementarity_eligibility"], "NOT_QUALIFIED")
        self.assertEqual(audit.classify_citation(row, "Byron", [["Byron", 0]])["entailment"],
                         "UNRESOLVED_HUMAN_NEEDED")

    def test_whole_normalized_token_occurrence(self):
        self.assertFalse(audit.literal_occurrence("no", "know about Norway"))
        self.assertTrue(audit.literal_occurrence("the Rocky Mountain Institute", "Rocky Mountain Institute."))

    def test_source_packet_does_not_depend_on_reference_or_performance(self):
        row = self.record()
        packet = audit.blinded_source_packet(row)
        changed = copy.deepcopy(row)
        changed.update(private_reference_answer="Mary", all_annotated_support_present=False,
                       baseline_initial_disagreement=False, final_em=100)
        self.assertEqual(packet, audit.blinded_source_packet(changed))
        self.assertEqual(set(packet), {"review_id", "question", "documents", "annotation_status"})

    def test_invalid_citation_cannot_be_valid_provenance(self):
        for citations in [[["Missing", 0]], [["Ada", -1]], [["Ada", True]], [["Ada", "0"]], ["Ada"]]:
            row = audit.classify_citation(self.record(), "Mary", citations)
            self.assertFalse(row["source_provenance_valid"])
            self.assertEqual(row["entailment"], "UNRESOLVED_WITH_MISSING_OR_INVALID_SOURCE")

    def test_no_citation_and_abstention_are_not_nonentailment_labels(self):
        self.assertEqual(audit.classify_citation(self.record(), "noanswer", [])["entailment"],
                         "ABSTENTION_NO_ANSWER_ASSERTION")
        self.assertEqual(audit.classify_citation(self.record(), "Mary", [])["entailment"],
                         "NO_CITATION_SUPPLIED")

    def test_untracked_scores_cannot_change_bounded_selection(self):
        rows = [self.record() for _ in range(8)]
        for i, row in enumerate(rows):
            row["id"] = str(i)
            row["all_annotated_support_present"] = bool(i % 2)
            row["baseline_initial_disagreement"] = bool(i // 2 % 2)
        selected = audit.select_old_review(rows)
        for i, row in enumerate(rows):
            row["final_em"] = i * 7
            row["final_f1"] = 100 - i
            row["private_reference_answer"] = "altered"
        self.assertEqual(selected, audit.select_old_review(rows[::-1]))

    def test_agent_labels_never_qualify_future_cohort(self):
        annotations = dict(reviewers=["a", "b"], adjudicated=True,
            reviewer_type="independent_human", outcome_blinded=True, union_sufficient=True,
            question_valid=True, reference_valid=True,
            both_articles_necessary=True, single_article_sufficient=False,
            alternative_source_answer_bearing=False)
        self.assertTrue(audit.qualified_by_independent_annotations(annotations))
        for field, value in [("reviewer_type", "agent"), ("reviewers", ["a", "a"]),
                             ("outcome_blinded", False), ("adjudicated", False),
                             ("question_valid", False), ("reference_valid", False),
                             ("alternative_source_answer_bearing", None)]:
            failed = {**annotations, field: value}
            self.assertFalse(audit.qualified_by_independent_annotations(failed))

    def test_no_provider_or_network_imports(self):
        parsed = ast.parse(MODULE.read_text())
        modules = set()
        for node in ast.walk(parsed):
            if isinstance(node, ast.Import):
                modules.update(a.name.split(".")[0] for a in node.names)
            if isinstance(node, ast.ImportFrom):
                modules.add((node.module or "").split(".")[0])
        self.assertFalse(modules & {"provider", "backend", "openai", "requests", "urllib", "httpx", "socket"})

    def test_historical_artifacts_cannot_be_output(self):
        with self.assertRaises(ValueError):
            audit.generate(output=audit.REPO / "WEBExtension/causal_study/paid_pilot")

    def test_real_corpus_integrity_and_complete_packages(self):
        directory = MODULE.parent / "source_artifacts"
        rows = audit.read_jsonl(directory / "source_observations.jsonl")
        packets = audit.read_jsonl(directory / "blinded_source_review.jsonl")
        self.assertEqual(len(rows), 308)
        self.assertEqual(len({r["id"] for r in rows}), 308)
        self.assertEqual(len(packets), 308)
        self.assertEqual({r["review_id"] for r in rows}, {r["review_id"] for r in packets})
        self.assertEqual(sum(r["all_annotated_support_present"] for r in rows
                             if r["cohort"] == "E8_OLD_300"), 76)
        manifest = json.loads((directory / "source_audit_manifest.json").read_text())
        for path, expected in manifest["source_hashes"].items():
            self.assertEqual(audit.file_hash(audit.REPO / path), expected)
        for path, expected in manifest["generated_hashes"].items():
            self.assertEqual(audit.file_hash(directory / path), expected)


if __name__ == "__main__":
    unittest.main()
