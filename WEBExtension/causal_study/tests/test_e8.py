"""Meaningful E8 guards: endpoint identity, linked states and EM/F1 separation."""
import importlib.util
import unittest
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "offline_e8.py"
SPEC = importlib.util.spec_from_file_location("offline_e8", PATH)
e8 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(e8)


def responses(*answers):
    return [{"answer": answer, "call_key": "call"+str(i)+answer} for i, answer in enumerate(answers)]


def condition(initial, revised, answer=None, receiver=False, communicate=True):
    answer = answer if answer is not None else (revised[1]["answer"] if receiver else e8.majority(revised))
    return {"responses": revised, "answer": answer, **e8.answer_score(answer, "Travis County"),
            "cost": {"call_keys": e8.keys(initial)+e8.keys(revised)},
            "deliveries": [{"sender": 0, "recipient": 1}] if communicate else []}


class E8Tests(unittest.TestCase):
    def test_em_and_f1_are_separate(self):
        self.assertEqual(e8.answer_score("Travis", "Travis County"), {"em": 0.0, "f1": 200/3})
        self.assertEqual(e8.answer_score("yes indeed", "yes")["f1"], 0.0)
        self.assertEqual(e8.answer_score("The Travis County!", "Travis County")["em"], 100.0)

    def test_majority_rejects_correct_candidate(self):
        initial = responses("noanswer", "noanswer", "Travis County")
        revised = responses("noanswer", "noanswer", "Travis County")
        row = e8.condition_record("E1", {"id": "q"}, "P_CW", condition(initial, revised), initial, "Travis County")
        self.assertTrue(row["aggregator_rejected_correct_candidate"])
        self.assertEqual(row["final_correctness_category"], "correct_proposal_available_but_rejected_by_majority")
        self.assertEqual(row["oracle_final_f1_minus_endpoint_f1"], 100)

    def test_lost_and_recovered_proposals(self):
        initial = responses("Travis County", "noanswer", "noanswer")
        revised = responses("noanswer", "noanswer", "noanswer")
        row = e8.condition_record("E1", {"id": "q"}, "P_CW", condition(initial, revised), initial, "Travis County")
        self.assertTrue(row["any_initial_correct_agent_became_incorrect"])
        self.assertEqual(row["final_correctness_category"], "correct_proposal_produced_but_lost_during_revision")
        corrected = responses("Travis County", "Travis County", "noanswer")
        recovered = e8.condition_record("E1", {"id": "q"}, "P_CW", condition(revised, corrected),
                                        revised, "Travis County")
        self.assertTrue(recovered["new_correct_proposal_after_communication"])
        self.assertEqual(recovered["final_correctness_category"], "correct_answer_recovered_through_communication")

    def test_same_question_is_not_sufficient_initial_link(self):
        initial = responses("noanswer", "noanswer", "noanswer")
        c = condition(initial, initial)
        c["cost"]["call_keys"][0] = "different-deployment-or-prompt"
        with self.assertRaisesRegex(ValueError, "call keys"):
            e8.condition_record("E3", {"id": "q"}, "R_crop", c, initial, "Travis County")

    def test_receiver_endpoint_does_not_invent_aggregator_rejection(self):
        initial = responses("Travis County", "noanswer", "noanswer")
        c = condition(initial, initial, receiver=True)
        row = e8.condition_record("E3", {"id": "q"}, "R_crop", c, initial, "Travis County", endpoint="receiver_1")
        self.assertIsNone(row["aggregator_rejected_correct_candidate"])
        self.assertEqual(row["aggregation_observability"], "NOT_APPLICABLE_RECEIVER_ENDPOINT")

    def test_detects_corrupted_majority_endpoint(self):
        initial = responses("noanswer", "noanswer", "Travis County")
        c = condition(initial, initial, "Travis County")
        with self.assertRaisesRegex(ValueError, "deterministic majority"):
            e8.condition_record("E5", {"id": "q"}, "P_Broadcast", c, initial, "Travis County")

    def test_duplicate_questions_cannot_inflate_summary(self):
        initial = responses("noanswer", "noanswer", "noanswer")
        row = e8.condition_record("E1", {"id": "q"}, "P_CW", condition(initial, initial), initial, "Travis County")
        with self.assertRaisesRegex(ValueError, "unique questions"):
            e8.summarize([row, row])


if __name__ == "__main__":
    unittest.main()
