"""Saved-trace pairing, accounting and observability invariants; zero provider calls."""
import copy
import importlib
import json
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / 'WEBExtension/causal_study/next_phase/deep_transitions.py'
deep = importlib.import_module('WEBExtension.causal_study.next_phase.deep_transitions')


class TestDeepTransitions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.gold, cls.paths, _, cls.provenance = deep.load_raw()
        cls.records, cls.missing = deep.reconstruct_records(cls.data, cls.gold, cls.paths)
        cls.before = [r for r in cls.records if r['experiment'] == 'E5' and r['condition'] == 'P_NoComm']
        cls.after = [r for r in cls.records if r['experiment'] == 'E5' and r['condition'] == 'P_Broadcast']

    def test_nonabstaining_vote_is_gold_free_and_fixed_order(self):
        function = getattr(deep, 'nonabstaining_vote', None)
        self.assertTrue(callable(function), 'Gold-free abstention reaggregation not implemented')
        responses = [{'answer':'noanswer'}, {'answer':'The Alpha'}, {'answer':'beta'}]
        self.assertEqual(function(responses), 'The Alpha')
        self.assertEqual(function([{'answer':'noanswer'}]*3), 'noanswer')
        self.assertEqual(function([{'answer':'unknown'}, {'answer':'noanswer'}]), 'unknown')
        self.assertEqual(function([{'answer':'','provider_refusal':True}, {'answer':'beta'}]), 'beta')
        with self.assertRaisesRegex(ValueError, 'explicit provider'):
            function([{'answer':''}, {'answer':'beta'}])

    def test_reaggregation_eligibility_and_endpoint_counts(self):
        diagnostic = deep.reaggregation_diagnostics(self.records, self.gold, draws=100)
        for regime, eligible, new_correct in [('P',20,62), ('R',33,73)]:
            selected = next(item for item in diagnostic if item['regime'] == regime)
            self.assertEqual(selected['eligible_two_distinct_nonabstaining'], eligible)
            comparison = next(item for item in selected['comparisons'] if
                              item['population'] == 'ALL_SAVED_QUESTIONS' and
                              item['after'] == regime+'_NoAbstentionVote')
            self.assertEqual(comparison['after_correct'], new_correct)
            self.assertEqual(comparison['vote_transition']['10']['n'], 0)

    def test_e3_eligibility_alignment_is_explicit(self):
        records = [r for r in self.records if r['experiment'] == 'E3-v2']
        summaries = deep.build_contrasts(records, draws=100)
        self.assertTrue(summaries)
        selected = next(s for s in summaries if s['after'] == 'R_crop')
        self.assertIn('baseline_questions_excluded_not_run', selected)
        self.assertEqual(selected['n'] + selected['baseline_questions_excluded_not_run'], 300)
        self.assertEqual(selected['n'], 299)
        self.assertEqual(len(self.missing), 2)
        self.assertEqual({r['id'] for r in self.missing}, {'5a8cfee555429941ae14df5c'})

    def test_exact_paired_accounting_not_just_marginals(self):
        summary = deep.paired_summary(self.before, self.after, draws=100)
        self.assertEqual([summary['coverage_transition'][c]['n'] for c in ['00','01','10','11']], [228,5,6,61])
        self.assertEqual([summary['vote_transition'][c]['n'] for c in ['00','01','10','11']], [265,25,1,9])
        self.assertEqual(summary['joint_coverage_vote']['11_01']['n'], 23)
        self.assertEqual(summary['joint_coverage_vote']['01_01']['n'], 2)
        self.assertEqual(summary['joint_coverage_vote']['10_10']['n'], 1)
        self.assertEqual(summary['em_delta_pp'], 8.0)

    def test_duplicate_questions_cannot_be_independent_rows(self):
        with self.assertRaisesRegex(ValueError, 'unique question'):
            deep.paired_summary(self.before + self.before[:1], self.after, draws=100)

    def test_matching_question_id_does_not_replace_matching_initial_call_keys(self):
        changed = copy.deepcopy(self.after[:1])
        changed[0]['initial_call_keys'][0] = 'different-prompt-or-generation'
        with self.assertRaisesRegex(ValueError, 'initial state'):
            deep.paired_summary(self.before[:1], changed, draws=100)

    def test_incomplete_question_pairing_rejected(self):
        with self.assertRaisesRegex(ValueError, 'identities'):
            deep.paired_summary(self.before[:-1], self.after, draws=100)

    def test_normalized_answer_agreement_and_fixed_node_ties(self):
        observed = deep.agreement(['The Beatles!', 'beatles', 'a Beatles'])
        self.assertTrue(observed['unanimous'])
        self.assertEqual(observed['pairwise_agreement'], 1.0)
        self.assertEqual(deep.old.majority([{'answer':'first'}, {'answer':'second'}, {'answer':'third'}]), 'first')

    def test_e3_receiver_is_never_a_majority_rejection(self):
        receiver = [r for r in self.records if r['experiment'] in {'E3','E3-v2'}]
        self.assertTrue(receiver)
        self.assertTrue(all(r['endpoint'] == 'receiver_1' and r['aggregator_rejected_correct_candidate'] is None for r in receiver))

    def test_exact_raw_line_and_initial_provenance_links(self):
        for experiment, path in self.paths.items():
            record = next(r for r in self.records if r['experiment'] == experiment)
            raw = json.loads(path.read_text().splitlines()[record['raw_line']-1])
            self.assertEqual(record['id'], raw['id'])
            if experiment.startswith('E3'):
                self.assertIn('/E1.jsonl:', record['initial_raw_link'])

    def test_reconstruction_does_not_drop_coverage_losses_or_unobserved_inference(self):
        self.assertEqual(sum(r['correct_initial_proposal_disappeared'] for r in self.after), 6)
        self.assertEqual(sum(r['previously_unavailable_correct_proposal_appeared'] for r in self.after), 5)
        self.assertTrue(all(r['causal_evidence_use'] == 'UNOBSERVABLE_FROM_ANSWER_AND_CITATION_TRACE' for r in self.records))
        self.assertEqual(len({r['id'] for r in self.records}), 300)

    def test_bootstrap_preserves_question_units_and_is_reproducible(self):
        first = deep.paired_mean_ci([100, -100, 0, 0], draws=500, seed=11)
        self.assertEqual(first, deep.paired_mean_ci([100, -100, 0, 0], draws=500, seed=11))
        self.assertLessEqual(first[0], 0)
        self.assertGreaterEqual(first[1], 0)
        self.assertEqual(deep.paired_mean_ci([0]*5, draws=100), [0,0])


if __name__ == '__main__':
    unittest.main()
