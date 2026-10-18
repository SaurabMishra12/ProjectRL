"""Meaningful guards against unit, voting and identifiability mistakes."""
import json, math, sys, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'analyses'))
from offline_audit import score,vote,graph_support,effect
from power_sensitivity import bounded_power

class ScientificContracts(unittest.TestCase):
    def test_special_answers_not_partial_matches(self):
        self.assertEqual(score('yes','yes please'),(0.,0.))
        self.assertEqual(score('the Paris','Paris'),(1.,1.))
        self.assertEqual(score('Che Guevara','Ernesto Che Guevara'),(0.,.8))
    def test_abstention_aware_counterexample(self):
        xs=['noanswer','noanswer','Paris']
        self.assertEqual(vote(xs,'original'),'noanswer')
        self.assertEqual(vote(xs,'aware'),'Paris')
        # On a required-abstention case the second decision is a false answer.
    def test_empty_refusals_are_not_silently_dropped_in_original(self):
        xs=['','','Paris']
        self.assertEqual(vote(xs,'original'),'')
        self.assertEqual(vote(xs,'aware'),'Paris')
        self.assertEqual(vote(xs,'strict'),'noanswer')
    def test_three_original_seats_define_strict_denominator(self):
        self.assertEqual(vote(['noanswer','Paris','noanswer'],'strict'),'noanswer')
        self.assertEqual(vote(['noanswer','Paris','Paris'],'strict'),'Paris')
    def test_instar_has_no_activity_fixed_alternative(self):
        g=graph_support()
        self.assertEqual(g['total_graphs'],64)
        self.assertEqual(g['two_edge_one_recipient_class_sizes'],[1,1,1])
        self.assertEqual(sum(len(x['graphs']) for x in g['groups']),64)
    def test_paired_uncertainty_uses_question_differences(self):
        d=effect([-1,0,1,0]);self.assertEqual(d['n'],4);self.assertEqual(d['mean'],0)
        self.assertLess(d['ci95'][0],0);self.assertGreater(d['ci95'][1],0)
    def test_ppo_clipped_improvement_is_flat(self):
        objective=lambda r:min(r, min(max(r,.8),1.2))
        self.assertEqual((objective(1.5+1e-6)-objective(1.5-1e-6))/(2e-6),0)
        self.assertGreater(min(max(1.5,.8),1.2),0)
    def test_no_invalid_sd2_simulation(self):
        with self.assertRaises(ValueError):bounded_power(100,2)
        d=bounded_power(951,.5,draws=2000)
        self.assertAlmostEqual(sum(d['probabilities']),1)
        self.assertTrue(.70<d['rejection_rate']<.90)
    def test_saved_outputs_independently_rescore(self):
        a=json.loads((ROOT/'results/offline_audit.json').read_text())
        self.assertEqual(a['score_mismatches'],[])
        self.assertEqual(a['E6_score_mismatches'],[])
        self.assertEqual(a['union_ids'],300)
        self.assertEqual(a['E5_recipient_count_mismatches'],912)
        self.assertEqual(a['E3v2_distinct_payloads'],299)
    def test_no_false_precision_from_duplicate_contexts(self):
        d=json.loads((ROOT/'results/history_checks.json').read_text())
        self.assertEqual(d['MuSiQue']['distinct_question_texts'],2410)
        self.assertEqual(d['MuSiQue']['confirm_pool_after_20_and_24'],2366)

if __name__=='__main__':unittest.main()
