"""Call-ledger recovery tests only; no generated model responses or network calls."""
import pathlib
import sys
import unittest
import tempfile
import json
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
import backend

class RecoveryTests(unittest.TestCase):
    def test_later_interrupted_attempt_is_not_hidden_by_earlier_rejection(self):
        self.assertTrue(hasattr(backend,'unresolved_attempts'),'Per-attempt recovery must exist')
        records=[{'event':'started','key':'same','attempt_id':'first'},
                 {'event':'rejected','key':'same','attempt_id':'first'},
                 {'event':'started','key':'same','attempt_id':'second'}]
        self.assertEqual([r['attempt_id'] for r in backend.unresolved_attempts(records)],['second'])

    def test_timeout_remains_unresolved(self):
        self.assertTrue(hasattr(backend,'unresolved_attempts'),'Per-attempt recovery must exist')
        records=[{'event':'started','key':'q','attempt_id':'a'},
                 {'event':'uncertain','key':'q','attempt_id':'a'}]
        self.assertEqual(len(backend.unresolved_attempts(records)),1)

    def test_uncertain_attempt_blocks_calls_on_same_instance(self):
        instance=backend.RealBackend.__new__(backend.RealBackend)
        instance.attempts=[{'event':'started','key':'q','attempt_id':'a'},
                           {'event':'uncertain','key':'q','attempt_id':'a'}]
        with self.assertRaisesRegex(RuntimeError,'Unresolved provider attempt'):
            instance.complete([],[],{})

    def test_exhausted_budget_rejects_before_client_or_network(self):
        from protocol import ROOT
        instance=backend.RealBackend.__new__(backend.RealBackend)
        instance.config=json.loads((ROOT/'config.example.json').read_text())
        instance.config['max_calls']=1
        instance.identity={}
        class IdentityOnlyTokenizer:
            identity={}
        instance.tokenizer=IdentityOnlyTokenizer()
        instance.attempts=[{'event':'started','key':'q','attempt_id':'a'},
                           {'event':'rejected','key':'q','attempt_id':'a'}]
        with tempfile.TemporaryDirectory() as temp:
            instance.cache_dir=pathlib.Path(temp)
            with self.assertRaisesRegex(RuntimeError,'Maximum call budget reached'):
                instance.complete([],[],{})
        # No client exists: an accidental network call would fail this test.

    def test_unique_prefix_facts_do_not_double_count_delivery(self):
        import experiments
        self.assertTrue(hasattr(experiments,'cumulative_deliveries'),'Prefix delivery helper must exist')
        # Metadata structure test only; values are identifiers, not benchmark text or predictions.
        rounds=[[{'sender':0,'recipient':1,'tokens':4,'text':'', 'facts':[{'title':'source','sentence_id':0,'text':''}]}],
                [{'sender':0,'recipient':1,'tokens':4,'text':'', 'facts':[{'title':'source','sentence_id':0,'text':''}]}]]
        merged=experiments.cumulative_deliveries(rounds)
        self.assertEqual(len(merged),2)
        self.assertEqual([d['round'] for d in merged],[1,2])
        self.assertEqual(experiments.recipient_fact_sets(merged)[1],{('source',0)})

if __name__=='__main__':unittest.main()
