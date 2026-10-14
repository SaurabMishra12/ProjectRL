import importlib.util
import pathlib
import unittest
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('protocol'), 'experiment protocol must exist')
        import protocol
        self.p = protocol

    def test_official_scoring_edges(self):
        self.assertEqual(self.p.score('yes indeed', 'yes')['f1'], 0)
        self.assertEqual(self.p.score('The U.S.', 'US')['em'], 100)
        self.assertEqual(self.p.score('', '')['f1'], 0)

    def test_partition_preserves_union_without_gold(self):
        docs=[{'title':str(i),'sentences':['real source sentence']} for i in range(10)]
        x=self.p.assign_evidence(docs,'P','qid',42)
        self.assertEqual(sorted(d['title'] for a in x for d in a), [str(i) for i in range(10)])
        self.assertEqual(x,self.p.assign_evidence(docs,'P','qid',42))
        self.assertEqual([len(a) for a in self.p.assign_evidence(docs,'R','qid',42)],[10]*3)

    def test_prompt_gold_and_condition_excluded(self):
        row={'id':'x','question':'Who wrote it?', 'answer':'SECRET GOLD', 'supporting_facts':[['GOLD',0]]}
        msgs=self.p.build_prompt(row['question'], 'retrieval', [], None, [])
        self.assertNotIn('SECRET GOLD',str(msgs))
        self.assertNotIn('GOLD',str(msgs))
        self.assertNotIn('clockwise',str(msgs))

    def test_invalid_completion_does_not_become_an_answer(self):
        with self.assertRaises(ValueError): self.p.parse_completion('network error', [])
        with self.assertRaises(ValueError): self.p.parse_completion('{"answer":"", "citations":[]}', [])

    def test_cycles_balance_and_reverse(self):
        cw, ccw=self.p.CYCLES.values()
        for edges in (cw,ccw):
            self.assertEqual(sorted(s for s,t in edges),[0,1,2])
            self.assertEqual(sorted(t for s,t in edges),[0,1,2])
        self.assertEqual(set(cw),set((t,s) for s,t in ccw))

    def test_paired_stats_reject_missing_pairs(self):
        with self.assertRaises(ValueError): self.p.paired_effect({'a':1},{'b':1})
        d=self.p.paired_effect({'a':10,'b':20},{'a':0,'b':10},draws=100)
        self.assertEqual(d['effect'],10)
        self.assertEqual(d['ci95'],[10,10])

if __name__=='__main__': unittest.main()
