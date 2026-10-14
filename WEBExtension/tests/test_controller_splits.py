import importlib.util
import pathlib
import sys
import unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from protocol import load_official
class SplitTests(unittest.TestCase):
 def setUp(self):
  self.assertIsNotNone(importlib.util.find_spec('controller_splits'),'Leakage-controlled splitter required')
  import controller_splits
  self.s=controller_splits
 def test_same_question_id_rejected_across_partitions(self):
  row=load_official()[0]
  with self.assertRaises(ValueError):self.s.assert_disjoint({'train':[row],'test':[row]})
 def test_same_candidate_document_rejected_even_with_distinct_ids(self):
  a=load_official()[0];b=load_official()[1]
  b={**b,'documents':a['documents']}
  with self.assertRaises(ValueError):self.s.assert_disjoint({'train':[a],'test':[b]})
 def test_heldout_exposes_gold_free_inputs_separately(self):
  row=load_official()[0]
  inputs=self.s.inference_inputs([row])
  self.assertEqual(set(inputs[0]),{'id','question','documents'})
  self.assertNotIn('answer',inputs[0]);self.assertNotIn('supporting_facts',inputs[0])
if __name__=='__main__':unittest.main()
