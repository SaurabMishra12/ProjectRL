"""Metadata accounting invariants; no model predictions or experiments generated."""
import pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
class CostTests(unittest.TestCase):
 def test_fanin_counts_one_receiver_and_no_untouched_initial_twice(self):
  import controller_study as study
  self.assertTrue(hasattr(study,'routed_cost'),'Actual recipient call accounting required')
  initial=[{'input_tokens':10,'output_tokens':2,'call_key':str(i),'cache_hit':False} for i in range(3)]
  updated=[{'input_tokens':20,'output_tokens':4,'call_key':'u'+str(i),'cache_hit':False} for i in range(3)]
  cost=study.routed_cost(initial,updated,[[0,2],[1,2]])
  self.assertEqual(cost['logical_calls'],4);self.assertEqual(cost['input_tokens'],50);self.assertEqual(cost['output_tokens'],10)
 def test_outstar_counts_two_receivers(self):
  import controller_study as study
  self.assertTrue(hasattr(study,'routed_cost'),'Actual recipient call accounting required')
  r={'input_tokens':10,'output_tokens':2,'call_key':'r','cache_hit':False}
  self.assertEqual(study.routed_cost([r]*3,[r]*3,[[0,1],[0,2]])['logical_calls'],5)
 def test_no_comm_counts_initials_only(self):
  import controller_study as study
  self.assertTrue(hasattr(study,'routed_cost'),'Actual recipient call accounting required')
  r={'input_tokens':10,'output_tokens':2,'call_key':'r','cache_hit':False}
  self.assertEqual(study.routed_cost([r]*3,[r]*3,[])['logical_calls'],3)
if __name__=='__main__':unittest.main()
