"""Software invariants; these do not create or claim research rollouts."""
import importlib.util
import pathlib
import sys
import unittest
import numpy as np
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
class LearnedControllerTests(unittest.TestCase):
 def setUp(self):
  self.assertIsNotNone(importlib.util.find_spec('learned_controller'),'Learned controller implementation required')
  import learned_controller
  self.c=learned_controller
 def test_clipped_positive_advantage_has_zero_actor_gradient(self):
  import torch
  logp=torch.tensor([np.log(1.5)],requires_grad=True)
  loss=self.c.clipped_actor_loss(logp,torch.tensor([0.]),torch.tensor([1.]),.2)
  loss.backward();self.assertAlmostEqual(float(logp.grad),0.)
 def test_actions_have_no_self_edges_and_true_stop_has_zero_quota(self):
  actions=self.c.action_space([64,128,256])
  self.assertEqual(actions[0],{'edges':[],'quota':0,'name':'NoComm'})
  self.assertEqual(len(actions),55)
  for action in actions:
   self.assertEqual(len(action['edges']),len(set(map(tuple,action['edges']))))
   self.assertTrue(all(s!=t and 0<=s<3 and 0<=t<3 for s,t in action['edges']))
 def test_inference_features_do_not_accept_gold(self):
  import inspect
  self.assertEqual(list(inspect.signature(self.c.observable_features).parameters),['question','state','tokenizer'])
 def test_ppo_preserves_finite_parameters(self):
  import torch
  model=self.c.CategoricalController(4,3,hidden=8)
  states=torch.zeros((8,4));dist,value=model(states)
  actions=dist.sample();old=dist.log_prob(actions).detach()
  data={'features':states,'actions':actions,'old_logp':old,'rewards':torch.linspace(-.2,.2,8),'old_values':value.detach()}
  stats=self.c.ppo_update(model,torch.optim.Adam(model.parameters(),lr=1e-3),data,epochs=2,batch_size=4)
  self.assertTrue(np.isfinite(stats['loss']))
  self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
if __name__=='__main__':unittest.main()
