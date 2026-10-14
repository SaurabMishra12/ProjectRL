"""Verify optimizer persistence wiring in an isolated process, with no inference."""
import subprocess
import sys
import unittest
from pathlib import Path

class OptimizerHookTests(unittest.TestCase):
    def test_install_archives_each_actual_software_step(self):
        script='''
import sys,tempfile
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,sys.argv[1])
import torch,controller_study as study
from analysis.durable_runtime import install
class Optimizer:
    def __init__(self):self.count=0
    def step(self):self.count+=1
    def state_dict(self):return {'fixture_step':self.count}
def update(model,optimizer,data):
    optimizer.step();optimizer.step();return {'fixture_steps':optimizer.count}
study.ppo_update=update
with tempfile.TemporaryDirectory() as folder:
    runtime=SimpleNamespace(session=Path(folder),seed=7,allowed={},
        event=lambda *a,**k:None,timing=lambda *a,**k:None)
    install(runtime)
    model=SimpleNamespace(state_dict=lambda:{'software_fixture_parameter':1},named_parameters=lambda:[])
    result=study.ppo_update(model,Optimizer(),{'software_fixture_batch':1})
    assert result['fixture_steps']==2
    dirs=list((Path(folder)/'optimizer_updates').glob('*'))
    assert len(dirs)==1,'Optimizer persistence hook was not installed'
    assert (dirs[0]/'actual_rollout_batch.pt').exists()
    steps=sorted(dirs[0].glob('step_*.pt'));assert len(steps)==2
    assert torch.load(steps[0],weights_only=True)['optimizer']['fixture_step']==1
    assert torch.load(steps[1],weights_only=True)['optimizer']['fixture_step']==2
'''
        root=str(Path(__file__).resolve().parents[1])
        result=subprocess.run([sys.executable,'-c',script,root],capture_output=True,text=True,timeout=120)
        self.assertEqual(result.returncode,0,result.stderr)

if __name__=='__main__':unittest.main()
