import ast
import tempfile
import unittest
from pathlib import Path

class ResumePhaseTests(unittest.TestCase):
    def test_frozen_evaluation_resumes_without_training(self):
        source=Path(__file__).resolve().parents[1]/'analysis/rate_limited_machine.py'
        node=next(n for n in ast.parse(source.read_text()).body
                  if isinstance(n,ast.FunctionDef) and n.name=='controller_resume_phase')
        ns={};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            self.assertEqual(ns['controller_resume_phase'](folder),'all')
            (folder/'evaluation_gate.json').write_text('{}')
            self.assertEqual(ns['controller_resume_phase'](folder),'evaluate')
