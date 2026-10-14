import sys, unittest, types
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from analysis.durable_runtime import rebind_json_writers
class RebindingTests(unittest.TestCase):
    def test_does_not_trigger_lazy_module_imports(self):
        lazy=types.ModuleType('lazy');calls=[]
        def getattr_(name):
            calls.append(name);raise RuntimeError('Lazy import must not run')
        lazy.__getattr__=getattr_
        regular=types.ModuleType('regular');old=lambda:None;new=lambda:None
        regular.write_json=old
        rebind_json_writers([None,lazy,regular],old,new)
        self.assertEqual(calls,[]);self.assertIs(regular.write_json,new)
if __name__=='__main__':unittest.main()
