"""Test operational retry functions without executing the CLI or making calls."""
import ast
import math
import time
from email.utils import formatdate, parsedate_to_datetime
import unittest
from pathlib import Path
from types import SimpleNamespace

SOURCE=Path(__file__).resolve().parents[1]/'analysis/rate_limited_machine.py'
def functions(namespace):
    tree=ast.parse(SOURCE.read_text())
    module=ast.Module(body=[node for node in tree.body if isinstance(node,ast.FunctionDef)
                           and node.name in ['guidance','paced_complete']],type_ignores=[])
    exec(compile(module,str(SOURCE),'exec'),namespace)
    return namespace

class RatePacingTests(unittest.TestCase):
    def error(self,status=429,body=None,headers=None):
        error=Exception('control-flow fixture')
        error.status_code=status;error.body=body;error.response=SimpleNamespace(headers=headers or {})
        return error
    def test_numeric_retry_after_seconds_and_milliseconds(self):
        ns=functions({'math':math,'time':time,'parsedate_to_datetime':parsedate_to_datetime})
        self.assertEqual(ns['guidance'](self.error(headers={'retry-after':'30'}))[0],30)
        self.assertEqual(ns['guidance'](self.error(headers={'retry-after-ms':'1250'}))[0],1.25)
        self.assertEqual(ns['guidance'](self.error())[0],60)
    def test_http_date_retry_after_is_honored(self):
        ns=functions({'math':math,'time':time,'parsedate_to_datetime':parsedate_to_datetime})
        target=time.time()+300
        delay=ns['guidance'](self.error(headers={'retry-after':formatdate(target,usegmt=True)}))[0]
        self.assertGreater(delay,298);self.assertLessEqual(delay,300)
    def retry(self,error,failures=1):
        counts=[]
        def original(*args,**kwargs):
            counts.append(1)
            if len(counts)<=failures:
                raise RuntimeError('backend rejected') from error
            return 'terminal fixture marker'
        ns=functions({'math':math,'original_complete':original,
                      'operational':{'maximum_429_retries_per_completion':4}})
        return ns['paced_complete'],counts
    def test_rejected_429_can_resume_same_arguments(self):
        run,counts=self.retry(self.error(body={'error':{'code':'rate_limit'}}))
        self.assertEqual(run(object(),[],[],{}),'terminal fixture marker');self.assertEqual(len(counts),2)
    def test_other_error_or_completion_body_is_never_retried(self):
        for error in [self.error(status=400),self.error(status=None),
                      self.error(body={'choices':[{'message':{'content':'fixture'}}]})]:
            run,counts=self.retry(error)
            with self.assertRaises(RuntimeError):run(object(),[],[],{})
            self.assertEqual(len(counts),1)
    def test_retry_bound(self):
        run,counts=self.retry(self.error(),failures=10)
        with self.assertRaises(RuntimeError):run(object(),[],[],{})
        self.assertEqual(len(counts),5)

if __name__=='__main__':unittest.main()
