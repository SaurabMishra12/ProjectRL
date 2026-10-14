"""Provider control-flow fixtures only, never research answers or datasets."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend import RealBackend, provider_refusal
from reporting import physical_usage

class ProviderRefusalTests(unittest.TestCase):
    def instance(self,path,raw=None,error=None,policy='abstain_with_reported_usage'):
        b=RealBackend.__new__(RealBackend)
        b.config={'model':'fixture','temperature':0,'max_tokens':10,'max_calls':2,'context_limit':1000,
                  'provider_refusal_policy':policy}
        b.identity={};b.run_dir=Path(path);b.cache_dir=b.run_dir/'responses';b.cache_dir.mkdir()
        b.ledger=b.run_dir/'calls.jsonl';b.attempts=[]
        b.tokenizer=SimpleNamespace(identity={},count=lambda s:len(s))
        calls=[]
        def create(**kw):
            calls.append(kw)
            if error:raise error
            return SimpleNamespace(model_dump=lambda **_:raw)
        b.client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        return b,calls
    def raw(self):
        return {'model':'fixture','choices':[{'finish_reason':'content_filter','message':{'content':''}}],
                'usage':{'prompt_tokens':7,'completion_tokens':2}}
    def test_output_filter_is_abstention_cached_with_real_usage(self):
        with tempfile.TemporaryDirectory() as p:
            b,calls=self.instance(p,self.raw())
            result=b.complete([],[],{})
            self.assertEqual(result['answer'],'');self.assertTrue(result['provider_refusal'])
            self.assertEqual(result['input_tokens'],7);self.assertEqual(result['output_tokens'],2)
            self.assertEqual(b.complete([],[],{})['answer'],'');self.assertEqual(len(calls),1)
            report=physical_usage(p)
            self.assertEqual(report['provider_refusal_calls'],1)
            self.assertEqual(report['input_tokens'],7)
    def test_http400_filter_body_is_preserved_not_synthesized(self):
        class ProviderError(Exception):
            status_code=400
            body={'error':{'code':'content_filter'},'usage':{'prompt_tokens':5,'completion_tokens':0}}
        with tempfile.TemporaryDirectory() as p:
            b,_=self.instance(p,error=ProviderError())
            result=b.complete([],[],{})
            self.assertTrue(result['provider_refusal']);self.assertEqual(result['output_tokens'],0)
            saved=json.loads(next(b.cache_dir.glob('*.json')).read_text())
            self.assertEqual(saved['response'],ProviderError.body)
            self.assertEqual(saved['provider_http_status'],400)
    def test_unknown_usage_is_not_zero(self):
        with tempfile.TemporaryDirectory() as p:
            raw=self.raw();raw.pop('usage');b,_=self.instance(p,raw)
            result=b.complete([],[],{});self.assertIsNone(result['input_tokens'])
            self.assertFalse(physical_usage(p)['usage_complete_for_successes'])
    def test_refusal_message_is_detected_and_truncation_is_not(self):
        self.assertTrue(provider_refusal({'choices':[{'finish_reason':'stop','message':{'refusal':'fixture refusal'}}]}))
        self.assertFalse(provider_refusal({'choices':[{'finish_reason':'length','message':{'content':''}}]}))
    def test_other_http400_and_strict_policy_still_fail(self):
        class ProviderError(Exception):
            status_code=400
            body={'error':{'code':'bad_request'}}
        with tempfile.TemporaryDirectory() as p:
            b,_=self.instance(p,error=ProviderError())
            with self.assertRaises(RuntimeError):b.complete([],[],{})
        with tempfile.TemporaryDirectory() as p:
            b,_=self.instance(p,self.raw(),policy='fail_closed')
            with self.assertRaises(ValueError):b.complete([],[],{})
    def test_http400_requires_content_filter_not_just_message_refusal(self):
        class ProviderError(Exception):
            status_code=400
            body={'choices':[{'finish_reason':'stop','message':{'refusal':'fixture'}}]}
        with tempfile.TemporaryDirectory() as p:
            b,_=self.instance(p,error=ProviderError())
            with self.assertRaises(RuntimeError):b.complete([],[],{})

if __name__=='__main__':unittest.main()
