"""Isolated fake transport tests; fixture outputs never enter experiment artifacts."""
import subprocess
import sys
import unittest
from pathlib import Path

class TransportRecoveryTests(unittest.TestCase):
    def test_committed_body_recovers_without_a_second_transport_call(self):
        script='''
import sys,tempfile,json,os
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,sys.argv[1])
import openai
from backend import RealBackend
from analysis.durable_runtime import Runtime,install
calls=[]
body={'model':'software-fixture','choices':[{'finish_reason':'stop','message':{'content':'{"answer":"software fixture","citations":[]}'}}],
      'usage':{'prompt_tokens':1,'completion_tokens':1}}
def create(**params):
    calls.append(params)
    return SimpleNamespace(http_response=SimpleNamespace(content=json.dumps(body).encode(),headers={},status_code=200),
                           parse=lambda:SimpleNamespace(model_dump=lambda **kw:body))
def client(**params):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw:None,
                    with_raw_response=SimpleNamespace(create=create))))
openai.OpenAI=client;os.environ['OPENAI_API_KEY']='software-fixture-placeholder'
cfg={'provider':'openai_compatible','model':'software-fixture','model_revision':'fixture',
     'base_url':'http://software-fixture.invalid','api_version':'fixture','replicate_id':'fixture',
     'max_calls':5,'max_tokens':2,'context_limit':1000,'temperature':0}
tokenizer=SimpleNamespace(identity={},count=lambda value:len(value))
with tempfile.TemporaryDirectory() as folder:
    path=Path(folder);runtime=Runtime(path,{'attempts':[]},'controller');install(runtime)
    backend=RealBackend(cfg,tokenizer,path)
    first=backend.complete([],[],{'stage':'software-fixture'})
    assert not first['cache_hit'];assert len(calls)==1
    staged=next((path/'network_attempts').glob('*.json'))
    canonical=next((path/'responses').glob('*.json'));canonical.unlink()
    ledger=path/'calls.jsonl';rows=[json.loads(l) for l in ledger.read_text().splitlines()]
    start=next(r for r in rows if r['event']=='started')
    ledger.write_text(json.dumps(start)+'\\n')
    # Simulate interruption after receipt/body fsync but before parsed-stage/cache.
    (path/'network_attempts'/(start['attempt_id']+'.json')).unlink()
    restored=RealBackend(cfg,tokenizer,path)
    second=restored.complete([],[],{'stage':'software-fixture'})
    assert second['cache_hit'];assert len(calls)==1
    assert next((path/'network_attempts').glob('*.body')).read_bytes()==json.dumps(body).encode()
    # A mismatched receipt must halt instead of fabricating recovery or calling again.
    next((path/'responses').glob('*.json')).unlink();ledger.write_text(json.dumps(start)+'\\n')
    stage=path/'network_attempts'/(start['attempt_id']+'.json')
    bad=json.loads(stage.read_text());bad['request_sha256']='wrong';stage.write_text(json.dumps(bad))
    try:RealBackend(cfg,tokenizer,path)
    except ValueError:pass
    else:raise AssertionError('Mismatched recovery should fail')
    assert len(calls)==1
    # A saved timeout diagnostic has no response body to recover. Without an
    # exact authorization it must reach the uncertainty guard, not be parsed
    # as a corrupt completion; with authorization one NEW attempt is permitted.
    from protocol import digest
    from backend import unresolved_attempts
    from analysis.durable_storage import original_unresolved_attempts
    bad.update(request_sha256=start['request_sha256'],http_status=None,
               provider_response=None,provider_response_sha256=digest(None))
    stage.write_text(json.dumps(bad))
    try:RealBackend(cfg,tokenizer,path)
    except RuntimeError:pass
    else:raise AssertionError('Unapproved timeout must remain blocked')
    assert len(calls)==1
    runtime.allowed[start['attempt_id']]=start
    resumed=RealBackend(cfg,tokenizer,path)
    replacement=resumed.complete([],[],{'stage':'software-fixture'})
    assert len(calls)==2 and not replacement['cache_hit']
    all_rows=[json.loads(l) for l in ledger.read_text().splitlines()]
    assert len(original_unresolved_attempts(all_rows))==1
    assert not unresolved_attempts(all_rows)
'''
        result=subprocess.run([sys.executable,'-c',script,str(Path(__file__).resolve().parents[1])],
                              capture_output=True,text=True,timeout=120)
        self.assertEqual(result.returncode,0,result.stderr)

if __name__=='__main__':unittest.main()
