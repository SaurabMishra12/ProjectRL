"""Operational Azure pacing only: identical requests, caches and scientific plans.

Two machine pipelines share adaptive start pacing and at most two network slots. Only explicit HTTP 429
rejections without a completion body can be retried; every attempt is recorded
by the original RealBackend. No timeout/filter/malformed result is retried.
"""
import argparse
import json
import math
import os
import runpy
import sys
import time
from email.utils import parsedate_to_datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from filelock import FileLock
from backend import RealBackend
from protocol import ROOT, digest, file_hash, write_json
from experiments import load_config, freeze
from controller_study import study_directory
from analysis.adaptive_pacing import AdaptivePacer

parser=argparse.ArgumentParser();parser.add_argument('pipeline',choices=['controller','controlled'])
args=parser.parse_args();cfg=load_config()
settings=json.loads((ROOT/'controller_config.json').read_text())
directory=(study_directory(cfg,settings)[0] if args.pipeline=='controller' else freeze(cfg)[0])
operational={'script_sha256':file_hash(Path(__file__)),
             'dispatcher_sha256':file_hash(ROOT/'analysis/adaptive_pacing.py'),
             'initial_start_interval_seconds':2,'minimum_start_interval_seconds':1,
             'maximum_429_retries_per_completion':4,'shared_request_serialization':False,
             'maximum_simultaneous_network_requests':2,'within_pipeline_calls_remain_sequential':True,
             'scientific_requests_changed':False,'scope':'Only explicit ungenerated HTTP429 responses; honor Retry-After',
             'costs':'Every rejected/retried attempt remains in calls.jsonl; unreported usage remains unknown'}
write_json(directory/'operational_rate_policy.json',operational)
rate_key=digest([cfg['base_url'],cfg['model']])[:16]
state_path=ROOT/'artifacts'/('azure_rate_'+rate_key+'.json')
pacer=AdaptivePacer(state_path)

def sanitized_body(body):
    encoded=json.dumps(body)
    for name,value in os.environ.items():
        if any(word in name.upper() for word in ['KEY','TOKEN','SECRET']) and len(value)>8:
            encoded=encoded.replace(value,'[REDACTED]')
    return json.loads(encoded)

def guidance(error):
    response=getattr(error,'response',None)
    headers=getattr(response,'headers',{}) or {}
    selected={k:headers.get(k) for k in ['retry-after','retry-after-ms','x-ratelimit-remaining-requests',
                'x-ratelimit-remaining-tokens','x-ratelimit-reset-requests','x-ratelimit-reset-tokens'] if headers.get(k) is not None}
    delay=60.
    for name,scale in [('retry-after',1),('retry-after-ms',.001)]:
        try:
            value=float(headers.get(name))*scale
            if math.isfinite(value) and value>=0:delay=max(1.,value);break
        except (ValueError,TypeError):
            if name=='retry-after':
                try:
                    delay=max(1.,parsedate_to_datetime(headers.get(name)).timestamp()-time.time())
                    break
                except (ValueError,TypeError,OverflowError):pass
    return delay,selected

original_init=RealBackend.__init__
original_complete=RealBackend.complete

def paced_init(self,*pos,**kw):
    original_init(self,*pos,**kw)
    create=self.client.chat.completions.create
    def paced_create(**params):
        def diagnose(data):
            write_json(self.run_dir/'rate_limit_diagnostic.json',data)
            print('Azure429: shared cooldown',round(data['retry_after_seconds'],2),
                  'seconds; interval',data['next_interval_seconds'],flush=True)
        return pacer.call(create,params,guidance,diagnose)
    self.client.chat.completions.create=paced_create

def paced_complete(self,*pos,**kw):
    for retry in range(operational['maximum_429_retries_per_completion']+1):
        try:return original_complete(self,*pos,**kw)
        except RuntimeError as error:
            cause=error
            while cause.__cause__ is not None:cause=cause.__cause__
            body=getattr(cause,'body',None)
            # A response carrying completion choices is never a request rejection.
            if (getattr(cause,'status_code',None)!=429 or (isinstance(body,dict) and body.get('choices'))
                    or retry==operational['maximum_429_retries_per_completion']):raise
            # Cooldown happens before the next network call under the shared lock.
            # The original backend has already preserved this rejected attempt.
RealBackend.__init__=paced_init
RealBackend.complete=paced_complete
if args.pipeline=='controller':
    sys.argv=['run_controller_machine.py','all']
    runpy.run_path(str(ROOT/'run_controller_machine.py'),run_name='__main__')
else:
    sys.argv=['run_controlled_machine.py']
    runpy.run_path(str(ROOT/'analysis/run_controlled_machine.py'),run_name='__main__')
