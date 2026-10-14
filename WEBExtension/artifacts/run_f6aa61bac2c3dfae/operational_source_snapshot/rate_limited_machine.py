"""Operational Azure pacing only: identical requests, caches and scientific plans.

Two machine pipelines share one request lock/cooldown. Only explicit HTTP 429
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
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from filelock import FileLock
from backend import RealBackend
from protocol import ROOT, digest, file_hash, write_json
from experiments import load_config, freeze
from controller_study import study_directory

parser=argparse.ArgumentParser();parser.add_argument('pipeline',choices=['controller','controlled'])
args=parser.parse_args();cfg=load_config()
settings=json.loads((ROOT/'controller_config.json').read_text())
directory=(study_directory(cfg,settings)[0] if args.pipeline=='controller' else freeze(cfg)[0])
operational={'script_sha256':file_hash(Path(__file__)),'initial_minimum_interval_seconds':4,
             'maximum_429_retries_per_completion':4,'shared_request_serialization':True,
             'scientific_requests_changed':False,'scope':'Only explicit ungenerated HTTP429 responses; honor Retry-After',
             'costs':'Every rejected/retried attempt remains in calls.jsonl; unreported usage remains unknown'}
write_json(directory/'operational_rate_policy.json',operational)
rate_key=digest([cfg['base_url'],cfg['model']])[:16]
state_path=ROOT/'artifacts'/('azure_rate_'+rate_key+'.json')
lock=FileLock(str(state_path)+'.lock',timeout=7200)

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
        except (ValueError,TypeError):pass
    return delay,selected

original_init=RealBackend.__init__
original_complete=RealBackend.complete

def paced_init(self,*pos,**kw):
    original_init(self,*pos,**kw)
    create=self.client.chat.completions.create
    def paced_create(**params):
        with lock:
            state=(json.loads(state_path.read_text()) if state_path.exists()
                   else {'interval_seconds':4.,'next_allowed':0.})
            while time.time()<state['next_allowed']:
                time.sleep(min(30.,state['next_allowed']-time.time()))
            try:
                return create(**params)
            except Exception as error:
                if getattr(error,'status_code',None)==429:
                    delay,headers=guidance(error)
                    state['interval_seconds']=min(60.,max(4.,state['interval_seconds']*1.5))
                    state['next_allowed']=time.time()+delay
                    write_json(self.run_dir/'rate_limit_diagnostic.json',{
                        'http_status':429,'retry_after_seconds':delay,'selected_headers':headers,
                        'provider_error_body':sanitized_body(getattr(error,'body',None)),
                        'next_interval_seconds':state['interval_seconds']})
                    print('Azure429: shared cooldown',round(delay,2),'seconds; interval',state['interval_seconds'],flush=True)
                raise
            finally:
                state['next_allowed']=max(state['next_allowed'],time.time()+state['interval_seconds'])
                write_json(state_path,state)
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
