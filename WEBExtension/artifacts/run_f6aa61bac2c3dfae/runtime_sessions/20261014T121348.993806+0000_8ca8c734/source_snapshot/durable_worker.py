"""Run the existing frozen pipeline with local durability, telemetry and scoped recovery."""
import argparse
import hashlib
import json
import os
import runpy
import signal
import sys
import threading
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from analysis.durable_storage import durable_json,durable_bytes,original_unresolved_attempts
from analysis.durable_runtime import Runtime,install,utc

def main():
    parser=argparse.ArgumentParser();parser.add_argument('pipeline',choices=['controller','controlled'])
    parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    active=json.loads((ROOT/'artifacts/active_machine_runs.json').read_text())
    directory=Path(active['controller_study' if args.pipeline=='controller' else 'controlled_run'])
    auth_path=Path(active['retry_authorization_file'])
    if hashlib.sha256(auth_path.read_bytes()).hexdigest()!=active['retry_authorization_sha256']:
        raise ValueError('Retry authorization changed')
    authorization=json.loads(auth_path.read_text())
    core=json.loads((directory/('manifest.json' if args.pipeline=='controller' else 'frozen_manifest.json')).read_text())
    plan=core if args.pipeline=='controller' else core['plan']
    from protocol import file_hash
    for name,sha in plan['sources'].items():
        if file_hash(ROOT/name)!=sha:raise ValueError('Frozen scientific source changed: '+name)
    cfg=json.loads((ROOT/'config.json').read_text())
    if cfg!=plan['base' if args.pipeline=='controller' else 'config']:raise ValueError('Frozen configuration changed')
    if args.pipeline=='controller' and json.loads((ROOT/'controller_config.json').read_text())!=plan['settings']:
        raise ValueError('Frozen controller settings changed')
    selected=[r for r in authorization['attempts'] if r['pipeline']==args.pipeline]
    if len({r['attempt_id'] for r in selected})!=len(selected):
        raise ValueError('Duplicate authorized attempt')
    expected_run=directory/'inference' if args.pipeline=='controller' else directory
    if any(Path(r['run_dir']).resolve()!=expected_run.resolve() for r in selected):
        raise ValueError('Retry authorization belongs to another run')
    for entry in selected:
        ledger=Path(entry['run_dir'])/'calls.jsonl'
        records=[json.loads(l) for l in ledger.read_text().splitlines()]
        old=next(r for r in records if r.get('attempt_id')==entry['attempt_id'] and r['event']=='started')
        if any(old[k]!=entry[k] for k in ['key','request_sha256','utc']):raise ValueError('Authorization scope changed')
        if file_hash(old['request_file'])!=entry['request_sha256']:raise ValueError('Original request changed')
        if not any(r.get('event')=='uncertain_retry_authorized' and r.get('attempt_id')==entry['attempt_id'] for r in records):
            line=json.dumps({'event':'uncertain_retry_authorized','utc':utc(),'attempt_id':entry['attempt_id'],
                'key':entry['key'],'authorization_file':str(auth_path),'authorization_sha256':file_hash(auth_path),
                'original_outcome':'UNKNOWN_POTENTIALLY_BILLED','original_usage':None})+'\n'
            with ledger.open('a') as f:f.write(line);f.flush();os.fsync(f.fileno())
    if args.verify_only:
        print('VERIFIED: frozen sources/configuration and '+str(len(selected))+' exact authorized retries; no inference')
        return
    runtime=Runtime(directory,authorization,args.pipeline)
    for name in ['analysis/durable_storage.py','analysis/durable_runtime.py',
                 'analysis/durable_worker.py','analysis/rate_limited_machine.py','analysis/adaptive_pacing.py']:
        source=ROOT/name
        durable_bytes(runtime.session/'source_snapshot'/source.name,source.read_bytes())
    durable_json(runtime.session/'operational_manifest.json',{'scientific_manifest_sha256':file_hash(
        directory/('manifest.json' if args.pipeline=='controller' else 'frozen_manifest.json')),
        'operational_sources':{name:file_hash(ROOT/name) for name in ['analysis/durable_storage.py',
             'analysis/durable_runtime.py','analysis/durable_worker.py','analysis/rate_limited_machine.py','analysis/adaptive_pacing.py']},
        'recovery_authorization_sha256':file_hash(auth_path),'scientific_requests_and_optimizer_math_changed':False})
    def requested_stop(signum,frame):
        runtime.stop_requested=True;runtime.event('stop_requested',signal=signum)
    signal.signal(signal.SIGTERM,requested_stop);signal.signal(signal.SIGINT,requested_stop)
    install(runtime)
    thread=threading.Thread(target=runtime.monitor,daemon=True);thread.start()
    status='FAILED'
    try:
        sys.argv=['rate_limited_machine.py',args.pipeline]
        runpy.run_path(str(ROOT/'analysis/rate_limited_machine.py'),run_name='__main__')
        status='COMPLETE'
    except SystemExit as error:
        status='COMPLETE' if error.code in [None,0] else 'FAILED'
        raise
    except BaseException as error:
        runtime.event('worker_error',error_type=type(error).__name__)
        print('Worker stopped:',type(error).__name__,flush=True)
        raise SystemExit(1)
    finally:
        runtime.finished.set()
        thread.join()
        runtime.close('STOPPED' if runtime.stop_requested else status)

if __name__=='__main__':main()
