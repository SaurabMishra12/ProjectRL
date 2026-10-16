"""Execute the user-authorized pilot with an exact permit and visible progress."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
from WEBExtension.web_reasoning import live
from WEBExtension.web_reasoning.preparation import verify_run,make_tokenizer,file_sha
from WEBExtension.web_reasoning.config import validate_config
from WEBExtension.web_reasoning.storage import Journal,atomic_json,digest
from WEBExtension.web_reasoning.backend import DurableBackend
from WEBExtension.web_reasoning.serialization import public_from_dict
from WEBExtension.web_reasoning.runner import initialize
from WEBExtension.web_reasoning.workflow import run_manifest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--permit',required=True);parser.add_argument('--initial-only',action='store_true');args=parser.parse_args()
    path=Path(args.manifest);root=path.parent;manifest=json.loads(path.read_text());config=validate_config(manifest['config'])
    permit=json.loads(Path(args.permit).read_text());live.require_live_permit(file_sha(path),permit,config)
    verify_run(root)
    from dotenv import dotenv_values
    key_name=manifest['sources']['provider']['api_key_env']
    if not os.environ.get(key_name):
        key=dotenv_values('.env').get(key_name)
        if key:os.environ[key_name]=key
    original=live.live_transport;events=Journal(root/'transport_observations.jsonl')
    def factory(descriptor):
        send=original(descriptor);next_call=0.
        def observed(request,attempt_id):
            nonlocal next_call
            while time.monotonic()<next_call:time.sleep(min(30.,next_call-time.monotonic()))
            started=time.perf_counter();status,headers,raw=send(request,attempt_id)
            finish=(raw.get('choices') or [{}])[0].get('finish_reason') if isinstance(raw,dict) else None
            usage=raw.get('usage') if isinstance(raw,dict) else None
            event={'attempt_id':attempt_id,'http_status':status,'finish_reason':finish,'usage':usage,'latency_seconds':time.perf_counter()-started,'reported_model':raw.get('model') if isinstance(raw,dict) else None}
            events.append(event);print(json.dumps({'PROVIDER_RESPONSE':event}),flush=True)
            delay=2.
            if status==429:
                try:delay=max(15.,float(next((v for k,v in headers.items() if k.lower()=='retry-after'),15)))
                except (ValueError,TypeError):delay=15.
            next_call=time.monotonic()+delay
            return status,headers,raw
        return observed
    live.live_transport=factory
    if args.initial_only:
        tokenizer=make_tokenizer(config,manifest['sources']);backend=DurableBackend(root/'inference',config,factory(manifest['sources']),tokenizer)
        public=public_from_dict(json.loads((root/'pilot_inputs.json').read_text())[0]);lookup={a.atom_id:a for s in public.sources for a in s.atoms}
        shards=json.loads((root/'pilot_N3_assignments.json').read_text())[public.key]
        initial=initialize(public,tuple(tuple(lookup[k] for k in shard) for shard in shards),backend,config)
        snapshot={'question_key':public.key,'initial_state_hash':digest(initial),'outcomes':[{'kind':s.outcome.kind,'reason':s.outcome.reason} for s in initial],'usage':backend.summary()}
        atomic_json(root/'initial_preflight.json',snapshot);print(json.dumps({'INITIAL_PREFLIGHT':snapshot}),flush=True)
    else:
        runs=run_manifest(path,'live','pilot',args.permit)
        print(json.dumps({'COMPLETE_QUESTION_RECORDS':len(runs),'mode':'REAL'}),flush=True)


if __name__=='__main__':main()
