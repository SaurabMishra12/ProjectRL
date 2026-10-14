"""Local persistence/timing hooks. Frozen model requests and PPO math are unchanged."""
import datetime
import hashlib
import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from analysis.durable_storage import (durable_json,durable_bytes,archive_file,
                                      fsync_directory,Journal,blocking_attempts,original_unresolved_attempts)

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()

class StopRequested(RuntimeError):pass

class Runtime:
    def __init__(self,directory,authorization,pipeline):
        self.directory=Path(directory);self.pipeline=pipeline
        self.session_id=utc().replace(':','').replace('-','')+'_'+uuid.uuid4().hex[:8]
        self.session=self.directory/'runtime_sessions'/self.session_id;self.session.mkdir(parents=True)
        self.events=Journal(self.session/'events.jsonl');self.timings=Journal(self.session/'timings.jsonl')
        self.outputs=Journal(self.session/'parsed_call_outputs.jsonl')
        self.start_perf=time.perf_counter();self.start_cpu=time.process_time();self.started_utc=utc()
        self.stop_requested=False;self.finished=threading.Event();self.phase='startup';self.seed=None
        self.allowed={r['attempt_id']:r for r in authorization['attempts'] if r['pipeline']==pipeline}
        self.auth=authorization
        durable_json(self.session/'session_start.json',{'utc':self.started_utc,'pid':os.getpid(),
            'pipeline':pipeline,'old_attempts_allowed_for_new_retry':list(self.allowed),
            'old_attempt_costs_remain_unknown':True,'prior_unobserved_timings_not_invented':True})
    def event(self,kind,**fields):self.events.append({'utc':utc(),'event':kind,**fields})
    def timing(self,kind,start,cpu,**fields):
        self.timings.append({'utc_end':utc(),'kind':kind,'wall_seconds':time.perf_counter()-start,
                            'client_process_cpu_seconds':time.process_time()-cpu,**fields})
    def heartbeat(self):
        counts={}
        for path in self.directory.glob('seed_*/*.jsonl'):
            counts[str(path.relative_to(self.directory))]=sum(1 for l in path.open() if l.endswith('\n'))
        for name in ['E1','E2','E3','E4','E5']:
            path=self.directory/(name+'.jsonl')
            if path.exists():counts[name]=sum(1 for l in path.open() if l.endswith('\n'))
        totals={'attempts':0,'returned_responses':0,'reported_input_tokens':0,'reported_output_tokens':0,
                'attempts_with_unknown_usage':0,'provider_refusals':0,'unresolved_original_liabilities':0}
        for path in [self.directory/'calls.jsonl',self.directory/'inference/calls.jsonl',
                     self.directory/'pilot/inference/calls.jsonl']:
            if not path.exists():continue
            records=[json.loads(l) for l in path.read_text().splitlines() if l]
            started={r['attempt_id'] for r in records if r['event']=='started'}
            successes=[r for r in records if r['event']=='success']
            known=[r for r in successes if r.get('usage') and r['usage'].get('prompt_tokens') is not None
                   and r['usage'].get('completion_tokens') is not None]
            totals['attempts']+=len(started);totals['returned_responses']+=len(successes)
            totals['reported_input_tokens']+=sum(r['usage']['prompt_tokens'] for r in known)
            totals['reported_output_tokens']+=sum(r['usage']['completion_tokens'] for r in known)
            totals['attempts_with_unknown_usage']+=len(started)-len(known)
            totals['provider_refusals']+=sum(bool(r.get('provider_refusal')) for r in successes)
            totals['unresolved_original_liabilities']+=len(original_unresolved_attempts(records))
        durable_json(self.directory/'runtime_progress.json',{'heartbeat_utc':utc(),'pid':os.getpid(),
            'state':'STOP_REQUESTED' if self.stop_requested else 'RUNNING','phase':self.phase,'seed':self.seed,
            'session':str(self.session),'session_wall_seconds':time.perf_counter()-self.start_perf,
            'session_client_cpu_seconds':time.process_time()-self.start_cpu,'counts':counts,'physical_usage':totals,
            'timing_scope':'New local session and inclusive stages; server compute timing is not exposed'})
    def monitor(self):
        while not self.finished.is_set():
            try:self.heartbeat()
            except Exception as error:self.event('heartbeat_error',error_type=type(error).__name__)
            self.finished.wait(30)
    def close(self,status):
        self.finished.set()
        self.timing('session',self.start_perf,self.start_cpu,status=status)
        durable_json(self.session/'session_end.json',{'utc':utc(),'status':status,'wall_seconds':time.perf_counter()-self.start_perf})
        durable_json(self.directory/'runtime_progress.json',{'heartbeat_utc':utc(),'pid':os.getpid(),
            'state':status,'session':str(self.session),'session_wall_seconds':time.perf_counter()-self.start_perf})

def install(runtime):
    import protocol,backend
    root=protocol.ROOT
    old_write=protocol.write_json
    def tracked_json(path,value):
        if Path(path).exists():
            previous=archive_file(path,runtime.session/'previous_json_versions')
            runtime.event('prior_json_version_archived',**previous)
        receipt=durable_json(path,value);runtime.event('json_committed',**receipt)
        return receipt
    protocol.write_json=tracked_json;backend.write_json=tracked_json
    import controller_study as study
    import experiments
    # Catch imports that obtained the former write_json function before install.
    import sys
    for module in list(sys.modules.values()):
        if module is not None and getattr(module,'write_json',None) is old_write:module.write_json=tracked_json
    backend.unresolved_attempts=lambda records:blocking_attempts(records,runtime.allowed)
    old_init=backend.RealBackend.__init__;old_complete=backend.RealBackend.complete
    def persisted_init(self,*args,**kwargs):
        run_dir=Path(args[2] if len(args)>2 else kwargs['run_dir'])
        # Recover a fully committed transport response if interruption occurred
        # before the original backend could commit its canonical response cache.
        ledger=run_dir/'calls.jsonl'
        if ledger.exists():
            rows=[json.loads(l) for l in ledger.read_text().splitlines() if l]
            for pending in original_unresolved_attempts(rows):
                cache=run_dir/'responses'/(pending['key']+'.json')
                staged=run_dir/'network_attempts'/(pending['attempt_id']+'.json')
                receipt=run_dir/'network_attempts'/(pending['attempt_id']+'.receipt.json')
                body_path=run_dir/'network_attempts'/(pending['attempt_id']+'.body')
                if cache.exists():continue
                if not staged.exists() and receipt.exists() and body_path.exists():
                    s=json.loads(receipt.read_text())
                    if protocol.file_hash(body_path)!=s.get('body_sha256'):raise ValueError('Transport body changed')
                    body=json.loads(body_path.read_bytes())
                    s.update(provider_response=body,provider_response_sha256=protocol.digest(body))
                    tracked_json(staged,s)
                if not staged.exists():continue
                s=json.loads(staged.read_text());body=s.get('provider_response')
                if s.get('request_sha256')!=pending['request_sha256'] or s.get('key')!=pending['key']:
                    raise ValueError('Staged response does not match original request')
                if not isinstance(body,dict) or protocol.digest(body)!=s.get('provider_response_sha256'):
                    raise ValueError('Staged provider response checksum mismatch')
                if s['http_status']!=200 and not(s['http_status']==400 and backend.provider_refusal(body)=='content_filter'):
                    continue
                request=json.loads(Path(pending['request_file']).read_text())
                if protocol.file_hash(pending['request_file'])!=pending['request_sha256']:raise ValueError('Request changed')
                saved={'key':pending['key'],'attempt_id':pending['attempt_id'],'request':request['request'],
                       'identity':request['identity'],'tokenizer':request['tokenizer'],'response':body,
                       'provider_http_status':s['http_status'],'latency_seconds':s['end_to_end_wall_seconds'],
                       'metadata':request['metadata'],'recovered_from_transport_stage':str(staged)}
                saved['content_hash']=protocol.digest(saved);tracked_json(cache,saved)
                runtime.event('transport_response_recovered',attempt_id=pending['attempt_id'])
        old_init(self,*args,**kwargs)
        raw_create=self.client.chat.completions.with_raw_response.create
        def capture(**params):
            pending=next(r for r in reversed(self.attempts) if r['event']=='started')
            start=time.perf_counter();cpu=time.process_time();started=utc()
            base=run_dir/'network_attempts'/pending['attempt_id']
            tracked_json(str(base)+'.dispatch.json',{'utc':started,'attempt_id':pending['attempt_id'],
                'key':pending['key'],'request_sha256':pending['request_sha256']})
            try:
                response=raw_create(**params)
                transport=response.http_response
                tracked_json(str(base)+'.receipt.json',{'attempt_id':pending['attempt_id'],'key':pending['key'],
                    'request_sha256':pending['request_sha256'],'http_status':transport.status_code,
                    'body_sha256':hashlib.sha256(transport.content).hexdigest(),
                    'end_to_end_wall_seconds':time.perf_counter()-getattr(self,'_local_call_start',start),
                    'utc_received':utc()})
                # Commit the exact body before parsing it or returning to the
                # original backend. No request/authentication headers are saved.
                durable_bytes(str(base)+'.body',transport.content)
                parsed=response.parse();body=parsed.model_dump(mode='json')
                headers={k:transport.headers.get(k) for k in ['x-request-id','apim-request-id',
                         'content-type','retry-after','retry-after-ms'] if transport.headers.get(k) is not None}
                status=transport.status_code
            except Exception as error:
                transport=getattr(error,'response',None)
                body=getattr(error,'body',None);status=getattr(error,'status_code',None)
                headers={k:transport.headers.get(k) for k in ['x-request-id','apim-request-id','retry-after',
                    'retry-after-ms'] if transport is not None and transport.headers.get(k) is not None}
                if transport is not None:durable_bytes(str(base)+'.body',transport.content)
                tracked_json(str(base)+'.json',{'attempt_id':pending['attempt_id'],'key':pending['key'],
                    'request_sha256':pending['request_sha256'],'utc_dispatch':started,'utc_received':utc(),
                    'http_status':status,'selected_response_headers':headers,'provider_response':body,
                    'provider_response_sha256':protocol.digest(body),'network_wall_seconds':time.perf_counter()-start,
                    'end_to_end_wall_seconds':time.perf_counter()-getattr(self,'_local_call_start',start),
                    'error_type':type(error).__name__})
                runtime.timing('network_error',start,cpu,attempt_id=pending['attempt_id'],http_status=status)
                raise
            tracked_json(str(base)+'.json',{'attempt_id':pending['attempt_id'],'key':pending['key'],
                'request_sha256':pending['request_sha256'],'utc_dispatch':started,'utc_received':utc(),
                'http_status':status,'selected_response_headers':headers,'provider_response':body,
                'provider_response_sha256':protocol.digest(body),'network_wall_seconds':time.perf_counter()-start,
                'end_to_end_wall_seconds':time.perf_counter()-getattr(self,'_local_call_start',start)})
            runtime.timing('network_response',start,cpu,attempt_id=pending['attempt_id'],http_status=status)
            return parsed
        self.client.chat.completions.create=capture
    def persisted_complete(self,*args,**kwargs):
        if runtime.stop_requested:raise StopRequested('Stop requested before any new attempt')
        if shutil.disk_usage(root).free<512*1024*1024:raise RuntimeError('Low disk space; no new provider attempt issued')
        start=time.perf_counter();cpu=time.process_time();self._local_call_start=start
        try:
            result=old_complete(self,*args,**kwargs)
            runtime.outputs.append({'utc':utc(),'metadata':args[2] if len(args)>2 else kwargs.get('metadata'),
                                    'result':result,'wall_seconds':time.perf_counter()-start})
            return result
        finally:runtime.timing('backend_complete_including_queue_and_cache',start,cpu)
    backend.RealBackend.__init__=persisted_init;backend.RealBackend.complete=persisted_complete
    def wrap(module,name):
        original=getattr(module,name)
        def timed(*args,**kwargs):
            start=time.perf_counter();cpu=time.process_time();previous=runtime.phase;runtime.phase=name
            try:return original(*args,**kwargs)
            finally:
                runtime.timing(name,start,cpu);runtime.phase=previous
        setattr(module,name,timed)
    for module,names in [(experiments,['initialize','update','run_question']),
                         (study,['policy_result','validate_policy','select_static','train','evaluate','pilot'])]:
        for name in names:wrap(module,name)
    old_seed=study.train_seed
    def seeded(seed,*args,**kwargs):
        runtime.seed=seed
        try:return old_seed(seed,*args,**kwargs)
        finally:runtime.seed=None
    study.train_seed=seeded
    old_checkpoint=study.save_checkpoint
    def checkpoint(path,*args,**kwargs):
        start=time.perf_counter();cpu=time.process_time();old_checkpoint(path,*args,**kwargs)
        with Path(path).open('rb') as f:os.fsync(f.fileno())
        fsync_directory(Path(path).parent)
        receipt=archive_file(path,Path(path).parent/'checkpoint_history')
        runtime.event('checkpoint_archived',**receipt);runtime.timing('checkpoint_save',start,cpu)
    study.save_checkpoint=checkpoint
    old_copy=shutil.copyfile
    def committed_copy(src,dst,*args,**kwargs):
        result=old_copy(src,dst,*args,**kwargs)
        with Path(dst).open('rb') as f:os.fsync(f.fileno())
        fsync_directory(Path(dst).parent)
        if Path(dst).suffix=='.pt':
            receipt=archive_file(dst,Path(dst).parent/'checkpoint_history')
            runtime.event('copied_checkpoint_committed',**receipt)
        return result
    shutil.copyfile=committed_copy
    old_ppo=study.ppo_update
    def ppo(model,optimizer,data,*args,**kwargs):
        import torch
        update_dir=runtime.session/'optimizer_updates'/('seed_'+str(runtime.seed)+'_'+uuid.uuid4().hex[:8]);update_dir.mkdir(parents=True)
        def save_obj(path,obj):
            temp=Path(str(path)+'.tmp');torch.save(obj,temp)
            with temp.open('rb') as f:os.fsync(f.fileno())
            os.replace(temp,path);fsync_directory(path.parent)
        save_obj(update_dir/'actual_rollout_batch.pt',data)
        original_step=optimizer.step;count=0;start=time.perf_counter();cpu=time.process_time()
        def step(*pos,**kw):
            nonlocal count
            result=original_step(*pos,**kw);count+=1
            save_obj(update_dir/f'step_{count:04d}.pt',{'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                'torch_rng':torch.get_rng_state(),'optimizer_step_in_update':count,'seed':runtime.seed,
                'gradients_at_step':{name:(p.grad.detach().clone() if p.grad is not None else None)
                                     for name,p in model.named_parameters()},
                'utc':utc(),'wall_seconds_since_update_start':time.perf_counter()-start})
            return result
        optimizer.step=step
        try:return old_ppo(model,optimizer,data,*args,**kwargs)
        finally:
            optimizer.step=original_step
            runtime.timing('ppo_update',start,cpu,optimizer_steps=count,archive=str(update_dir))
    study.ppo_update=ppo
