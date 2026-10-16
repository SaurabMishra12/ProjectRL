"""Independent serial attempt ledger. Transport injection does not authorize paid calls."""
from dataclasses import dataclass,asdict
import json
from pathlib import Path
import time
import uuid
from filelock import FileLock
from .config import validate_config
from .contracts import Atom
from .parsing import parse_outcome
from .storage import atomic_json,canonical,digest,Journal
from .accounting import valid_usage,cost_summary


class BudgetExceeded(RuntimeError):pass
class UncertainRequest(RuntimeError):pass
class ProviderRejected(RuntimeError):pass


@dataclass(frozen=True)
class Request:
    messages:list
    model_identity:dict
    tokenizer_identity:dict
    decoding:dict

    @property
    def key(self):return digest(asdict(self))
    def provider_body(self):
        return {'model':self.model_identity['model'],'messages':self.messages,
                'temperature':self.decoding.get('temperature',0),'max_tokens':self.decoding['max_tokens']}


class DurableBackend:
    def __init__(self,directory,config,transport,tokenizer):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.config=validate_config(config);self.transport=transport;self.tokenizer=tokenizer;self.references=[]
        self.lock=FileLock(str(self.directory/'runtime.lock'),timeout=0)
        self.journal=Journal(self.directory/'calls.jsonl')
        self.logical=Journal(self.directory/'logical_refs.jsonl')
        with self.lock:
            path=self.directory/'config.json';value=self.config.to_dict()
            if path.exists() and canonical(json.loads(path.read_text()))!=canonical(value):raise ValueError('immutable_config_changed')
            if not path.exists():atomic_json(path,value)
            self._recover()

    def _read(self):
        try:return self.journal.read()
        except (ValueError,OSError) as e:raise UncertainRequest('damaged_attempt_ledger') from e

    def _checked(self,path):
        try:
            saved=json.loads(path.read_text())
            if saved.get('sha256')!=digest({k:v for k,v in saved.items() if k!='sha256'}):raise ValueError('hash')
            return saved
        except (OSError,ValueError,TypeError) as e:raise UncertainRequest('corrupt_or_missing_saved_response') from e

    def _finish(self,saved):
        status,raw=saved['status'],saved['raw']
        if status in (200,201) or (isinstance(raw,dict) and raw.get('choices')):
            terminal='success'
        elif status==429 and isinstance(raw,dict) and isinstance(raw.get('error'),dict) and not raw.get('choices'):
            terminal='rejected'
        else:raise UncertainRequest('unconfirmed_generation_status')
        rows=self._read()
        if not any(e.get('attempt_id')==saved['attempt_id'] and e['event']==terminal for e in rows):
            self.journal.append({'event':terminal,'attempt_id':saved['attempt_id'],'key':saved['key'],
                                 'usage':raw.get('usage') if isinstance(raw,dict) else None,'status':status})
        if terminal=='success':atomic_json(self.directory/'responses'/(saved['key']+'.json'),saved)
        return terminal

    def _recover(self):
        rows=self._read();closed={e['attempt_id']:e for e in rows if e['event'] in ('success','rejected')}
        for start in [e for e in rows if e['event']=='started']:
            terminal=closed.get(start['attempt_id'])
            if terminal and terminal['event']=='rejected':continue
            cache=self.directory/'responses'/(start['key']+'.json')
            raw=self.directory/'attempts'/(start['attempt_id']+'.json')
            if terminal and cache.exists():
                saved=self._checked(cache)
            elif raw.exists():saved=self._checked(raw)
            else:raise UncertainRequest('unresolved_execution_blocks_restart')
            if saved['key']!=start['key'] or saved['attempt_id']!=start['attempt_id'] or saved['request_sha256']!=start['request_sha256']:
                raise UncertainRequest('saved_response_identity_mismatch')
            self._finish(saved)

    def _reserve(self,request):
        rows=self._read();starts={e['attempt_id']:e for e in rows if e['event']=='started'}
        terminal={e['attempt_id']:e for e in rows if e['event'] in ('success','rejected')}
        reserve_in=self.tokenizer.count(canonical(request.messages))+self.config.framing_allowance
        reserve_out=request.decoding['max_tokens']
        if type(reserve_out) is not int or not 0<reserve_out<=self.config.max_output_per_call:raise ValueError('output_cap_mismatch')
        used_in=used_out=completed=0
        for aid,start in starts.items():
            event=terminal.get(aid)
            if event is None:raise UncertainRequest('unresolved_execution')
            if event['event']=='rejected':continue
            completed+=1;usage=event.get('usage') or {}
            a,b=usage.get('prompt_tokens'),usage.get('completion_tokens')
            used_in+=a if valid_usage(a) else start['reserved_input']
            used_out+=b if valid_usage(b) else start['reserved_output']
        if (len(starts)+1>self.config.max_attempts or completed+1>self.config.max_completions or
            used_in+reserve_in>self.config.max_input_tokens or used_out+reserve_out>self.config.max_output_tokens):
            raise BudgetExceeded('persistent_resource_ceiling')
        return reserve_in,reserve_out

    def complete(self,request,logical_slot):
        if not isinstance(logical_slot,dict) or set(logical_slot)&{'key','attempt_id','cached'}:
            raise ValueError('logical_slot_cannot_override_physical_identity')
        with self.lock:
            self._recover()
            if self.config.mode=='REAL' and request.tokenizer_identity.get('kind')!='pinned':raise ValueError('real_tokenizer_required')
            if self.tokenizer is None or canonical(request.tokenizer_identity)!=canonical(self.tokenizer.identity):
                raise ValueError('tokenizer_identity_mismatch')
            key=request.key;cache=self.directory/'responses'/(key+'.json');cached=cache.exists()
            if cached:
                saved=self._checked(cache)
                if saved['key']!=key:raise UncertainRequest('cache_key_mismatch')
            else:
                if self.transport is None:raise ValueError('transport_not_configured')
                prior_rejections=sum(e['event']=='rejected' and e['key']==key for e in self._read())
                if prior_rejections>self.config.max_retries:raise ProviderRejected('retry_limit_across_restart')
                while True:
                    ri,ro=self._reserve(request);aid=uuid.uuid4().hex
                    request_obj=asdict(request);atomic_json(self.directory/'requests'/(key+'.json'),request_obj)
                    req_hash=digest(request_obj)
                    self.journal.append({'event':'started','attempt_id':aid,'key':key,'request_sha256':req_hash,
                                         'reserved_input':ri,'reserved_output':ro,'mode':self.config.mode})
                    start=time.perf_counter()
                    try:status,headers,raw=self.transport(request.provider_body(),aid)
                    except Exception as error:
                        self.journal.append({'event':'transport_error','attempt_id':aid,'error_type':type(error).__name__})
                        raise UncertainRequest('unknown_transport_status') from error
                    saved={'key':key,'attempt_id':aid,'request_sha256':req_hash,'raw':raw,'status':status,
                           'headers':{k:v for k,v in headers.items() if k.lower() in ('retry-after','x-request-id','apim-request-id')},
                           'latency_seconds':time.perf_counter()-start,'mode':self.config.mode}
                    saved['sha256']=digest(saved);atomic_json(self.directory/'attempts'/(aid+'.json'),saved)
                    if self._finish(saved)=='success':break
                    prior_rejections+=1
                    if prior_rejections>self.config.max_retries:raise ProviderRejected('explicit_rejection_retry_limit')
            slot={'key':key,'attempt_id':saved['attempt_id'],'cached':cached,**logical_slot}
            self.references.append(slot);self.logical.append(slot)
            payload=json.loads(request.messages[-1]['content']);visible=[]
            for item in payload.get('visible_sources',[]):
                visible.append(Atom(**{**item,'context':tuple(item.get('context',{}).items())}))
            return parse_outcome(saved['raw'],tuple(visible),request.decoding['dataset'])

    def summary(self):return cost_summary(self._read(),self.references)
