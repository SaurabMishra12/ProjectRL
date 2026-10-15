"""One serial, durable budget shared by every E6/E7 paid request in this study.

Only explicit ungenerated 429 responses are retried. Unknown execution status
halts the entire ledger, including after process restart. No SDK retry layer.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import time
import uuid
from collections import Counter
from email.utils import parsedate_to_datetime
from pathlib import Path

from filelock import FileLock


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def digest(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('w') as f:
        f.write(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


class BudgetExceeded(RuntimeError): pass
class UncertainRequest(RuntimeError): pass
class ProviderRejected(RuntimeError): pass


def retry_delay(headers, fallback=2.0):
    """Honor both standard date/seconds Retry-After and Azure milliseconds."""
    headers = {k.lower(): v for k, v in headers.items()}
    delays = [float(fallback)]
    for key, scale in [('retry-after-ms', .001), ('retry-after', 1)]:
        if key not in headers: continue
        try: delays.append(max(0., float(headers[key]) * scale))
        except (ValueError, TypeError):
            if key == 'retry-after':
                try:
                    date = parsedate_to_datetime(headers[key])
                    delays.append(max(0., (date - dt.datetime.now(dt.timezone.utc)).total_seconds()))
                except (ValueError, TypeError, OverflowError): pass
    return max(delays)


def azure_transport(config):
    """Existing Azure v1 deployment; credentials remain only in memory."""
    import httpx
    from WEBExtension.machine_connection import azure_connection
    if config.get('provider') != 'azure_v1':
        raise ValueError('This pilot is frozen to the existing Azure v1 provider')
    connection = azure_connection()
    endpoint = config['base_url'].rstrip('/') + '/chat/completions'
    client = httpx.Client(timeout=httpx.Timeout(180., connect=20.), follow_redirects=False,
                          transport=httpx.HTTPTransport(retries=0))
    def send(request, request_id):
        response = client.post(endpoint, json=request,
                               headers={'Authorization': 'Bearer ' + connection['api_key'],
                                        'Content-Type': 'application/json',
                                        'x-ms-client-request-id': request_id})
        headers = dict(response.headers)
        try: raw = response.json()
        except ValueError: raw = {'non_json_body': response.text}
        return response.status_code, headers, raw
    return send


class BudgetBackend:
    def __init__(self, directory, config, *, transport=None, token_count=None, sleep=time.sleep):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config = dict(config)
        for key, ceiling in [('max_completions',150), ('max_attempts',180)]:
            if type(config.get(key)) is not int or not 0 < config[key] <= ceiling:
                raise ValueError('Approved hard ceiling: ' + key)
        if config.get('min_interval_seconds', 0) < 0: raise ValueError('Negative pacing')
        self.lock = FileLock(str(self.directory/'budget.lock'), timeout=0)
        self.sleep, self.token_count = sleep, token_count
        self.transport = transport
        self.ledger = self.directory/'ledger.jsonl'
        # This identity freezes decoding, model, limits, pacing and provider location.
        self.identity = self.config
        with self.lock:
            p = self.directory/'frozen_config.json'
            if p.exists() and json.loads(p.read_text()) != self.identity:
                raise ValueError('Frozen provider configuration differs; no alternate budget namespace')
            if not p.exists(): atomic_json(p,self.identity)
            self._recover()

    def _records(self):
        if not self.ledger.exists(): return []
        try: return [json.loads(line) for line in self.ledger.read_text().splitlines() if line]
        except ValueError as exc:
            raise UncertainRequest('Damaged ledger; manual reconciliation required') from exc

    def _append(self, obj):
        obj = {**obj, 'utc': dt.datetime.now(dt.timezone.utc).isoformat()}
        with self.ledger.open('a') as f:
            f.write(canonical(obj)+'\n'); f.flush(); os.fsync(f.fileno())

    def _recover(self):
        records = self._records()
        closed = {r['attempt_id'] for r in records if r['event'] in ['success','rejected']}
        unresolved = [r for r in records if r['event']=='started' and r['attempt_id'] not in closed]
        for started in unresolved:
            p = self.directory/'responses'/(started['request_hash']+'.json')
            saved = json.loads(p.read_text()) if p.exists() else None
            if saved and saved.get('attempt_id')==started['attempt_id']:
                self._validate_saved(saved, started['request_hash'])
                if saved['http_status'] in [200,201] or (isinstance(saved['raw'],dict) and saved['raw'].get('choices')):
                    self._append({'event':'success','attempt_id':started['attempt_id'],
                                  'request_hash':started['request_hash'], 'usage':saved['raw'].get('usage'),
                                  'latency_seconds':saved['latency_seconds'], 'recovered':True})
                    continue
            raise UncertainRequest('Unresolved provider attempt ' + started['attempt_id'] +
                                   '; no paid calls until provider-side reconciliation')

    def _validate_saved(self, saved, request_hash):
        content_hash = saved.get('content_hash')
        if content_hash != digest({k:v for k,v in saved.items() if k!='content_hash'}):
            raise ValueError('Cached response checksum mismatch')
        if saved.get('request_hash') != request_hash or saved.get('identity') != self.identity:
            raise ValueError('Cached response identity mismatch')

    def summary(self):
        records = self._records()
        started = [r for r in records if r['event']=='started']
        successes = [r for r in records if r['event']=='success']
        rejected = [r for r in records if r['event']=='rejected']
        resolved = {r['attempt_id'] for r in successes+rejected}
        uncertain = [r for r in started if r['attempt_id'] not in resolved]
        usages = [r.get('usage') or {} for r in successes]
        input_known = [u['prompt_tokens'] for u in usages if isinstance(u.get('prompt_tokens'),int)]
        output_known = [u['completion_tokens'] for u in usages if isinstance(u.get('completion_tokens'),int)]
        counts = Counter(r['request_hash'] for r in started)
        return {'successful_completions':len(successes), 'http_attempts':len(started),
                'retries':sum(n-1 for n in counts.values()),'rejected_attempts':len(rejected),
                'uncertain_requests':len(uncertain),
                'input_tokens':sum(input_known) if len(input_known)==len(successes) else None,
                'output_tokens':sum(output_known) if len(output_known)==len(successes) else None,
                'input_tokens_lower_bound':sum(input_known),'output_tokens_lower_bound':sum(output_known),
                'missing_input_usage':len(successes)-len(input_known),
                'missing_output_usage':len(successes)-len(output_known),
                'successful_http_latency_seconds':sum(r.get('latency_seconds',0) for r in successes),
                'remaining_completions':self.config['max_completions']-len(successes),
                'remaining_http_attempts':self.config['max_attempts']-len(started),
                'monetary_cost':None, 'monetary_cost_status':'UNVERIFIED_DEPLOYMENT_PRICE',
                'uncertain_attempt_ids':[r['attempt_id'] for r in uncertain]}

    def complete(self, messages, metadata=None):
        with self.lock:
            self._recover()
            request = {k:self.config[k] for k in ['model','temperature','max_tokens']}
            request['messages'] = messages
            # Internal scope prevents reusing stochastic states across distinct
            # question/placement/node assignments, even if two prompts coincide.
            # It is never inserted into the model-visible messages.
            request_hash = digest({'identity':self.identity, 'request':request, 'scope':metadata or {}})
            p = self.directory/'responses'/(request_hash+'.json')
            if p.exists():
                saved = json.loads(p.read_text()); self._validate_saved(saved,request_hash)
                if saved['request'] != request: raise ValueError('Cached prompt mismatch')
                return {**saved,'cache_hit':True}
            # Word/byte fallback is conservative gating only; never reported as provider tokens.
            estimate = (sum(self.token_count(m['content']) for m in messages) if self.token_count else
                        sum(len(m['content'].encode('utf-8')) for m in messages)) + 128
            if estimate+self.config['max_tokens'] > self.config['context_limit']:
                raise BudgetExceeded('Declared context ceiling would be exceeded; no silent truncation')
            for retry in range(self.config.get('max_retries_per_request',2)+1):
                current = self.summary()
                if current['successful_completions'] >= self.config['max_completions']:
                    raise BudgetExceeded('Successful completion ceiling reached')
                if current['http_attempts'] >= self.config['max_attempts']:
                    raise BudgetExceeded('HTTP attempt ceiling reached')
                records = self._records()
                estimated_spent = sum(r.get('reserved_tokens',0) for r in records if r['event']=='started')
                reserve = estimate+self.config['max_tokens']
                if estimated_spent+reserve > self.config.get('max_estimated_tokens',2000000):
                    raise BudgetExceeded('Conservative input-plus-output token reservation ceiling reached')
                last = next((r for r in reversed(records) if r['event']=='started'),None)
                if last:
                    elapsed = time.time()-last['unix_time']
                    delay = max(0.,self.config.get('min_interval_seconds',2.)-elapsed)
                    if delay: self.sleep(delay)
                attempt_id = uuid.uuid4().hex
                atomic_json(self.directory/'requests'/(attempt_id+'.json'),
                            {'attempt_id':attempt_id,'request_hash':request_hash, 'request':request,
                             'identity':self.identity,'metadata':metadata or {}})
                self._append({'event':'started','attempt_id':attempt_id,'request_hash':request_hash,
                              'unix_time':time.time(),'reserved_tokens':reserve,'estimated_input_tokens':estimate,
                              'retry_index':retry,'metadata':metadata or {}})
                start = time.perf_counter()
                try:
                    if self.transport is None: self.transport = azure_transport(self.config)
                    status, headers, raw = self.transport(request, attempt_id)
                except Exception as exc:
                    self._append({'event':'uncertain','attempt_id':attempt_id,'request_hash':request_hash,
                                  'error_type':type(exc).__name__})
                    raise UncertainRequest('Request execution status unknown; raw exception suppressed') from None
                elapsed = time.perf_counter()-start
                selected_headers = {k.lower():v for k,v in headers.items() if k.lower() in
                                    ['x-request-id','apim-request-id','x-ms-request-id','retry-after',
                                     'retry-after-ms','x-ratelimit-remaining-requests','x-ratelimit-remaining-tokens']}
                saved = {'attempt_id':attempt_id,'request_hash':request_hash,'request':request,
                         'identity':self.identity,'http_status':status,'headers':selected_headers,
                         'raw':raw,'latency_seconds':elapsed,'metadata':metadata or {}}
                saved['content_hash'] = digest(saved)
                atomic_json(self.directory/'attempts'/(attempt_id+'.json'),saved)
                # Count filtered completion-shaped HTTP400 bodies conservatively as completions.
                completed = status in [200,201] or (status==400 and isinstance(raw,dict) and bool(raw.get('choices')))
                if completed:
                    atomic_json(p,saved)
                    self._append({'event':'success','attempt_id':attempt_id,'request_hash':request_hash,
                                  'http_status':status,'usage':raw.get('usage') if isinstance(raw,dict) else None,
                                  'latency_seconds':elapsed,'provider_request_ids':selected_headers})
                    return {**saved,'cache_hit':False}
                if status in [400,401,403,404,405,413,415,422,429]:
                    self._append({'event':'rejected','attempt_id':attempt_id,'request_hash':request_hash,
                                  'http_status':status,'latency_seconds':elapsed})
                    if status==429 and retry < self.config.get('max_retries_per_request',2):
                        self.sleep(retry_delay(selected_headers,2.**(retry+1)))
                        continue
                    raise ProviderRejected('Provider rejected HTTP '+str(status)+'; response saved, no further retry')
                self._append({'event':'uncertain','attempt_id':attempt_id,'request_hash':request_hash,
                              'http_status':status,'latency_seconds':elapsed})
                raise UncertainRequest('HTTP '+str(status)+' may have generated a completion; stop and reconcile')
