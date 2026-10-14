"""Real OpenAI-compatible/Azure inference, exact prompt cache, append-only call ledger."""
from __future__ import annotations
import datetime
import json
import os
import time
import uuid
from pathlib import Path
from protocol import canonical, digest, file_hash, write_json, parse_completion

def unresolved_attempts(records):
    """Track attempts, not prompt keys: a later retry is a separate liability."""
    started = {r['attempt_id']: r for r in records if r['event'] == 'started'}
    finished = {r['attempt_id'] for r in records
                if r['event'] in ['success', 'rejected', 'reconciled_not_generated']}
    return [r for aid, r in started.items() if aid not in finished]

def reconcile_no_generation(run_dir, attempt_id, provider_evidence_path):
    """Human-use only after provider logs confirm this attempt produced no completion.

    Never call automatically. Evidence is hashed, not printed; a timeout alone is not evidence.
    If a billed/completed response exists, recover that response rather than mark it ungenerated.
    """
    path = Path(run_dir) / 'calls.jsonl'
    records = [json.loads(l) for l in path.read_text().splitlines() if l]
    attempts = {r['attempt_id']:r for r in unresolved_attempts(records)}
    if attempt_id not in attempts: raise ValueError('Attempt is not unresolved')
    evidence = Path(provider_evidence_path)
    if not evidence.is_file() or evidence.stat().st_size == 0: raise ValueError('Require saved provider reconciliation evidence')
    event = {'event':'reconciled_not_generated','key':attempts[attempt_id]['key'], 'attempt_id':attempt_id,
             'evidence_path':str(evidence.resolve()), 'evidence_sha256':file_hash(evidence),
             'attestation':'Human verified provider logs: this attempt did not generate a completion',
             'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with open(path,'a') as f: f.write(canonical(event)+'\n'); f.flush(); os.fsync(f.fileno())

class RealBackend:
    def __init__(self, config, tokenizer, run_dir):
        from openai import AzureOpenAI, OpenAI
        self.config, self.tokenizer, self.run_dir = config, tokenizer, Path(run_dir)
        self.cache_dir = self.run_dir / 'responses'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = self.run_dir / 'calls.jsonl'
        if config['provider'] not in ['azure', 'openai_compatible']:
            raise ValueError('Only real inference providers supported; no mock/synthetic fallback')
        if not config['model_revision'] or config['max_calls'] <= 0:
            raise ValueError('Specify immutable model/deployment revision and positive maximum calls')
        self.identity = {k: config[k] for k in ['provider', 'model', 'model_revision', 'base_url', 'api_version', 'replicate_id']}
        if config['provider'] == 'azure':
            endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT')
            key = os.environ.get('AZURE_OPENAI_API_KEY')
            if (not endpoint or not key) and config.get('credential_source') == 'project_existing':
                from machine_connection import azure_connection
                connection = azure_connection()
                endpoint, key = connection['azure_endpoint'], connection['api_key']
            if not key or not endpoint: raise RuntimeError('Set Azure endpoint and key in environment; never in notebook')
            self.identity['endpoint'] = endpoint
            self.client = AzureOpenAI(api_key=key, azure_endpoint=endpoint, api_version=config['api_version'], max_retries=0, timeout=120)
        else:
            key = os.environ.get('OPENAI_API_KEY')
            if not key: raise RuntimeError('Set OPENAI_API_KEY for your real compatible server')
            self.identity['endpoint'] = config['base_url']
            self.client = OpenAI(api_key=key, base_url=config['base_url'] or None, max_retries=0, timeout=120)
        self.attempts = []
        if self.ledger.exists(): self.attempts = [json.loads(l) for l in self.ledger.read_text().splitlines() if l]
        for attempt in unresolved_attempts(self.attempts):
            path = self.cache_dir / (attempt['key'] + '.json')
            saved = json.loads(path.read_text()) if path.exists() else None
            if (saved and saved.get('attempt_id') == attempt['attempt_id']
                    and saved.get('content_hash') == digest({k:v for k,v in saved.items() if k != 'content_hash'})):
                # Crash after atomic response save but before success-ledger append: recover actual response.
                self.append({'event':'success','key':attempt['key'],'attempt_id':attempt['attempt_id'],
                             'raw_sha256':file_hash(path),'usage':saved['response'].get('usage'), 'recovered':True})
            else:
                raise RuntimeError('Unresolved/uncertain provider attempt ' + attempt['attempt_id'] +
                                   ': reconcile provider logs before resume; no automatic retry')

    def append(self, event):
        event['utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with open(self.ledger, 'a') as f: f.write(canonical(event) + '\n'); f.flush(); os.fsync(f.fileno())
        self.attempts.append(event)

    def complete(self, messages, allowed_documents, metadata):
        if unresolved_attempts(self.attempts):
            raise RuntimeError('Unresolved provider attempt; stop this backend and reconcile before any further calls')
        params = {'model': self.config['model'], 'temperature': self.config['temperature'],
                  'max_tokens': self.config['max_tokens'], 'messages': messages}
        key = digest({'identity': self.identity, 'request': params, 'tokenizer': self.tokenizer.identity})
        path = self.cache_dir / (key + '.json')
        cached = path.exists()
        if cached:
            saved = json.loads(path.read_text())
            content_hash = saved.get('content_hash')
            if content_hash != digest({k:v for k,v in saved.items() if k != 'content_hash'}):
                raise ValueError('Cached raw response checksum mismatch')
            if saved['key'] != key or saved['request'] != params or saved['identity'] != self.identity:
                raise ValueError('Cache identity mismatch')
        else:
            attempted = sum(x['event'] == 'started' for x in self.attempts)
            if attempted >= self.config['max_calls']: raise RuntimeError('Maximum call budget reached')
            # Estimate includes provider framing reserve; never reported as measured input usage.
            estimate = sum(self.tokenizer.count(m['content']) for m in messages) + 128
            if estimate + self.config['max_tokens'] > self.config['context_limit']:
                raise RuntimeError('Prompt exceeds declared context limit; no silent truncation')
            attempt_id = uuid.uuid4().hex
            request_file = self.run_dir / 'requests' / (attempt_id + '.json')
            write_json(request_file, {'attempt_id':attempt_id,'key':key,'request':params,
                                     'identity':self.identity,'tokenizer':self.tokenizer.identity,'metadata':metadata})
            self.append({'event': 'started', 'key': key, 'attempt_id':attempt_id, 'metadata': metadata,
                         'request_file':str(request_file),'request_sha256':file_hash(request_file),
                         'request_estimated_input_tokens': estimate})
            start = time.perf_counter()
            try:
                response = self.client.chat.completions.create(**params)
                raw = response.model_dump(mode='json')
                saved = {'key': key, 'attempt_id':attempt_id, 'request': params, 'identity': self.identity,
                         'tokenizer': self.tokenizer.identity, 'response': raw,
                         'latency_seconds': time.perf_counter() - start, 'metadata': metadata}
                saved['content_hash'] = digest(saved)
                write_json(path, saved)
                self.append({'event': 'success', 'key': key, 'attempt_id':attempt_id, 'raw_sha256': file_hash(path), 'usage': raw.get('usage')})
            except Exception as exc:
                status = getattr(exc, 'status_code', None)
                event = 'rejected' if status in [400,401,403,404,422,429] else 'uncertain'
                self.append({'event': event, 'key': key, 'attempt_id':attempt_id,
                             'error_type': type(exc).__name__, 'http_status':status})
                raise RuntimeError('Real inference failed; no fabricated response or automatic retry') from exc
        raw = saved['response']
        choices = raw.get('choices', [])
        if not choices or choices[0].get('finish_reason') != 'stop':
            raise ValueError('Incomplete/refused/truncated provider response; raw preserved')
        text = choices[0]['message'].get('content')
        if not isinstance(text, str): raise ValueError('No textual provider completion')
        parsed = parse_completion(text, allowed_documents)
        usage = raw.get('usage') or {}
        return {**parsed, 'raw_text': text, 'call_key': key, 'cache_hit': cached,
                'input_tokens': usage.get('prompt_tokens'), 'output_tokens': usage.get('completion_tokens'),
                'provider_usage': usage, 'returned_model': raw.get('model'),
                'latency_seconds': saved['latency_seconds'] if not cached else None}
