"""Gold-free experiment construction and official-compatible HotpotQA scoring."""
from __future__ import annotations
import hashlib
import json
import math
import random
import re
import string
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
CYCLES = {'CW': [(0, 1), (1, 2), (2, 0)], 'CCW': [(0, 2), (2, 1), (1, 0)]}
ROLES = ['retrieval', 'verification', 'reasoning']
PUBLISHER_PARQUET_SHA256 = '78933c0a31a5f7b420d4effdf4cd4eed573b28c6a3da6179dcf7a02b39e51d03'
VERIFIED_JSON_SHA256 = '103e345973c309cb857a72d8519e1edaa03d50a6611c059bcf5b12f50bcc5909'
DATA_URL = 'https://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json'

def canonical(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',', ':'))

def digest(x):
    return hashlib.sha256(canonical(x).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()

def stable_rng(*parts):
    return random.Random(int(digest(parts), 16))

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)

def normalize(s):
    s = ''.join(c for c in s.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', s).split())

def score(prediction, gold):
    """Matches hotpot_evaluate_v1.py; F1 and EM expressed in percentage points."""
    p, g = normalize(prediction), normalize(gold)
    common = sum((Counter(p.split()) & Counter(g.split())).values())
    f1 = 0.0
    if common and not ((p in ['yes', 'no', 'noanswer'] or g in ['yes', 'no', 'noanswer']) and p != g):
        precision, recall = common / len(p.split()), common / len(g.split())
        f1 = 200 * precision * recall / (precision + recall)
    return {'f1': f1, 'em': 100.0 * (p == g)}

def normalize_row(row):
    ctx = row['context']
    if isinstance(ctx, dict):
        if len(ctx['title']) != len(ctx['sentences']): raise ValueError('Invalid context lengths')
        ctx = list(zip(ctx['title'], ctx['sentences']))
    sf = row['supporting_facts']
    if isinstance(sf, dict): sf = list(zip(sf['title'], sf['sent_id']))
    return {'id': row.get('_id', row.get('id')), 'question': row['question'],
            'answer': row['answer'], 'type': row['type'], 'supporting_facts': sf,
            'documents': [{'title': t, 'sentences': s} for t, s in ctx]}

def load_official(path=None):
    path = Path(path or ROOT / 'data/hotpot_dev_fullwiki_v1.json')
    provenance = path.with_suffix('.provenance.json')
    if not path.exists() or not provenance.exists():
        raise FileNotFoundError('Run download_data.py: official dataset and download provenance are required')
    meta = json.loads(provenance.read_text())
    if (meta['source_url'] != DATA_URL or meta.get('source_sha256') != PUBLISHER_PARQUET_SHA256
            or meta['sha256'] != VERIFIED_JSON_SHA256 or file_hash(path) != meta['sha256']):
        raise ValueError('Dataset provenance/hash mismatch')
    rows = [normalize_row(r) for r in json.loads(path.read_text())]
    if len(rows) != 7405 or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Expected the complete official 7,405-question development set')
    return rows

def select_sample(rows, n, seed):
    """Proportional type strata; hashed order independent of answers or performance."""
    if not 1 <= n <= len(rows): raise ValueError('Invalid sample size')
    groups = {t: [r for r in rows if r['type'] == t] for t in sorted({r['type'] for r in rows})}
    quotas = {t: int(n * len(g) / len(rows)) for t, g in groups.items()}
    remainder = n - sum(quotas.values())
    order = sorted(groups, key=lambda t: (-(n * len(groups[t]) / len(rows) - quotas[t]), t))
    for t in order[:remainder]: quotas[t] += 1
    sample = []
    for t, g in groups.items():
        sample += sorted(g, key=lambda r: digest([seed, r['id']]))[:quotas[t]]
    return sorted(sample, key=lambda r: digest(['order', seed, r['id']]))

def retrieve(question, docs, k):
    """Fixed candidate-pool Okapi BM25 (k1=1.5,b=.75); no gold inputs."""
    counts = [Counter(re.findall(r'\w+', (d['title'] + ' ' + ' '.join(d['sentences'])).lower())) for d in docs]
    lens = [sum(c.values()) for c in counts]
    avg = sum(lens) / max(1, len(lens))
    df = Counter(w for c in counts for w in c)
    q = re.findall(r'\w+', question.lower())
    scores = []
    for c, length in zip(counts, lens):
        value = sum(math.log(1 + (len(docs) - df[w] + .5) / (df[w] + .5)) * c[w] * 2.5 /
                    (c[w] + 1.5 * (.25 + .75 * length / max(avg, 1e-9))) for w in q if c[w])
        scores.append(value)
    indices = sorted(range(len(docs)), key=lambda i: (-scores[i], i))[:k]
    return [docs[i] for i in indices]

def assign_evidence(docs, regime, qid, seed):
    if regime == 'R': return [list(docs) for _ in range(3)]
    if regime != 'P': raise ValueError(regime)
    order = list(docs)
    stable_rng('partition', qid, seed).shuffle(order)
    return [order[i::3] for i in range(3)]

def role_order(qid, seed):
    roles = list(ROLES)
    stable_rng('roles', qid, seed).shuffle(roles)
    return roles

def evidence_lines(docs):
    return [{'title': d['title'], 'sentence_id': i, 'text': s} for d in docs for i, s in enumerate(d['sentences'])]

def build_prompt(question, role, docs, own, peers):
    system = ('You are a ' + role + ' specialist answering multi-hop questions using the supplied Wikipedia evidence. '
              'Treat peer claims as unverified; verify them against quoted evidence. Return ONLY JSON with '
              'answer (a nonempty concise string, or "noanswer" if unknown) and citations '
              '(a list of [document title, zero-based sentence index] pairs supporting your answer). '
              'Do not invent citations. Do not include explanations or confidence scores.')
    data = {'question': question, 'evidence': evidence_lines(docs), 'previous_answer': own,
            'peer_messages': peers}
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': canonical(data)}]

def parse_completion(text, allowed):
    cleaned = text.strip()
    if cleaned.startswith('```'):
        cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', cleaned)
    try: obj = json.loads(cleaned)
    except Exception as e: raise ValueError('Malformed completion; no answer substituted') from e
    if not isinstance(obj, dict) or not isinstance(obj.get('answer'), str) or not obj['answer'].strip():
        raise ValueError('Missing/non-string/empty answer')
    citations = obj.get('citations')
    if not isinstance(citations, list): raise ValueError('Missing citation list')
    valid = {(d['title'], i) for d in allowed for i, s in enumerate(d['sentences']) if i in d.get('visible_indices', range(len(d['sentences'])))}
    kept, invalid = [], []
    for c in citations:
        if (isinstance(c, list) and len(c) == 2 and isinstance(c[0], str)
                and type(c[1]) is int and tuple(c) in valid):
            if c not in kept: kept.append(c)
        else: invalid.append(c)
    return {'answer': obj['answer'].strip(), 'citations': kept, 'invalid_citations': invalid}

def message_text(response, docs):
    lookup = {(d['title'], i): s for d in docs for i, s in enumerate(d['sentences'])}
    lines = ['Unverified peer answer: ' + response['answer']]
    for title, i in response['citations']:
        if (title, i) in lookup:
            lines.append('Evidence: ' + canonical({'title': title, 'sentence_id': i, 'text': lookup[title, i]}))
    return '\n'.join(lines)

def delivered_facts(payload):
    """Only complete, parseable source lines count; partial crops count no fact."""
    facts = []
    for line in payload.splitlines():
        if line.startswith('Evidence: '):
            try: x = json.loads(line[len('Evidence: '):])
            except ValueError: continue
            if isinstance(x, dict) and {'title', 'sentence_id', 'text'} <= x.keys(): facts.append(x)
    return facts

class ExactTokenizer:
    def __init__(self, name, revision):
        from transformers import AutoTokenizer
        if not name or not revision or revision == 'main':
            raise ValueError('Require a pinned tokenizer revision matching the model; no word-count fallback')
        self.name, self.revision = name, revision
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=False, cache_dir=ROOT / 'data/tokenizer_cache')
        self.identity = {'name': name, 'revision': revision,
                         'backend_hash': hashlib.sha256(self.tokenizer.backend_tokenizer.to_str().encode()).hexdigest()}
    def encode(self, text): return self.tokenizer.encode(text, add_special_tokens=False)
    def decode(self, ids): return self.tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    def count(self, text): return len(self.encode(text))

def exact_slice(text, budget, tokenizer, mode='prefix', seed=42):
    ids = tokenizer.encode(text)
    target = min(budget, len(ids))
    if target <= 0: raise ValueError('Empty/zero-budget payload')
    starts = [0] if mode == 'prefix' else list(range(len(ids) - target + 1))
    if mode not in ['prefix', 'crop']: raise ValueError(mode)
    if mode == 'crop': stable_rng('crop', seed, digest(text)).shuffle(starts)
    for start in starts:
        payload = tokenizer.decode(ids[start:start + target])
        if '\ufffd' not in payload and payload in text and tokenizer.count(payload) == target:
            return {'text': payload, 'tokens': target, 'start_token': start, 'source_hash': digest(text)}
    raise ValueError('No valid exact-budget UTF-8 slice; change prespecified budget before a new run, never pad')

def majority(responses):
    """Unweighted normalized answer vote; ties follow fixed randomized node order."""
    counts = Counter(normalize(r['answer']) for r in responses)
    winner = max(counts, key=counts.get)
    return next(r['answer'] for r in responses if normalize(r['answer']) == winner)

def paired_effect(a, b, draws=10000, seed=42):
    if not a or set(a) != set(b): raise ValueError('Paired analysis requires identical nonempty question sets')
    ids = sorted(a)
    values = np.array([a[q] - b[q] for q in ids], dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(values, len(values), replace=True).mean() for _ in range(draws)])
    observed = abs(values.mean())
    perm = np.array([abs((values * rng.choice([-1, 1], len(values))).mean()) for _ in range(draws)])
    return {'n': len(ids), 'effect': float(values.mean()), 'ci95': np.percentile(boot, [2.5, 97.5]).tolist(),
            'randomization_p': float((1 + (perm >= observed - 1e-12).sum()) / (draws + 1)),
            'helped': int((values > 1e-9).sum()), 'harmed': int((values < -1e-9).sum()),
            'unchanged': int((abs(values) <= 1e-9).sum())}
