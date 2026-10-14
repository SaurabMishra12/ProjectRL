"""Freeze outcome-independent subsets and import authentic response caches only."""
import json
from collections import Counter
from pathlib import Path
from controller_splits import load_partition, choose, assert_disjoint, inference_inputs
from protocol import ROOT, digest, file_hash
from analysis.durable_storage import durable_json, durable_bytes


def freeze_subsets(parent, settings):
    parent = Path(parent)
    old = json.loads((parent / 'manifest.json').read_text())
    parts = {name: load_partition(parent, name, gold=True)
             for name in ('train', 'validation', 'heldout', 'pilot')}
    if settings['test_questions'] != len(parts['heldout']):
        raise ValueError('The held-out sample must remain unchanged')
    for name, count_key, offset in [('train', 'train_questions', 2),
                                    ('validation', 'validation_questions', 1)]:
        rows = parts[name]
        if not 0 < settings[count_key] <= len(rows):
            raise ValueError('Subset size outside original partition')
        parts[name] = choose(rows, settings[count_key], settings['split_seed'] + offset,
                             [], Counter(r['type'] for r in rows))
    checks = assert_disjoint(parts)
    spec = {'version': 'title_disjoint_v1', 'config': settings, 'source': old['spec']['source']}
    directory = ROOT / 'artifacts' / ('controller_splits_' + digest(spec)[:16])
    if (directory / 'manifest.json').exists():
        existing = json.loads((directory / 'manifest.json').read_text())
        if existing['spec'] != spec or existing.get('parent_split_manifest_sha256') != file_hash(parent / 'manifest.json'):
            raise ValueError('Existing subset plan differs')
        for name, sha in existing['files'].items():
            if file_hash(directory / name) != sha:
                raise ValueError('Existing subset file changed')
        return directory
    files = {}
    for name, rows in parts.items():
        for suffix, value in [('gold', rows), ('inputs', inference_inputs(rows))]:
            path = directory / (name + '_' + suffix + '.json')
            durable_json(path, value)
            files[path.name] = file_hash(path)
    manifest = {**old, 'spec': spec, 'files': files, 'leakage_checks': checks,
                'partitions': {name: {'n': len(rows), 'types': dict(Counter(r['type'] for r in rows)),
                                      'ids': [r['id'] for r in rows]} for name, rows in parts.items()},
                'parent_split_manifest': str(parent / 'manifest.json'),
                'parent_split_manifest_sha256': file_hash(parent / 'manifest.json'),
                'subset_selection': 'Fixed type-stratified deterministic hash ordering; no predictions or scores used',
                'heldout_unchanged': files['heldout_gold.json'] == old['files']['heldout_gold.json']}
    if not manifest['heldout_unchanged']:
        raise ValueError('Held-out file changed')
    durable_json(directory / 'manifest.json', manifest)
    return directory


def import_responses(source, destination):
    source, destination = Path(source), Path(destination)
    entries = []
    for path in sorted((source / 'responses').glob('*.json')):
        raw = json.loads(path.read_text())
        if raw.get('content_hash') != digest({k: v for k, v in raw.items() if k != 'content_hash'}):
            raise ValueError('Original response checksum mismatch')
        if path.stem != raw['key']:
            raise ValueError('Original response key mismatch')
        target = destination / 'responses' / path.name
        durable_bytes(target, path.read_bytes())
        entries.append({'key': raw['key'], 'original_attempt_id': raw['attempt_id'],
                        'source': str(path), 'sha256': file_hash(path),
                        'reported_usage': raw['response'].get('usage')})
    durable_json(destination / 'cache_import_manifest.json', {
        'source_run': str(source), 'source_ledger': str(source / 'calls.jsonl'),
        'source_ledger_sha256': file_hash(source / 'calls.jsonl'), 'responses': entries,
        'accounting': 'Original calls remain owned by the preserved source ledger. Imported responses cost no new API call; do not double-count them.'})
    return len(entries)
