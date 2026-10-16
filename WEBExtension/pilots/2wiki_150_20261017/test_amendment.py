"""Protect reuse, failed-gate provenance, and fixed assigned denominators."""
import importlib
import json
from dataclasses import replace

import pytest

from WEBExtension.web_reasoning.backend import DurableBackend, Request
from WEBExtension.web_reasoning.config import RunConfig
from WEBExtension.web_reasoning.storage import digest, atomic_json
from WEBExtension.web_reasoning.preparation import file_sha
from WEBExtension.web_reasoning.tokenization import UTF8FixtureTokenizer


def impl():
    return importlib.import_module('WEBExtension.pilots.2wiki_150_20261017.amendment')


def test_import_preserves_physical_history_and_reuses_without_transport(tmp_path):
    # Dropping the source ledger would hide paid calls; changing identity would regenerate.
    tok = UTF8FixtureTokenizer()
    old = RunConfig(mode='MOCK', model='fixture', max_completions=2)
    request = Request([{'role':'user','content':'{"visible_sources":[]}'}],
                      {'model':'fixture'}, tok.identity, {'dataset':'2wiki','max_tokens':100})
    def send(body, aid):
        return 200, {}, {'choices':[{'finish_reason':'stop','message':{'content':'{"answer":"yes"}'}}],
                         'usage':{'prompt_tokens':12,'completion_tokens':6}}
    source = tmp_path/'source'
    first = DurableBackend(source, old, send, tok)
    first.complete(request, {'qid':'fixture'})
    expected = {str(p.relative_to(source)):file_sha(p) for p in source.rglob('*')
                if p.is_file() and p.name not in ('runtime.lock','config.json')}
    dest = tmp_path/'dest'
    impl().import_cache(source, dest, expected)
    backend = DurableBackend(dest, replace(old, max_completions=10), None, tok)
    backend.complete(request, {'qid':'fixture'})
    assert backend.summary()['physical_completions'] == 1
    assert backend.summary()['physical_attempts'] == 1
    assert (dest/'calls.jsonl').read_bytes() == (source/'calls.jsonl').read_bytes()
    assert all(file_sha(source/p) == sha for p,sha in expected.items())


def test_corrupt_import_fails_before_any_cache_is_copied(tmp_path):
    source=tmp_path/'source';source.mkdir()
    (source/'calls.jsonl').write_text('original')
    expected={'calls.jsonl':file_sha(source/'calls.jsonl')}
    (source/'calls.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='origin_hash_mismatch'):
        impl().import_cache(source,tmp_path/'dest',expected)
    assert not (tmp_path/'dest/calls.jsonl').exists()


def test_completed_result_import_keeps_original_manifest_and_checks_hash(tmp_path):
    row={'question_key':'q','phase':'pilot','arms':{}}
    atomic_json(tmp_path/'q.json',{'result':row,'sha256':digest(row),'manifest_sha256':'old'})
    assert impl().load_result(tmp_path/'q.json','old') == row
    with pytest.raises(ValueError, match='result_identity_mismatch'):
        impl().load_result(tmp_path/'q.json','new')


def test_failed_gate_remains_false_with_authorized_expansion():
    gate={'pass':False,'reasons':['below_dense_sink:complementary:em']}
    record=impl().authorization_record(gate,'oldhash')
    assert record['historical_gate'] == gate
    assert record['historical_manifest_sha256'] == 'oldhash'
    assert record['user_authorized_total'] == 150
    assert record['expansion_despite_failed_gate'] is True
    assert gate['pass'] is False


def test_partial_scores_keep_all_150_assigned_questions():
    core=importlib.import_module('WEBExtension.pilots.2wiki_revised_20261016.core')
    row={'arms':{name:{'readouts':{rule:{'em':100.,'f1':50.} for rule in core.READOUTS},
                      'logical_call_keys':[], **{k:0 for k in impl().COST_METRICS}} for name in core.ARMS}}
    stats=impl().arm_stats([row],150)
    assert stats['instar']['n'] == 150
    assert stats['instar']['completed_records'] == 1
    assert stats['instar']['readouts']['sink1']['em'] == pytest.approx(100/150)
    assert stats['instar']['readouts']['sink1']['f1'] == pytest.approx(50/150)


def test_extra_selection_excludes_old_source_overlap_and_incomplete_support():
    def row(key,title):
        return {'_id':key,'type':'comparison','question':key,'context':[[title,['text']]],
                'supporting_facts':[[title,0]]}
    rows=[row('overlap','Old'),row('good1','New1'),row('good2','New2'),row('bad','Bad')]
    rows[-1]['supporting_facts']=[['Bad',2]]
    previous=[{'question':'old question','sources':[{'title':'Old'}]}]
    selected,_,_=impl().select_extra(iter(rows),previous,2)
    assert {r['_id'] for r in selected} == {'good1','good2'}
