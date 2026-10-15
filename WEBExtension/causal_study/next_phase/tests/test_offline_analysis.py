import copy
import pytest

from WEBExtension.causal_study.provider import digest
from WEBExtension.causal_study.next_phase.binding_design import make_case, make_manifest, messages
from WEBExtension.causal_study.next_phase.offline_analysis import score_records, validate_records


def fixtures():
    case = make_case('q', 'What city?',
        [{'answer': 'The Paris!', 'citations': [['France', 0]]},
         {'answer': 'Lyon', 'citations': [['Lyon', 0]]},
         {'answer': 'noanswer', 'citations': []}],
        [[{'title': 'France', 'sentence_id': 0, 'text': 'Paris is the capital.'}],
         [{'title': 'Lyon', 'sentence_id': 0, 'text': 'Lyon is a city.'}], []])
    manifest = make_manifest([case], n=1)
    q = manifest['questions'][0]
    record = {'question_id': 'q', 'manifest_hash': manifest['manifest_sha256'],
              'execution_mode': 'UNIT_TEST_FIXTURE_ONLY', 'arms': {}}
    for arm, node in [('recorded', 0), ('exchanged', 1)]:
        choice = 'C' + str(q['candidate_order'].index(node))
        record['arms'][arm] = {'choice': choice, 'selected_node': node,
            'selected_frozen_answer': q['candidates'][node]['answer'],
            'prompt_sha256': digest(messages(q, arm))}
    record['content_hash'] = digest(record)
    return manifest, [record], {'q': {'answer': 'Paris'}}


def resign(record):
    record.pop('content_hash', None)
    record['content_hash'] = digest(record)


def test_offline_score_maps_candidate_ids_and_uses_official_normalization():
    m, records, labels = fixtures()
    result = score_records(m, records, labels, fixture_only=True)
    assert result['status'] == 'UNIT_TEST_ONLY_NOT_SCIENTIFIC_EVIDENCE'
    assert result['rows'][0]['recorded_em'] == 100
    assert result['rows'][0]['recorded_f1'] == 100
    assert result['rows'][0]['exchanged_em'] == 0
    assert result['analysis']['em_delta_pp'] == 100


def test_mock_outputs_never_become_scientific_effects():
    m, records, labels = fixtures()
    records[0]['execution_mode'] = 'MOCK_ONLY'
    resign(records[0])
    assert validate_records(m, records, expected_mode='MOCK_ONLY')['paired_questions'] == 1
    with pytest.raises(ValueError, match='Mock'):
        score_records(m, records, labels)
    with pytest.raises(ValueError, match='Mock'):
        score_records(m, records, labels, fixture_only=True)


@pytest.mark.parametrize('mutation', ['missing_arm', 'answer', 'prompt', 'choice', 'node'])
def test_resigned_but_inconsistent_outputs_are_rejected(mutation):
    m, records, _ = fixtures()
    r = records[0]
    if mutation == 'missing_arm':
        del r['arms']['exchanged']
    else:
        field, value = {'answer': ('selected_frozen_answer', 'Rome'),
                        'prompt': ('prompt_sha256', 'another-prompt'),
                        'choice': ('choice', 'C8'), 'node': ('selected_node', 2)}[mutation]
        r['arms']['recorded'][field] = value
    resign(r)
    with pytest.raises(ValueError):
        validate_records(m, records, expected_mode='UNIT_TEST_FIXTURE_ONLY')


def test_question_pairing_never_silently_drops_or_duplicates_questions():
    m, records, labels = fixtures()
    for bad in [[], records + copy.deepcopy(records)]:
        with pytest.raises(ValueError):
            score_records(m, bad, labels, fixture_only=True)
    with pytest.raises(ValueError):
        score_records(m, records, {}, fixture_only=True)
