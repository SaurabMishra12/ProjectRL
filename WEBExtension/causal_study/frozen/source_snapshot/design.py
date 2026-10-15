"""Pure, deterministic E6/E7 design. Labels never enter a model payload.

No network, credentials, provider, caches or filesystem writes belong here. Every
model-visible payload is built from an explicit allowlist. ``token_count`` is an
optional exact tokenizer callable; word counts are never labelled model tokens.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import itertools
import json
import random
import re
import string
from typing import Callable

VERSION = 'causal-pilot-e6-matched-revision-v1'
ROLES = ('retrieval', 'verification', 'reasoning')
E6_COMPLETIONS_PER_QUESTION = 18


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def _rng(*parts):
    return random.Random(int(digest(parts), 16))


def _normalize(text):
    text = ''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())


def _facts(documents):
    return [{'title': d['title'], 'sentence_id': i, 'text': text}
            for d in documents for i, text in enumerate(d['sentences'])]


def _length(text, token_count=None):
    return {'characters': len(text), 'words': len(text.split()),
            'tokens': int(token_count(text)) if token_count is not None else None}


def _normalized_row(row):
    """Accept the existing normalized representation or publisher JSON rows."""
    documents = row.get('documents')
    if documents is None:
        context = row['context']
        if isinstance(context, dict):
            if len(context['title']) != len(context['sentences']):
                raise ValueError('Mismatched context arrays')
            context = zip(context['title'], context['sentences'])
        documents = [{'title': t, 'sentences': list(s)} for t, s in context]
    sf = row.get('supporting_facts', [])
    if isinstance(sf, dict):
        if len(sf['title']) != len(sf['sent_id']):
            raise ValueError('Mismatched supporting-fact arrays')
        sf = list(zip(sf['title'], sf['sent_id']))
    return {'id': row.get('id', row.get('_id')), 'question': row['question'],
            'type': row.get('type', 'unknown'), 'answer': row.get('answer'),
            'supporting_facts': [list(f) for f in sf],
            'documents': copy.deepcopy(documents)}


def _eligibility(row):
    if not row['id'] or not row['question'] or not isinstance(row['answer'], str):
        return 'missing_identity_question_or_label'
    documents = row['documents']
    titles = [d['title'] for d in documents]
    if len(set(titles)) != len(titles):
        return 'duplicate_candidate_titles'
    if len(documents) < 6:
        return 'fewer_than_six_candidate_articles'
    if any(not isinstance(d.get('sentences'), list) or not d['sentences'] or
           any(not isinstance(s, str) or not s.strip() for s in d['sentences']) for d in documents):
        return 'invalid_or_empty_source_sentences'
    support = {t for t, _ in row['supporting_facts']}
    if len(support) != 2:
        return 'not_exactly_two_annotated_support_articles'
    if not support.issubset(titles):
        return 'annotated_support_article_absent_from_candidates'
    docs = {d['title']: d for d in documents}
    if any(type(i) is not int or not 0 <= i < len(docs[t]['sentences'])
           for t, i in row['supporting_facts']):
        return 'invalid_annotated_sentence_index'
    return None


def _balanced_sample(rows, n, seed):
    """Proportional type strata; no answers or predictions enter the rank."""
    types = sorted({r['type'] for r in rows})
    groups = {t: [r for r in rows if r['type'] == t] for t in types}
    quotas = {t: n * len(g) // len(rows) for t, g in groups.items()}
    remainder = n - sum(quotas.values())
    for t in sorted(types, key=lambda t: (-(n * len(groups[t]) / len(rows) - quotas[t]), t))[:remainder]:
        quotas[t] += 1
    selected = []
    for t in types:
        selected.extend(sorted(groups[t], key=lambda r: digest([seed, 'sample', r['id']]))[:quotas[t]])
    return sorted(selected, key=lambda r: digest([seed, 'order', r['id']]))


def build_manifest(rows, n=8, seed=20261015, blocked_rows=(), token_count: Callable | None = None,
                   source=None):
    """Freeze selection, six-document unions, and paired assignments before outcomes.

    ``blocked_rows`` supports rows with only id/question; all supplied exposure IDs
    and normalized questions are excluded. Candidate-title overlaps are reported,
    not excluded: the pilot makes no document-disjoint held-out-test claim.
    """
    if type(n) is not int or n < 1 or n * E6_COMPLETIONS_PER_QUESTION > 150:
        raise ValueError('E6 sample exceeds the 150-completion pilot authorization')
    rows = [_normalized_row(r) for r in rows]
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate source question IDs')
    blocked_rows = list(blocked_rows)
    blocked_ids = {r.get('id', r.get('_id')) for r in blocked_rows}
    blocked_questions = {_normalize(r['question']) for r in blocked_rows if r.get('question')}
    blocked_titles = {d['title'] for r in blocked_rows for d in r.get('documents', [])}
    reasons = Counter()
    eligible = []
    exposed = 0
    seen_questions = set()
    for row in sorted(rows, key=lambda r: digest([seed, 'eligibility', r['id']])):
        if row['id'] in blocked_ids or _normalize(row['question']) in blocked_questions:
            exposed += 1
            continue
        reason = _eligibility(row)
        if reason:
            reasons[reason] += 1
            continue
        normalized_question = _normalize(row['question'])
        if normalized_question in seen_questions:
            reasons['duplicate_normalized_question'] += 1
            continue
        seen_questions.add(normalized_question)
        eligible.append(row)
    if n > len(eligible):
        raise ValueError(f'Insufficient eligible questions: requested {n}, found {len(eligible)}; reasons={dict(reasons)}')
    questions, labels = [], {}
    base_roles = list(ROLES)
    _rng(seed, 'base_roles').shuffle(base_roles)
    for ordinal, row in enumerate(_balanced_sample(eligible, n, seed)):
        qid = row['id']
        supports = sorted({t for t, _ in row['supporting_facts']}, key=lambda t: digest([seed, qid, 'support_order', t]))
        source_documents = {d['title']: d for d in row['documents']}
        distractors = sorted((d for d in row['documents'] if d['title'] not in supports),
                             key=lambda d: digest([seed, qid, 'distractor_selection', d['title']]))[:4]
        documents = [source_documents[t] for t in supports] + distractors
        # Union order is neutral; membership of support articles is never exposed.
        documents.sort(key=lambda d: digest([seed, qid, 'union_order', d['title']]))
        support_docs = [source_documents[t] for t in supports]
        coloc_owner = ordinal % 3
        dist_owner = (coloc_owner + (1 if ordinal % 2 == 0 else 2)) % 3
        # Roles rotate over all node identities independently of question content.
        # A separate Latin schedule prevents the colocated owner always taking
        # the same specialist role as role rotation.
        roles = [base_roles[(node + ordinal // 3) % 3] for node in range(3)]
        colocated = [[] for _ in range(3)]
        colocated[coloc_owner] = list(support_docs)
        others = [i for i in range(3) if i != coloc_owner]
        colocated[others[0]] = distractors[:2]
        colocated[others[1]] = distractors[2:]
        distributed = [[] for _ in range(3)]
        distributed[coloc_owner].append(support_docs[0])
        distributed[dist_owner].append(support_docs[1])
        remaining_owner = next(i for i in range(3) if i not in (coloc_owner, dist_owner))
        distributed[coloc_owner].append(distractors[0])
        distributed[dist_owner].append(distractors[1])
        distributed[remaining_owner] = distractors[2:]
        placements = {}
        for name, assignment in [('colocated', colocated), ('distributed', distributed)]:
            for node, group in enumerate(assignment):
                _rng(seed, qid, name, node, 'document_order').shuffle(group)
            placements[name] = {'assignments': copy.deepcopy(assignment), 'roles': list(roles),
                                'evidence_lengths': [_length(canonical(_facts(g)), token_count) for g in assignment]}
        placement_order = ['colocated', 'distributed']
        _rng(seed, qid, 'placement_execution_order').shuffle(placement_order)
        mode_order = ['nocomm', 'broadcast']
        _rng(seed, qid, 'revision_execution_order').shuffle(mode_order)
        questions.append({'id': qid, 'question': row['question'], 'type': row['type'],
                          'documents': copy.deepcopy(documents), 'placements': placements,
                          'execution_order': placement_order, 'revision_order': mode_order})
        labels[qid] = {'answer': row['answer'], 'supporting_facts': row['supporting_facts'],
                       'support_titles': supports}
    protocol = {
        'version': VERSION, 'seed': seed, 'n_questions': n,
        'e6_completions_per_question': E6_COMPLETIONS_PER_QUESTION,
        'e6_successful_completions': n * E6_COMPLETIONS_PER_QUESTION,
        'e7_paid_completions': 0, 'successful_completion_ceiling': 150, 'http_attempt_ceiling': 180,
        'primary': '(distributed_comm - distributed_nocomm) - (colocated_comm - colocated_nocomm)',
        'nocomm': 'one independent initial generation then one own-state revision with no peer context per node',
        'comm': 'same frozen initial state, then one simultaneous full directed broadcast revision per node',
        'aggregation': 'normalized unweighted majority; three-way ties resolved by fixed node order, identical in every arm',
        'initial_majority': 'secondary descriptive zero-additional-call anchor',
        'e7_status': 'LOCAL_VALIDATION_ONLY_PAID_INFERENCE_DEFERRED',
        'source': source, 'token_length_method': 'exact_provided_callable' if token_count else 'unavailable_words_characters_only',
        'scope': 'controlled evidence availability, supplied HotpotQA candidates; exploratory pilot, no confirmatory claim',
    }
    selection = {'source_questions': len(rows), 'excluded_exposed': exposed,
                 'known_exposure_id_count': len(blocked_ids - {None}),
                 'known_exposure_normalized_question_count': len(blocked_questions),
                 'eligible_questions': len(eligible), 'ineligible_reasons': dict(sorted(reasons.items())),
                 'sample_type_counts': dict(sorted(Counter(q['type'] for q in questions).items())),
                 'selected_candidate_title_overlap_with_exposure': sum(
                     bool({d['title'] for d in q['documents']} & blocked_titles) for q in questions),
                 'no_document_disjoint_or_unexposed_backbone_claim': True,
                 'selection_rank': 'SHA256(seed,sample,id) within proportional type strata; SHA256(seed,order,id) execution order'}
    manifest = {'protocol': protocol, 'questions': questions, 'private': {'labels': labels}, 'selection': selection}
    manifest['manifest_sha256'] = digest(manifest)
    return manifest



def validate_manifest(manifest):
    """Fail closed on assignment/content corruption before inference is possible."""
    questions = manifest['questions']
    if not questions or len({q['id'] for q in questions}) != len(questions):
        raise ValueError('Manifest requires unique nonempty question set')
    coloc_owners, support_roles, support_positions = Counter(), Counter(), Counter()
    token_lengths, character_lengths = [], []
    for q in questions:
        if set(q['placements']) != {'colocated', 'distributed'}:
            raise ValueError('Missing E6 placement')
        documents = q['documents']
        union = {d['title']: canonical(d) for d in documents}
        if len(documents) != 6 or len(union) != 6:
            raise ValueError('Require the same six distinct source documents')
        label = manifest['private']['labels'][q['id']]
        supports = set(label['support_titles'])
        for placement, state in q['placements'].items():
            if len(state['assignments']) != 3 or sorted(state['roles']) != sorted(ROLES):
                raise ValueError('Require three agents and all three specialist roles')
            titles, counts = [], []
            for node, group in enumerate(state['assignments']):
                if len(group) != 2:
                    raise ValueError('Require two documents per agent')
                count = 0
                for slot, doc in enumerate(group):
                    if union.get(doc['title']) != canonical(doc):
                        raise ValueError('Assignment differs from frozen source content')
                    titles.append(doc['title'])
                    if doc['title'] in supports:
                        count += 1
                        support_roles[placement + ':' + state['roles'][node]] += 1
                        support_positions[placement + ':' + str(slot)] += 1
                counts.append(count)
                length = state['evidence_lengths'][node]
                character_lengths.append(length['characters'])
                if length['tokens'] is not None:
                    token_lengths.append(length['tokens'])
            if set(titles) != set(union) or len(titles) != len(set(titles)):
                raise ValueError('Placements do not preserve the source union')
            expected = [0, 0, 2] if placement == 'colocated' else [0, 1, 1]
            if sorted(counts) != expected:
                raise ValueError('Incorrect annotated support placement')
            if placement == 'colocated':
                coloc_owners[str(counts.index(2))] += 1
        if q['placements']['colocated']['roles'] != q['placements']['distributed']['roles']:
            raise ValueError('Role changed between paired placements')
    required = len(questions) * E6_COMPLETIONS_PER_QUESTION
    if manifest['protocol']['e6_successful_completions'] != required or required > 150:
        raise ValueError('Manifest request projection violates authorized count')
    if manifest['protocol']['n_questions'] != len(questions):
        raise ValueError('Manifest sample count differs')
    # Extra keys attached by downstream preparation may carry provenance; the
    # manifest hash binds the original builder output, rather than this report.
    return {'questions': len(questions), 'required_completions': required,
            'equal_union_and_source_content': True, 'two_documents_per_agent': True,
            'annotated_support_placement_compliance': 1.0,
            'colocated_owner_counts': dict(sorted(coloc_owners.items())),
            'support_article_role_counts': dict(sorted(support_roles.items())),
            'support_article_position_counts': dict(sorted(support_positions.items())),
            'local_evidence_characters_min_max': [min(character_lengths), max(character_lengths)],
            'local_evidence_tokens_min_max': [min(token_lengths), max(token_lengths)] if token_lengths else None}


def _response(response):
    if not isinstance(response, dict) or not isinstance(response.get('answer'), str) or not response['answer'].strip():
        raise ValueError('Response requires a nonempty string answer')
    citations = response.get('citations')
    if not isinstance(citations, list):
        raise ValueError('Response requires a citation list')
    citations = [list(c) for c in citations if isinstance(c, (list, tuple)) and len(c) == 2
                 and isinstance(c[0], str) and type(c[1]) is int]
    return {'answer': response['answer'].strip(), 'citations': citations}


def _messages(question, role, evidence, previous, peers):
    system = (
        'You are a ' + role + ' specialist answering multi-hop questions from Wikipedia source evidence. '
        'Use the local evidence and any source evidence supplied in peer messages. '
        'Peer answer proposals are unverified. Reconsider a previous response when one is supplied. '
        'Return ONLY JSON with answer (a nonempty concise string, or "noanswer" if unknown) '
        'and citations (a list of [document title, zero-based sentence index] pairs). '
        'Cite only source sentences actually shown in this request. Do not invent citations. '
        'Do not include explanations or confidence scores.'
    )
    payload = {'question': question, 'evidence': evidence, 'previous_response': previous, 'peer_messages': peers}
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': canonical(payload)}]


def initial_messages(question, placement, node):
    state = question['placements'][placement]
    return _messages(question['question'], state['roles'][node], _facts(state['assignments'][node]), None, [])


def revision_messages(question, placement, node, initial_responses, mode):
    if mode not in ('nocomm', 'broadcast') or len(initial_responses) != 3:
        raise ValueError('Require three frozen initial responses and a valid revision mode')
    state = question['placements'][placement]
    peers = []
    if mode == 'broadcast':
        for sender in range(3):
            if sender == node:
                continue
            response = _response(initial_responses[sender])
            cited = set(map(tuple, response['citations']))
            facts = [f for f in _facts(state['assignments'][sender])
                     if (f['title'], f['sentence_id']) in cited]
            peers.append({'sender': sender, 'answer': response['answer'], 'evidence': facts})
    return _messages(question['question'], state['roles'][node], _facts(state['assignments'][node]),
                     _response(initial_responses[node]), peers)


def majority(responses):
    """No gold access or LLM call; normalized ties use the fixed node sequence."""
    if not responses:
        raise ValueError('Cannot aggregate no responses')
    counts = Counter(_normalize(r['answer']) for r in responses)
    winner = max(counts, key=counts.get)
    return next(r['answer'] for r in responses if _normalize(r['answer']) == winner)


def interaction(scores):
    return ((scores['distributed_comm'] - scores['distributed_nocomm']) -
            (scores['colocated_comm'] - scores['colocated_nocomm']))


def _similar(a, b):
    if _normalize(a) == _normalize(b):
        return True
    left, right = set(re.findall(r'\w+', a.casefold())), set(re.findall(r'\w+', b.casefold()))
    return bool(left and right) and len(left & right) / len(left | right) >= 0.6


def build_e7(question, private_label, initial_responses, token_count=None):
    """Construct whole-sentence source controls without looking at response quality.

    Automatic support provenance is narrower than semantic non-support. Every
    accepted plan records this limitation and requires review before paid use.
    The pilot reserves no E7 paid calls; this function supports local validation.
    """
    if len(initial_responses) != 3:
        raise ValueError('E7 requires three frozen distributed initial responses')
    for r in initial_responses:
        _response(r)
    placement = question['placements']['distributed']
    support = set(map(tuple, private_label['supporting_facts']))
    support_titles = set(private_label['support_titles'])
    facts = _facts(question['documents'])
    support_facts = [f for f in facts if (f['title'], f['sentence_id']) in support]
    candidates = []
    for receiver in range(3):
        visible = {d['title'] for d in placement['assignments'][receiver]}
        local_support = [f for f in support_facts if f['title'] in visible]
        local_facts = _facts(placement['assignments'][receiver])
        missing = [f for f in support_facts if f['title'] not in visible
                   and not any(_similar(f['text'], seen['text']) for seen in local_facts)]
        if len(visible & support_titles) != 1:
            continue
        distractors = [f for f in facts if f['title'] not in support_titles and f['title'] not in visible]
        for relevant, redundant, distractor in itertools.product(missing, local_support, distractors):
            triple = [relevant, redundant, distractor]
            if any(_similar(a['text'], b['text']) for a, b in itertools.combinations(triple, 2)):
                continue
            # Neither control may repeat another annotated missing supporting sentence.
            if any(_similar(control['text'], f['text']) for control in (redundant, distractor) for f in missing):
                continue
            lengths = [token_count(canonical([f])) if token_count else len(f['text'].split()) for f in triple]
            if min(lengths) <= 0 or max(lengths) / min(lengths) > 1.5:
                continue
            # Whole-source-sentence matching; no clipping or synthetic filler.
            candidates.append((max(lengths) / min(lengths), max(lengths) - min(lengths),
                               digest([question['id'], receiver, triple]), receiver, triple))
    if not candidates:
        return {'id': question['id'], 'eligible': False,
                'reason': 'NO_DISTINCT_WHOLE_SENTENCE_TRIPLE_WITH_LENGTH_RATIO_AT_MOST_1_5'}
    _, _, _, receiver, triple = min(candidates)
    arms = {}
    for name, fact in zip(('relevant', 'redundant', 'distractor'), triple):
        arms[name] = {'facts': [copy.deepcopy(fact)], 'length': _length(canonical([fact]), token_count)}
    plan = {'id': question['id'], 'eligible': True, 'receiver': receiver, 'placement': 'distributed',
            'arms': arms, 'frozen_initial_sha256': digest(_response(initial_responses[receiver])),
            'one_source_slot': True, 'updates_per_arm': 1,
            'length_matching_metric': 'exact_packet_tokens' if token_count else 'sentence_words_proxy',
            'maximum_length_ratio': 1.5,
            'semantic_nonsupport_verified': False,
            'paid_ready': False,
            'limitations': ['Non-annotated distractor status does not prove semantic irrelevance.',
                            'Lexical distinctness checks cannot prove absence of paraphrased supporting facts.',
                            'Manual blind source-content review is required before paid E7; pilot E7 is deferred.',
                            'Whole-sentence length matching permits residual packet-token differences; report all differences.']}
    plan['plan_sha256'] = digest(plan)
    return plan


def e7_messages(question, plan, initial_response, arm):
    if not plan.get('eligible') or arm not in ('relevant', 'redundant', 'distractor'):
        raise ValueError('Require an eligible E7 plan and valid arm')
    own = _response(initial_response)
    if digest(own) != plan['frozen_initial_sha256']:
        raise ValueError('E7 receiver initial state changed')
    receiver = plan['receiver']
    placement = question['placements']['distributed']
    # Treatment labels, construction annotations and source role are never shown.
    peers = [{'sender': 'evidence_source', 'evidence': copy.deepcopy(plan['arms'][arm]['facts'])}]
    return _messages(question['question'], placement['roles'][receiver],
                     _facts(placement['assignments'][receiver]), own, peers)
