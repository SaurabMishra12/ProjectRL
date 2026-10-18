import itertools
import pytest
from WEBExtension.causal_study.external_validation import contracts as c


def p(text,kind='answer'):return c.Proposal(text,kind)


def test_contract_semantics_are_not_interchangeable():
    assert c.hotpot_proposal({'answer':'noanswer'}).kind=='abstain'
    assert c.hotpot_proposal({'answer':'','provider_refusal':True}).kind=='refusal'
    assert c.hotpot_proposal({'answer':''}).kind=='empty'
    assert c.hotpot_proposal({}).kind=='missing'
    assert c.hotpot_proposal({'answer':None}).kind=='malformed'
    assert c.hotpot_proposal({'answer':'A'}).kind=='answer'
    with pytest.raises(ValueError):c.hotpot_proposal({'answer':'Paris','provider_refusal':True})


def test_abstention_aware_is_unsafe_for_unsupported_questions():
    proposals=[p('noanswer','abstain'),p('noanswer','abstain'),p('Paris')]
    assert c.readout(proposals,'original').kind=='abstain'
    assert c.readout(proposals,'aware').text=='Paris'
    assert c.readout(proposals,'strict_majority').kind=='abstain'
    rows=[{'answerable':False,'correct':False,'decision':c.readout(proposals,'aware')}]
    assert c.metrics(rows)['false_answer_rate']==1


def test_majority_denominator_includes_unavailable_seats():
    a=[p('Paris'),p('noanswer','abstain'),p('','refusal')]
    assert c.readout(a,'aware').text=='Paris'
    assert c.readout(a,'strict_majority').kind=='abstain'
    assert c.readout([p('Paris'),p('Paris'),p('','missing')],'strict_majority').text=='Paris'


def test_lexical_ties_are_order_invariant():
    for perm in itertools.permutations([p('Zebra'),p('Paris'),p('noanswer','abstain')]):
        assert c.readout(perm,'lexical').text=='Paris'


def test_original_preserves_first_seat_and_refusal_empty_category():
    assert c.readout([p('Zebra'),p('Paris'),p('Berlin')],'original').text=='Zebra'
    assert c.readout([p('','refusal'),p('','refusal'),p('Paris')],'original').kind=='refusal'


def test_empty_and_missing_are_not_semantic_abstentions():
    result=c.readout([p('','missing'),p('','empty'),p('bad','malformed')],'aware')
    assert result.kind=='invalid'
    assert c.metrics([{'answerable':False,'correct':False,'decision':result}])['valid_abstention_rate_unsupported']==0


def test_normalized_empty_is_a_valid_answer():
    assert c.readout([p('A'),p('A'),p('noanswer','abstain')],'aware').kind=='answer'


def test_risk_coverage_handles_no_answers_and_missing_labels():
    m=c.metrics([{'answerable':True,'correct':False,'decision':p('noanswer','abstain')}])
    assert m['coverage']==0 and m['selective_risk'] is None and m['false_answer_rate'] is None
    with pytest.raises(ValueError):c.metrics([{'answerable':None,'correct':False,'decision':p('Paris')}])


def test_gsm_none_is_a_parse_failure_not_expressed_abstention():
    assert c.gsm_proposal(None).kind=='malformed'
    assert c.gsm_proposal('12').kind=='answer'
    assert c.readout([c.gsm_proposal(None),p('2'),p('3')],'original',normalize=str).text=='2'


def test_interaction_keeps_both_contexts_within_question_cluster():
    # Per-question interaction: ((CA-PA)-(CO-PO)), averaged over paired contexts.
    rows=[{'id':'q','context':'sufficient','CA':1,'PA':0,'CO':0,'PO':0},
          {'id':'q','context':'insufficient','CA':0,'PA':1,'CO':0,'PO':0}]
    assert c.cluster_interactions(rows)=={'q':0}
    with pytest.raises(ValueError):c.cluster_interactions(rows+rows[:1])


def test_musique_boolean_answerability_is_separate_from_world_answer():
    assert c.musique_proposal({'predicted_answer':'Paris','predicted_answerable':False}).kind=='abstain'
    assert c.musique_proposal({'predicted_answer':'Paris','predicted_answerable':True}).kind=='answer'
    assert c.musique_proposal({'predicted_answer':'Paris','predicted_answerable':'false'}).kind=='malformed'
    assert c.musique_proposal({'predicted_answer':'','predicted_answerable':True}).kind=='empty'


def test_all_provider_refusals_are_not_credited_as_semantic_abstention():
    result=c.readout([p('','refusal')]*3,'aware')
    assert result.kind=='refusal'
    assert c.metrics([{'answerable':False,'correct':False,'decision':result}])['valid_abstention_rate_unsupported']==0


def test_musique_explicit_contract_rejects_true_noanswer_and_bad_citations():
    assert c.musique_proposal({'predicted_answer':'noanswer','predicted_answerable':True}).kind=='malformed'
    assert c.musique_proposal({'predicted_answer':'Paris','predicted_answerable':True,'predicted_support_idxs':[99]},source_ids={0,1}).kind=='malformed'
    assert c.musique_proposal({'predicted_answer':'Paris','predicted_answerable':True,'predicted_support_idxs':[0]},source_ids={0,1}).kind=='answer'


def test_power_support_is_realizable_by_the_frozen_readouts():
    a=p('noanswer','abstain');correct=p('Paris');wrong=p('London')
    # Each context's four cells are evaluated from the same frozen arm proposals.
    def utility(proposals,rule,sufficient):
        d=c.readout(proposals,rule)
        return int(d.kind=='answer' and d.text=='Paris') if sufficient else int(d.kind=='abstain')
    arms={True:([a,a,correct],[correct]*3),False:([a]*3,[a,a,wrong])}
    values=[]
    for sufficient,(peer,private) in arms.items():
        values.append((utility(peer,'aware',sufficient)-utility(private,'aware',sufficient))-(utility(peer,'original',sufficient)-utility(private,'original',sufficient)))
    assert sum(values)/2==1


def test_prospective_arm_failure_has_zero_utility_under_every_readout():
    ps=[p('Paris'),p('Paris'),p('','missing')]
    for rule in c.RULES:
        assert c.prospective_utility(ps,rule,True,['Paris'])==0
        assert c.prospective_utility(ps,rule,False,['Paris'])==0
        assert c.prospective_utility([p('Paris')]*3,rule,True,['Paris'],upstream_failed=True)==0
    assert c.prospective_utility([p('Paris')]*3,'aware',True,['Paris'])==1


def test_musique_provider_envelope_rejects_bad_metadata_and_conflicts():
    with pytest.raises(ValueError):c.musique_proposal({'provider_refusal':'true','predicted_answerable':True,'predicted_answer':'Paris'})
    with pytest.raises(ValueError):c.musique_proposal({'provider_refusal':True,'predicted_answerable':True,'predicted_answer':'Paris'})
    assert c.musique_proposal({'provider_refusal':True}).kind=='refusal'


def test_question_unit_deduplication_is_gold_free_and_preserves_two_contexts():
    from WEBExtension.causal_study.external_validation.reanalyse import ranked_question_units
    rows=[{'id':'q1','question':'Who did this?','answer':'A','answerable':True},
          {'id':'q1','question':'Who did this?','answer':'A','answerable':False},
          {'id':'q2','question':'Who did this?','answer':'B','answerable':True},
          {'id':'q2','question':'Who did this?','answer':'B','answerable':False},
          {'id':'q3','question':'What is this?','answer':'C','answerable':True}]
    ids,groups=ranked_question_units(rows)
    changed=[dict(r,answer='wrong',answerable=not r['answerable']) for r in rows]
    assert ranked_question_units(changed)==(ids,groups)
    assert len(ids)==2 and len(set(ids)&{'q1','q2'})==1
    assert any(set(g)=={'q1','q2'} for g in groups)
