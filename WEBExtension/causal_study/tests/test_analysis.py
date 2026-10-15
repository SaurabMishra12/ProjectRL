from WEBExtension.causal_study.analyze import proposal_utilization


def test_new_exact_answer_is_distinguished_from_selecting_existing_proposal():
    records=[{'question_id':'q','initial':{
        'colocated':[{'answer':'Paris'},{'answer':'noanswer'},{'answer':'noanswer'}],
        'distributed':[{'answer':'Lyon'},{'answer':'noanswer'},{'answer':'noanswer'}]},
        'cells':{'colocated_comm':{'answer':'Paris','f1':100,'any_revised_exact':True},
                 'distributed_comm':{'answer':'Paris','f1':100,'any_revised_exact':True}}}]
    labels={'q':{'answer':'Paris'}}
    rows=proposal_utilization(records,labels)
    assert rows==[
        {'question_id':'q','placement':'colocated','initial_any_exact':True,'broadcast_any_exact':True,
         'broadcast_selected_initial_proposal':True,'broadcast_equals_initial_oracle_best_f1':True,
         'broadcast_introduced_exact_without_initial_exact':False,'initial_oracle_best_f1':100.0,'broadcast_f1':100},
        {'question_id':'q','placement':'distributed','initial_any_exact':False,'broadcast_any_exact':True,
         'broadcast_selected_initial_proposal':False,'broadcast_equals_initial_oracle_best_f1':False,
         'broadcast_introduced_exact_without_initial_exact':True,'initial_oracle_best_f1':0.0,'broadcast_f1':100}]
