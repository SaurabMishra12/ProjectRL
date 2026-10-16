def test_unknown_usage_is_not_zero():
    from WEBExtension.web_reasoning.accounting import cost_summary
    rows=[{'event':'started','attempt_id':'1','key':'x'}, {'event':'success','attempt_id':'1','key':'x','usage':None}]
    r=cost_summary(rows,[{'key':'x','arm':'a'},{'key':'x','arm':'b'}])
    assert r['physical_attempts']==1 and r['logical_references']==2
    assert r['input_usage_missing']==1 and r['known_input_tokens_lower_bound']==0
