import pytest

def records(split):
    return [{'question_key':split,'split':split,'mode':'MOCK','feature_version':'observable_web_v2_32','prompt_version':'web_prompt_v1','features':[[0.0]*32,[0.1]*32],'sink_features':[0.0]*32,'outcomes':[{'em':100,'input_tokens':10,'output_tokens':3,'calls':1},{'em':0,'input_tokens':10,'output_tokens':3,'calls':1}],'cost_records':[{'key':split,'input_tokens':20,'output_tokens':6,'calls':2}]}]

def settings():
    return dict(optimizer='adam',learning_rate=.001,epochs=2,seeds=[7],lambda_input_tokens=.01,lambda_output_tokens=.01,lambda_calls=.01,input_normalizer=100,output_normalizer=100,call_normalizer=1,feature_version='observable_web_v2_32',prompt_version='web_prompt_v1',evaluation_keys=['heldout'],engineering=True)

def test_training_selection_save_reload_and_leakage(tmp_path):
    from WEBExtension.web_reasoning.training import train_router,load_checkpoint
    c=train_router(records('train'),records('validation'),settings())
    p=tmp_path/'router.npz';c.save(p)
    model,meta=load_checkpoint(p,'observable_web_v2_32','web_prompt_v1',mode='MOCK')
    assert meta['mode']=='MOCK' and meta['costs']['input_tokens']==40
    with pytest.raises(ValueError):load_checkpoint(p,'wrong','web_prompt_v1',mode='MOCK')
    with pytest.raises(ValueError):load_checkpoint(p,'observable_web_v2_32','web_prompt_v1',mode='REAL')
    bad=records('train');bad[0]['question_key']='heldout'
    with pytest.raises(ValueError):train_router(bad,records('validation'),settings())

def test_optimizer_must_be_declared():
    from WEBExtension.web_reasoning.training import train_router
    s=settings();s.pop('optimizer',None)
    with pytest.raises(ValueError):train_router(records('train'),records('validation'),s)
