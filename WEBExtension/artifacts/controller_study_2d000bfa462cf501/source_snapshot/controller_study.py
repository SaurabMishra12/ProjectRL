"""Real-data PPO collection, validation-only selection and frozen held-out policy comparison."""
from __future__ import annotations
import json
import os
import shutil
from pathlib import Path
import numpy as np
import torch
from filelock import FileLock
from protocol import ROOT, digest, canonical, file_hash, write_json, exact_slice, message_text, stable_rng, ExactTokenizer
from experiments import initialize, update, no_peer_update, condition, measured_cost
from backend import RealBackend
from controller_splits import prepare_splits, load_partition
from learned_controller import FEATURE_VERSION, action_space, observable_features, CategoricalController, ppo_update

def study_directory(base,settings):
    splits=prepare_splits(settings)
    sources={p.name:file_hash(p) for p in ROOT.glob('*.py') if 'build_notebooks' not in p.name}
    plan={'base':base,'settings':settings,'splits_sha256':file_hash(splits/'manifest.json'),
          'sources':sources,'feature_version':FEATURE_VERSION,'actions':action_space(settings['quotas']),
          'scientific_scope':'one-step categorical contextual-bandit SparseComm PPO; no multi-round stopping attribution'}
    directory=ROOT/'artifacts'/('controller_study_'+digest(plan)[:16]);directory.mkdir(parents=True,exist_ok=True)
    manifest=directory/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=plan:raise ValueError('Study plan changed')
    if not manifest.exists():
        write_json(manifest,plan)
        snapshot=directory/'source_snapshot';snapshot.mkdir()
        for filename in sources:shutil.copyfile(ROOT/filename,snapshot/filename)
    return directory,splits

def forecast(base,settings):
    directory,splits=study_directory(base,settings)
    n=settings['train_questions'];v=settings['validation_questions'];t=settings['test_questions']
    seeds=len(settings['training_seeds']);epochs=settings['collection_epochs'];regimes=len(settings['regimes'])
    upper=regimes*(3*(n+v+t)+3*n*epochs*seeds+3*v*epochs*seeds+3*v*len(settings['static_candidates'])
                   +t*(3*seeds+3*seeds+9))
    return {'study_dir':str(directory),'split_dir':str(splits),'train':n,'validation':v,'heldout':t,
            'seeds':settings['training_seeds'],'conservative_max_calls_before_reuse':upper,
            'configured_operational_cap':settings['max_physical_calls'],
            'note':'Upper bound; identical initial and receiver prompts reuse real cached responses. Pilot costs are separate.'}

def runtime(base,settings,directory):
    cfg={**base,'replicate_id':'controller_frozen_environment','max_calls':settings['max_physical_calls']}
    tokenizer=ExactTokenizer(cfg['tokenizer'],cfg['tokenizer_revision'])
    return cfg,tokenizer,RealBackend(cfg,tokenizer,directory/'inference')

def apply_action(inputs,state,action,cfg,tokenizer,backend):
    if not action['edges']:return state['responses'],[]
    selected=[exact_slice(message_text(r,state['pool']),action['quota'],tokenizer,'prefix',cfg['crop_seed'])
              for r in state['responses']]
    return update(inputs,state,action['edges'],selected,backend,'controller_update')

def observed_reward(result,baseline,state,tokenizer,settings):
    full=2*sum(tokenizer.count(message_text(r,state['pool'])) for r in state['responses'])
    receiver_calls=result['cost']['logical_calls']-3
    return ((result['f1']-baseline['f1'])/100 - settings['lambda_payload']*result['payload_tokens']/max(1,full)
            - settings['lambda_calls']*receiver_calls/3)

def policy_result(inputs,gold,state,action,cfg,tokenizer,backend,settings):
    responses,deliveries=apply_action(inputs,state,action,cfg,tokenizer,backend)
    baseline=condition(gold,state,state['responses'],[],include_initial=False)
    result=condition(gold,state,responses,deliveries,include_initial=bool(action['edges']))
    result['action']=action
    result['utility_gain']=observed_reward(result,baseline,state,tokenizer,settings)
    return result

def append(path,record):
    with open(path,'a') as f:f.write(canonical(record)+'\n');f.flush();os.fsync(f.fileno())

def records(path):
    if not Path(path).exists():return []
    rows=[json.loads(l) for l in Path(path).read_text().splitlines() if l]
    if len({r['key'] for r in rows})!=len(rows):raise ValueError('Duplicate study checkpoint')
    return rows

def save_checkpoint(path,model,optimizer,epoch,phase,feature_width,actions,plan_hash):
    obj={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch,'phase':phase,
         'feature_width':feature_width,'n_actions':len(actions),'actions':actions,'torch_rng':torch.get_rng_state(),
         'plan_hash':plan_hash,'feature_version':FEATURE_VERSION}
    temp=Path(str(path)+'.tmp');torch.save(obj,temp);temp.replace(path)

def load_checkpoint(path,plan_hash,hidden):
    obj=torch.load(path,map_location='cpu',weights_only=True)
    if obj['plan_hash']!=plan_hash or obj['feature_version']!=FEATURE_VERSION:raise ValueError('Checkpoint/protocol mismatch')
    model=CategoricalController(obj['feature_width'],obj['n_actions'],hidden)
    model.load_state_dict(obj['model']);model.eval()
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4);optimizer.load_state_dict(obj['optimizer'])
    return model,optimizer,obj

def validate_policy(model,path,inputs,golds,cfg,tokenizer,backend,settings,actions):
    done={r['key']:r for r in records(path)}
    for row in inputs:
        for regime in settings['regimes']:
            key=row['id']+'_'+regime
            if key in done:continue
            state=initialize(row,regime,cfg,backend)
            features=observable_features(row['question'],state,tokenizer)
            index,_,_=model.choose(features,deterministic=True)
            result=policy_result(row,golds[row['id']],state,actions[index],cfg,tokenizer,backend,settings)
            record={'key':key,'id':row['id'],'regime':regime,'action_index':index,'result':result}
            append(path,record);done[key]=record
    expected={r['id']+'_'+g for r in inputs for g in settings['regimes']}
    if set(done)!=expected:raise ValueError('Validation records do not match frozen validation sample')
    return {'mean_utility_gain':float(np.mean([r['result']['utility_gain'] for r in done.values()])),
            'mean_f1':float(np.mean([r['result']['f1'] for r in done.values()])),'n_contexts':len(done)}

def train_seed(seed,directory,splits,cfg,tokenizer,backend,settings,plan_hash):
    folder=directory/f'seed_{seed}';folder.mkdir(exist_ok=True)
    actions=action_space(settings['quotas']);train=load_partition(splits,'train')
    golds={r['id']:r for r in load_partition(splits,'train',gold=True)}
    val=load_partition(splits,'validation');val_gold={r['id']:r for r in load_partition(splits,'validation',gold=True)}
    latest=folder/'latest.pt';selection_path=folder/'selection.json'
    if latest.exists():
        model,optimizer,obj=load_checkpoint(latest,plan_hash,settings['hidden']);torch.set_rng_state(obj['torch_rng'])
    else:
        torch.manual_seed(seed)
        state=initialize(train[0],settings['regimes'][0],cfg,backend)
        width=len(observable_features(train[0]['question'],state,tokenizer))
        model=CategoricalController(width,len(actions),settings['hidden'])
        optimizer=torch.optim.Adam(model.parameters(),lr=settings['learning_rate'])
        save_checkpoint(latest,model,optimizer,0,'collect',width,actions,plan_hash)
        obj=torch.load(latest,map_location='cpu',weights_only=True)
    selection=json.loads(selection_path.read_text()) if selection_path.exists() else None
    for epoch in range(obj['epoch'],settings['collection_epochs']):
        if obj['phase']=='collect':
            path=folder/f'rollout_epoch_{epoch}.jsonl';saved=records(path);done={r['key']:r for r in saved}
            if saved:torch.set_rng_state(torch.tensor(saved[-1]['torch_rng_after'],dtype=torch.uint8))
            for row in train:
                for regime in settings['regimes']:
                    key=row['id']+'_'+regime
                    if key in done:continue
                    state=initialize(row,regime,cfg,backend)
                    features=observable_features(row['question'],state,tokenizer)
                    index,logp,value=model.choose(features,deterministic=False)
                    result=policy_result(row,golds[row['id']],state,actions[index],cfg,tokenizer,backend,settings)
                    record={'key':key,'id':row['id'],'regime':regime,'features':features.tolist(),'action_index':index,
                            'old_logp':logp,'old_value':value,'reward':result['utility_gain'],'result':result,
                            'torch_rng_after':torch.get_rng_state().tolist()}
                    append(path,record);done[key]=record
            expected={r['id']+'_'+g for r in train for g in settings['regimes']}
            if set(done)!=expected:raise ValueError('Training rollout sample changed/incomplete')
            batch=list(done.values())
            data={'features':torch.tensor([r['features'] for r in batch]),
                  'actions':torch.tensor([r['action_index'] for r in batch],dtype=torch.long),
                  'old_logp':torch.tensor([r['old_logp'] for r in batch]),
                  'old_values':torch.tensor([r['old_value'] for r in batch]),
                  'rewards':torch.tensor([r['reward'] for r in batch])}
            model.train();stats=ppo_update(model,optimizer,data,epochs=settings['ppo_epochs'],batch_size=settings['batch_size'],
                                          clip=settings['clip'],entropy_coef=settings['entropy_coef'],target_kl=settings['target_kl'])
            model.eval();write_json(folder/f'optimizer_epoch_{epoch}.json',stats)
            save_checkpoint(latest,model,optimizer,epoch,'validate',obj['feature_width'],actions,plan_hash)
        metrics=validate_policy(model,folder/f'validation_epoch_{epoch}.jsonl',val,val_gold,cfg,tokenizer,backend,settings,actions)
        if selection is None or metrics['mean_utility_gain']>selection['metrics']['mean_utility_gain']:
            chosen=folder/'selected.pt';shutil.copyfile(latest,chosen)
            selection={'seed':seed,'epoch':epoch,'metrics':metrics,'checkpoint':str(chosen),
                       'checkpoint_sha256':file_hash(chosen),'plan_hash':plan_hash,'selection_sample':'validation only'}
            write_json(selection_path,selection)
        write_json(folder/f'validation_metrics_epoch_{epoch}.json',metrics)
        save_checkpoint(latest,model,optimizer,epoch+1,'collect',obj['feature_width'],actions,plan_hash)
        obj={'epoch':epoch+1,'phase':'collect','feature_width':obj['feature_width']}
        print(f'Controller seed {seed}: epoch {epoch+1}/{settings["collection_epochs"]}; validation F1={metrics["mean_f1"]:.2f}',flush=True)
    if selection is None:raise ValueError('No complete validation-selected checkpoint')
    if file_hash(Path(selection['checkpoint']))!=selection['checkpoint_sha256']:raise ValueError('Selected checkpoint changed')
    write_json(folder/'training_status.json',{'status':'COMPLETE','selected_checkpoint_sha256':selection['checkpoint_sha256']})
    return selection

def static_actions(settings):
    supports={'CW':[[0,1],[1,2],[2,0]],'CCW':[[0,2],[2,1],[1,0]],'OUTSTAR':[[0,1],[0,2]]}
    return [{'name':name,'edges':supports[name],'quota':max(settings['quotas'])} for name in settings['static_candidates']]

def select_static(directory,splits,cfg,tokenizer,backend,settings):
    path=directory/'static_validation.jsonl';done={r['key']:r for r in records(path)}
    rows=load_partition(splits,'validation');golds={r['id']:r for r in load_partition(splits,'validation',gold=True)}
    for row in rows:
        for regime in settings['regimes']:
            state=initialize(row,regime,cfg,backend)
            for action in static_actions(settings):
                key=row['id']+'_'+regime+'_'+action['name']
                if key in done:continue
                result=policy_result(row,golds[row['id']],state,action,cfg,tokenizer,backend,settings)
                record={'key':key,'action':action,'result':result};append(path,record);done[key]=record
    expected={r['id']+'_'+g+'_'+a['name'] for r in rows for g in settings['regimes'] for a in static_actions(settings)}
    if set(done)!=expected:raise ValueError('Static validation incomplete')
    scores={a['name']:float(np.mean([r['result']['utility_gain'] for r in done.values() if r['action']['name']==a['name']])) for a in static_actions(settings)}
    selected=max(static_actions(settings),key=lambda a:scores[a['name']])
    result={'action':selected,'validation_utility_by_candidate':scores,'selection':'prespecified candidates; validation only'}
    write_json(directory/'static_selection.json',result);return result

def train(base,settings,execute=False):
    if not execute:return forecast(base,settings)
    directory,splits=study_directory(base,settings)
    with FileLock(str(directory/'study.lock'),timeout=0):
        if (directory/'evaluation_gate.json').exists():raise RuntimeError('Held-out gate frozen: training/selection is closed for this study')
        cfg,tokenizer,backend=runtime(base,settings,directory)
        plan_hash=file_hash(directory/'manifest.json')
        try:
            write_json(directory/'training_status.json',{'status':'RUNNING'})
            selections=[train_seed(s,directory,splits,cfg,tokenizer,backend,settings,plan_hash) for s in settings['training_seeds']]
            static=select_static(directory,splits,cfg,tokenizer,backend,settings)
            write_json(directory/'training_status.json',{'status':'COMPLETE','selections':selections,'static_selection':static})
        except Exception as e:
            write_json(directory/'training_status.json',{'status':'INCOMPLETE','error_type':type(e).__name__});raise
    return str(directory)

def selected_models(directory,settings):
    plan_hash=file_hash(directory/'manifest.json');models={};checks={}
    for seed in settings['training_seeds']:
        folder=directory/f'seed_{seed}'
        if not (folder/'training_status.json').exists() or json.loads((folder/'training_status.json').read_text())['status']!='COMPLETE':
            raise ValueError('Training seed is not complete')
        selection=json.loads((folder/'selection.json').read_text());path=Path(selection['checkpoint'])
        if file_hash(path)!=selection['checkpoint_sha256']:raise ValueError('Validation-selected checkpoint changed')
        models[seed],_,_=load_checkpoint(path,plan_hash,settings['hidden']);checks[str(seed)]=selection['checkpoint_sha256']
    return models,checks

def evaluate(base,settings,execute=False):
    if not execute:return forecast(base,settings)
    directory,splits=study_directory(base,settings)
    with FileLock(str(directory/'study.lock'),timeout=0):
        status=directory/'training_status.json'
        if not status.exists() or json.loads(status.read_text())['status']!='COMPLETE':raise ValueError('Complete training/selection before held-out evaluation')
        models,hashes=selected_models(directory,settings)
        static=json.loads((directory/'static_selection.json').read_text())['action']
        gate={'plan_sha256':file_hash(directory/'manifest.json'),'checkpoint_hashes':hashes,
              'static_selection_sha256':file_hash(directory/'static_selection.json'),'split_manifest_sha256':file_hash(splits/'manifest.json')}
        gate_path=directory/'evaluation_gate.json'
        if gate_path.exists() and json.loads(gate_path.read_text())!=gate:raise ValueError('Frozen evaluation gate differs')
        if not gate_path.exists():write_json(gate_path,gate)
        cfg,tokenizer,backend=runtime(base,settings,directory)
        # Held-out labels enter only now, after checkpoint/action-space/static selection are frozen.
        rows=load_partition(splits,'heldout');golds={r['id']:r for r in load_partition(splits,'heldout',gold=True)}
        path=directory/'E5.jsonl';done={r['key']:r for r in records(path)}
        try:
            write_json(directory/'E5_status.json',{'status':'RUNNING','completed':len(done)})
            actions=action_space(settings['quotas'])
            for row in rows:
                key=row['id']
                if key in done:continue
                record={'key':key,'id':key,'type':golds[key]['type'],'conditions':{},'learned_action_indices':{}}
                for regime in settings['regimes']:
                    state=initialize(row,regime,cfg,backend);features=observable_features(row['question'],state,tokenizer)
                    stop={'name':'NoComm','edges':[],'quota':0}
                    record['conditions'][regime+'_NoComm']=policy_result(row,golds[key],state,stop,cfg,tokenizer,backend,settings)
                    record['conditions'][regime+'_Static']=policy_result(row,golds[key],state,static,cfg,tokenizer,backend,settings)
                    broadcast={'name':'Broadcast','edges':[[s,t] for s in range(3) for t in range(3) if s!=t], 'quota':max(settings['quotas'])}
                    record['conditions'][regime+'_Broadcast']=policy_result(row,golds[key],state,broadcast,cfg,tokenizer,backend,settings)
                    revised=no_peer_update(row,state,backend)
                    result=condition(golds[key],state,revised,[])
                    result['action']={'name':'SelfUpdate','edges':[],'quota':0}
                    initial=record['conditions'][regime+'_NoComm']
                    result['utility_gain']=observed_reward(result,initial,state,tokenizer,settings)
                    record['conditions'][regime+'_SelfUpdate']=result
                    for seed,model in models.items():
                        index,_,_=model.choose(features,deterministic=True);action=actions[index]
                        learned=policy_result(row,golds[key],state,action,cfg,tokenizer,backend,settings)
                        record['conditions'][regime+f'_Learned_seed{seed}']=learned
                        record['learned_action_indices'][regime+f'_seed{seed}']=index
                        eligible=[a for a in actions if len(a['edges'])==len(action['edges']) and a['quota']==action['quota']]
                        random_action=stable_rng('matched_random',key,regime,seed).choice(eligible)
                        record['conditions'][regime+f'_MatchedRandom_seed{seed}']=policy_result(row,golds[key],state,random_action,cfg,tokenizer,backend,settings)
                append(path,record);done[key]=record
                print(f'Held-out controller evaluation: {len(done)}/{len(rows)}',flush=True)
            if set(done)!={r['id'] for r in rows}:raise ValueError('Incomplete/changed held-out sample')
            write_json(directory/'E5_status.json',{'status':'COMPLETE','n_questions':len(rows),'sha256':file_hash(path),'gate':gate})
        except Exception as e:
            write_json(directory/'E5_status.json',{'status':'INCOMPLETE','completed':len(done),'error_type':type(e).__name__});raise
    return str(directory)

def pilot(base,settings,execute=False):
    if not execute:return {'pilot_questions':settings['pilot_questions'],'status':'NOT_RUN','note':'Real inference only; use pilot to inspect format/cost, not favorable scores'}
    directory,splits=study_directory(base,settings)
    with FileLock(str(directory/'study.lock'),timeout=0):
        cfg,tokenizer,backend=runtime(base,settings,directory/'pilot')
        rows=load_partition(splits,'pilot');golds={r['id']:r for r in load_partition(splits,'pilot',gold=True)}
        path=directory/'pilot.jsonl';done={r['key']:r for r in records(path)}
        for row in rows:
            if row['id'] in done:continue
            out={'key':row['id'],'results':{}}
            for regime in settings['regimes']:
                state=initialize(row,regime,cfg,backend)
                out['results'][regime]=[policy_result(row,golds[row['id']],state,a,cfg,tokenizer,backend,settings) for a in static_actions(settings)]
            append(path,out);done[row['id']]=out
        write_json(directory/'pilot_status.json',{'status':'COMPLETE','n_questions':len(done),'sha256':file_hash(path),
                   'purpose':'Invariants, parse validity and measured cost; not primary research results'})
    return str(directory)
