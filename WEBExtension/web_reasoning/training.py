"""Offline matched-utility regression. This module never generates missing outcomes."""
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
import numpy as np
import torch
from .learned import PooledSenderScorer,FEATURE_VERSION
from .storage import digest,fsync_dir


@dataclass
class Checkpoint:
    weights:dict
    metadata:dict
    validation_utility:float
    sha256:str

    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        temporary=None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as f:
                temporary=Path(f.name)
                np.savez(f,metadata=np.array(json.dumps(self.metadata,sort_keys=True)),sha256=np.array(self.sha256),**self.weights)
                f.flush();os.fsync(f.fileno())
            os.replace(temporary,path);fsync_dir(path.parent)
        finally:
            if temporary and temporary.exists():temporary.unlink()


def checkpoint_hash(weights,metadata):return digest({'weights':{k:v.tolist() for k,v in weights.items()},'metadata':metadata})


def load_checkpoint(path,expected_feature_version,expected_prompt_version,mode='REAL'):
    with np.load(path,allow_pickle=False) as saved:
        metadata=json.loads(saved['metadata'].item());expected=saved['sha256'].item()
        weights={k:np.array(saved[k]) for k in saved.files if k not in ('metadata','sha256')}
    if checkpoint_hash(weights,metadata)!=expected:raise ValueError('checkpoint_hash_mismatch')
    if metadata['feature_version']!=expected_feature_version or metadata['prompt_version']!=expected_prompt_version or metadata['mode']!=mode:
        raise ValueError('incompatible_checkpoint_contract')
    model=PooledSenderScorer();model.load_state_dict({k:torch.tensor(v) for k,v in weights.items()},strict=True);model.eval()
    return model,metadata


def choose_checkpoint(validation_scores,tie_rule='sha256'):
    if not validation_scores or tie_rule!='sha256':raise ValueError('invalid_checkpoint_selection')
    return min(validation_scores,key=lambda c:(-c.validation_utility,c.sha256))


def train_router(training_records,validation_records,settings):
    required={'learning_rate','epochs','seeds','lambda_input_tokens','lambda_output_tokens','lambda_calls',
              'input_normalizer','output_normalizer','call_normalizer','feature_version','prompt_version','evaluation_keys','engineering','optimizer'}
    if not required<=set(settings) or settings.get('optimizer','adam')!='adam':raise ValueError('complete_training_settings_required')
    if type(settings['epochs']) is not int or not 1<=settings['epochs']<=10000 or not settings['seeds'] or len(settings['seeds'])!=len(set(settings['seeds'])):
        raise ValueError('invalid_training_schedule')
    for key in ('learning_rate','input_normalizer','output_normalizer','call_normalizer'):
        if not math.isfinite(settings[key]) or settings[key]<=0:raise ValueError('positive_training_scale_required')
    for key in ('lambda_input_tokens','lambda_output_tokens','lambda_calls'):
        if not math.isfinite(settings[key]) or settings[key]<0:raise ValueError('nonnegative_penalty_required')
    if settings['feature_version']!=FEATURE_VERSION or type(settings['engineering']) is not bool:raise ValueError('invalid_feature_contract')
    train_keys={r['question_key'] for r in training_records};val_keys={r['question_key'] for r in validation_records}
    if not training_records or not validation_records or len(train_keys)!=len(training_records) or len(val_keys)!=len(validation_records) or train_keys&val_keys or (train_keys|val_keys)&set(settings['evaluation_keys']):
        raise ValueError('training_validation_evaluation_overlap')
    mode='MOCK' if settings['engineering'] else 'REAL';costs={};prepared=[]
    def utility(outcome):
        if not isinstance(outcome.get('em'),(int,float)) or not 0<=outcome['em']<=100:raise ValueError('invalid_training_score')
        reward=outcome['em']/100.
        for field,weight,scale in [('input_tokens','lambda_input_tokens','input_normalizer'),('output_tokens','lambda_output_tokens','output_normalizer'),('calls','lambda_calls','call_normalizer')]:
            if type(outcome.get(field)) is not int or outcome[field]<0:raise ValueError('unknown_training_cost')
            reward-=settings[weight]*outcome[field]/settings[scale]
        return reward
    for rows,split in ((training_records,'train'),(validation_records,'validation')):
        block=[]
        for row in rows:
            if row.get('split')!=split or row.get('mode')!=mode or row.get('feature_version')!=FEATURE_VERSION or row.get('prompt_version')!=settings['prompt_version']:
                raise ValueError('incompatible_training_record')
            features=np.array(row['features'],dtype=np.float32);sink=np.array(row['sink_features'],dtype=np.float32)
            if features.shape!=(2,32) or sink.shape!=(32,) or len(row['outcomes'])!=2 or not np.isfinite(features).all() or not np.isfinite(sink).all():raise ValueError('complete_n3_sender_pairs_required')
            if not row.get('cost_records'):raise ValueError('training_overhead_ledger_required')
            for cost in row['cost_records']:
                if any(type(cost.get(k)) is not int or cost[k]<0 for k in ('input_tokens','output_tokens','calls')):raise ValueError('unknown_training_overhead')
                if cost['key'] in costs and costs[cost['key']]!=cost:raise ValueError('conflicting_training_cost_record')
                costs[cost['key']]=cost
            block.append((torch.from_numpy(features),torch.from_numpy(sink),torch.tensor([utility(o) for o in row['outcomes']],dtype=torch.float32)))
        prepared.append(block)
    summaries={field:sum(c[field] for c in costs.values()) for field in ('input_tokens','output_tokens','calls')}
    candidates=[]
    for seed in settings['seeds']:
        torch.manual_seed(seed);model=PooledSenderScorer();optimizer=torch.optim.Adam(model.parameters(),lr=settings['learning_rate'])
        for epoch in range(settings['epochs']):
            optimizer.zero_grad();loss=torch.stack([((model(x,sink)-target)**2).mean() for x,sink,target in prepared[0]]).mean()
            loss.backward();optimizer.step()
            model.eval()
            with torch.no_grad():score=float(np.mean([float(target[int(model(x,sink).argmax())]) for x,sink,target in prepared[1]]))
            weights={k:v.detach().cpu().numpy().copy() for k,v in model.state_dict().items()}
            metadata={'feature_version':FEATURE_VERSION,'prompt_version':settings['prompt_version'],'mode':mode,'seed':seed,'epoch':epoch,
                      'training_keys':sorted(train_keys),'validation_keys':sorted(val_keys),'costs':summaries,
                      'settings_sha256':digest(settings),'record_sha256':digest([training_records,validation_records]),'validation_utility':score}
            candidates.append(Checkpoint(weights,metadata,score,checkpoint_hash(weights,metadata)));model.train()
    return choose_checkpoint(candidates)
