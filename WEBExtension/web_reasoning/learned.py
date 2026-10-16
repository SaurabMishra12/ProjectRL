"""Fixed-width per-agent features; shared scoring has no pairwise adjacency state."""
import hashlib
import math
import numpy as np
import torch
from torch import nn
from .packets import tokens
from .policies import jaccard

FEATURE_VERSION='observable_web_v2_32'


class PooledSenderScorer(nn.Module):
    def __init__(self,feature_dim=32,hidden_dim=64):
        super().__init__()
        self.encode=nn.Sequential(nn.Linear(feature_dim,hidden_dim),nn.Tanh())
        self.score=nn.Sequential(nn.Linear(3*hidden_dim,hidden_dim),nn.Tanh(),nn.Linear(hidden_dim,1))
    def forward(self,features,sink_features):
        if features.ndim!=2 or features.shape[0]<1 or sink_features.shape!=(features.shape[1],):raise ValueError('invalid_set_features')
        encoded=self.encode(features)
        pooled=encoded.mean(dim=0,keepdim=True).expand_as(encoded)
        sink=self.encode(sink_features).expand_as(encoded)
        return self.score(torch.cat((encoded,pooled,sink),dim=1)).squeeze(-1)


def hash_bins(words,seed):
    words=set(words);result=np.zeros(8,dtype=np.float32)
    for word in words:
        h=hashlib.sha256(f'{seed}:{word}'.encode()).digest();result[h[0]%8]+=1 if h[1]%2 else -1
    return result/max(1,len(words))


def make_features(metadata,sink,question,delivered,seed=20261016):
    by_id={m.agent_id:m for m in metadata}
    if sink not in by_id or len(by_id)!=len(metadata):raise ValueError('invalid_metadata_population')
    needs=tokens(' '.join(by_id[sink].needs));query=tokens(question);sink_tags=tokens(' '.join(by_id[sink].entity_tags))
    def row(m):
        tags=tokens(' '.join(m.entity_tags));need_tokens=tokens(' '.join(m.needs))
        x=np.zeros(32,dtype=np.float32)
        if m.status in ('supported','partial','no_evidence','conflict'):x[('supported','partial','no_evidence','conflict').index(m.status)]=1
        x[4]=min(1,max(0,math.log1p(m.valid_fact_count)/math.log(5)))
        x[5]=jaccard(tags,needs);x[6]=jaccard(tags,query)
        x[7]=len(set(m.fact_hashes)&set(delivered))/max(1,len(m.fact_hashes))
        x[8:16]=hash_bins(tags,seed);x[16:24]=hash_bins(need_tokens,seed);x[24:32]=hash_bins(sink_tags,seed)
        return x
    nodes=tuple(sorted(i for i in by_id if i!=sink))
    if not nodes:raise ValueError('sender_required')
    return np.stack([row(by_id[i]) for i in nodes]),row(by_id[sink]),nodes


def sender_features(metadata,question,delivered,sink=0,seed=20261016):
    return make_features(metadata,sink,question,delivered,seed)[0]


def rank_senders(model,metadata,sink,k,question,delivered,seed=20261016):
    features,sink_features,nodes=make_features(metadata,sink,question,delivered,seed)
    if type(k) is not int or not 0<=k<=len(nodes):raise ValueError('invalid_sender_count')
    with torch.no_grad():scores=model(torch.from_numpy(features),torch.from_numpy(sink_features)).numpy()
    by_id=dict(zip(nodes,map(float,scores)))
    return tuple(sorted(sorted(nodes,key=lambda node:(-by_id[node],node))[:k]))
