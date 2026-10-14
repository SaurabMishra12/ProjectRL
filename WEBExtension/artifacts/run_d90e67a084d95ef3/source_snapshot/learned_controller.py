"""CPU-capable, gold-free categorical SparseComm contextual-bandit PPO."""
import itertools
import math
import re
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from protocol import CYCLES, normalize, message_text, digest

FEATURE_VERSION='observable_web_v1'

def action_space(quotas):
    if not quotas or any(type(q)!=int or q<=0 for q in quotas):raise ValueError('Require positive integer quotas')
    edges=[(s,t) for s in range(3) for t in range(3) if s!=t]
    supports=[list(x) for x in itertools.combinations(edges,2)]
    supports += [CYCLES['CW'],CYCLES['CCW'],edges]
    actions=[{'edges':[],'quota':0,'name':'NoComm'}]
    for i,support in enumerate(supports):
        for quota in sorted(set(quotas)):
            actions.append({'edges':[list(e) for e in support],'quota':quota,'name':f'support_{i}_quota_{quota}'})
    return actions

def lexical_overlap(a,b):
    a=set(re.findall(r'\w+',a.lower()));b=set(re.findall(r'\w+',b.lower()))
    return len(a&b)/max(1,len(a|b))

def observable_features(question,state,tokenizer):
    """Whitelisted observable input only. No correctness/history/gold ID/hash feature."""
    responses=state['responses'];assigned=state['assignment']
    own=[{(d['title'],i) for d in docs for i in range(len(d['sentences']))} for docs in assigned]
    cited=[set(map(tuple,r['citations'])) for r in responses]
    features=[math.log1p(tokenizer.count(question))/10, len({normalize(r['answer']) for r in responses})/3]
    for i,r in enumerate(responses):
        features += [len(assigned[i])/10,len(cited[i])/10,len(r['invalid_citations'])/10,
                     min(tokenizer.count(message_text(r,state['pool'])),4096)/4096,
                     float(normalize(r['answer'])=='noanswer'), lexical_overlap(question,r['answer'])]
        features += [float(state['roles'][i]==role) for role in ['retrieval','verification','reasoning']]
    for s in range(3):
        for t in range(3):
            if s==t:continue
            features += [float(normalize(responses[s]['answer'])!=normalize(responses[t]['answer'])),
                         len(cited[s]-own[t])/max(1,len(cited[s])),
                         len(own[s]&own[t])/max(1,len(own[s]|own[t])),
                         lexical_overlap(question,message_text(responses[s],state['pool']))]
    return np.asarray(features,dtype=np.float32)

class CategoricalController(nn.Module):
    def __init__(self,n_features,n_actions,hidden=64):
        super().__init__()
        self.trunk=nn.Sequential(nn.Linear(n_features,hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh())
        self.actor=nn.Linear(hidden,n_actions);self.critic=nn.Linear(hidden,1)
        nn.init.orthogonal_(self.actor.weight,gain=.01);nn.init.zeros_(self.actor.bias)
    def forward(self,features):
        hidden=self.trunk(features)
        return Categorical(logits=self.actor(hidden)),self.critic(hidden).squeeze(-1)
    @torch.no_grad()
    def choose(self,features,deterministic=False):
        dist,value=self(torch.as_tensor(features,dtype=torch.float32))
        action=dist.logits.argmax() if deterministic else dist.sample()
        return int(action),float(dist.log_prob(action)),float(value)

def clipped_actor_loss(logp,old_logp,advantages,clip=.2):
    ratio=torch.exp(logp-old_logp)
    return -torch.minimum(ratio*advantages,torch.clamp(ratio,1-clip,1+clip)*advantages).mean()

def ppo_update(model,optimizer,data,epochs=4,batch_size=32,clip=.2,entropy_coef=.01,target_kl=.03):
    advantage=(data['rewards']-data['old_values']).detach()
    if len(advantage)>1:advantage=(advantage-advantage.mean())/(advantage.std(unbiased=False)+1e-8)
    losses=[];kls=[];updates=0
    for epoch in range(epochs):
        for ids in torch.randperm(len(advantage)).split(batch_size):
            dist,value=model(data['features'][ids]);logp=dist.log_prob(data['actions'][ids])
            logratio=logp-data['old_logp'][ids]
            approx_kl=(torch.exp(logratio)-1-logratio).mean()
            if float(approx_kl.detach())>target_kl:
                return {'loss':float(np.mean(losses)) if losses else 0.,'updates':updates,'early_kl_stop':True,'max_kl':max(kls+[float(approx_kl.detach())])}
            actor=clipped_actor_loss(logp,data['old_logp'][ids],advantage[ids],clip)
            value_loss=.5*((value-data['rewards'][ids])**2).mean()
            loss=actor+.5*value_loss-entropy_coef*dist.entropy().mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite PPO objective')
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step()
            losses.append(float(loss.detach()));kls.append(float(approx_kl.detach()));updates+=1
    return {'loss':float(np.mean(losses)),'updates':updates,'early_kl_stop':False,'max_kl':max(kls)}
