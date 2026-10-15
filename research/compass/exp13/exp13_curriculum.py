#!/usr/bin/env python3
"""Compass 13: test length-three swap composition from zero-to-two examples.
No explicit operator in the model. Exact frozen checkpoints/other scripts
are in the verified Compass 13 conversation archive.
"""
import argparse, json, random, time, itertools
from collections import Counter
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

N=4
PAD=4
Q_TOKEN=5
PAIRS=[(a,b) for a in range(N) for b in range(a+1,N)]

def final_state(ops):
    state=list(range(N))
    for a,b in ops: state[a],state[b]=state[b],state[a]
    return tuple(state)

def cycle_type(state):
    visited=set(); sizes=[]
    for i in range(N):
        if i not in visited:
            n=0; at=i
            while at not in visited:
                visited.add(at);n+=1;at=state[at]
            sizes.append(n)
    return ','.join(map(str,sorted(sizes,reverse=True)))

def dataset():
    examples=[]
    for slots in itertools.product(range(-1,len(PAIRS)),repeat=3):
        n=sum(s!=-1 for s in slots)
        ops=[PAIRS[i] for i in slots if i!=-1]
        flat=[]
        for i in slots:
            flat.extend(PAIRS[i] if i!=-1 else (PAD,PAD))
        state=final_state(ops)
        for q in range(N):
            examples.append(dict(slots=slots,length=n,x=flat,q=q,y=state[q],
                                 cycle_type=cycle_type(state),world=state))
    return examples

class Block(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.attn=nn.MultiheadAttention(d,4,batch_first=True,dropout=0.0)
        self.ln1=nn.LayerNorm(d);self.ln2=nn.LayerNorm(d)
        self.ff=nn.Sequential(nn.Linear(d,3*d),nn.GELU(),nn.Linear(3*d,d))
    def forward(self,x):
        n=x.shape[1]
        mask=torch.triu(torch.ones(n,n,device=x.device,dtype=torch.bool),1)
        z=self.ln1(x)
        x=x+self.attn(z,z,z,attn_mask=mask,need_weights=False)[0]
        return x+self.ff(self.ln2(x))

class CausalWorld(nn.Module):
    def __init__(self,d=48,depth=3):
        super().__init__()
        self.d=d;self.depth=depth
        self.tok=nn.Embedding(6,d);self.pos=nn.Embedding(7,d)
        self.qemb=nn.Embedding(N,d)
        self.blocks=nn.ModuleList([Block(d) for _ in range(depth)])
        self.head=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,2*d),
                                nn.GELU(),nn.Linear(2*d,N))
    def forward(self,ops,q):
        t=torch.cat([ops,torch.full((len(ops),1),Q_TOKEN,device=ops.device,dtype=torch.long)],dim=1)
        x=self.tok(t)+self.pos(torch.arange(7,device=ops.device))[None,:,:]
        x=x.clone();x[:,-1,:]=x[:,-1,:]+self.qemb(q)
        for block in self.blocks:x=block(x)
        return self.head(x[:,-1,:])

@torch.no_grad()
def evaluate(model,x,q,y,idx,batch=2048):
    model.eval()
    predictions=torch.cat([model(x[s],q[s]).argmax(-1) for s in idx.split(batch)])
    return int((predictions==y[idx]).sum()),int(len(idx)),predictions

def run(cfg):
    torch.set_num_threads(cfg.threads)
    out=Path(cfg.out);out.mkdir(exist_ok=True,parents=True)
    rows=dataset()
    x=torch.tensor([r['x'] for r in rows]);q=torch.tensor([r['q'] for r in rows])
    y=torch.tensor([r['y'] for r in rows])
    splits={n:torch.tensor([i for i,r in enumerate(rows) if r['length']==n]) for n in range(4)}
    train_mask={
       'pairs_only':splits[2],
       'single_plus_pairs':torch.cat([splits[1],splits[2]]),
       'all_short':torch.cat([splits[0],splits[1],splits[2]])}
    counts={str(k):len(v) for k,v in splits.items()}
    four_counts={k:sum(rows[i]['cycle_type']=='4' for i in v.tolist()) for k,v in splits.items()}
    assert counts=={'0':4,'1':72,'2':432,'3':864}
    assert four_counts[0]==four_counts[1]==four_counts[2]==0
    report={'study':'Compass 13 emergence of structural composition',
            'spec':vars(cfg),'split_counts':counts,'four_cycle_counts':four_counts,
            'warnings':['Synthetic development universe; examples sharing worlds are dependent',
                        'Triple sequences withheld from training, but primitive tokens reused',
                        'Accuracy does not establish natural pre-question computation reuse'],
            'runs':[]}
    (out/'data_spec.json').write_text(json.dumps(
        {'PAIRS':PAIRS,'split_counts':counts,'four_cycle_counts':four_counts},indent=2)+'\n')
    for arm in cfg.arms:
        for seed in cfg.seeds:
            start=time.perf_counter();torch.manual_seed(seed);random.seed(seed)
            model=CausalWorld(cfg.width,cfg.depth)
            opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=1e-4)
            chosen=train_mask[arm];trace=[]
            for step in range(1,cfg.steps+1):
                model.train()
                ix=chosen[torch.randint(len(chosen),(cfg.batch,))]
                loss=F.cross_entropy(model(x[ix],q[ix]),y[ix])
                opt.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                opt.step()
                if step in [1,100,200,400,800,1200,2000,cfg.steps] or step%cfg.log_every==0:
                    seen,_,_=evaluate(model,x,q,y,chosen)
                    triple,_,_=evaluate(model,x,q,y,splits[3])
                    four=torch.tensor([i for i in splits[3].tolist() if rows[i]['cycle_type']=='4'])
                    four_ok,_,_=evaluate(model,x,q,y,four)
                    rec={'step':step,'train':seen,'train_n':len(chosen),
                         'triple':triple,'triple_n':864,'four_cycle':four_ok,
                         'four_cycle_n':len(four)}
                    trace.append(rec)
                    if step%cfg.log_every==0 or step==cfg.steps:
                        print(arm,seed,rec,flush=True)
            _,_,pred=evaluate(model,x,q,y,torch.arange(len(rows)))
            results={str(n):{'correct':int((pred[splits[n]]==y[splits[n]]).sum()),
                             'total':len(splits[n])} for n in range(4)}
            cycles={c:{'correct':sum(int(pred[i]==y[i]) for i in splits[3].tolist()
                                      if rows[i]['cycle_type']==c),
                       'total':sum(rows[i]['cycle_type']==c for i in splits[3].tolist())}
                    for c in sorted(set(rows[i]['cycle_type'] for i in splits[3].tolist()))}
            r={'arm':arm,'seed':seed,
               'parameters':sum(p.numel() for p in model.parameters()),
               'final':results,'triple_by_cycle_type':cycles,'trace':trace,
               'seconds':round(time.perf_counter()-start,2)}
            torch.save({'state':model.state_dict(),'width':cfg.width,'depth':cfg.depth,
                        'seed':seed,'arm':arm},out/f'{arm}_seed{seed}.pt')
            report['runs'].append(r)
            (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
            print('FINAL',arm,seed,results,cycles,flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',default='results')
    p.add_argument('--arms',nargs='+',default=['pairs_only','all_short'])
    p.add_argument('--seeds',nargs='+',type=int,default=[19,29,41])
    p.add_argument('--steps',type=int,default=1200)
    p.add_argument('--width',type=int,default=48)
    p.add_argument('--depth',type=int,default=3)
    p.add_argument('--lr',type=float,default=.003)
    p.add_argument('--batch',type=int,default=128)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--log-every',type=int,default=400)
    run(p.parse_args())
