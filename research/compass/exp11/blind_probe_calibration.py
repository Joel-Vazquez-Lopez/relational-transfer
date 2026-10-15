#!/usr/bin/env python3
"""Compass 11: pre-question decodability false-positive calibration.

Frozen Investigation 04 learned Transformers were trained on all 36 contexts.
External probes fitted on context subsets; probe heldout is NOT model OOD.
"""
import argparse, json, random, time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from learned_mechanism_controls import LearnedControl, make_data

class Probe(nn.Module):
    def __init__(self,d,width=128):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,width),nn.GELU(),
                               nn.Linear(width,width//2),nn.GELU(),nn.Linear(width//2,16))
    def forward(self,x):
        return self.net(x).reshape(-1,4,4)

def features(model,ops,kind):
    if kind=='cache_native':
        assert model.mode=='cache'
        with torch.no_grad(): return model.prefix(ops)
    if kind=='reconstruct_raw':
        assert model.mode=='reconstruct'
        with torch.no_grad():
            idx=torch.arange(4,device=ops.device)
            return (model.token(ops)+model.pos(idx)[None,:,:]).reshape(len(ops),-1)
    if kind=='onehot_raw':
        return F.one_hot(ops,4).float().reshape(len(ops),-1)
    raise ValueError(kind)

def train_probe(z,labels,training_contexts,probe_seed,steps=600):
    torch.manual_seed(probe_seed)
    net=Probe(z.shape[1]); opt=torch.optim.AdamW(net.parameters(),lr=.015,weight_decay=.0001)
    x=z[training_contexts];y=labels[training_contexts]
    for _ in range(steps):
        logits=net(x)
        loss=F.cross_entropy(logits.reshape(-1,4),y.flatten())
        opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad(): scores=net(z).argmax(-1)
    return scores

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='results_3seeds')
    p.add_argument('--steps',type=int,default=600)
    p.add_argument('--seeds',type=int,nargs='+',default=[19,29,41])
    p.add_argument('--out',default='probe_results.json')
    p.add_argument('--split-seed',type=int,default=555)
    a=p.parse_args()
    torch.set_num_threads(1)
    X,Q,Y,G=make_data()
    ops=X[::4];gold=Y.reshape(-1,4)
    assert len(ops)==36
    ids=list(range(36));random.Random(a.split_seed).shuffle(ids)
    test=sorted(ids[:9]);train=sorted(ids[9:]);allidx=list(range(36))
    results=[]
    for seed in a.seeds:
        for kind,mode in [('cache_native','cache'),('reconstruct_raw','reconstruct'),('onehot_raw','reconstruct')]:
            ck=torch.load(Path(a.root)/f'{mode}_seed{seed}.pt',map_location='cpu',weights_only=True)
            m=LearnedControl(mode,d=ck['d']);m.load_state_dict(ck['state_dict']);m.eval()
            with torch.no_grad(): z=features(m,ops,kind)
            for split,indices in [('all_contexts',allidx),('nine_probe_contexts_heldout',train)]:
                start=time.perf_counter()
                scores=train_probe(z,gold,indices,seed+101,steps=a.steps)
                matched=scores==gold
                row={'base_seed':seed,'feature':kind,'schedule':mode,'split':split,
                     'feature_width':int(z.shape[-1]),'probe_steps':a.steps,
                     'train_correct':int(matched[indices].sum()),'train_n':len(indices)*4,
                     'probe_heldout_correct':int(matched[test].sum()),'probe_heldout_n':36,
                     'full_correct':int(matched.sum()),'full_n':144,
                     'seconds':round(time.perf_counter()-start,2)}
                results.append(row);print(row,flush=True)
    out={'meaning':'Externally trained probes: only cache_native precomputes the relation before query in the underlying model',
         'probe_holdout_seed':a.split_seed,'holdout_contexts':test,'train_contexts':train,
         'warning':'Frozen base Transformer weights already trained on all 36 worlds. Heldout only for the probe.',
         'results':results}
    Path(a.out).write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
