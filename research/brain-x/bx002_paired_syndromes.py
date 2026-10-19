"""BX-002 paired syndromes: synthetic development, not natural language.

Run: python brain_x_bx002.py --steps 1200 --seeds 17 29 --outdir /mnt/data/bx002
Requires: Python 3, NumPy, CPU PyTorch.
"""
import argparse, csv, hashlib, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn

N=5
EDGES=np.asarray([(i,(i+1)%N) for i in range(N)],dtype=np.int64)
ASSIGN=np.asarray([[int(j>>i&1) for i in range(N)] for j in range(1<<N)],dtype=np.int64)

def worlds(n,seed):
    rng=np.random.default_rng(seed)
    bits=rng.integers(0,2,(n,N),dtype=np.int64)
    local=bits^(rng.random((n,N))<.35).astype(np.int64)
    parity=(bits[:,EDGES[:,0]]^bits[:,EDGES[:,1]] ^ (rng.random((n,N))<.08).astype(np.int64))
    fake=rng.integers(0,2,(n,N),dtype=np.int64)
    query=rng.integers(0,N,n,dtype=np.int64)
    labels=bits[np.arange(n),query]
    return dict(bits=bits,local=local,valid=parity,random=fake,query=query,labels=labels)

def encode(local,edges,query,include_edges=True):
    num=len(query)
    x=np.zeros((num,2*N+1,15),dtype=np.float32)
    for i in range(N):
        x[:,i,0]=1; x[:,i,3+i]=1; x[:,i,13]=2*local[:,i]-1
        x[:,N+i,1]=1; a,b=EDGES[i]; x[:,N+i,3+a]=1; x[:,N+i,8+b]=1
        x[:,N+i,13]= (2*edges[:,i]-1) if include_edges else 0
    x[:,2*N,2]=1
    x[np.arange(num),2*N,3+query]=1
    return x

def marginal_decode(local,edge,query,edge_flip=.08):
    pred=np.empty(len(query),dtype=np.int64)
    synd=ASSIGN[:,EDGES[:,0]]^ASSIGN[:,EDGES[:,1]]
    for start in range(0,len(query),512):
        o=local[start:start+512];e=edge[start:start+512];q=query[start:start+512]
        l=(ASSIGN[None,:,:]==o[:,None,:]).sum(2)
        z=(synd[None,:,:]==e[:,None,:]).sum(2)
        score=l*np.log(.65/.35)+z*np.log((1-edge_flip)/edge_flip)
        w=np.exp(score-score.max(1,keepdims=True));w/=w.sum(1,keepdims=True)
        marg=w@ASSIGN
        pred[start:start+len(q)]=(marg[np.arange(len(q)),q]>.5).astype(np.int64)
    return pred

def vote_baseline(local,edge,query):
    # Both neighbors send noisy proposals of the queried node through their parity edges.
    rows=np.arange(len(query));prev=(query-1)%N;nxt=(query+1)%N
    direct=local[rows,query]
    left=local[rows,prev]^edge[rows,prev]
    right=local[rows,nxt]^edge[rows,query]
    return ((direct+left+right)>=2).astype(np.int64)

class Learner(nn.Module):
    def __init__(self,layers=3,width=48):
        super().__init__()
        self.embed=nn.Linear(15,width)
        self.pos=nn.Embedding(2*N+1,width)
        self.tr=nn.TransformerEncoder(nn.TransformerEncoderLayer(width,4,width*2,dropout=0,batch_first=True),layers,enable_nested_tensor=False)
        self.out=nn.Linear(width,2)
    def forward(self,x):
        z=self.embed(x)+self.pos(torch.arange(x.size(1),device=x.device))
        return self.out(self.tr(z)[:,-1])

def predict(net,x):
    net.eval()
    with torch.no_grad():
        p=torch.cat([net(torch.from_numpy(k)).argmax(-1) for k in np.array_split(x, max(1,(len(x)+255)//256))]).numpy()
    return p

def fit(seed,steps,mode,layer=3,width=48, train_size=22000):
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    w=worlds(train_size,100000+seed)
    # Keep all quantities and labels identical between node-only and normal arms.
    rng=np.random.default_rng(300000+seed)
    use_valid=rng.random(train_size)<.80
    edges=np.where(use_valid[:,None],w['valid'],w['random'])
    x=encode(w['local'],edges,w['query'],include_edges=(mode=='edge'))
    xt=torch.from_numpy(x);yt=torch.from_numpy(w['labels'])
    net=Learner(layer,width);optim=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.001)
    records=[]
    for step in range(1,steps+1):
        ix=torch.randint(train_size,(128,))
        logits=net(xt[ix]);loss=nn.functional.cross_entropy(logits,yt[ix]);optim.zero_grad();loss.backward();optim.step()
        if step in (steps//2,steps):records.append({'step':step,'train_minibatch_loss':round(float(loss.item()),6)})
    return net, records

def paired_counts(p1,p0,y):
    v=(p1==y);r=(p0==y)
    return dict(n=len(y),both_correct=int(np.sum(v&r)),valid_only=int(np.sum(v&~r)),random_only=int(np.sum(~v&r)),both_wrong=int(np.sum(~v&~r)),valid_accuracy=float(v.mean()),random_accuracy=float(r.mean()),valid_minus_random=float(v.mean()-r.mean()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=1200);p.add_argument('--layers',type=int,default=3);p.add_argument('--width',type=int,default=48);p.add_argument('--seeds',type=int,nargs='+',default=[17,29]);p.add_argument('--test-size',type=int,default=4000);p.add_argument('--outdir',default='/mnt/data/bx002'); a=p.parse_args()
    torch.set_num_threads(2)
    out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True)
    wd=worlds(a.test_size,802002); y=wd['labels'];q=wd['query'];loc=wd['local'];v=wd['valid'];rand=wd['random']
    xx=encode(loc,v,q);xxrandom=encode(loc,rand,q);xxnode=encode(loc,v,q,include_edges=False)
    baseline=loc[np.arange(len(y)),q]
    exact=marginal_decode(loc,v,q)
    vote=vote_baseline(loc,v,q)
    # Fixture digests detect accidental changes, not statistical validity.
    fixture_hash=hashlib.sha256(b''.join(wd[k].tobytes() for k in ['bits','local','valid','random','query','labels'])).hexdigest()
    result=dict(experiment='BX-002',status='open_synthetic_development',description='Paired same-world valid versus independently randomized syndrome under one frozen model; extra node-only matched-budget model',seed_test=802002,fixture_sha256=fixture_hash,parameters=dict(n=N,local_flip=.35,edge_flip=.08,train_edge_valid_probability=.8,train_size=22000,test_size=a.test_size,steps=a.steps,layers=a.layers,width=a.width,model='PyTorch TransformerEncoder',batch_size=128,optimizer='AdamW',learning_rate=.002),controls=dict(local_accuracy=float(np.mean(baseline==y)),exact_valid_accuracy=float(np.mean(exact==y)),majority_valid_accuracy=float(np.mean(vote==y)),local_vs_exact=paired_counts(exact,baseline,y),majority_vs_local=paired_counts(vote,baseline,y)),models=[])
    raw={'id':np.arange(len(y)), 'query':q,'gold':y,'local':baseline,'oracle':exact,'majority':vote}
    for seed in a.seeds:
        for mode in ('edge','node'):
            net,losses=fit(seed,a.steps,mode,a.layers,a.width)
            a_valid=predict(net,xx if mode=='edge' else xxnode)
            a_random=predict(net,xxrandom if mode=='edge' else xxnode)
            row=dict(seed=seed,mode=mode,losses=losses,paired=paired_counts(a_valid,a_random,y),valid_vs_local=paired_counts(a_valid,baseline,y),valid_vs_bayes=paired_counts(a_valid,exact,y),random_vs_local=paired_counts(a_random,baseline,y))
            print(json.dumps({'seed':seed,'mode':mode,'paired':row['paired'],'valid_vs_local':row['valid_vs_local']}),flush=True)
            result['models'].append(row)
            raw[f'{mode}{seed}_valid']=a_valid;raw[f'{mode}{seed}_random']=a_random
    outjson=out/'results.json';outcsv=out/'paired_predictions.csv'
    with outcsv.open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(list(raw))
        for i in range(len(y)):writer.writerow([int(raw[k][i]) for k in raw])
    result['prediction_csv_sha256']=hashlib.sha256(outcsv.read_bytes()).hexdigest()
    outjson.write_text(json.dumps(result,indent=2)+'\n')
    print('SAVED',outjson,outcsv,flush=True)
if __name__=='__main__':main()
