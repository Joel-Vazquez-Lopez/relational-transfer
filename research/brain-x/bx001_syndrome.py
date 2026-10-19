"""Brain-X BX-001: open synthetic developmental, NOT natural-language model evidence."""
import argparse, json, random
import numpy as np
import torch
from torch import nn

N=5; EDGE=np.array([(i,(i+1)%N) for i in range(N)],dtype=np.int64)
ALL=np.array([[int(j>>i&1) for i in range(N)] for j in range(2**N)],dtype=np.int64)

def sample(num, seed, true_parity=True, p=.35, q=.08):
    rng=np.random.default_rng(seed)
    b=rng.integers(0,2,(num,N)); obs=b^ (rng.random((num,N))<p)
    ep=b[:,EDGE[:,0]]^b[:,EDGE[:,1]]^(rng.random((num,N))<q)
    if not true_parity: ep=rng.integers(0,2,(num,N))
    query=rng.integers(0,N,num)
    x=np.zeros((num,2*N+1,15),dtype=np.float32)
    for i in range(N):
        x[:,i,0]=1; x[:,i,3+i]=1; x[:,i,13]=2*obs[:,i]-1
        x[:,N+i,1]=1; a,c=EDGE[i]; x[:,N+i,3+a]=1; x[:,N+i,8+c]=1; x[:,N+i,13]=2*ep[:,i]-1
    x[:,2*N,2]=1; x[np.arange(num),2*N,3+query]=1
    return x,b[np.arange(num),query].astype(np.int64),obs.astype(np.int64),ep.astype(np.int64),query

def oracle(obs,ep,query,p=.35,q=.08):
    # Exact Bayes marginal decoder, enumerating all 2**N latent assignments.
    nd=obs.shape[0]; out=np.empty(nd,dtype=np.int64)
    for k in range(0,nd,1000):
        o=obs[k:k+1000];e=ep[k:k+1000]; qu=query[k:k+1000]
        local=(ALL[None,:,:]==o[:,None,:]).sum(axis=2)
        predicted=(ALL[:,EDGE[:,0]] ^ ALL[:,EDGE[:,1]])
        checks=(predicted[None,:,:]==e[:,None,:]).sum(axis=2)
        s=local*np.log((1-p)/p)+checks*np.log((1-q)/q)
        weights=np.exp(s-s.max(axis=1,keepdims=True)); weights/=weights.sum(axis=1,keepdims=True)
        post=weights@ALL; out[k:k+len(qu)]=(post[np.arange(len(qu)),qu]>.5).astype(np.int64)
    return out

class Transformer(nn.Module):
    def __init__(self, layers=2):
        super().__init__(); dim=40
        self.emb=nn.Linear(15,dim)
        self.pos=nn.Embedding(11,dim)
        self.enc=nn.TransformerEncoder(nn.TransformerEncoderLayer(dim,4,80,dropout=0,batch_first=True),layers,enable_nested_tensor=False)
        self.head=nn.Linear(dim,2)
    def forward(self,x):
        h=self.emb(x)+self.pos(torch.arange(x.size(1),device=x.device))
        return self.head(self.enc(h)[:,-1])

def fit(seed, true_parity, steps, layers=2, train_num=20000, test_num=5000):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    tr=sample(train_num,100000+seed,true_parity);te=sample(test_num,200000+seed,true_parity)
    X=torch.from_numpy(tr[0]);Y=torch.from_numpy(tr[1]); Xt=torch.from_numpy(te[0]);Yt=torch.from_numpy(te[1]);
    model=Transformer(layers=layers);opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.001)
    checkpoints=[]
    for t in range(1,steps+1):
        sel=torch.randint(train_num,(128,)); logits=model(X[sel]);loss=nn.functional.cross_entropy(logits,Y[sel]); opt.zero_grad();loss.backward();opt.step()
        if t in sorted(set([steps//2,steps])):
            model.eval()
            with torch.no_grad():
                preds=torch.cat([model(xx).argmax(1) for xx in Xt.split(512)])
                accuracy=float((preds==Yt).float().mean())
            checkpoints.append({"steps":t,"test_accuracy":round(accuracy,5)})
            model.train()
    loc=(te[2][np.arange(len(te[4])),te[4]]==te[1]).mean()
    if true_parity:
        orpred=oracle(te[2],te[3],te[4]); oracc=float((orpred==te[1]).mean())
    else: oracc=float(loc)
    return {"seed":seed,"layers":layers,"parity":"valid" if true_parity else "shuffled","train_num":train_num,"test_num":test_num,"checks":checkpoints,"local_accuracy":round(float(loc),5),"bayes_accuracy":round(oracc,5)}

def main():
    a=argparse.ArgumentParser(); a.add_argument('--steps',type=int,default=600);a.add_argument('--layers',type=int,default=2);a.add_argument('--seeds',type=int,nargs='+',default=[17,29]);a.add_argument('--output',default=None);args=a.parse_args()
    torch.set_num_threads(2)
    rows=[]
    for seed in args.seeds:
        for valid in [True,False]:
            r=fit(seed,valid,args.steps,args.layers);rows.append(r);print(json.dumps(r),flush=True)
    out={"experiment":"BX-001","status":"synthetic_development","params":{"nodes":N,"ring_edges":N,"p_node_flip":.35,"p_parity_flip":.08,"layers":args.layers,"hidden_dim":40,"training_steps":args.steps,"train_cases":20000,"eval_cases":5000},"results":rows,"notes":"Synthetic strings encoded as feature vectors; no authentic text; trained independent models; fixed topology; Bayes oracle has full generative likelihood; high prompt counts are not independent linguistic families."}
    if args.output:
        with open(args.output,'w') as f:json.dump(out,f,indent=2)
if __name__=='__main__': main()
