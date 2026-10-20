"""Unmodified Block/Encoder/Agents class definitions from the existing pilot. No checkpoints loaded."""
import math
import torch
from torch import nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self,d,h):
        super().__init__(); self.h=h; self.dh=d//h
        self.qkv=nn.Linear(d,3*d); self.proj=nn.Linear(d,d); self.n1=nn.LayerNorm(d)
        self.ff=nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d)); self.n2=nn.LayerNorm(d)
    def forward(self,x,ablate_head=None,ablate_mlp=False):
        z=self.n1(x); q,k,v=self.qkv(z).chunk(3,-1); B,T,D=x.shape
        q=q.view(B,T,self.h,self.dh).transpose(1,2); k=k.view(B,T,self.h,self.dh).transpose(1,2); v=v.view(B,T,self.h,self.dh).transpose(1,2)
        a=(q@k.transpose(-2,-1)/math.sqrt(self.dh)).softmax(-1); heads=a@v
        if ablate_head is not None: heads=heads.clone(); heads[:,ablate_head]=0
        x=x+self.proj(heads.transpose(1,2).reshape(B,T,D))
        if not ablate_mlp: x=x+self.ff(self.n2(x))
        return x

class Encoder(nn.Module):
    def __init__(self,in_dim,n_tok,d,h,l):
        super().__init__(); self.inp=nn.Linear(in_dim,d); self.pos=nn.Parameter(torch.randn(1,n_tok+1,d)*.02); self.cls=nn.Parameter(torch.randn(1,1,d)*.02)
        self.blocks=nn.ModuleList([Block(d,h) for _ in range(l)]); self.norm=nn.LayerNorm(d)
    def forward(self,x,ablation=None,return_layers=False):
        B=x.shape[0]; x=torch.cat([self.cls.expand(B,-1,-1),self.inp(x)],1)+self.pos
        layers=[]
        for li,b in enumerate(self.blocks):
            ah=ablation[1] if ablation and ablation[0]=='head' and ablation[1]//100==li else None
            if ah is not None: ah=ah%100
            am=bool(ablation and ablation==('mlp',li))
            x=b(x,ah,am); layers.append(x)
        out=self.norm(x[:,0]); return (out,layers) if return_layers else out

class Agents(nn.Module):
    def __init__(self,cfg):
        super().__init__(); d=cfg['d_model']; h=cfg['n_heads']; l=cfg['n_layers']; V=cfg['vocab_size']; L=cfg['message_length']
        self.V=V; self.L=L; self.speaker=Encoder(14,3,d,h,l); self.emit=nn.Linear(d,L*V)
        self.sym=nn.Parameter(torch.randn(V,d)*.1); self.mpos=nn.Parameter(torch.randn(1,L+1,d)*.02); self.mcls=nn.Parameter(torch.randn(1,1,d)*.02)
        self.msg_blocks=nn.ModuleList([Block(d,h) for _ in range(l)]); self.msg_norm=nn.LayerNorm(d); self.candidate=Encoder(14,3,d,h,l)
    def messages(self,source,tau=.7,noise=True):
        logits=self.emit(self.speaker(source)).view(-1,self.L,self.V)
        if noise:
            u=torch.rand_like(logits).clamp(1e-6,1-1e-6); logits=(logits-torch.log(-torch.log(u)))/tau
        soft=logits.softmax(-1); ids=soft.argmax(-1); hard=F.one_hot(ids,self.V).float(); return hard+soft-soft.detach(),ids
    def listen(self,msg,cands,msg_ablation=None):
        B=msg.shape[0]; x=torch.cat([self.mcls.expand(B,-1,-1),msg@self.sym],1)+self.mpos
        for li,b in enumerate(self.msg_blocks):
            ah=msg_ablation[1] if msg_ablation and msg_ablation[0]=='head' and msg_ablation[1]//100==li else None
            if ah is not None: ah%=100
            x=b(x,ah,bool(msg_ablation and msg_ablation==('mlp',li)))
        q=self.msg_norm(x[:,0]); cv=self.candidate(cands.reshape(-1,3,14)).reshape(B,4,-1)
        return torch.einsum('bd,bkd->bk',F.normalize(q,dim=-1),F.normalize(cv,dim=-1))*10
    def forward(self,s,c,tau=.7,noise=True,ablation=None):
        m,ids=self.messages(s,tau,noise); return self.listen(m,c,ablation),ids
