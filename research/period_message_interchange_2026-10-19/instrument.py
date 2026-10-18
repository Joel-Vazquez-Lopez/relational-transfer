"""Native Qwen projections/SDPA, with optional edge-specific donor K/V."""
from contextlib import contextmanager
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import scaled_dot_product_attention
from core import mask,BLOCKS
CUT=14

class Attention(nn.Module):
 def __init__(self,original,rec,block,row,donor,port,kind):
  super().__init__();self.original=original;self.rec=rec;self.block=block;self.row=row;self.donor=donor;self.port=port;self.kind=kind
 def __call__(self,x,mask=None,cache=None):
  assert cache is None
  a=self.original;B,L,D=x.shape
  q,k,v=a.q_proj(x),a.k_proj(x),a.v_proj(x)
  q=q.reshape(B,L,a.n_heads,-1).transpose(0,2,1,3)
  k=k.reshape(B,L,a.n_kv_heads,-1).transpose(0,2,1,3)
  v=v.reshape(B,L,a.n_kv_heads,-1).transpose(0,2,1,3)
  q=a.rope(q);k=a.rope(k)
  anchors=mx.array([self.row['period']-1,self.row['period']])
  self.rec['kv']=mx.stack([mx.take(k[0],anchors,axis=1),mx.take(v[0],anchors,axis=1)])
  output=scaled_dot_product_attention(q,k,v,cache=None,scale=a.scale,mask=mask)
  if self.donor is not None and self.block>=CUT:
   index=self.row['period'] if self.port=='period' else self.row['period']-1
   site=1 if self.port=='period' else 0
   dk=mx.array(self.donor[self.block-CUT,0,:,site,:]).astype(k.dtype)
   dv=mx.array(self.donor[self.block-CUT,1,:,site,:]).astype(v.dtype)
   # Copy-on-update: other keys/values and every prefix query stay recipient-native.
   kk=mx.concatenate([k[:,:,:index,:],dk[None,:,None,:],k[:,:,index+1:,:]],axis=2) if 'K' in self.kind else k
   vv=mx.concatenate([v[:,:,:index,:],dv[None,:,None,:],v[:,:,index+1:,:]],axis=2) if 'V' in self.kind else v
   edited=scaled_dot_product_attention(q,kk,vv,cache=None,scale=a.scale,mask=mask)
   selector=(mx.arange(L)>self.row['period'])[None,None,:,None]
   output=mx.where(selector,edited,output)
  output=output.transpose(0,2,1,3).reshape(B,L,-1)
  return a.o_proj(output)

class Block(nn.Module):
 def __init__(self,original,rec,m,native):
  super().__init__();self.original=original;self.rec=rec;self.m=m;self.native=native;self.use_sliding=False
 def __call__(self,x,mask=None,cache=None):
  assert cache is None;self.rec['incoming']=x
  out=self.original(x,mask if self.native else self.m,cache=None);self.rec['outgoing']=out;return out

@contextmanager
def instrument(model,row,graph='baseline',donor=None,port='period',kind='',native=False,attention_hook=True):
 layers=model.model.layers;old=[];records={};masks={}
 for i,b in enumerate(layers):
  assert not getattr(b,'use_sliding',False);rec={};records[i]=rec
  a=mask(row,graph,CUT,i);key=a.tobytes()
  if key not in masks:masks[key]=mx.array(a)
  att=b.self_attn;old.append((i,b,att))
  if attention_hook:b.self_attn=Attention(att,rec,i,row,donor,port,kind)
  layers[i]=Block(b,rec,masks[key],native)
 try:yield records
 finally:
  for i,b,att in old:b.self_attn=att;layers[i]=b
