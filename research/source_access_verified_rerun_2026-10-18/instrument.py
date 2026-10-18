"""Native block wrapper; original attention/MLP operations are preserved."""
import contextlib
import mlx.core as mx
import mlx.nn as nn
from core import BLOCKS, mask

class Capture(nn.Module):
    def __init__(self,original,record,name):
        super().__init__(); self.original=original; self.record=record; self.name=name
    def __call__(self,*args,**kwargs):
        y=self.original(*args,**kwargs); self.record[self.name]=y; return y

class Block(nn.Module):
    def __init__(self,original,record,attention_mask,native):
        super().__init__(); self.original=original; self.record=record
        self.attention_mask=attention_mask; self.native=native
        self.use_sliding=getattr(original,'use_sliding',False)
    def __call__(self,x,mask=None,cache=None):
        assert cache is None, 'This study forbids cached/native-length execution'
        self.record['incoming']=x
        y=self.original(x,mask if self.native else self.attention_mask,cache=cache)
        self.record['outgoing']=y; return y

@contextlib.contextmanager
def instrument(model,row,condition,native=False):
    layers=model.model.layers; assert len(layers)==BLOCKS
    old=[]; records={}; masks={}
    for i,b in enumerate(layers):
        assert not getattr(b,'use_sliding',False), 'Sliding attention requires a separate design'
        rec={}; records[i]=rec
        a=mask(row,condition,i); key=a.tobytes()
        if key not in masks: masks[key]=mx.array(a)
        attn,mlp=b.self_attn,b.mlp
        b.self_attn=Capture(attn,rec,'attention'); b.mlp=Capture(mlp,rec,'mlp')
        old.append((i,b,attn,mlp)); layers[i]=Block(b,rec,masks[key],native)
    try: yield records
    finally:
        for i,b,attn,mlp in old:
            b.self_attn,b.mlp=attn,mlp; layers[i]=b
