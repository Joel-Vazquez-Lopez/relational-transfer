"""Wrap native block boundaries; change only the supplied attention mask."""
import contextlib
import mlx.core as mx
import mlx.nn as nn
from core import BLOCKS, mask

class Block(nn.Module):
    def __init__(self, original, record, attention_mask, native):
        super().__init__()
        self.original=original; self.record=record; self.attention_mask=attention_mask
        self.native=native; self.use_sliding=getattr(original,'use_sliding',False)
    def __call__(self,x,mask=None,cache=None):
        assert cache is None
        self.record['incoming']=x
        y=self.original(x,mask if self.native else self.attention_mask,cache=cache)
        self.record['outgoing']=y
        return y

@contextlib.contextmanager
def instrument(model,row,mode,cut,native=False):
    layers=model.model.layers; assert len(layers)==BLOCKS
    old=[]; records={}; masks={}
    for i,b in enumerate(layers):
        assert not getattr(b,'use_sliding',False)
        a=mask(row,mode,cut,i); key=a.tobytes()
        if key not in masks: masks[key]=mx.array(a)
        rec={}; records[i]=rec; old.append((i,b))
        layers[i]=Block(b,rec,masks[key],native)
    try: yield records
    finally:
        for i,b in old: layers[i]=b
