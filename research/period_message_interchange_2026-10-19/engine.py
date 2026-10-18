"""Bounded experiment engine. No model load at import."""
import numpy as np
from core import ROOT,LENGTH,BLOCKS
from instrument import instrument,CUT
import hashlib

def digest(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
class Engine:
 def __init__(self,lock):
  import mlx.core as mx
  from mlx_lm import load
  self.mx=mx;self.model,self.tok=load(str(ROOT/lock['model']['path']),tokenizer_config={'trust_remote_code':False});self.model.eval();mx.set_cache_limit(256*1024**2)
 def compute(self,row,graph='baseline',donor=None,port='period',kind='',attention_hook=True,native=False):
  mx=self.mx;e=row['prompt_end'];p=row['period'];ids=row['token_ids'];x=mx.array([ids+[row['pad']]*(LENGTH-len(ids))])
  with instrument(self.model,row,graph,donor,port,kind,native,attention_hook) as rec:
   y=self.model(x);logits=y[0,e].astype(mx.float32)
   selected=[rec[0]['incoming'][0,[p-1,p,e]].astype(mx.float32)]+[rec[i]['outgoing'][0,[p-1,p,e]].astype(mx.float32) for i in range(BLOCKS)]
   prefixes=[rec[i]['outgoing'][0,:p+1].astype(mx.float32) for i in range(BLOCKS)]
   kv=[rec[i]['kv'].astype(mx.float32) for i in range(CUT,BLOCKS)] if attention_hook else []
   mx.eval(logits,*selected,*prefixes,*kv)
   result=np.asarray(logits).copy();trace=np.stack([np.asarray(a).copy() for a in selected]);payload=dict(logits=result,trace=trace)
   if kv:payload['kv']=np.stack([np.asarray(a).copy() for a in kv])
   hashes=[digest(np.asarray(a)) for a in prefixes]
  assert all(np.isfinite(a).all() for a in payload.values());mx.clear_cache();return payload,hashes
