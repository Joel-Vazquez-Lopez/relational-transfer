"""Native forward with residuals at sentence period, question end and assistant end."""
import numpy as np
from core import ROOT,LENGTH,BLOCKS,digest
class Engine:
 def __init__(self,lock):
  import mlx.core as mx
  from mlx_lm import load
  self.mx=mx;self.model,self.tok=load(str(ROOT/lock['model']['path']),tokenizer_config={'trust_remote_code':False});self.model.eval();mx.set_cache_limit(256*1024**2)
 def compute(self,row,mode='baseline',cut=28,wrapped=True,native=False,physical_length=LENGTH):
  from instrument import instrument
  mx=self.mx;ids=row['token_ids'];e=row['prompt_end'];p=row['period'];q=row['question_end'];assert len(ids)<physical_length
  x=mx.array([ids+[row['pad']]*(physical_length-len(ids))]);arrays={};hashes={}
  if wrapped:
   assert physical_length==LENGTH
   with instrument(self.model,row,mode,cut,native=native) as records:
    y=self.model(x);logits=y[0,e].astype(mx.float32);sites=[p,q,e]
    selected=[records[0]['incoming'][0,sites].astype(mx.float32)]+[records[b]['outgoing'][0,sites].astype(mx.float32) for b in range(BLOCKS)]
    prefix=[records[b]['outgoing'][0,:p].astype(mx.float32) for b in range(BLOCKS)]
    qprefix=[records[b]['outgoing'][0,:q+1].astype(mx.float32) for b in range(BLOCKS)]
    null=[records[b]['outgoing'][0,p:e+1].astype(mx.float32) for b in range(BLOCKS)] if cut==0 and mode=='closed' else []
    mx.eval(logits,*selected,*prefix,*qprefix,*null);arrays['trace']=np.stack([np.asarray(a).copy() for a in selected]);hashes['source_prefix']=[digest(np.asarray(a)) for a in prefix];hashes['question_prefix']=[digest(np.asarray(a)) for a in qprefix]
    if null:hashes['disconnected_suffix']=digest(np.stack([np.asarray(a) for a in null]))
    result=np.asarray(logits).copy()
  else:
   y=self.model(x);z=y[0,e].astype(mx.float32);mx.eval(z);result=np.asarray(z).copy();arrays['trace']=np.empty((0,),np.float32)
  assert np.isfinite(result).all() and all(np.isfinite(a).all() for a in arrays.values());mx.clear_cache();return result,arrays,hashes
