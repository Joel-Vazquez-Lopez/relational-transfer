"""Copied native engine, unchanged computation; no model load at import."""
import hashlib
import numpy as np
from core import ROOT,LENGTH,BLOCKS
def digest(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

class Engine:
    def __init__(self,model_name,lock):
        import mlx.core as mx
        from mlx_lm import load
        self.mx=mx
        self.model,self.tok=load(str(ROOT/lock['models'][model_name]['path']),tokenizer_config={'trust_remote_code':False})
        self.model.eval(); mx.set_cache_limit(256*1024**2)
    def compute(self,row,mode,cut,wrapped=True,native=False):
        from instrument import instrument
        mx=self.mx; ids=row['token_ids']; p=row['period']; e=row['prompt_end']
        x=mx.array([ids+[row['pad']]*(LENGTH-len(ids))])
        arrays={}; hashes={}
        if wrapped:
            with instrument(self.model,row,mode,cut,native=native) as records:
                y=self.model(x); logits=y[0,e].astype(mx.float32)
                selected=[records[0]['incoming'][0,[p,e]].astype(mx.float32)] + [records[b]['outgoing'][0,[p,e]].astype(mx.float32) for b in range(BLOCKS)]
                # Verify unmodified prefixes at every block. Store hashes of exact arrays.
                before=[records[b]['outgoing'][0,:p].astype(mx.float32) for b in range(BLOCKS)]
                null=[records[b]['outgoing'][0,p:e+1].astype(mx.float32) for b in range(BLOCKS)] if cut==0 and mode in ['closed','frozen'] else []
                mx.eval(logits,*selected,*before,*null)
                arrays['trace']=np.stack([np.asarray(a).copy() for a in selected])
                hashes['prefix']=[digest(np.asarray(a)) for a in before]
                hashes['prefix_without_neighbor']=[digest(np.asarray(a)[:-1]) for a in before]
                if null: arrays['null_states']=np.stack([np.asarray(a).copy() for a in null])
                result=np.asarray(logits).copy()
        else:
            y=self.model(x); logits=y[0,e].astype(mx.float32); mx.eval(logits)
            result=np.asarray(logits).copy(); arrays['trace']=np.empty((0,),dtype=np.float32)
        assert np.isfinite(result).all() and all(np.isfinite(a).all() for a in arrays.values())
        mx.clear_cache()
        return result,arrays,hashes
