"""Real MLX backend. Imports/load occur only for an authorized actual run."""
from pathlib import Path
import numpy as np
from core import BLOCKS,LENGTH

ROOT=Path(__file__).resolve().parent.parent.parent

class Engine:
    def __init__(self,model_name,lock):
        import mlx.core as mx
        from mlx_lm import load
        self.mx=mx
        self.model,self.tok=load(str(ROOT/lock['models'][model_name]['path']),tokenizer_config={'trust_remote_code':False})
        self.model.eval();mx.set_cache_limit(256*1024**2)
        self.decode=self.tok.decode

    def compute(self,row,condition,prefix=(),wrapped=True,native=False,capture=True):
        from instrument import instrument
        mx=self.mx; ids=row['token_ids']+list(prefix)
        assert len(ids)<=LENGTH
        x=mx.array([ids+[row['pad']]*(LENGTH-len(ids))])
        end=len(ids)-1; original_end=row['prompt_end'];s=row['suffix_start']; arrays={}
        if wrapped:
            with instrument(self.model,row,condition,native=native) as records:
                all_logits=self.model(x)
                last_logits=all_logits[0,end].astype(mx.float32)
                original_logits=all_logits[0,original_end].astype(mx.float32)
                selected=[]; prefixes=[]; suffixes=[]
                if capture:
                    selected=[records[b][p][0,original_end].astype(mx.float32) for b in range(BLOCKS) for p in ['incoming','attention','mlp','outgoing']]
                    prefixes=[records[b]['outgoing'][0,:s].astype(mx.float32) for b in range(BLOCKS)]
                    if condition['name']=='cut_0': suffixes=[records[b]['outgoing'][0,s:original_end+1].astype(mx.float32) for b in range(BLOCKS)]
                mx.eval(last_logits,original_logits,*selected,*prefixes,*suffixes)
                if capture:
                    arrays['last']=np.stack([np.asarray(a).copy() for a in selected]).reshape(BLOCKS,4,-1)
                    arrays['prefix']=np.stack([np.asarray(a).copy() for a in prefixes])
                    if suffixes: arrays['suffix']=np.stack([np.asarray(a).copy() for a in suffixes])
                result=np.asarray(last_logits).copy(); original=np.asarray(original_logits).copy()
        else:
            all_logits=self.model(x)
            last_logits=all_logits[0,end].astype(mx.float32)
            original_logits=all_logits[0,original_end].astype(mx.float32)
            mx.eval(last_logits,original_logits)
            result=np.asarray(last_logits).copy();original=np.asarray(original_logits).copy()
        assert np.isfinite(result).all() and np.isfinite(original).all()
        assert all(np.isfinite(a).all() for a in arrays.values())
        mx.clear_cache()
        return result,arrays,original
