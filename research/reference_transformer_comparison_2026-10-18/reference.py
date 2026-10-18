"""Tracr reference programs and a float64 Torch execution adapter.
Reference sorting/duplicate routines follow InterpBench cases59/108 (MIT).
"""
import sys,math
from pathlib import Path
import numpy as np
import torch
RUNTIME=Path(__file__).resolve().parent.parent/'transformer_comparison_feasibility_2026-10-18/runtime'
sys.path.insert(0,str(RUNTIME))
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
from tracr.rasp import rasp
from tracr.compiler import compiling

torch.set_num_threads(2)


def program(kind):
    if kind=='sorting':
        unique_keys=rasp.SequenceMap(lambda x,i:x+i*.00001,rasp.tokens,rasp.indices).named('unique_keys')
        lt=rasp.Select(unique_keys,unique_keys,rasp.Comparison.LT).named('lt_selector')
        rank=rasp.SelectorWidth(lt).named('sorted_position')
        selector=rasp.Select(rank,rasp.indices,rasp.Comparison.EQ).named('sorted_sequence_selector')
        sorted_seq=rasp.Aggregate(selector,rasp.tokens).named('sorted_sequence')
        matches=rasp.SequenceMap(lambda a,b:a==b,sorted_seq,rasp.indices).named('matches_index')
        failures=rasp.Select(matches,rasp.Map(lambda _:False,rasp.indices),rasp.Comparison.EQ)
        return rasp.Map(lambda x:x==0,rasp.SelectorWidth(failures)).named('is_permutation')
    equal=rasp.Select(rasp.tokens,rasp.tokens,rasp.Comparison.EQ)
    counts=rasp.SelectorWidth(equal).named('duplicate_counts')
    at_least_two=rasp.Select(counts,rasp.Map(lambda _:2,rasp.indices),rasp.Comparison.GEQ)
    duplicate=rasp.Map(lambda x:1 if x>0 else 0,rasp.SelectorWidth(at_least_two))
    return rasp.Map(lambda x:x==0,duplicate).named('is_permutation')


class Reference:
    def __init__(self,kind,n):
        self.kind,self.n=kind,n
        self.compiled=compiling.compile_rasp_to_model(program(kind),set(range(n)),n)
        c=self.compiled.model_config
        assert not c.layer_norm and not c.causal
        self.config={'heads':c.num_heads,'layers':c.num_layers,'key_size':c.key_size,'mlp_size':c.mlp_hidden_size}
        self.params={k:{a:torch.tensor(np.asarray(v),dtype=torch.float64) for a,v in obj.items()} for k,obj in self.compiled.params.items()}
        self.labels=list(self.compiled.residual_labels)
        self.width=len(self.labels)
        self.positive=next(i for i,s in enumerate(self.labels) if s.startswith('is_permutation') and s.endswith(':True'))
        self.negative=next(i for i,s in enumerate(self.labels) if s.startswith('is_permutation') and s.endswith(':False'))
        self.token_columns=[i for i,s in enumerate(self.labels) if s.startswith('tokens:') and s.split(':',1)[1] in [str(j) for j in range(n)]]
        self.zero_column=self.labels.index('tokens:0')
        target='sorted_position' if kind=='sorting' else 'duplicate_counts'
        cols=[i for i,s in enumerate(self.labels) if s==target or s.startswith(target+':') or s.startswith(target+'_')]
        assert cols,(target,self.labels)
        writes=[]
        for k in range(2*c.num_layers):
            layer=k//2
            key=f'transformer/layer_{layer}/'+('attn/linear' if k%2==0 else 'mlp/linear_2')
            if float(self.params[key]['w'][:,cols].abs().max())>0 or float(self.params[key]['b'][cols].abs().max())>0:writes.append(k+1)
        assert writes,target
        self.cut=max(writes)

    def encode(self,seqs):
        ids=np.array([self.compiled.input_encoder.encode(['compiler_bos']+list(a)) for a in seqs])
        h=self.params['token_embed']['embeddings'][ids]+self.params['pos_embed']['embeddings'][torch.arange(self.n+1)]
        return h

    def op(self,h,k):
        layer=k//2
        if k%2==0:
            base=f'transformer/layer_{layer}/attn/'
            p=self.params
            q,k1,v=[(h@p[base+name]['w']+p[base+name]['b']).reshape(h.shape[0],h.shape[1],self.config['heads'],self.config['key_size']) for name in ['query','key','value']]
            a=torch.softmax(torch.einsum('bthd,bshd->bhts',q,k1)/math.sqrt(self.config['key_size']),dim=-1)
            z=torch.einsum('bhts,bshd->bthd',a,v).reshape(h.shape[0],h.shape[1],-1)
            return h+z@p[base+'linear']['w']+p[base+'linear']['b']
        base=f'transformer/layer_{layer}/mlp/'
        a,b=self.params[base+'linear_1'],self.params[base+'linear_2']
        return h+torch.relu(h@a['w']+a['b'])@b['w']+b['b']

    def trace(self,seqs):
        h=self.encode(seqs);states=[h]
        for k in range(2*self.config['layers']):h=self.op(h,k);states.append(h)
        return states

    def margin(self,h):return (h[:,1:,self.positive]-h[:,1:,self.negative]).mean(1)

    def resume(self,h,stage):
        for k in range(stage,2*self.config['layers']):h=self.op(h,k)
        return self.margin(h)

    def clamp_tokens_zero(self,h):
        z=h.clone();z[:,1:,self.token_columns]=0;z[:,1:,self.zero_column]=1
        return z

    def compare_upstream(self,seq):
        ids=jnp.asarray([self.compiled.input_encoder.encode(['compiler_bos']+list(seq))])
        params=jax.tree_util.tree_map(lambda a:jnp.asarray(a,dtype=jnp.float64),self.compiled.params)
        ref=self.compiled.forward(params,ids).transformer_output
        ours=self.trace([seq])
        expected=[ref.input_embeddings]+list(ref.residuals)
        return max(float(np.max(abs(a.numpy()-np.asarray(b)))) for a,b in zip(ours,expected))


def preflight():
    result=[]
    with torch.no_grad():
        for kind in ['sorting','counting']:
            m=Reference(kind,3)
            for seq in [[0,1,2],[0,0,2]]:
                err=m.compare_upstream(seq)
                assert err<1e-8,err
            h=m.trace([[0,1,2]])[m.cut]
            y=float(m.resume(m.clamp_tokens_zero(h),m.cut)[0])
            expected=-1 if kind=='sorting' else 1
            assert y*expected>.5,(kind,m.cut,y,m.labels)
            result.append({'program':kind,'N':3,'width':m.width,'config':m.config,'cut':m.cut,'clamped_margin':y,'upstream_max_error':err})
    import json
    (Path(__file__).resolve().parent/'PREFLIGHT.json').write_text(json.dumps({'infrastructure_N3_only':True,'checks':result},indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':preflight()
