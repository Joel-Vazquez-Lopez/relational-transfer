"""Frozen audit of identical-prefix measurements under varying execution shapes."""
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'component_reuse_pilot_2026-10-18'
sys.path.insert(0,str(SOURCE))
import run as instrument_source

LOCK=json.loads((HERE/'PROTOCOL.json').read_text())
OLD=json.loads((SOURCE/'PROTOCOL.json').read_text())
BASE=json.loads((SOURCE/'BASELINES.json').read_text())['rows']
MODELS={'qwen7b':HERE.parent/'stronger_model_development_2026-10-15/models/qwen7b',
        'llama3b':HERE.parent/'llama_gemma_cross_family_2026-10-15/model'}
PARTS=instrument_source.PARTS
SITES=instrument_source.SITES


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def measure(a,b):
    a,b=a.astype(np.float64),b.astype(np.float64)
    na,nb=np.linalg.norm(a),np.linalg.norm(b)
    return {'absolute_l2':float(np.linalg.norm(a-b)),
            'relative_l2':float(np.linalg.norm(a-b)/max(nb,1e-30)),
            'max_coordinate':float(np.max(np.abs(a-b))),
            'cosine':float(a@b/(na*nb)) if na*nb else None}


def capture(model,ids,positions):
    with instrument_source.instrument(model,capture=True) as records:
        h=model.model(mx.array([ids]))
        mx.eval(h)
        rows=[]
        for l in range(len(model.model.layers)):
            parts=[]
            for part in PARTS:
                v=records[l][part][0,mx.array(positions)].astype(mx.float32)
                mx.eval(v)
                parts.append(np.asarray(v).copy())
            rows.append(np.stack(parts))
        return np.stack(rows)


def main(key):
    assert key in LOCK['models']
    output=HERE/f'RAW_{key}.json'
    assert not output.exists() and not (HERE/f'STARTED_{key}.json').exists(),'refusing a second audit run'
    metadata={'protocol_sha256':sha(HERE/'PROTOCOL.json'),'runner_sha256':sha(__file__),
              'instrument_source_sha256':sha(SOURCE/'run.py'),'baseline_source_sha256':sha(SOURCE/'BASELINES.json'),
              'model_config_sha256':sha(MODELS[key]/'config.json'),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    save(HERE/f'STARTED_{key}.json',metadata)
    start=time.monotonic()
    model,tok=load(str(MODELS[key]),tokenizer_config={'trust_remote_code':False})
    model.eval();mx.set_cache_limit(256*1024*1024)
    suffix=[tok.encode(s,add_special_tokens=False) for s in LOCK['suffixes']]
    assert all(len(s)==1 for s in suffix) and suffix[0]!=suffix[1]
    suffix=[s[0] for s in suffix]
    q=model.model.layers[0].self_attn.q_proj
    weight=mx.dequantize(q.weight,q.scales,q.biases,group_size=q.group_size,bits=q.bits).astype(mx.float32)
    mx.eval(weight)
    saved,measurements,projections,conditions,families={},{},{},[],[]
    fingerprints={}
    forwards=0
    for fid in LOCK['families']:
        fam=next(f for f in OLD['families'] if f['id']==fid)
        data=instrument_source.build(tok,fam,'active_a','agent','normal')
        if key=='qwen7b':
            source=next(r for r in BASE if r['family']==fid and r['condition']=='active_a' and r['question_key']=='agent' and r['mapping']=='normal')
            assert data['ids']==source['ids']
        positions=[data['sites'][s] for s in SITES]
        prefix=data['ids'][:data['sites']['period']+1]
        P=len(prefix)
        lengths=sorted(set(n for n in [P,P+1,60,63,64,65,67,68,96,128] if n>=P))
        local={}
        def run(label,ids,kind,length,suf,save_projection=True):
            nonlocal forwards
            assert ids[:P]==prefix and max(positions)<P
            array=capture(model,ids,positions);forwards+=1
            assert forwards<=LOCK['maximum_model_forwards_per_model']
            fingerprint=hashlib.sha256(array.tobytes()).hexdigest()
            archive_key=fingerprints.get(fingerprint)
            if archive_key is None:
                archive_key=f'{fid}__{label}'
                saved[archive_key]=array
                fingerprints[fingerprint]=archive_key
            local[label]=array
            conditions.append({'family':fid,'label':label,'kind':kind,'length':length,'suffix':suf,'ids':ids,
                               'archive_key':archive_key,'tensor_sha256':fingerprint})
            if save_projection:
                emb=model.model.embed_tokens(mx.array([ids]))
                norm=model.model.layers[0].input_layernorm(emb)
                native=q(norm)
                selected=norm[0,mx.array(positions)]
                rowwise=mx.concatenate([q(selected[i:i+1]) for i in range(len(positions))],axis=0)
                dense=norm.astype(mx.float32)@weight.T
                if 'bias' in q:dense=dense+q.bias.astype(mx.float32)
                values=[emb[0,mx.array(positions)],selected,native[0,mx.array(positions)],rowwise,dense[0,mx.array(positions)]]
                mx.eval(*values)
                projections[(fid,label)]=[np.asarray(v).astype(np.float32).copy() for v in values]
            mx.clear_cache()
            print(key,fid,label,'length',len(ids),'forwards',forwards,flush=True)
            return array
        for n,s in itertools.product(lengths,range(2)):
            run(f'n{n}_s{s}',prefix+[suffix[s]]*(n-P),'sweep',n,s)
        for n in [P,63,67]:
            if n in lengths:run(f'repeat_n{n}',prefix+[suffix[0]]*(n-P),'repeat',n,0,False)
        # Compare every state to the shortest prefix reference, without layer selection.
        reference=local[f'n{P}_s0']
        comparisons=[]
        def compare(label,a,b,comparison):
            for l,p,s in itertools.product(range(a.shape[0]),range(4),range(6)):
                comparisons.append({'family':fid,'label':label,'comparison':comparison,'layer_zero_indexed':l,
                                    'part':PARTS[p],'site':SITES[s],**measure(a[l,p,s],b[l,p,s])})
        for n,s in itertools.product(lengths,range(2)):
            label=f'n{n}_s{s}'
            compare(label,local[label],reference,'versus_prefix_only')
        for n in lengths:
            compare(f'n{n}',local[f'n{n}_s1'],local[f'n{n}_s0'],'same_shape_content')
        for before,after in zip(lengths,lengths[1:]):
            compare(f'n{before}_to_n{after}',local[f'n{after}_s0'],local[f'n{before}_s0'],'adjacent_lengths')
        for n in [P,63,67]:
            if n in lengths:compare(f'repeat_n{n}',local[f'repeat_n{n}'],local[f'n{n}_s0'],'repeat')
        if key=='qwen7b':
            for question,mapping in itertools.product(OLD['questions'],OLD['code_maps']):
                item=instrument_source.build(tok,fam,'active_a',question,mapping)
                for padded in [False,True]:
                    ids=item['ids']
                    assert len(ids)<=128
                    if padded:ids=ids+[suffix[0]]*(128-len(ids))
                    run(f'real_{question}_{mapping}_{padded}',ids,'real_padded' if padded else 'real_native',len(ids),None,False)
            for padded in [False,True]:
                ref=local[f'real_agent_normal_{padded}']
                for question,mapping in itertools.product(OLD['questions'],OLD['code_maps']):
                    label=f'real_{question}_{mapping}_{padded}'
                    compare(label,local[label],ref,'real_padded' if padded else 'real_native')
        measurements[fid]=comparisons
        # Isolate the first Q projection under identical prefix inputs.
        project=[]
        pref=projections[(fid,f'n{P}_s0')]
        for n,s in itertools.product(lengths,range(2)):
            label=f'n{n}_s{s}'
            value=projections[(fid,label)]
            for part,(a,b) in enumerate(zip(value,pref)):
                for si in range(6):
                    project.append({'label':label,'length':n,'suffix':s,'stage':['embedding','input_rmsnorm','q_native','q_rowwise','q_dequantized_float32'][part],
                                    'site':SITES[si],**measure(a[si],b[si])})
        families.append({'id':fid,'prefix_ids':prefix,'prefix_length':P,'prefix_text':tok.decode(prefix),
                         'sites':data['sites'],'lengths':lengths,'projection_metrics':project})
    # Retain tensors in an archive; all numerical comparisons are separately explicit.
    saved['parts']=np.array(PARTS);saved['sites']=np.array(SITES)
    raw=HERE/f'STATES_{key}.npz';np.savez_compressed(raw,**saved)
    save(output,{'model':key,'metadata':metadata,'families':families,'conditions':conditions,
                 'measurements':measurements,'model_forwards':forwards,'states_sha256':sha(raw),
                 'elapsed_seconds':time.monotonic()-start,'suffix_token_ids':suffix})
    print(key,'COMPLETE',forwards,'forwards',round(time.monotonic()-start,1),'seconds',flush=True)


if __name__=='__main__':main(sys.argv[1])
