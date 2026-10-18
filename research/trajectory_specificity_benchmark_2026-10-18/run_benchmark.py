"""Frozen native/fixed-shape trajectory specificity benchmark."""
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'component_reuse_pilot_2026-10-18/run.py'
spec=importlib.util.spec_from_file_location('component_instrument',SOURCE)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
LOCK=json.loads((HERE/'PROTOCOL.json').read_text())
MODELS={'qwen7b':HERE.parent/'stronger_model_development_2026-10-15/models/qwen7b',
        'llama3b':HERE.parent/'llama_gemma_cross_family_2026-10-15/model'}
SITES=['person_a','person_b','verb','period']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def capture(model,ids,positions):
    with helper.instrument(model,capture=True) as records:
        h=model.model(mx.array([ids]));mx.eval(h)
        selected=mx.array(positions)
        emb=model.model.embed_tokens(mx.array([ids]))[0,selected].astype(mx.float32)
        mx.eval(emb)
        layers=[np.asarray(emb).copy()]
        for l in range(len(model.model.layers)):
            v=records[l]['outgoing'][0,selected].astype(mx.float32)
            mx.eval(v);layers.append(np.asarray(v).copy())
        arr=np.stack(layers)
        assert np.isfinite(arr).all()
        return arr


def main(key):
    assert key in LOCK['models']
    out=HERE/f'RAW_{key}.json'
    assert not out.exists() and not (HERE/f'STARTED_{key}.json').exists(),'no second run'
    metadata={'protocol_sha256':sha(HERE/'PROTOCOL.json'),'runner_sha256':sha(__file__),
              'analysis_script_before_run_sha256':sha(HERE/'analyze.py'),
              'helper_sha256':sha(SOURCE),'model_config_sha256':sha(MODELS[key]/'config.json'),
              'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    save(HERE/f'STARTED_{key}.json',metadata)
    t=time.monotonic();model,tok=load(str(MODELS[key]),tokenizer_config={'trust_remote_code':False})
    model.eval();mx.set_cache_limit(256*1024*1024)
    suffix=[tok.encode(s,add_special_tokens=False) for s in LOCK['dummy_tokens']]
    assert all(len(x)==1 for x in suffix) and suffix[0]!=suffix[1]
    suffix=[x[0] for x in suffix]
    arrays,conditions,families={},[],[]
    count=0
    for fam in LOCK['families']:
        fid=fam['id'];items={};local={}
        def run(label,ids,data):
            nonlocal count
            positions=[data['sites'][s] for s in SITES]
            p=data['sites']['period']+1
            assert ids[:p]==data['ids'][:p]
            arr=capture(model,ids,positions);count+=1
            assert count<=LOCK['max_model_forwards_per_model']
            arrays[f'{fid}__{label}']=arr;local[label]=arr
            conditions.append({'family':fid,'label':label,'ids':ids,'length':len(ids),
                               'sentence':data['sentence'],'prompt':data['prompt'],'sites':data['sites'],
                               'archive_key':f'{fid}__{label}'})
            mx.clear_cache()
            print(key,fid,label,'length',len(ids),'forward',count,flush=True)
            return arr
        for cell in LOCK['cells']:
            data=helper.build(tok,fam,cell,'agent','normal');items[cell]=data
            ids=data['ids'];assert len(ids)<=128
            run(f'{cell}__native',ids,data)
            run(f'{cell}__fixed',ids+[suffix[0]]*(128-len(ids)),data)
        data=items['active_a'];P=data['sites']['period']+1;prefix=data['ids'][:P]
        null_available=P<=64
        if null_available:
            for n in [64,65]:run(f'null_{n}',prefix+[suffix[0]]*(n-P),data)
        other=run('fixed_content_control',prefix+[suffix[1]]*(128-P),data)
        reference=local['active_a__fixed']
        difference=np.linalg.norm((other-reference).astype(np.float64),axis=-1)
        denom=np.maximum(np.linalg.norm(reference.astype(np.float64),axis=-1),1e-30)
        # First person in active_a / passive_b has the same causal token prefix.
        aa,pb=items['active_a'],items['passive_b']
        posa,posb=aa['sites']['person_a'],pb['sites']['person_a']
        same=aa['ids'][:posa+1]==pb['ids'][:posb+1]
        families.append({'family':fam,'native_lengths':{k:len(v['ids']) for k,v in items.items()},
                         'sentence_prefix_length':P,'null_available':null_available,
                         'same_shape_content_max_relative':float(np.max(difference/denom)),
                         'first_person_active_passive_prefix_identical':same})
    arrays['sites']=np.array(SITES)
    raw=HERE/f'STATES_{key}.npz';np.savez_compressed(raw,**arrays)
    save(out,{'model':key,'metadata':metadata,'families':families,'conditions':conditions,
              'model_forwards':count,'states_sha256':sha(raw),'elapsed_seconds':time.monotonic()-t,
              'technical_pass':all(f['same_shape_content_max_relative']<=1e-6 for f in families),
              'answers_scored':0,'causal_interventions':0})
    print(key,'COMPLETE',count,'forwards',round(time.monotonic()-t,1),'seconds',flush=True)


if __name__=='__main__':main(sys.argv[1])
