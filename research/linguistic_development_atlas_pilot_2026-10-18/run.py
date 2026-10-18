"""One bounded atlas pilot. --preflight performs no model load or inference.

Execution requires ANNOTATION_DECISION.json and a previously written LOCK.json.
Each forward is durably journaled before execution. No automatic reruns/resumes.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DESIGN = ROOT/'research/linguistic_development_atlas_design_2026-10-18'
MODELS = {'qwen7b': 'research/stronger_model_development_2026-10-15/models/qwen7b',
          'llama3b': 'research/llama_gemma_cross_family_2026-10-15/model'}
WIDTHS = {'qwen7b':3584,'llama3b':3072}
SCRIPT_FILES = ['run.py','instrument.py','analyze.py','PROTOCOL.json','ANNOTATION_REVIEW.md','ANNOTATION_AUDIT.json']


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()


def save(p,obj):
    p=Path(p); temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');os.replace(temp,p)


def inputs():
    rows=[json.loads(x) for x in (DESIGN/'DRAFT_STIMULI.jsonl').read_text().splitlines()]
    tokens=[json.loads(x) for x in (DESIGN/'DRAFT_TOKENIZATION.jsonl').read_text().splitlines()]
    return rows,{(r['model'],r['id']):r for r in tokens}


def preflight():
    """Validate local inputs and freeze implementation before any outputs."""
    assert not (HERE/'LOCK.json').exists(), 'lock already exists; do not overwrite'
    rows,tokens=inputs();assert len(rows)==64 and len(tokens)==128
    from analyze import self_check
    assert all(self_check().values())
    old=json.loads((DESIGN/'MANIFEST.json').read_text())
    for n in ['DRAFT_STIMULI.jsonl','DRAFT_TOKENIZATION.jsonl','TOKENIZER_AUDIT.json','DESIGN.md']:
        assert sha(DESIGN/n)==old['files'][n]['sha256'],n
    from transformers import AutoTokenizer
    metadata={}
    prior=json.loads((ROOT/'research/prefix_shape_audit_2026-10-18/SOURCE_MANIFEST.json').read_text())
    for m,path in MODELS.items():
        p=ROOT/path;t=AutoTokenizer.from_pretrained(p,local_files_only=True,trust_remote_code=False)
        pad=t.encode('x',add_special_tokens=False);assert len(pad)==1
        for r in rows:
            d=tokens[m,r['id']]
            assert t.encode(d['prompt'],add_special_tokens=False)==d['token_ids']
            assert len(d['token_ids'])<128
            assert max(d['retained_positions'])==d['original_prompt_end']
        files=sorted([*p.glob('*.safetensors'),*p.glob('*.json'),*p.glob('*.jinja')])
        metadata[m]=dict(model_path=path,padding_id=pad[0],width=WIDTHS[m],blocks=28,
                         files={str(f.relative_to(ROOT)):dict(bytes=f.stat().st_size,sha256=sha(f)) for f in files})
    # Avoid hidden runtime inference: only static source files read here.
    import sysconfig
    package=Path(sysconfig.get_paths()['purelib'])/'mlx_lm/models'
    native={str(p.relative_to(ROOT)):sha(p) for p in [package/'qwen2.py',package/'llama.py',package/'base.py']}
    lock=dict(status='FROZEN_BEFORE_ANY_MODEL_OUTPUT',date='2026-10-18',
        user_authorization='User: go dor it; bounded pilot authorized. Annotation status recorded separately.',
        source_design_commit='b6f0847105c62813bba36eba622928f0d8220d15',
        annotation_review_required_or_explicitly_waived=True,
        files={n:sha(HERE/n) for n in SCRIPT_FILES},
        input_files={str((DESIGN/n).relative_to(ROOT)):sha(DESIGN/n) for n in ['DRAFT_STIMULI.jsonl','DRAFT_TOKENIZATION.jsonl','TOKENIZER_AUDIT.json','DESIGN.md']},
        models=metadata,native_source_files=native,
        runtime={p:importlib.metadata.version(p) for p in ['mlx','mlx-lm','numpy','transformers','matplotlib']},
        prior_source_manifest_sha256=sha(ROOT/'research/prefix_shape_audit_2026-10-18/SOURCE_MANIFEST.json'),
        available_bytes=shutil.disk_usage(HERE).free,max_main_forwards=128,max_validation_forwards=4)
    assert lock['available_bytes']>3*1024**3,'Insufficient storage for planned arrays and temporary overhead'
    save(HERE/'LOCK.json',lock)
    print('PREFLIGHT_FROZEN',json.dumps({m:d['padding_id'] for m,d in metadata.items()}),flush=True)


def assert_lock():
    lock=json.loads((HERE/'LOCK.json').read_text())
    for p,h in lock['files'].items():assert sha(HERE/p)==h,('changed frozen code',p)
    for p,h in lock['input_files'].items():assert sha(ROOT/p)==h,('changed input',p)
    for p,h in lock['native_source_files'].items():assert sha(ROOT/p)==h,('changed runtime',p)
    return lock


def lse(x):
    x=np.asarray(x,dtype=np.float64);hi=np.max(x)
    return float(hi+np.log(np.exp(x-hi).sum()))


def score(logits,row,aliases,decode):
    ids={k:sorted({v[0] for v in forms.values()}) for k,forms in aliases.items()}
    assert not set(ids['Yes'])&set(ids['No'])
    scores={k:lse(logits[v]) for k,v in ids.items()}
    best=int(np.argmax(logits));expected=row['expected'];other='No' if expected=='Yes' else 'Yes'
    margin=scores[expected]-scores[other];denom=lse(logits)
    top=np.argsort(logits)[-20:][::-1]
    return dict(expected=expected,argmax_id=best,argmax_token=decode([best]),
        actual_class=next((k for k,vs in ids.items() if best in vs),None),
        class_logit_logsumexp=scores,class_probability={k:float(np.exp(v-denom)) for k,v in scores.items()},
        gold_margin=margin,pass_competence=bool(best in ids[expected] and margin>0),
        top20=[dict(id=int(i),token=decode([int(i)]),logit=float(logits[i])) for i in top])


def execute(model_name):
    lock=assert_lock()
    decision=json.loads((HERE/'ANNOTATION_DECISION.json').read_text())
    assert decision['status'] in ['human_confirmed','human_explicitly_waived'], 'human annotation gate pending'
    assert decision['review_sha256']==sha(HERE/'ANNOTATION_REVIEW.md')
    out=HERE/model_name;out.mkdir(exist_ok=True)
    assert not (out/'STARTED.json').exists(),'Refusing a second run/resume'
    save(out/'STARTED.json',dict(lock_sha256=sha(HERE/'LOCK.json'),annotation_decision_sha256=sha(HERE/'ANNOTATION_DECISION.json'),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    import mlx.core as mx
    from mlx_lm import load
    from instrument import instrument
    from analyze import technical_controls
    model,tok=load(str(ROOT/MODELS[model_name]),tokenizer_config={'trust_remote_code':False})
    model.eval();mx.set_cache_limit(256*1024**2)
    rows,tokens=inputs();aliases=json.loads((DESIGN/'TOKENIZER_AUDIT.json').read_text())['models'][model_name]['candidate_alias_ids']
    pad=lock['models'][model_name]['padding_id']
    raw=[];calls=[];controls={}

    def forward(row,kind,wrapped):
        assert len(calls)<66
        calls.append(dict(number=len(calls)+1,id=row['id'],kind=kind,wrapped=wrapped))
        save(out/'FORWARD_JOURNAL.json',calls) # consumes budget even if the call raises
        d=tokens[model_name,row['id']];ids=d['token_ids']+[pad]*(128-len(d['token_ids']))
        x=mx.array([ids]);pos=mx.array(d['retained_positions']);recon=0.0
        arrays=None
        if wrapped:
            with instrument(model) as records:
                logits_mx=model(x)[0,d['original_prompt_end']].astype(mx.float32)
                selected=[]
                for li in range(28):
                    for part in ['incoming','attention','mlp','outgoing']:
                        selected.append(records[li][part][0,pos,:].astype(mx.float32))
                mx.eval(logits_mx,*selected)
                pieces=np.stack([np.asarray(v).copy() for v in selected]).reshape(28,4,len(pos),-1)
                arrays=dict(residual=np.concatenate([pieces[0,0][None],pieces[:,3]],axis=0),
                            attention=pieces[:,1],mlp=pieces[:,2])
                error=np.linalg.norm((pieces[:,3]-pieces[:,0]-pieces[:,1]-pieces[:,2]).astype(np.float64),axis=-1)
                scale=np.maximum(np.linalg.norm(pieces[:,3].astype(np.float64),axis=-1),1e-12)
                recon=float(np.max(error/scale))
                arrays['reconstruction_error_norm']=error
                del selected,pieces,records
        else:
            logits_mx=model(x)[0,d['original_prompt_end']].astype(mx.float32);mx.eval(logits_mx)
        logits=np.asarray(logits_mx).copy()
        assert np.isfinite(logits).all()
        if arrays is not None:assert all(np.isfinite(a).all() for a in arrays.values())
        mx.clear_cache()
        return logits,arrays,recon

    def one(row):
        logits,arrays,recon=forward(row,'main',True)
        np.savez(out/(row['id']+'.npz'),**arrays)
        np.save(out/(row['id']+'_logits.npy'),logits)
        result=dict(**row,**score(logits,row,aliases,tok.decode),max_reconstruction_relative_error=recon)
        raw.append(result);save(out/'RAW.json',dict(model=model_name,rows=raw))
        print('MAIN',model_name,row['id'],repr(result['argmax_token']),row['expected'],round(result['gold_margin'],5),result['pass_competence'],flush=True)
        return logits,arrays

    dev=[r for r in rows if r['split']=='development'];held=[r for r in rows if r['split']=='held_out']
    unwrapped,_,_=forward(dev[0],'unwrapped_validation',False)
    original,original_arrays=one(dev[0])
    repeat,repeat_arrays,_=forward(dev[0],'wrapped_repeat_validation',True)
    controls['wrapped_logit_max_absolute_error']=float(np.max(np.abs(unwrapped-original)))
    controls['repeat_logit_max_absolute_error']=float(np.max(np.abs(repeat-original)))
    controls['repeat_state_max_relative_error']=max(float(np.max(np.linalg.norm((repeat_arrays[k]-original_arrays[k]).astype(np.float64),axis=-1))/max(np.max(np.linalg.norm(original_arrays[k].astype(np.float64),axis=-1)),1e-12)) for k in ['residual','attention','mlp'])
    np.savez(out/'VALIDATION_LOGITS.npz',unwrapped=unwrapped,wrapped=original,repeat=repeat)
    del original_arrays,repeat_arrays
    validation=(controls['wrapped_logit_max_absolute_error']<=.05 and
                controls['repeat_logit_max_absolute_error']<=.05 and
                controls['repeat_state_max_relative_error']<=1e-6)
    if validation:
        for row in dev[1:]:one(row)
        controls['development']=technical_controls(model_name,dev,out,tokens)
        prefix_ok=controls['development']['prefix_pass']
        eligible=[f for f in ['a01','a02'] if all(r['pass_competence'] for r in raw if r['family']==f)]
    else:prefix_ok=False;eligible=[]
    save(out/'CONTROLS.json',controls)
    allowed=validation and prefix_ok and bool(eligible)
    save(out/'DEVELOPMENT_DECISION.json',dict(validation_pass=validation,prefix_pass=prefix_ok,
        fully_competent_development_families=eligible,held_out_authorized_by_frozen_gate=allowed,
        frozen_before_held_out=True,forward_count=len(calls)))
    print('DEVELOPMENT_DECISION',model_name,allowed,eligible,flush=True)
    if allowed:
        for row in held:one(row)
        controls['held_out']=technical_controls(model_name,held,out,tokens)
        save(out/'CONTROLS.json',controls)
    save(out/'COMPLETE.json',dict(status='complete' if allowed else 'stopped_at_development_gate',
        main_forwards=len(raw),validation_forwards=len(calls)-len(raw),total_forwards=len(calls),
        lock_sha256=sha(HERE/'LOCK.json'),annotation_status=decision['status']))
    print('COMPLETE',model_name,len(calls),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--preflight',action='store_true');p.add_argument('--model',choices=MODELS);a=p.parse_args()
    if a.preflight:preflight()
    else:
        assert a.model
        execute(a.model)
