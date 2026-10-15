"""Resumable sequential MLX inference. Never reads sealed or new review data."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

from design import ROOT, sentence, sha

def file_sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def jobs(cfg):
    out=[]
    for e in cfg['events']:
        for form in cfg['forms']:
            for wording,question in cfg['legacy_questions'].items():
                role='actor' if wording.startswith('actor_') else 'patient'
                out.append(dict(**e,panel='legacy',form=form,wording=wording,role=role,order='ab',question=question.format(verb=e['verb'])))
    for e in cfg['events']:
        if e['id'] not in cfg['strong_event_ids']: continue
        for form in cfg['forms']:
            for wording,qs in cfg['strong_questions'].items():
                for role,q in qs.items():
                    for panel,orders in [('strong_free',['none']),('strong_list',['ab','ba'])]:
                        for order in orders:
                            out.append(dict(**e,panel=panel,form=form,wording=wording,role=role,order=order,question=q.format(verb=e['verb'])))
    assert len(out)==1152
    for x in out:
        x['key']='|'.join(x[k] for k in ['panel','id','form','wording','role','order'])
    assert len(set(x['key'] for x in out))==len(out)
    return out

def prompt(tok,cfg,r):
    s=sentence(r,r['form'])
    if r['panel']=='legacy':
        system='Answer with exactly one of the two candidate names. Output only the name.'
        user=f"Sentence: {s}\nCandidates: {r['a']}, {r['b']}\nQuestion: {r['question']}"
        suffix=''
    else:
        system=cfg['strong_system']
        candidates=''
        if r['panel']=='strong_list':
            names=(r['a'],r['b']) if r['order']=='ab' else (r['b'],r['a'])
            candidates=f'\nCandidates: {names[0]}, {names[1]}'
        user=f"Sentence: {s}{candidates}\nQuestion: {r['question']}"
        suffix=cfg['strong_assistant_prefix']
    return tok.apply_chat_template([dict(role='system',content=system),dict(role='user',content=user)],tokenize=False,add_generation_prompt=True)+suffix

def run(name,smoke=False):
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    cfg=json.loads((ROOT/'PROTOCOL.json').read_text())
    lock=json.loads((ROOT/'SOURCE_LOCK.json').read_text())
    for rel,h in lock.items(): assert file_sha(ROOT/rel)==h,rel
    mx.set_cache_limit(256*1024*1024)
    path=ROOT/'models'/name
    model,tok=load(str(path),tokenizer_config={'trust_remote_code':False})
    model.eval()
    config=json.loads((path/'config.json').read_text())
    assert config['quantization']=={'group_size':64,'bits':4}
    ids={}
    for e in cfg['events']:
        for n in [e['actor'],e['patient']]:
            bare=tok.encode(n,add_special_tokens=False); spaced=tok.encode(' '+n,add_special_tokens=False)
            assert len(bare)==len(spaced)==1,(n,bare,spaced)
            ids[n]=sorted(set(bare+spaced))
    eos=set(tok.eos_token_ids)
    t0=time.monotonic()
    if smoke:
        tokens=tok.encode('The capital of France is',add_special_tokens=False)
        uncached=model(mx.array([tokens]))[0,-1].astype(mx.float32)
        cache=make_prompt_cache(model)
        cached=model(mx.array([tokens]),cache=cache)[0,-1].astype(mx.float32)
        diff=float(mx.max(mx.abs(uncached-cached)).item())
        assert diff<1e-3,diff
        report=dict(model=name,max_cached_uncached_difference=diff,next_token=tok.decode([int(cached.argmax().item())]),
                    memory_gb=mx.get_peak_memory()/1e9,seconds=time.monotonic()-t0)
        (ROOT/f'SMOKE_{name}.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)
        return
    meta=dict(model=name,hub_id=cfg['models'][name][0],revision=cfg['models'][name][1],
              source_lock=lock,files={p.name:file_sha(p) for p in path.iterdir() if p.is_file()},
              versions={p:importlib.metadata.version(p) for p in ['mlx','mlx-lm','transformers','numpy']},
              device=mx.device_info(),name_token_ids=ids,quantization=config['quantization'])
    dest=ROOT/f'{name}.jsonl'
    done={}
    if dest.exists():
        for line in dest.read_text().splitlines():
            r=json.loads(line); assert r['key'] not in done
            done[r['key']]=r
        assert json.loads((ROOT/f'{name}_META.json').read_text())==meta
    else:
        (ROOT/f'{name}_META.json').write_text(json.dumps(meta,indent=2)+'\n')
    rows=jobs(cfg)
    with dest.open('a',buffering=1) as f:
        for i,r in enumerate(rows):
            if r['key'] in done: continue
            start=time.monotonic()
            p=prompt(tok,cfg,r); tokens=tok.encode(p,add_special_tokens=False)
            assert len(tokens)<512
            cache=make_prompt_cache(model) if r['panel']=='strong_free' else None
            z=model(mx.array([tokens]),cache=cache)[0,-1].astype(mx.float32)
            actor_score=float(mx.logsumexp(z[mx.array(ids[r['actor']])]).item())
            patient_score=float(mx.logsumexp(z[mx.array(ids[r['patient']])]).item())
            first=int(z.argmax().item()); generated=[]; token=first
            if r['panel']=='strong_free':
                for _ in range(12):
                    if token in eos: break
                    generated.append(token)
                    znext=model(mx.array([[token]]),cache=cache)[0,-1]
                    token=int(znext.argmax().item())
            else:
                generated=[first]
            answer=tok.decode(generated).strip()
            record=dict(**r,prompt=p,prompt_sha256=hashlib.sha256(p.encode()).hexdigest(),
                        token_ids_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                        prompt_tokens=len(tokens),actor_margin=actor_score-patient_score,
                        actor_score=actor_score,patient_score=patient_score,
                        first_token_id=first,first_answer=tok.decode([first]).strip(),
                        generated_token_ids=generated,answer=answer,gold=r[r['role']],
                        correct=answer==r[r['role']],valid=answer in [r['actor'],r['patient']],
                        ended_eos=(token in eos) if r['panel']=='strong_free' else None,
                        seconds=time.monotonic()-start,peak_memory_gb=mx.get_peak_memory()/1e9)
            f.write(json.dumps(record)+'\n')
            del z,cache
            if (i+1)%48==0:
                mx.clear_cache()
                print(json.dumps(dict(model=name,completed=i+1,total=len(rows),elapsed_s=round(time.monotonic()-t0,1),peak_memory_gb=mx.get_peak_memory()/1e9)),flush=True)
    assert len(dest.read_text().splitlines())==len(rows)
    (ROOT/f'{name}_COMPLETE.json').write_text(json.dumps(dict(rows=len(rows),sha256=file_sha(dest),elapsed_s=time.monotonic()-t0,peak_memory_gb=mx.get_peak_memory()/1e9),indent=2)+'\n')
    del model,tok
    gc.collect(); mx.clear_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('model',choices=['qwen05','qwen15','qwen7b']);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();run(a.model,a.smoke)
