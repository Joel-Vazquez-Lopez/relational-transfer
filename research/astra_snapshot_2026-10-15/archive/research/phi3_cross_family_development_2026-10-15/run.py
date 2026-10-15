"""One frozen Phi diagnostic. Imports only old prompt construction, never sealed data."""
import argparse,gc,hashlib,importlib.metadata,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
QWEN=ROOT.parent/'stronger_model_development_2026-10-15'
sys.path.insert(0,str(QWEN))
from design import sentence
from run import jobs as qwen_jobs, prompt, file_sha

def jobs(cfg):
    rows=qwen_jobs(cfg)
    extras=[]
    for r in rows:
        if r['panel']=='legacy':
            # Reconstruct the semantic content with reversed lists below.
            x=dict(r,order='ba');x['key']=x['key'][:-2]+'ba';extras.append(x)
    rows+=extras
    assert len(rows)==1728 and len(set(x['key'] for x in rows))==1728
    return rows

def make_prompt(tok,cfg,r):
    if r['panel']!='legacy' or r['order']=='ab':return prompt(tok,cfg,r)
    s=sentence(r,r['form'])
    return tok.apply_chat_template([{'role':'system','content':'Answer with exactly one of the two candidate names. Output only the name.'},
        {'role':'user','content':f"Sentence: {s}\nCandidates: {r['b']}, {r['a']}\nQuestion: {r['question']}"}],tokenize=False,add_generation_prompt=True)

def paths_for(tok,name):
    paths=[tok.encode(name,add_special_tokens=False),tok.encode(' '+name,add_special_tokens=False)]
    assert len(paths[0])==1 and len(paths[1])==2 and paths[1][1:]==paths[0],(name,paths)
    assert paths[0][0]!=paths[1][0]
    return paths

def run(smoke):
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    cfg=json.loads((ROOT/'PROTOCOL.json').read_text())
    lock=json.loads((ROOT/'SOURCE_LOCK.json').read_text())
    for f,h in lock.items():assert file_sha(ROOT/f)==h,f
    p=ROOT/'model';weight=p/'model.safetensors'
    assert weight.stat().st_size==cfg['weight_bytes'] and file_sha(weight)==cfg['weight_sha256']
    model,tok=load(str(p),tokenizer_config={'trust_remote_code':False});model.eval()
    mx.set_cache_limit(256*1024*1024)
    conf=json.loads((p/'config.json').read_text());assert conf['quantization']=={'bits':4,'group_size':64}
    paths={n:paths_for(tok,n) for e in cfg['events'] for n in [e['actor'],e['patient']]}
    prefixes={tuple(v[1][:-1]) for v in paths.values()};assert len(prefixes)==1
    space=list(next(iter(prefixes)));assert len(space)==1
    eos=set(tok.eos_token_ids)|{conf['eos_token_id']}
    def scores(tokens,z):
        z=z.astype(mx.float32);lp=z-mx.logsumexp(z)
        later=model(mx.array([tokens+space]))[0,-1].astype(mx.float32)
        slp=later-mx.logsumexp(later)
        return lp,slp
    if smoke:
        tokens=tok.encode(tok.apply_chat_template([{'role':'user','content':'What is the capital of France? Answer with only the city name.'}],tokenize=False,add_generation_prompt=True),add_special_tokens=False)
        unc=model(mx.array([tokens]))[0,-1].astype(mx.float32);cache=make_prompt_cache(model)
        cached=model(mx.array([tokens]),cache=cache)[0,-1].astype(mx.float32)
        diff=float(mx.max(mx.abs(unc-cached)).item());assert diff<1e-3,diff
        lp,slp=scores(tokens,cached)
        # Independent teacher-forced validation of both complete candidate paths.
        residual=[]
        for name in ['Amy','Susan']:
            for path in paths[name]:
                logits=model(mx.array([tokens+path[:-1]]))[0].astype(mx.float32)
                full_lp=logits-mx.logsumexp(logits,axis=-1,keepdims=True)
                direct=sum(float(full_lp[len(tokens)-1+i,t].item()) for i,t in enumerate(path))
                shared=float(lp[path[0]].item())+(float(slp[path[1]].item()) if len(path)>1 else 0)
                residual.append(abs(direct-shared))
        assert max(residual)<1e-3,residual
        ids=[];token=int(cached.argmax().item())
        for _ in range(12):
            if token in eos:break
            ids.append(token);z=model(mx.array([[token]]),cache=cache)[0,-1];token=int(z.argmax().item())
        response=tok.decode(ids).strip();assert 'Paris' in response,response
        report=dict(cached_uncached_max_abs=diff,path_scoring_max_abs=max(residual),response=response,eos_ids=sorted(eos),
                    native_template=tok.chat_template,name_paths=paths,peak_memory_gb=mx.get_peak_memory()/1e9)
        (ROOT/'SMOKE.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True);return
    assert (ROOT/'SMOKE.json').exists()
    meta=dict(hub_id=cfg['hub_id'],revision=cfg['revision'],source_lock=lock,model_files={x.name:file_sha(x) for x in p.iterdir() if x.is_file()},
              versions={x:importlib.metadata.version(x) for x in ['mlx','mlx-lm','transformers','numpy']},name_paths=paths,eos_ids=sorted(eos),
              template=tok.chat_template,device=mx.device_info(),quantization=conf['quantization'])
    out=ROOT/'phi3.jsonl';done=set()
    if out.exists():
        for line in out.read_text().splitlines():
            r=json.loads(line);assert r['key'] not in done;done.add(r['key'])
        assert json.loads((ROOT/'META.json').read_text())==meta
    else:(ROOT/'META.json').write_text(json.dumps(meta,indent=2)+'\n')
    start=time.monotonic()
    with out.open('a',buffering=1) as f:
        for i,r in enumerate(jobs(cfg)):
            if r['key'] in done:continue
            t=time.monotonic();ptext=make_prompt(tok,cfg,r);tokens=tok.encode(ptext,add_special_tokens=False);assert len(tokens)<512
            cache=make_prompt_cache(model);z=model(mx.array([tokens]),cache=cache)[0,-1].astype(mx.float32)
            lp,slp=scores(tokens,z);candidate={};path_scores={}
            for role in ['actor','patient']:
                bare,spaced=paths[r[role]]
                v=[float(lp[bare[0]].item()),float((lp[spaced[0]]+slp[spaced[1]]).item())]
                path_scores[role]=v;candidate[role]=float(mx.logsumexp(mx.array(v)).item())
            first=int(z.argmax().item());token=first;ids=[]
            for _ in range(12):
                if token in eos:break
                ids.append(token);znext=model(mx.array([[token]]),cache=cache)[0,-1];token=int(znext.argmax().item())
            answer=tok.decode(ids).strip()
            record=dict(**r,prompt=ptext,prompt_tokens=len(tokens),prompt_sha256=hashlib.sha256(ptext.encode()).hexdigest(),
                token_ids_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),actor_margin=candidate['actor']-candidate['patient'],
                actor_score=candidate['actor'],patient_score=candidate['patient'],path_log_probabilities=path_scores,
                first_token_id=first,first_answer=tok.decode([first]).strip(),generated_token_ids=ids,answer=answer,gold=r[r['role']],
                correct=answer==r[r['role']],valid=answer in [r['actor'],r['patient']],ended_eos=token in eos,
                seconds=time.monotonic()-t,peak_memory_gb=mx.get_peak_memory()/1e9)
            f.write(json.dumps(record)+'\n');del cache,z,lp,slp
            if (i+1)%48==0:
                mx.clear_cache();print(json.dumps(dict(completed=i+1,total=1728,seconds=round(time.monotonic()-start,1),peak_memory_gb=mx.get_peak_memory()/1e9)),flush=True)
    assert len(out.read_text().splitlines())==1728
    (ROOT/'COMPLETE.json').write_text(json.dumps(dict(rows=1728,sha256=file_sha(out),elapsed_s=time.monotonic()-start,peak_memory_gb=mx.get_peak_memory()/1e9),indent=2)+'\n')
    del model;gc.collect();mx.clear_cache()
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');run(parser.parse_args().smoke)
