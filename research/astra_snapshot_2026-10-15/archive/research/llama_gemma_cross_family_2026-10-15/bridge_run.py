"""Approved local Llama comparison. Does not read credentials or new/sealed data."""
import argparse,hashlib,importlib.metadata,importlib.util,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
QWEN=ROOT.parent/'stronger_model_development_2026-10-15'
sys.path.insert(0,str(QWEN))
from run import file_sha
spec=importlib.util.spec_from_file_location('phi_prompt_helpers',ROOT.parent/'phi3_cross_family_development_2026-10-15'/'run.py')
helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)

def main(smoke=False):
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    cfg=json.loads((ROOT/'BRIDGE_PROTOCOL.json').read_text())
    for f,h in json.loads((ROOT/'BRIDGE_SOURCE_LOCK.json').read_text()).items():assert file_sha(ROOT/f)==h,f
    weight=ROOT/'model'/'model.safetensors'
    assert weight.stat().st_size==cfg['weight_bytes'] and file_sha(weight)==cfg['weight_sha256']
    model,tok=load(str(ROOT/'model'),tokenizer_config={'trust_remote_code':False});model.eval();mx.set_cache_limit(256*1024*1024)
    conf=json.loads((ROOT/'model'/'config.json').read_text());assert conf['quantization']=={'bits':4,'group_size':64}
    ids={}
    for e in cfg['events']:
        for n in [e['actor'],e['patient']]:
            a=tok.encode(n,add_special_tokens=False);b=tok.encode(' '+n,add_special_tokens=False)
            assert len(a)==len(b)==1,(n,a,b);ids[n]=sorted(set(a+b))
    e=conf['eos_token_id'];eos=set(tok.eos_token_ids)|set(e if isinstance(e,list) else [e])
    if smoke:
        text=tok.apply_chat_template([{'role':'user','content':'What is the capital of France? Answer only the city name.'}],tokenize=False,add_generation_prompt=True)
        tokens=tok.encode(text,add_special_tokens=False);cache=make_prompt_cache(model)
        a=model(mx.array([tokens]))[0,-1].astype(mx.float32);b=model(mx.array([tokens]),cache=cache)[0,-1].astype(mx.float32)
        diff=float(mx.max(mx.abs(a-b)).item());assert diff<1e-3,diff
        generated=[];token=int(b.argmax().item())
        for _ in range(12):
            if token in eos:break
            generated.append(token);z=model(mx.array([[token]]),cache=cache)[0,-1];token=int(z.argmax().item())
        response=tok.decode(generated).strip();assert 'Paris' in response,response
        report=dict(cached_uncached_max_abs=diff,response=response,eos_ids=sorted(eos),name_ids=ids,peak_memory_gb=mx.get_peak_memory()/1e9)
        (ROOT/'LOCAL_SMOKE.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));return
    assert (ROOT/'LOCAL_SMOKE.json').exists()
    meta=dict(hub_id=cfg['hub_id'],revision=cfg['revision'],quantization=conf['quantization'],native_chat_template=tok.chat_template,
        files={p.name:file_sha(p) for p in (ROOT/'model').iterdir() if p.is_file()},
        versions={n:importlib.metadata.version(n) for n in ['mlx','mlx-lm','transformers','numpy']},
        device=mx.device_info(),eos_ids=sorted(eos),name_token_ids=ids)
    out=ROOT/'llama_no_prefill.jsonl';done=set()
    if out.exists():
        for line in out.read_text().splitlines():
            r=json.loads(line);assert r['key'] not in done;done.add(r['key'])
        assert json.loads((ROOT/'BRIDGE_META.json').read_text())==meta
    else:(ROOT/'BRIDGE_META.json').write_text(json.dumps(meta,indent=2)+'\n')
    started=time.monotonic()
    with out.open('a',buffering=1) as f:
        for i,r in enumerate([r for r in helpers.jobs(cfg) if r['panel']=='strong_free']):
            if r['key'] in done:continue
            text=helpers.make_prompt(tok,cfg,r);tokens=tok.encode(text,add_special_tokens=False);assert len(tokens)<512
            cache=make_prompt_cache(model);z=model(mx.array([tokens]),cache=cache)[0,-1].astype(mx.float32)
            lp=z-mx.logsumexp(z)
            scores={role:float(mx.logsumexp(lp[mx.array(ids[r[role]])]).item()) for role in ['actor','patient']}
            assert all(math.isfinite(v) for v in scores.values())
            first=int(z.argmax().item());token=first;generated=[]
            for _ in range(12):
                if token in eos:break
                generated.append(token);next_logits=model(mx.array([[token]]),cache=cache)[0,-1];token=int(next_logits.argmax().item())
            answer=tok.decode(generated).strip()
            record=dict(**r,prompt=text,prompt_tokens=len(tokens),prompt_sha256=hashlib.sha256(text.encode()).hexdigest(),
                token_ids_sha256=hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),actor_margin=scores['actor']-scores['patient'],
                actor_score=scores['actor'],patient_score=scores['patient'],first_token_id=first,first_answer=tok.decode([first]).strip(),
                generated_token_ids=generated,answer=answer,gold=r[r['role']],correct=answer==r[r['role']],valid=answer in [r['actor'],r['patient']],
                ended_eos=token in eos,peak_memory_gb=mx.get_peak_memory()/1e9)
            f.write(json.dumps(record)+'\n');del cache,z,lp
            if (i+1)%48==0:
                mx.clear_cache();print(json.dumps(dict(completed=i+1,total=192,elapsed_s=round(time.monotonic()-started,1),peak_memory_gb=mx.get_peak_memory()/1e9)),flush=True)
    assert len(out.read_text().splitlines())==192
    (ROOT/'BRIDGE_COMPLETE.json').write_text(json.dumps(dict(records=192,sha256=file_sha(out),seconds=time.monotonic()-started,peak_memory_gb=mx.get_peak_memory()/1e9),indent=2)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');main(parser.parse_args().smoke)
