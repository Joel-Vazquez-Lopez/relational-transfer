"""One exposed-reference engineering forward, no new linguistic experiment."""
import gc
import hashlib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent

def resources():
    pressure=int(subprocess.check_output(['/usr/sbin/sysctl','-n','kern.memorystatus_vm_pressure_level'],text=True))
    text=subprocess.check_output(['/usr/sbin/sysctl','vm.swapusage'],text=True)
    m=re.search(r'used\s*=\s*([0-9.]+)([MG])',text);assert m
    swap=float(m[1])*(2**20 if m[2]=='M' else 2**30)
    free=shutil.disk_usage(HERE).free
    assert pressure==1 and swap<4*2**30 and free>8*2**30,(pressure,swap,free)
    return dict(memory_pressure=pressure,swap_bytes=swap,free_disk_bytes=free)

def main():
    assert not (HERE/'STARTED.json').exists(),'One exposed-reference attempt; no overwrite'
    initial=resources();start=time.monotonic()
    (HERE/'STARTED.json').write_text(json.dumps(dict(initial=initial,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),criterion='One saved exposed Qwen reference, full vocabulary max error<=.05, recorded MLX peak<3GiB, no pressure/swap guard failure',role_study='Llama native gate failed; no causal interpretation or new linguistic outputs in this engineering check'),indent=2)+'\n')
    paths=[ROOT/'research/stronger_model_development_2026-10-15/models/qwen7b'/n for n in ['config.json','model.safetensors']]
    paths += [ROOT/'research/three_role_native_development_2026-10-20'/n for n in ['CONTROL_INPUT.json','REFERENCE_CONTROL.npz']]
    hashes={}
    for path in paths:
        digest=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(2**20),b''):digest.update(block)
        hashes[str(path.relative_to(ROOT))]=digest.hexdigest()
    (HERE/'SOURCE_HASHES.json').write_text(json.dumps(hashes,indent=2)+'\n')
    import mlx.core as mx
    from mlx_lm.utils import load_model
    from mlx_lm.models.base import create_attention_mask
    mx.set_memory_limit(3*2**30);mx.set_cache_limit(32*2**20)
    ref=ROOT/'research/three_role_native_development_2026-10-20'
    row=json.loads((ref/'CONTROL_INPUT.json').read_text())
    model,config=load_model(ROOT/'research/stronger_model_development_2026-10-15/models/qwen7b',lazy=True)
    assert not config.get('tie_word_embeddings',False) and len(model.model.layers)==28
    observations=[]
    def record(stage):
        r=resources();assert r['swap_bytes']-initial['swap_bytes']<2**30
        r.update(stage=stage,active_mlx_bytes=mx.get_active_memory(),peak_mlx_bytes=mx.get_peak_memory(),elapsed_seconds=time.monotonic()-start)
        assert r['active_mlx_bytes']<3*2**30 and r['elapsed_seconds']<120,r
        observations.append(r)
        (HERE/'PROGRESS.json').write_text(json.dumps(observations,indent=2)+'\n')
    record('lazy_load')
    ids=row['token_ids'];x=mx.array([ids+[row['pad']]*(128-len(ids))])
    h=model.model.embed_tokens(x);mx.eval(h)
    model.model.embed_tokens=None;gc.collect();mx.clear_cache();record('embedding')
    mask=create_attention_mask(h,None)
    for i in range(28):
        block=model.model.layers[i]
        h=block(h,mask,cache=None);mx.eval(h)
        model.model.layers[i]=None;del block;gc.collect();mx.clear_cache();record(f'block_{i+1}')
    h=model.model.norm(h);mx.eval(h)
    y=model.lm_head(h);z=y[0,len(ids)-1].astype(mx.float32);mx.eval(z)
    logits=np.asarray(z).copy();np.savez_compressed(HERE/'STREAM_REFERENCE.npz',logits=logits)
    record('readout')
    with np.load(ref/'REFERENCE_CONTROL.npz',allow_pickle=False) as a:expected=a['logits'].astype(np.float32)
    error=float(np.max(np.abs(logits-expected)))
    result=dict(status='PASS' if error<=.05 and mx.get_peak_memory()<3*2**30 else 'FAIL',full_vocabulary_max_error=error,exact_equal=bool(np.array_equal(logits,expected)),argmax_equal=bool(logits.argmax()==expected.argmax()),peak_mlx_bytes=mx.get_peak_memory(),elapsed_seconds=time.monotonic()-start,observations=observations,exposed_reference_id=row['id'],additional_linguistic_test_prompts=0,novel_method_claim=False,scope='Sequential native layers with evaluated outputs and freed previous modules; existing quantized weights, unchanged token shape and per-layer operations. Only one exposed prompt validated. No KV cache, patch or future full-generation equivalence established.')
    (HERE/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='observations'}))
    assert result['status']=='PASS'

if __name__=='__main__':main()
