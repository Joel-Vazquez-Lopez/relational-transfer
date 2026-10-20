"""One four-row batch against four exposed singleton vocabulary vectors."""
import json,time
from pathlib import Path
import numpy as np
from check import resources
from stream import forward

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent

def main():
    assert not (HERE/'BATCH_STARTED.json').exists()
    initial=resources();start=time.monotonic()
    (HERE/'BATCH_STARTED.json').write_text(json.dumps(dict(criterion='Four previously exposed prompts, fixed4x128 streaming; each full vocabulary error<=.05; peak MLX<3GiB; no new linguistic test prompts',initial=initial),indent=2)+'\n')
    import mlx.core as mx
    mx.set_memory_limit(3*2**30);mx.set_cache_limit(32*2**20)
    prior=ROOT/'research/three_role_native_development_2026-10-20'
    answers=[json.loads(l) for l in (prior/'ANSWERS.jsonl').read_text().splitlines()][:4]
    index={r['id']:r for r in json.loads((prior/'INPUTS.json').read_text())}
    rows=[index[r['id']] for r in answers]
    inputs=[r['token_ids']+[r['pad']]*(128-len(r['token_ids'])) for r in rows]
    observations=[]
    def check(cut):
        r=resources();assert r['swap_bytes']-initial['swap_bytes']<2**30 and time.monotonic()-start<120
        assert mx.get_active_memory()<3*2**30
        observations.append(dict(cut=cut,active_mlx_bytes=mx.get_active_memory(),peak_mlx_bytes=mx.get_peak_memory(),**r))
    logits=forward(inputs,ROOT/'research/stronger_model_development_2026-10-15/models/qwen7b',check=check)
    selected=mx.stack([logits[i,len(row['token_ids'])-1].astype(mx.float32) for i,row in enumerate(rows)]);mx.eval(selected)
    actual=np.asarray(selected).copy();np.savez_compressed(HERE/'BATCH_REFERENCE.npz',logits=actual)
    comparisons=[]
    for i,(row,answer) in enumerate(zip(rows,answers)):
        with np.load(prior/answer['base_payload'],allow_pickle=False) as z:expected=z['logits'].astype(np.float32)
        error=float(abs(actual[i]-expected).max())
        comparisons.append(dict(id=row['id'],max_error=error,exact_equal=bool(np.array_equal(actual[i],expected)),argmax_equal=bool(actual[i].argmax()==expected.argmax())))
    result=dict(status='PASS' if all(r['max_error']<=.05 for r in comparisons) else 'FAIL',comparisons=comparisons,elapsed_seconds=time.monotonic()-start,peak_mlx_bytes=mx.get_peak_memory(),observations=observations,new_linguistic_test_prompts=0)
    (HERE/'BATCH_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='observations'}));assert result['status']=='PASS'

if __name__=='__main__':main()
