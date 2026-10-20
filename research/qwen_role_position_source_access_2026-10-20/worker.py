"""Bounded native MLX worker; exact whole-region prompt-state interchange."""
import gc
import sys
import traceback
from core import *

def main():
    import mlx.core as mx
    from stream import forward
    lock=json.loads((HERE/'LOCK.json').read_text())
    mx.set_memory_limit(3*1024**3); mx.set_cache_limit(256*1024**2)
    # A fresh lazy model is evaluated and freed one native layer at a time.

    count=0
    for line in sys.stdin:
        try:
            req=json.loads(line);count+=1;assert count<=WORKER_CAP
            row,prefix=req['row'],req['prefix'];ids=row['token_ids']+prefix
            n=len(row['token_ids']);assert len(ids)<LENGTH
            x=mx.array([ids+[row['pad']]*(LENGTH-len(ids))])
            edit=row.get('edit');collected={};readouts={};proof={};replacement=None
            if edit:
                with np.load(HERE/edit['donor_payload'],allow_pickle=False) as z:
                    donor=z['states'][CUTS.index(edit['cut'])].astype(np.float32)
                assert donor.shape==(n,3584)
                left,right=(0,row['period']+1) if edit['region']=='prefix' else (row['period']+1,n) if edit['region']=='suffix' else (0,n)
                assert edit['region'] in ['prefix','suffix','both']
                replacement=mx.array(donor[left:right])

            def tap(cut,value):
                if edit and cut==edit['cut']:
                    before=value[0,:n].astype(mx.float32);tail=value[0,n:].astype(mx.float32)
                    mx.eval(before,tail)
                    proof['pre_sha256']=array_sha(np.asarray(before))
                    proof['tail_pre_sha256']=array_sha(np.asarray(tail))
                    y=mx.concatenate([value[:,:left],replacement.astype(value.dtype)[None],value[:,right:]],axis=1)
                    after=y[0,:n].astype(mx.float32);tail_after=y[0,n:].astype(mx.float32)
                    mx.eval(after,tail_after)
                    proof.update(post_sha256=array_sha(np.asarray(after)),tail_post_sha256=array_sha(np.asarray(tail_after)),left=left,right=right,cut=cut)
                    value=y
                if req['capture']:
                    readouts[cut]=value[0,n-1].astype(mx.float32)
                    if cut in CUTS: collected[cut]=value[0,:n].astype(mx.float32)
                return value

            y=forward([ids+[row['pad']]*(LENGTH-len(ids))],ROOT/lock['model']['path'],tap=tap if req['capture'] or edit else None)
            logits=y[0,len(ids)-1].astype(mx.float32);original=y[0,n-1].astype(mx.float32)
            mx.eval(logits,original,*collected.values(),*readouts.values())
            arrays=dict(logits=np.asarray(logits).copy(),original_logits=np.asarray(original).copy())
            if req['capture']:
                arrays['states']=np.stack([np.asarray(collected[c]).copy() for c in CUTS])
                arrays['readout']=np.stack([np.asarray(readouts[c]).copy() for c in range(29)])
            assert arrays['logits'].shape==(152064,)
            assert all(np.isfinite(a).all() for a in arrays.values())
            source_hashes=[array_sha(arrays['states'][i,:row['period']+1]) for i in range(len(CUTS))] if req['capture'] else []
            for k,v in list(arrays.items()):
                half=v.astype(np.float16)
                if np.isfinite(half).all() and np.array_equal(half.astype(np.float32),v):arrays[k]=half
            p=Path(req['path']);assert not p.exists();np.savez_compressed(p,**arrays)
            del x,y,logits,original,arrays,collected,readouts,replacement
            gc.collect();mx.clear_cache()
            metrics=dict(active_mlx_bytes=mx.get_active_memory(),peak_mlx_bytes=mx.get_peak_memory(),cache_mlx_bytes=mx.get_cache_memory())
            assert metrics['active_mlx_bytes']<3*1024**3
            print(json.dumps(dict(ok=True,prefix_hashes=source_hashes,patch_proof=proof,metrics=metrics)),flush=True)
        except BaseException:
            print(json.dumps(dict(ok=False,error=traceback.format_exc())),flush=True);raise

if __name__=='__main__':main()
