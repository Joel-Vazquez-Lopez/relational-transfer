"""Short-lived native MLX process. Only serializable local requests are accepted."""
import contextlib
import gc
import json
import sys
import traceback
from pathlib import Path
import numpy as np
from core import HERE, ROOT, LENGTH, WORKER_CAP
from replacement import replace


def main():
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    lock = json.loads((HERE/'LOCK.json').read_text())
    mx.set_memory_limit(6*1024**3)
    mx.set_cache_limit(256*1024**2)
    model, tok = load(str(ROOT/lock['model']['path']), tokenizer_config={'trust_remote_code':False})
    model.eval()

    class Boundary(nn.Module):
        def __init__(self, original, record, index, edit, vectors):
            super().__init__()
            self.original, self.record = original, record
            self.use_sliding = getattr(original,'use_sliding',False)
            self.index,self.edit,self.vectors=index,edit,vectors
        def __call__(self, x, mask=None, cache=None):
            assert cache is None
            self.record['incoming'] = x
            y = self.original(x, mask, cache=cache)
            if self.edit is not None and self.index==13:
                y=replace(mx,y,self.edit['recipient_positions'],self.vectors)
            self.record['outgoing'] = y
            return y

    @contextlib.contextmanager
    def capture(edit, vectors):
        layers = model.model.layers
        assert len(layers) == 28
        originals, records = list(layers), [{} for _ in layers]
        for i,b in enumerate(originals):
            layers[i] = Boundary(b, records[i],i,edit,vectors)
        try:
            yield records
        finally:
            for i,b in enumerate(originals):
                layers[i] = b

    count = 0
    for line in sys.stdin:
        request = json.loads(line)
        try:
            count += 1
            assert count <= WORKER_CAP
            row, prefix = request['row'], request['prefix']
            edit=row.get('edit');vectors=None
            if edit is not None:
                with np.load(HERE/edit['donor_payload'],allow_pickle=False) as saved:
                    vectors=saved['trace'][14,edit['donor_sites']].astype(np.float32)
            ids = row['token_ids'] + prefix
            assert len(ids) < LENGTH
            x = mx.array([ids + [row['pad']]*(LENGTH-len(ids))])
            end, original_end = len(ids)-1, len(row['token_ids'])-1
            traces, prefix_hashes = np.empty((0,),np.float32), []
            if request['capture'] or edit is not None:
                if request['capture']:assert not prefix
                with capture(edit,vectors) as records:
                    y = model(x)
                    z = y[0,end].astype(mx.float32)
                    original = y[0,original_end].astype(mx.float32)
                    sites = row['anchors'] + row['neighbors'] + [row['period'],original_end]
                    selected = ([records[0]['incoming'][0,sites].astype(mx.float32)] + [r['outgoing'][0,sites].astype(mx.float32) for r in records]) if request['capture'] else []
                    sources = [r['outgoing'][0,:row['period']+1].astype(mx.float32) for r in records] if request['capture'] else []
                    mx.eval(z,original,*selected,*sources)
                    if selected:traces = np.stack([np.asarray(a).copy() for a in selected])
                    import hashlib
                    prefix_hashes = [hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest() for a in sources]
                    logits, origin = np.asarray(z).copy(), np.asarray(original).copy()
            else:
                y = model(x)
                z, original = y[0,end].astype(mx.float32), y[0,original_end].astype(mx.float32)
                mx.eval(z,original)
                logits, origin = np.asarray(z).copy(), np.asarray(original).copy()
            assert logits.shape == origin.shape == (152064,)
            assert np.isfinite(logits).all() and np.isfinite(origin).all() and np.isfinite(traces).all()
            path = Path(request['path'])
            assert not path.exists()
            # Conversion is used only if it is exactly lossless.
            arrays = dict(logits=logits, original_logits=origin, trace=traces)
            for key,value in list(arrays.items()):
                with np.errstate(over='ignore',invalid='ignore'):
                    half=value.astype(np.float16)
                if np.isfinite(half).all() and np.array_equal(half.astype(value.dtype),value):
                    arrays[key]=half
            np.savez_compressed(path, **arrays)
            del x,y,z,original,logits,origin,traces,arrays
            if request['capture'] or edit is not None:
                del records,selected,sources
            gc.collect(); mx.clear_cache()
            metrics = dict(active_mlx_bytes=mx.get_active_memory(),peak_mlx_bytes=mx.get_peak_memory(),cache_mlx_bytes=mx.get_cache_memory())
            assert metrics['active_mlx_bytes'] < 6*1024**3
            print(json.dumps(dict(ok=True,prefix_hashes=prefix_hashes,metrics=metrics)),flush=True)
        except BaseException:
            print(json.dumps(dict(ok=False,error=traceback.format_exc())),flush=True)
            raise

if __name__ == '__main__':
    main()
