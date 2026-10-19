"""Short-lived owner of the unchanged MLX engine; line-delimited local IPC."""
import gc, json, sys, traceback
from pathlib import Path
import numpy as np
from engine import Engine
from core import HERE

def main():
    lock=json.loads((HERE/'LOCK.json').read_text())
    engine=Engine(lock)
    engine.mx.set_memory_limit(6*1024**3)  # MLX guideline, not a hard OS limit.
    for line in sys.stdin:
        request=json.loads(line)
        try:
            logits,arrays,hashes=engine.compute(request['row'],request['mode'],request['cut'],request['wrapped'],request['native'],request['physical_length'])
            path=Path(request['path']);assert not path.exists()
            np.savez(path,logits=logits,trace=arrays['trace'])
            del logits,arrays
            gc.collect();engine.mx.clear_cache()
            metrics={'active_mlx_bytes':engine.mx.get_active_memory(),'peak_mlx_bytes':engine.mx.get_peak_memory(),'cache_mlx_bytes':engine.mx.get_cache_memory()}
            assert metrics['active_mlx_bytes']<6*1024**3,metrics
            print(json.dumps({'ok':True,'path':str(path),'hashes':hashes,'metrics':metrics}),flush=True)
        except BaseException:
            print(json.dumps({'ok':False,'error':traceback.format_exc()}),flush=True)
            raise

if __name__=='__main__':main()
