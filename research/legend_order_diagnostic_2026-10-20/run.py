"""Exactly33 native forwards, no interventions, tuning, retries or hidden generation."""
import gc,json,re,shutil,subprocess,time,traceback
from core import *
def append(path,obj):
    with path.open('a') as f:f.write(json.dumps(obj,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def resource_check():
    free=shutil.disk_usage(HERE).free
    s=subprocess.check_output(['/usr/sbin/sysctl','vm.swapusage'],text=True)
    match=re.search(r'used\s*=\s*([0-9.]+)([MG])',s);assert match,s
    used=float(match[1])*(1024**2 if match[2]=='M' else 1024**3)
    assert free>4*1024**3,('Disk reserve',free)
    assert used<2*1024**3,('Swap limit',used)
    return dict(free_disk_bytes=free,swap_used_bytes=used)
def main():
    lock=json.loads((HERE/'LOCK.json').read_text())
    for n,h in lock['files'].items():assert sha(HERE/n)==h,n
    for n,h in lock['native_sources'].items():assert sha(ROOT/n)==h,n
    for n,d in lock['model']['files'].items():assert sha(ROOT/n)==d['sha256'],n
    assert (HERE/'FREEZE_RECEIPT.json').exists()
    assert not (HERE/'STARTED.json').exists(),'No automatic retry'
    save(HERE/'STARTED.json',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),lock_sha256=sha(HERE/'LOCK.json')))
    count=0;raw=[]
    try:
        resource_check()
        import mlx.core as mx
        from mlx_lm import load
        model,tok=load(str(ROOT/lock['model']['path']),tokenizer_config={'trust_remote_code':False});model.eval();mx.set_memory_limit(6*1024**3);mx.set_cache_limit(256*1024**2)
        (HERE/'payloads').mkdir(exist_ok=True)
        def forward(row,kind):
            nonlocal count
            assert count<33
            limits=resource_check();count+=1
            entry=dict(call=count,id=row['id'],kind=kind,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**limits)
            append(HERE/'BUDGET.jsonl',entry)
            ids=row['token_ids'];assert len(ids)<LENGTH
            x=mx.array([ids+[row['pad']]*(LENGTH-len(ids))]);y=model(x);z=y[0,row['prompt_end']].astype(mx.float32);mx.eval(z);logits=np.asarray(z).copy()
            assert logits.shape==(152064,) and np.isfinite(logits).all()
            path=HERE/'payloads'/f'CALL_{count:03d}.npz';assert not path.exists();np.savez_compressed(path,logits=logits)
            s=score(logits,row,tok.decode)
            keys=['family','role','query','mapping','legend_order','last_class','last_symbol','expected','expected_symbol'] if kind=='scientific' else ['family','expected','expected_symbol']
            record={**entry,**{k:row[k] for k in keys},**s,'payload':str(path.relative_to(HERE)),'payload_sha256':sha(path)}
            raw.append(record);append(HERE/'RAW.jsonl',record)
            del x,y,z
            gc.collect();mx.clear_cache()
            metrics=dict(call=count,active_mlx_bytes=mx.get_active_memory(),peak_mlx_bytes=mx.get_peak_memory(),cache_mlx_bytes=mx.get_cache_memory())
            append(HERE/'RESOURCE_METRICS.jsonl',metrics)
            assert metrics['active_mlx_bytes']<6*1024**3
            print('SAVED',count,'of33',kind,flush=True)
            return logits
        control=json.loads((HERE/'CONTROL_INPUT.json').read_text());z=forward(control,'exposed_equivalence_control')
        with np.load(HERE/'REFERENCE_CONTROL.npz',allow_pickle=False) as ref:error=float(np.max(np.abs(z-ref['logits'].astype(np.float32))))
        save(HERE/'CONTROL_RESULT.json',dict(passed=error<=.05,max_full_logit_error=error,tolerance=.05));assert error<=.05,'Exposed numerical control failed'
        for row in json.loads((HERE/'INPUTS.json').read_text()):forward(row,'scientific')
        assert count==len(raw)==33
        save(HERE/'COLLECTION.json',dict(status='COMPLETE_FIXED_32',model_forwards=count,scientific=32,exposed_control=1,no_interventions=True))
    except BaseException:
        save(HERE/'ERROR.json',dict(traceback=traceback.format_exc(),attempts=count,completed=len(raw),no_automatic_retry=True));raise
if __name__=='__main__':main()
