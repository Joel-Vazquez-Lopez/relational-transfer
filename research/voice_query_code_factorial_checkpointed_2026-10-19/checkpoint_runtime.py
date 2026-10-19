"""Bounded workers and disk-backed baselines; no new scientific conditions."""
import atexit,json,os,re,select,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
from core import HERE,ROOT,sha,save

PREVIOUS=ROOT/'research/voice_query_code_factorial_resumed_2026-10-19'
CHUNK=32

def append(p,x):
    with p.open('a') as f:
        f.write(json.dumps(x,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())

def resources():
    free=shutil.disk_usage(HERE).free
    text=subprocess.check_output(['/usr/sbin/sysctl','vm.swapusage'],text=True)
    match=re.search(r'used\s*=\s*([0-9.]+)([MG])',text);assert match,text
    used=float(match[1])*(1024**2 if match[2]=='M' else 1024**3)
    assert free>4*1024**3,('Disk below4GiB reserve',free)
    assert used<2*1024**3,('Swap exceeds2GiB stop threshold',used)
    return {'disk_free_bytes':free,'swap_used_bytes':used}

class DiskBases:
    def __init__(self,root):self.root=Path(root);self.records={}
    def __setitem__(self,key,value):self.records[key]=value['result']
    def __getitem__(self,key):
        r=self.records[key]
        with np.load(self.root/r['payload'],allow_pickle=False) as z:
            return {'result':r,'logits':z['logits'].astype(np.float32),'trace':z['trace'].astype(np.float32),'hashes':r['hashes']}

def previous_records():
    records=[json.loads(s) for s in (PREVIOUS/'qwen7b/RAW.jsonl').read_text().splitlines()]
    assert len(records)==248
    for r in records:assert sha(PREVIOUS/r['payload'])==r['payload_sha256']
    return records

def replay(root,record,entry):
    for key in ('call','id','mode','cut','kind','wrapped','native','physical_length'):
        assert record[key]==entry[key],(key,record[key],entry[key])
    path=Path(root)/record['payload'];assert not path.exists()
    os.link(PREVIOUS/record['payload'],path)
    assert sha(path)==record['payload_sha256']
    with np.load(path,allow_pickle=False) as z:
        arrays={'logits':z['logits'].astype(np.float32),'trace':z['trace'].astype(np.float32)}
    return {'result':record,'hashes':record['hashes'],**arrays}

class WorkerEngine:
    def __init__(self,lock):
        from transformers import AutoTokenizer
        self.lock=lock;self.tok=AutoTokenizer.from_pretrained(str(ROOT/lock['model']['path']),local_files_only=True,trust_remote_code=False)
        self.worker=None;self.done=0;self.worker_id=0;self.dispatches=0
        self.logs=None;(HERE/'WORKERS').mkdir(exist_ok=True)
        self.old=previous_records()
        for r in self.old:
            for t in r['top12']:assert self.tok.decode([t['id']])==t['token']
        self.control=self.old[2]
        self.controlrow=next(r for r in json.loads((HERE/'INPUTS.json').read_text()) if r['id']==self.control['id'])
        atexit.register(self.close)
    def close(self):
        if self.worker is not None:
            try:
                self.worker.stdin.close();self.worker.wait(timeout=10)
            except (OSError,subprocess.TimeoutExpired):
                self.worker.terminate()
                try:self.worker.wait(timeout=5)
                except subprocess.TimeoutExpired:self.worker.kill();self.worker.wait()
            if self.logs:self.logs.close()
            self.worker=None
    def dispatch(self,row,mode,cut,wrapped,native,length,kind,logical_call=None):
        assert self.dispatches<897
        self.dispatches+=1
        path=HERE/'WORKERS'/f'DISPATCH_{self.dispatches:04d}.npz'
        entry={'dispatch':self.dispatches,'worker':self.worker_id,'kind':kind,'logical_call':logical_call,'id':row['id'],'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**resources()}
        append(HERE/'DISPATCHES.jsonl',entry)
        request={'row':row,'mode':mode,'cut':cut,'wrapped':wrapped,'native':native,'physical_length':length,'path':str(path)}
        self.worker.stdin.write(json.dumps(request)+'\n');self.worker.stdin.flush()
        started=time.monotonic();lastcheck=started
        while not select.select([self.worker.stdout],[],[],.5)[0]:
            if time.monotonic()-lastcheck>=5:resources();lastcheck=time.monotonic()
            assert time.monotonic()-started<90,'Worker timeout; no retry'
            assert self.worker.poll() is None,'Worker exited; no retry'
        line=self.worker.stdout.readline();assert line,'Worker closed output; no retry'
        response=json.loads(line);assert response['ok'],response
        with np.load(path,allow_pickle=False) as z:
            logits=z['logits'].copy();trace=z['trace'].copy()
        append(HERE/'WORKER_METRICS.jsonl',{**entry,**response['metrics']})
        if kind=='restart_equivalence':
            with np.load(PREVIOUS/self.control['payload'],allow_pickle=False) as z:
                le=float(np.max(np.abs(logits-z['logits'].astype(np.float32))))
                te=float(np.max(np.abs(trace-z['trace'].astype(np.float32))))
            ok=le<=.05 and te<=1e-6 and response['hashes']==self.control['hashes']
            append(HERE/'WORKER_GATES.jsonl',{'worker':self.worker_id,'logit_error':le,'trace_error':te,'all_hashes_equal':response['hashes']==self.control['hashes'],'passed':ok,'payload':str(path.relative_to(HERE)),'payload_sha256':sha(path)})
            assert ok,('Restart equivalence failed',le,te)
        else:path.unlink()
        return logits,{'trace':trace},response['hashes']
    def start(self):
        self.close();resources();self.worker_id+=1;assert self.worker_id<=28
        self.logs=(HERE/'WORKERS'/f'WORKER_{self.worker_id:03d}.log').open('x')
        self.worker=subprocess.Popen([sys.executable,'-u',str(HERE/'gpu_worker.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.logs,text=True,bufsize=1,cwd=ROOT)
        self.dispatch(self.controlrow,'baseline',28,True,False,128,'restart_equivalence')
    def compute(self,row,mode,cut,wrapped,native,physical_length):
        if self.done%CHUNK==0:self.start()
        result=self.dispatch(row,mode,cut,wrapped,native,physical_length,'continuation',249+self.done)
        self.done+=1
        if self.done%CHUNK==0:self.close()
        return result
