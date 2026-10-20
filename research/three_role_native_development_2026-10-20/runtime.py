"""Journaled fixed-shape forwards with bounded workers and resource guards."""
import atexit
import json
import re
import select
import shutil
import subprocess
import sys
import time
import numpy as np
from core import *


def resources():
    free = shutil.disk_usage(HERE).free
    text = subprocess.check_output(['/usr/sbin/sysctl','vm.swapusage'],text=True)
    m = re.search(r'used\s*=\s*([0-9.]+)([MG])',text)
    assert m,text
    swap = float(m[1])*(1024**2 if m[2]=='M' else 1024**3)
    assert free > 4*1024**3,('Disk reserve',free)
    assert swap < 2*1024**3,('Swap reserve',swap)
    return dict(free_disk_bytes=free,swap_used_bytes=swap)


class Engine:
    def __init__(self, lock):
        self.lock,self.process,self.log = lock,None,None
        self.calls,self.worker,self.worker_calls = 0,0,0
        self.started = time.monotonic()
        self.control = json.loads((HERE/'CONTROL_INPUT.json').read_text())
        (HERE/'payloads').mkdir(exist_ok=True)
        (HERE/'workers').mkdir(exist_ok=True)
        atexit.register(self.close)

    def close(self):
        if self.process is not None:
            try:
                self.process.stdin.close(); self.process.wait(timeout=10)
            except (OSError,subprocess.TimeoutExpired):
                self.process.terminate()
                try:self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
            self.process=None
            self.log.close()

    def start(self):
        self.close(); resources()
        self.worker += 1; self.worker_calls = 0
        self.log = (HERE/'workers'/f'{self.worker:03d}.log').open('x')
        self.process = subprocess.Popen([sys.executable,'-u',str(HERE/'worker.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1,cwd=ROOT)
        data,record = self.dispatch(self.control,(),False,'exposed_restart_control',start=False)
        with np.load(HERE/'REFERENCE_CONTROL.npz',allow_pickle=False) as z:
            error=float(np.max(np.abs(data['logits']-z['logits'].astype(np.float32))))
        append(HERE/'CONTROLS.jsonl',dict(check='exposed_restart_equivalence',worker=self.worker,call=record['call'],error=error,passed=error<=.05))
        assert error<=.05,('Worker equivalence',error)

    def dispatch(self,row,prefix,capture,kind,start=True):
        assert time.monotonic()-self.started < 2700,'45 minute bound'
        if start and (self.process is None or self.worker_calls >= WORKER_CAP):
            self.start()
        assert self.calls < self.lock['max_model_forwards']
        limits=resources();self.calls+=1;self.worker_calls+=1
        path=HERE/'payloads'/f'CALL_{self.calls:04d}.npz'
        entry=dict(call=self.calls,worker=self.worker,id=row['id'],prefix=list(prefix),capture=capture,kind=kind,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**limits)
        append(HERE/'BUDGET.jsonl',entry)
        self.process.stdin.write(json.dumps(dict(row=row,prefix=list(prefix),capture=capture,path=str(path)))+'\n');self.process.stdin.flush()
        started=time.monotonic();lastcheck=started
        try:
            while not select.select([self.process.stdout],[],[],.5)[0]:
                if time.monotonic()-lastcheck >= 5:
                    resources();lastcheck=time.monotonic()
                assert time.monotonic()-started<90,'Per-forward90s limit'
                assert self.process.poll() is None,'Worker exited; no retry'
            line=self.process.stdout.readline();assert line,'Worker closed output'
            response=json.loads(line);assert response['ok'],response
            with np.load(path,allow_pickle=False) as z:
                data={k:z[k].astype(np.float32) for k in z.files}
            rec={**entry,**response,'payload':str(path.relative_to(HERE)),'payload_sha256':sha(path)}
            append(HERE/'FORWARDS.jsonl',rec)
            return data,rec
        except BaseException:
            self.close()
            raise
