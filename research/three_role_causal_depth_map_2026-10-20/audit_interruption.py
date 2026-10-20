"""No inference: audit the critical-pressure stop and recover any written reference."""
import json,hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent

def main():
 checks=[]
 def check(name,ok,**meta):
  checks.append(dict(check=name,passed=bool(ok),**meta));assert ok,(name,meta)
 lock=json.loads((HERE/'LOCK.json').read_text())
 for f,h in lock['files'].items():check('frozen_source',hashlib.sha256((HERE/f).read_bytes()).hexdigest()==h,file=f)
 budget=[json.loads(s) for s in (HERE/'BUDGET.jsonl').read_text().splitlines()];error=json.loads((HERE/'ERROR.json').read_text())
 check('one_exposed_attempt',len(budget)==error['attempts']==1 and budget[0]['kind']=='exposed_restart_control')
 check('critical_pressure_stop',"('Memory pressure', 4)" in error['traceback'])
 check('no_scientific_collection',error['completed_answers']==0 and not (HERE/'ANSWERS.jsonl').exists() and not (HERE/'DONOR_CACHE.jsonl').exists())
 check('no_controller_completion',not (HERE/'FORWARDS.jsonl').exists())
 payloads=[]
 for p in sorted((HERE/'payloads').glob('*.npz')):
  with np.load(p,allow_pickle=False) as a,np.load(HERE/'REFERENCE_CONTROL.npz',allow_pickle=False) as b:
   le=float(np.max(np.abs(a['logits'].astype(np.float32)-b['logits'].astype(np.float32))))
   check('recovered_reference_payload',a['trace'].size==0 and a['logits'].shape==(152064,) and le<=.05,file=p.name,error=le)
  payloads.append(dict(path=str(p.relative_to(HERE)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),recovered_after_controller_stop=True,logit_reference_error=le))
 check('no_retry',len(list((HERE/'workers').glob('*.log')))==1 and error['no_automatic_retry'])
 out=dict(passed=True,n_checks=len(checks),checks=checks,attempted_model_forwards=1,controller_completed_forwards=0,recovered_payloads=payloads,scientific_answers=0,interventions=0,status='INCONCLUSIVE_CRITICAL_MEMORY_PRESSURE_STOP')
 (HERE/'INTERRUPTION_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='checks'},indent=2))
if __name__=='__main__':main()
