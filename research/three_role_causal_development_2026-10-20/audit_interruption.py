"""Audit exactly the interrupted attempt; never treats it as a completed experiment."""
import hashlib,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent

def main():
    checks=[]
    def check(name,ok,**meta):
        checks.append(dict(check=name,passed=bool(ok),**meta));assert ok,(name,meta)
    lock=json.loads((HERE/'LOCK.json').read_text())
    for f,h in lock['files'].items():check('frozen_source',hashlib.sha256((HERE/f).read_bytes()).hexdigest()==h,file=f)
    f=[json.loads(s) for s in (HERE/'FORWARDS.jsonl').read_text().splitlines()]
    b=[json.loads(s) for s in (HERE/'BUDGET.jsonl').read_text().splitlines()]
    err=json.loads((HERE/'ERROR.json').read_text())
    check('one_attempt_one_completion',len(f)==len(b)==err['attempts']==1)
    check('reference_only',f[0]['kind']=='exposed_restart_control' and not f[0]['capture'])
    check('journal',all(f[0][k]==v for k,v in b[0].items()))
    check('raw_hash',hashlib.sha256((HERE/f[0]['payload']).read_bytes()).hexdigest()==f[0]['payload_sha256'])
    with np.load(HERE/f[0]['payload'],allow_pickle=False) as a,np.load(HERE/'REFERENCE_CONTROL.npz',allow_pickle=False) as r:
        check('full_vector_reference',a['logits'].shape==(152064,) and np.array_equal(a['logits'].astype(np.float32),r['logits'].astype(np.float32)))
        check('no_scientific_trace',a['trace'].size==0)
    check('no_scientific_answers',err['completed_answers']==0 and not (HERE/'ANSWERS.jsonl').exists())
    check('no_source_recapture',not (HERE/'DONOR_CACHE.jsonl').exists())
    check('no_causal_conclusion',not (HERE/'RESULTS.json').exists())
    check('guard_reason',"Swap reserve" in err['traceback'] and '2608206970.88' in err['traceback'])
    check('no_retry',err['no_automatic_retry'] and len(list((HERE/'workers').glob('*.log')))==1)
    receipt=json.loads((HERE/'FREEZE_RECEIPT.json').read_text())
    start=json.loads((HERE/'STARTED.json').read_text())
    check('frozen_before_start',receipt['before_causal_inference'] and receipt['utc'][:19]<start['utc'][:19])
    out=dict(passed=True,n_checks=len(checks),checks=checks,actual_model_forwards=1,scientific_answers=0,interventions=0,heldout_forwards=0,llama_forwards=0,status='INCONCLUSIVE_RESOURCE_STOP',scope='Audits preservation and reference equivalence only. Not an audit of completed scientific results.')
    (HERE/'INTERRUPTION_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='checks'},indent=2))
if __name__=='__main__':main()
