"""One323-pass maximum run; raw before scores, attempts before dispatch."""
from pathlib import Path
import json,time,hashlib,shutil,traceback
import numpy as np
from core import HERE,ROOT,save,sha,score,sensitivity

def append(p,x):
 with Path(p).open('a') as f:f.write(json.dumps(x,allow_nan=False)+'\n');f.flush()
def lossless(a):
 with np.errstate(over='ignore',invalid='ignore'):b=a.astype(np.float16)
 return b if np.isfinite(b).all() and np.array_equal(a,b.astype(a.dtype)) else a

def classify(raw):
 units=[]
 for fam,voice in sorted({(r['family'],r['voice']) for r in raw if r['kind']=='scientific'}):
  cells={m:[r for r in raw if r['kind']=='scientific' and r['family']==fam and r['voice']==voice and r['condition']==m] for m in ['native','F','N','P_V','P_K','P_KV','P_V_voice','N_V','N_K','N_KV']}
  metrics={m:sensitivity(rr) for m,rr in cells.items()};d={m:v['D'] for m,v in metrics.items()};b=d['F'];valid=b>=.5 and metrics['F']['all_ordinal_pairs_positive']
  E={m:((d['N'] if m.startswith('N_') else b)-v)/(2*b) if valid else None for m,v in d.items() if m not in ['native','F','N']}
  base={(r['role'],r['query']):r['yes_minus_no'] for r in cells['F']}
  shifts={m:[((-1)**(r['role']+r['query']))*(base[r['role'],r['query']]-r['yes_minus_no']) for r in cells[m]] for m in ['P_V','P_K']}
  nuisance=max(abs(base[r['role'],r['query']]-r['yes_minus_no']) for r in cells['P_V_voice'])
  control=bool(valid and abs(E['P_V_voice'])<=.10 and nuisance<=.5*b)
  hv=bool(control and E['P_V']>=.25 and E['P_V']-E['P_K']>=.25 and E['P_V']-E['N_V']>=.25 and min(shifts['P_V'])>.1)
  hk=bool(control and E['P_K']>=.25 and E['P_K']-E['P_V']>=.25 and E['P_K']-E['N_K']>=.25 and min(shifts['P_K'])>.1)
  hs=bool(valid and all(abs(E[m])<=.1 for m in ['P_V','P_K','P_KV']))
  units.append(dict(family=fam,voice=voice,D=d,metrics=metrics,E=E,donor_shifts=shifts,voice_max_absolute_shift=nuisance,voice_control=control,HV=hv,HK=hk,HS=hs,kv_interaction=d['P_KV']-d['P_K']-d['P_V']+b))
 prediction=next((h for h in ['HV','HK','HS'] if all(r[h] for r in units)),None)
 return dict(units=units,classification=prediction)

def main():
 lock=json.loads((HERE/'LOCK.json').read_text())
 for f,h in lock['files'].items():assert sha(ROOT/f)==h,('frozen file changed',f)
 for f,v in lock['model']['files'].items():assert sha(ROOT/f)==v['sha256'],f
 assert (HERE/'FREEZE_RECEIPT.json').exists()
 assert not (HERE/'STARTED.json').exists(),'Never automatically resume'
 save(HERE/'STARTED.json',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),lock_sha256=sha(HERE/'LOCK.json')))
 from engine import Engine
 engine=Engine(lock);rows=json.loads((HERE/'INPUTS.json').read_text());threshold=json.loads((HERE/'READOUT_CALIBRATION.json').read_text())['threshold']
 out=HERE/'qwen7b';out.mkdir();raw=[];budget=[];controls=[];bases={};prediction=None
 def call(row,condition,graph='baseline',donor=None,port='period',kind='',attention_hook=True,native=False,validation=False):
  assert len(budget)<323;assert shutil.disk_usage(HERE).free>1024**3
  e=dict(call=len(budget)+1,id=row['id'],condition=condition,graph=graph,port=port,edit=kind,kind='validation' if validation else 'scientific',family=row['family'],split=row['split'],voice=row['voice'],role=row['role'],query=row['query'],utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
  budget.append(e);append(HERE/'BUDGET.jsonl',e)
  arrays,hashes=engine.compute(row,graph,donor,port,kind,attention_hook,native)
  path=out/f"CALL_{e['call']:04d}.npz";assert not path.exists();np.savez_compressed(path,**{k:lossless(v) for k,v in arrays.items()})
  # Persist score-independent full payload before interpreting it.
  s=score(arrays['logits'],row,engine.tok.decode);v=s['class_logits']['Yes']-s['class_logits']['No'];cal='Yes' if v>threshold else 'No' if v<threshold else None
  record=dict(**e,**s,yes_minus_no=v,calibrated_class=cal,calibrated_correct=cal==row['expected'],threshold=threshold,payload=str(path.relative_to(HERE)),payload_sha256=sha(path),prefix_hashes=hashes)
  raw.append(record);append(HERE/'RAW.jsonl',record)
  print('CALL',e['call'],row['id'],condition,s['actual'],round(v,4),flush=True)
  return dict(record=record,arrays=arrays,hashes=hashes)
 def check(label,passed,**kwargs):
  finding=dict(check=label,passed=bool(passed),**kwargs);controls.append(finding);save(HERE/'CONTROLS.json',controls);assert passed,finding
 def same(name,a,b,trace=False):
  err=float(np.max(np.abs(a['arrays']['logits']-b['arrays']['logits'])));check(name+'_logits',err<=.05,error=err)
  if trace:
   err=float(np.max(np.abs(a['arrays']['trace']-b['arrays']['trace'])));check(name+'_trace',err<=1e-6,error=err)
 for split in ['development','evaluation']:
  rr=[r for r in rows if r['split']==split]
  if split=='development':
   native_ref=call(rr[0],'validation_native',attention_hook=False,native=True,validation=True)
   frozen_ref=call(rr[0],'validation_F',graph='frozen',attention_hook=False,validation=True)
  for row in rr:
   for condition,graph in [('native','baseline'),('F','frozen'),('N','neighbor')]:
    a=call(row,condition,graph);bases[row['id'],condition]=a
    if condition!='native':
     ref=bases[row['id'],'native'];err=float(np.max(np.abs(a['arrays']['trace'][:15]-ref['arrays']['trace'][:15])));check('before_cut',err<=1e-6,id=row['id'],condition=condition,error=err)
  if split=='development':
   same('native_wrapper',native_ref,bases[rr[0]['id'],'native'],True);same('F_wrapper',frozen_ref,bases[rr[0]['id'],'F'],True)
   a=bases[rr[0]['id'],'F'];identity=call(rr[0],'validation_identity',graph='frozen',donor=a['arrays']['kv'],kind='KV',validation=True);same('identity',identity,a,True)
  for row in rr:
   qother=next(r for r in rr if r['family']==row['family'] and r['voice']==row['voice'] and r['role']==row['role'] and r['query']!=row['query'])
   for condition,idx in [('F',1),('N',0)]:
    x=bases[row['id'],condition]['arrays']['kv'][:,:,:,idx];y=bases[qother['id'],condition]['arrays']['kv'][:,:,:,idx]
    check('donor_question_invariance',np.array_equal(x,y),id=row['id'],condition=condition)
  for row in rr:
   opposite=next(r for r in rr if r['family']==row['family'] and r['voice']==row['voice'] and r['role']!=row['role'] and r['query']==row['query'])
   voice=next(r for r in rr if r['family']==row['family'] and r['voice']!=row['voice'] and r['role']==row['role'] and r['query']==row['query'])
   for condition,base,donor,port,edit in [('P_V','F',opposite,'period','V'),('P_K','F',opposite,'period','K'),('P_KV','F',opposite,'period','KV'),('P_V_voice','F',voice,'period','V'),('N_V','N',opposite,'neighbor','V'),('N_K','N',opposite,'neighbor','K'),('N_KV','N',opposite,'neighbor','KV')]:
    a=call(row,condition,graph='frozen' if base=='F' else 'neighbor',donor=bases[donor['id'],base]['arrays']['kv'],port=port,kind=edit);b=bases[row['id'],base]
    check('prefix_through_period',a['hashes']==b['hashes'],id=row['id'],condition=condition)
    err=float(np.max(np.abs(a['arrays']['trace'][:15]-b['arrays']['trace'][:15])));check('donation_before_cut',err<=1e-6,id=row['id'],condition=condition,error=err)
    check('unchanged_source_anchor_states',np.array_equal(a['arrays']['trace'][:,:2],b['arrays']['trace'][:,:2]),id=row['id'],condition=condition)
  result=classify([r for r in raw if r['split']==split]);save(HERE/(split.upper()+'_SUMMARY.json'),result)
  if split=='development':
   prediction=result['classification'];save(HERE/'PREDICTION_BEFORE_EVALUATION.json',dict(prediction=prediction,raw_sha256=sha(HERE/'RAW.jsonl'),attempts=len(budget),evaluation_outputs_observed=False))
  else:
   passed=bool(prediction and all(r[prediction] for r in result['units']));save(HERE/'DECISION.json',dict(status='NARROW_PROSPECTIVE_PORT_SIGNATURE' if passed else 'NO_GENERALIZING_PORT_SIGNATURE',prediction=prediction,passed=passed,attempts=len(budget),technical_failures=[r for r in controls if not r['passed']],new_mechanism_claim=False))
 assert len(budget)==323
 print('COMPLETE',json.dumps(json.loads((HERE/'DECISION.json').read_text())),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  save(HERE/'ERROR.json',dict(error=traceback.format_exc(),no_automatic_resume=True));raise
