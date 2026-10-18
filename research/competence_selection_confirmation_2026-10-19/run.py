"""Bounded collection: all outcomes, no response-dependent eligibility during inference."""
import argparse,json,shutil,time,traceback,hashlib
from pathlib import Path
import numpy as np
from core import HERE,ROOT,sha,save,score
from engine import Engine

def append(p,x):
 with Path(p).open('a') as f:f.write(json.dumps(x,allow_nan=False)+'\n');f.flush()
def lossless(a):
 with np.errstate(over='ignore',invalid='ignore'):b=a.astype(np.float16)
 return b if np.isfinite(b).all() and np.array_equal(b.astype(a.dtype),a) else a

def execute(model,engine_factory=Engine,root=HERE,validate=True):
 root=Path(root);lock=json.loads((root/'LOCK.json').read_text())
 if validate:
  for n,h in lock['files'].items():assert sha(root/n)==h,n
  for n,h in lock['native_sources'].items():assert sha(ROOT/n)==h,n
  assert (root/'FREEZE_RECEIPT.json').exists()
 out=root/model;out.mkdir(exist_ok=True);assert not (out/'STARTED.json').exists(),'No automatic restart'
 save(out/'STARTED.json',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),lock_sha256=sha(root/'LOCK.json')))
 budget=[json.loads(s) for s in (root/'BUDGET.jsonl').read_text().splitlines()] if (root/'BUDGET.jsonl').exists() else []
 rows=[r for r in json.loads((root/'INPUTS.json').read_text()) if r['model']==model]
 threshold=json.loads((root/'READOUT_CALIBRATION.json').read_text())[model]['threshold'];checks=[];raw=[];baselines={};count=0
 def control(name,passed,**kw):
  x=dict(check=name,passed=bool(passed),**kw);checks.append(x);append(out/'CONTROLS.jsonl',x)
  assert passed,('Numerical control failed',x)
 engine=engine_factory(model,lock)
 def forward(row,mode='baseline',cut=28,kind='scientific',wrapped=True,native=False):
  nonlocal count
  assert len(budget)<lock['max_forwards'] and count<lock['max_per_model'];assert shutil.disk_usage(root).free>1024**3
  entry=dict(call=len(budget)+1,model=model,id=row['id'],mode=mode,cut=cut,kind=kind,wrapped=wrapped,native=native,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
  budget.append(entry);count+=1;append(root/'BUDGET.jsonl',entry)
  logits,arrays,hashes=engine.compute(row,mode,cut,wrapped,native)
  trace=arrays['trace'];depths=list(range(29)) if mode=='baseline' else [0,7,14,21,28]
  stored=trace if mode=='baseline' or not trace.size else trace[depths]
  packed=dict(logits=lossless(logits),trace=lossless(stored),depths=np.array(depths,dtype=np.int16))
  if 'null_states' in arrays:hashes['null_states']=hashlib.sha256(np.ascontiguousarray(arrays['null_states']).tobytes()).hexdigest()
  payload=f"CALL_{entry['call']:04d}.npz";path=out/payload;assert not path.exists();np.savez_compressed(path,**packed)
  scored=score(logits,row,engine.tok.decode);s=scored['class_logits']['Yes']-scored['class_logits']['No'];cal='Yes' if s>threshold else 'No' if s<threshold else None
  result={**entry,**scored,'family':row['family'],'role':row['role'],'query':row['query'],'voice':row['voice'],'yes_minus_no':s,'threshold':threshold,'calibrated_class':cal,'calibrated_correct':cal==row['expected'],'payload':payload,'payload_sha256':sha(path),'prefix_hashes':hashes,'full_trace_sha256':hashlib.sha256(np.ascontiguousarray(trace).tobytes()).hexdigest(),'stored_dtypes':{k:str(v.dtype) for k,v in packed.items()}}
  raw.append(result);append(out/'RAW.jsonl',result)
  if kind=='scientific' and mode!='baseline':
   ref=baselines[row['id']]
   control('untouched_prefix',hashes['prefix']==ref['hashes']['prefix'],call=entry['call'])
   err=float(np.max(np.abs(trace[:cut+1]-ref['arrays']['trace'][:cut+1])))
   control('before_cut',err<=1e-6,call=entry['call'],error=err)
  if count%24==0:print('PROGRESS',model,count,'of',lock['max_per_model'],flush=True)
  return dict(logits=logits,arrays=arrays,hashes=hashes,result=result)
 try:
  first=rows[0];u=forward(first,kind='validation_unwrapped',wrapped=False);w=forward(first,kind='validation_wrapped_native',native=True)
  for row in rows:baselines[row['id']]=forward(row)
  repeat=forward(first,kind='validation_repeat');ref=baselines[first['id']]
  for name,a,b in [('wrapper',u,w),('dense',w,ref),('repeat',repeat,ref)]:
   err=float(np.max(np.abs(a['logits']-b['logits'])));control(name,err<=.05,error=err)
  err=float(np.max(np.abs(repeat['arrays']['trace']-ref['arrays']['trace'])));control('repeat_trace',err<=1e-6,error=err)
  for fam in sorted({r['family'] for r in rows}):
   for r in [0,1]:
    for v in ['active','passive']:
     a,b=[baselines[x['id']]['arrays']['trace'][:,0] for x in rows if (x['family'],x['role'],x['voice'])==(fam,r,v)]
     err=float(np.max(np.abs(a-b)));control('question_invariant_period',err<=1e-6,family=fam,role=r,voice=v,error=err)
  # Continue regardless of answer correctness. Only numerical failures stop this panel.
  for row in rows:
   for mode in ['frozen','closed','irrelevant']:forward(row,mode,14)
   if row['family']==first['family'] and row['role']==0 and row['query']==0:
    sham=forward(row,'closed',28,kind='validation_sham');ref=baselines[row['id']];err=float(np.max(np.abs(sham['logits']-ref['logits'])))
    control('sham_logits',err<=.05,voice=row['voice'],error=err)
    err=float(np.max(np.abs(sham['arrays']['trace']-ref['arrays']['trace'])));control('sham_trace',err<=1e-6,voice=row['voice'],error=err)
  nulls={}
  for row in rows:
   if row['family']==first['family']:nulls[row['id']]=forward(row,'closed',0,kind='validation_disconnected')
  for v in ['active','passive']:
   for q in [0,1]:
    a,b=[nulls[x['id']] for x in rows if x['family']==first['family'] and x['voice']==v and x['query']==q]
    err=float(np.max(np.abs(a['logits']-b['logits'])));eq=a['hashes']['null_states']==b['hashes']['null_states'];control('cut0_disconnected',err<=.05 and eq,voice=v,query=q,error=err,suffix_hash_equal=eq)
  assert count==781 and len(raw)==781
  save(out/'DECISION.json',dict(status='COMPLETE_FIXED_PANEL',attempts=count,payloads=len(raw),controls=len(checks),all_controls_passed=all(c['passed'] for c in checks)))
  print('COMPLETE',model,count,'forwards',len(checks),'controls',flush=True)
 except BaseException:
  save(out/'ERROR.json',dict(error=traceback.format_exc(),attempts=count,completed=len(raw),no_automatic_retry=True));raise
 finally:save(out/'CONTROLS.json',checks)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('model',choices=['qwen7b','llama3b']);execute(p.parse_args().model)
