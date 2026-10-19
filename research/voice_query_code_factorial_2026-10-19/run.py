"""One bounded native/masked factorial collection; no behavioral exclusions or retries."""
import json,time,shutil,traceback
from pathlib import Path
import numpy as np
from core import *
from engine import Engine

def append(p,x):
 with Path(p).open('a') as f:f.write(json.dumps(x,allow_nan=False)+'\n');f.flush()
def lossless(a):
 with np.errstate(over='ignore',invalid='ignore'):b=a.astype(np.float16)
 return b if np.isfinite(b).all() and np.array_equal(b.astype(a.dtype),a) else a

def execute(root=HERE,engine_factory=Engine,validate=True):
 root=Path(root);lock=json.loads((root/'LOCK.json').read_text())
 if validate:
  for n,h in lock['files'].items():assert sha(root/n)==h,n
  for n,h in lock['native_sources'].items():assert sha(ROOT/n)==h,n
  assert (root/'FREEZE_RECEIPT.json').exists()
 out=root/'qwen7b';out.mkdir(exist_ok=True);assert not (out/'STARTED.json').exists(),'No automatic retry/resume'
 save(out/'STARTED.json',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),lock_sha256=sha(root/'LOCK.json')))
 rows=json.loads((root/'INPUTS.json').read_text());controlrow=json.loads((root/'CONTROL_INPUT.json').read_text());engine=engine_factory(lock);raw=[];budget=[];checks=[];bases={}
 def check(name,passed,**kw):
  x=dict(check=name,passed=bool(passed),**kw);checks.append(x);append(out/'CONTROLS.jsonl',x);assert passed,('Technical gate failed',x)
 def forward(row,mode='baseline',cut=28,kind='scientific',wrapped=True,native=False,physical_length=LENGTH):
  assert len(budget)<lock['max_forwards'];assert shutil.disk_usage(root).free>1024**3,'Below1GiB reserve'
  entry=dict(call=len(budget)+1,id=row['id'],mode=mode,cut=cut,kind=kind,wrapped=wrapped,native=native,physical_length=physical_length,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
  budget.append(entry);append(root/'BUDGET.jsonl',entry)
  logits,arrays,hashes=engine.compute(row,mode,cut,wrapped,native,physical_length);trace=arrays['trace'];depths=list(range(29)) if mode=='baseline' else [0,7,14,21,28];saved=trace if mode=='baseline' or not trace.size else trace[depths]
  payload=f'qwen7b/CALL_{entry["call"]:04d}.npz';path=root/payload;assert not path.exists();np.savez_compressed(path,logits=lossless(logits),trace=lossless(saved),depths=np.array(depths,dtype=np.int16))
  scored=score(logits,row,engine.tok.decode);metadata={k:row[k] for k in ['family','split','panel','role','query','voice','question_voice','lexicon','mapping','position','expected_symbol']};record={**entry,**metadata,**scored,'payload':payload,'payload_sha256':sha(path),'hashes':hashes,'full_trace_sha256':digest(trace)};raw.append(record);append(out/'RAW.jsonl',record)
  if mode!='baseline' and kind=='scientific':
   base=bases[row['id']];check('untouched_source_prefix',hashes['source_prefix']==base['hashes']['source_prefix'],call=entry['call']);err=float(np.max(np.abs(trace[:cut+1]-base['trace'][:cut+1])));check('before_cut',err<=1e-6,call=entry['call'],error=err)
  if len(raw)%32==0:print('PROGRESS',len(raw),'of',lock['max_forwards'],row['split'],mode,flush=True)
  return dict(logits=logits,trace=trace,hashes=hashes,result=record)
 try:
  a=forward(controlrow,kind='shape128',wrapped=False,physical_length=128);b=forward(controlrow,kind='shape192',wrapped=False)
  err=float(np.max(np.abs(a['logits']-b['logits'])));check('exposed128_192_logits',err<=.05,error=err)
  if validate:
   source=json.loads((root/'SOURCE_MANIFEST.json').read_text())['prior_shape_control'];p=ROOT/source['path'];assert sha(p)==source['sha256']
   with np.load(p) as z:err=float(np.max(np.abs(a['logits']-z['logits'].astype(np.float32))))
   check('saved_exposed128_reproduction',err<=.05,error=err)
  first=next(r for r in rows if r['panel']=='factorial' and r['lexicon']=='AB' and r['mapping']==0 and r['position']==0 and r['question_voice']=='active')
  unwrapped=forward(first,kind='validation_unwrapped',wrapped=False);native=forward(first,kind='validation_wrapped_native',native=True)
  for phase in ['development','evaluation']:
   rr=[r for r in rows if r['split']==phase];bases={}
   for row in rr:bases[row['id']]=forward(row)
   if phase=='development':
    repeat=forward(first,kind='validation_repeat');ref=bases[first['id']]
    for name,a,b in [('wrapper',unwrapped,native),('dense',native,ref),('repeat',repeat,ref)]:
     err=float(np.max(np.abs(a['logits']-b['logits'])));check(name,err<=.05,error=err)
    err=float(np.max(np.abs(repeat['trace']-ref['trace'])));check('repeat_trace',err<=1e-6,error=err)
   for fam in sorted({r['family'] for r in rr}):
    fr=[r for r in rr if r['family']==fam and r['panel']=='factorial']
    for role in [0,1]:
     for v in ['active','passive']:
      for pos in [0,1]:
       pair=[bases[r['id']]['trace'][:,0] for r in fr if (r['role'],r['voice'],r['position'])==(role,v,pos)];err=max(float(np.max(np.abs(x-pair[0]))) for x in pair);check('sentence_period_future_invariance',err<=1e-6,family=fam,role=role,voice=v,position=pos,error=err)
       for q in [0,1]:
        for qv in ['active','passive']:
         pairs=[bases[r['id']]['hashes']['question_prefix'] for r in fr if (r['role'],r['voice'],r['position'],r['query'],r['question_voice'])==(role,v,pos,q,qv)];check('full_question_prefix_legend_invariance',all(x==pairs[0] for x in pairs),family=fam,role=role,voice=v,position=pos,query=q,question_voice=qv)
   canonical=[r for r in rr if r['panel']=='factorial' and r['lexicon']=='AB' and r['mapping']==0 and r['position']==0]
   for row in canonical:
    for mode in ['frozen','closed','irrelevant']:forward(row,mode,14)
    if phase=='development' and row['family']==first['family'] and row['role']==0 and row['query']==0 and row['question_voice']=='active':
     sham=forward(row,'closed',28,kind='validation_sham');ref=bases[row['id']];err=float(np.max(np.abs(sham['logits']-ref['logits'])));check('sham_logits',err<=.05,voice=row['voice'],error=err);err=float(np.max(np.abs(sham['trace']-ref['trace'])));check('sham_trace',err<=1e-6,voice=row['voice'],error=err)
   if phase=='development':
    nullrows=[r for r in canonical if r['family']==first['family'] and r['question_voice']=='active'];null={r['id']:forward(r,'closed',0,kind='validation_disconnected') for r in nullrows}
    for v in ['active','passive']:
     for q in [0,1]:
      a,b=[null[r['id']] for r in nullrows if r['voice']==v and r['query']==q];err=float(np.max(np.abs(a['logits']-b['logits'])));same=a['hashes']['disconnected_suffix']==b['hashes']['disconnected_suffix'];check('cut0_role_disconnection',err<=.05 and same,voice=v,query=q,error=err,suffix_hash_equal=same)
   from analyze import grouped,routing
   groups=grouped([r for r in raw if r['kind']=='scientific' and r['split']==phase]);profiles=[patterns(groups,f) for f in sorted({r['family'] for r in rr})];routes=[routing(groups,f) for f in sorted({r['family'] for r in rr})]
   save(out/(phase.upper()+'_SUMMARY.json'),dict(groups=groups,profiles=profiles,routing=routes))
   if phase=='development':save(out/'PREDICTION_BEFORE_EVALUATION.json',dict(behavior=choose(profiles),routing=choose_routes(routes),raw_sha256=sha(out/'RAW.jsonl'),evaluation_outputs_seen=False))
  assert len(raw)==1119 and len(budget)==1119
  save(out/'DECISION.json',dict(status='COMPLETE_FIXED_PANEL',attempts=len(budget),payloads=len(raw),controls=len(checks),all_controls_passed=all(c['passed'] for c in checks)));print('COMPLETE1119',len(checks),'controls',flush=True)
 except BaseException:
  save(out/'ERROR.json',dict(error=traceback.format_exc(),attempts=len(budget),completed=len(raw),no_model_retry=True));raise
 finally:save(out/'CONTROLS.json',checks)

def choose_routes(rr):
 keys=['source_voice_dependence','query_match_dependence'];counts={k:sum(r['patterns'][k] for r in rr) for k in keys};c=[k for k,v in counts.items() if v>=2];assert len(c)<=1;return dict(candidate=c[0] if c else None,counts=counts)
if __name__=='__main__':execute()
