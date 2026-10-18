"""Frozen saved-data probe diagnostic, never imports or loads an LLM."""
from pathlib import Path
import json,hashlib,itertools,warnings,csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent

def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fit(X,y):
 X=np.asarray(X,dtype=np.float64);y=np.asarray(y);mu=X.mean(0);scale=float(np.sqrt(np.mean(np.sum((X-mu)**2,axis=1))))
 classes=np.unique(y);prior=np.array([np.mean(y==i) for i in classes])
 if scale==0:return lambda T:np.tile(prior,(len(T),1)),[],classes
 clf=LogisticRegression(C=1,solver='lbfgs',tol=1e-10,max_iter=2000)
 with warnings.catch_warnings(record=True) as ww:
  warnings.simplefilter('always');clf.fit((X-mu)/scale,y)
 return lambda T:clf.predict_proba((np.asarray(T)-mu)/scale),[str(w.message) for w in ww],classes

def metrics(p,y,classes):
 ix=[list(classes).index(i) for i in y];return dict(ce=float(-np.log(np.maximum(p[np.arange(len(y)),ix],1e-12)).mean()),correct=int(np.sum(classes[p.argmax(1)]==y)),n=len(y))
def main():
 lock=json.loads((HERE/'LOCK.json').read_text())
 for f,h in lock['files'].items():assert sha(ROOT/f)==h,f
 assert not (HERE/'RESULTS.json').exists()
 detail=[];brief=[];fits=0;warn=[]
 tokens={ (r['model'],r['id']):r for r in map(json.loads,(ROOT/'research/linguistic_development_atlas_design_2026-10-18/DRAFT_TOKENIZATION.jsonl').read_text().splitlines())}
 permutations=sorted(set(itertools.permutations([0,0,1,2])))
 for model in ['qwen7b','llama3b']:
  directory=ROOT/'research/linguistic_development_atlas_continuation_2026-10-18'/model
  rows=[r for r in json.loads((directory/'RAW.json').read_text())['rows'] if r['question_key']=='agent']
  data={}
  for family in ['a01','a02']:
   rr=[r for r in rows if r['family']==family];rr.sort(key=lambda r:('within'!=r['frame'],r['swap']))
   assert [r['frame'] for r in rr]==['within','within','across','across']
   assert len({r['question'] for r in rr})==len({r['expected'] for r in rr})==1
   data[family]={}
   for site in ['period','answer']:
    states=[]
    for r in rr:
     t=tokens[model,r['id']];pos=t['sentence_positions'][-1] if site=='period' else t['original_prompt_end'];j=t['retained_positions'].index(pos)
     with np.load(directory/(r['id']+'.npz')) as z:
      x=z['residual'][:,j].astype(np.float64);a=z['attention'][:,j].astype(np.float64)
      states.append(dict(incoming=x[:-1],attention=x[:-1]+a,outgoing=x[1:]))
    data[family][site]={k:np.stack([s[k] for s in states]) for k in states[0]}
  for training,test in [('a01','a02'),('a02','a01')]:
   y=np.array([0,0,1,2]);prior=np.tile([.5,.25,.25],(4,1));base=metrics(prior,y,np.arange(3))
   for site in ['period','answer']:
    for layer in range(28):
     X=data[training][site];T=data[test][site]
     decoder,ww,classes=fit(X['outgoing'][:,layer],y);fits+=1;warn+=ww
     for boundary in ['incoming','attention','outgoing']:
      prob=decoder(T[boundary][:,layer]);m=metrics(prob,y,classes)
      detail.append(dict(study='atlas',model=model,training=training,test=test,site=site,layer=layer,boundary=boundary,decoder_fit='outgoing_shared',prior_ce=base['ce'],gain=base['ce']-m['ce'],**m,probabilities=prob.tolist(),labels=y.tolist()))
     # Same-boundary fitted decoders, separately labeled.
     for boundary in ['incoming','attention']:
      g,ww,cl=fit(X[boundary][:,layer],y);fits+=1;warn+=ww;prob=g(T[boundary][:,layer]);m=metrics(prob,y,cl)
      detail.append(dict(study='atlas',model=model,training=training,test=test,site=site,layer=layer,boundary=boundary,decoder_fit='same_boundary',prior_ce=base['ce'],gain=base['ce']-m['ce'],**m,probabilities=prob.tolist(),labels=y.tolist()))
     permresults=[]
     for yp in permutations:
      g,ww,cl=fit(X['outgoing'][:,layer],yp);fits+=1;warn+=ww;pp=g(T['outgoing'][:,layer]);m=metrics(pp,y,cl);permresults.append(dict(training_labels=yp,**m))
     actual=next(v['ce'] for v in permresults if list(v['training_labels'])==y.tolist())
     detail.append(dict(study='atlas_permutation',model=model,training=training,test=test,site=site,layer=layer,actual_ce=actual,rank_among_12=1+sum(v['ce']<actual-1e-10 for v in permresults),all=permresults))
  brief.append(dict(model=model,study='atlas',n_physical_sentences=8,n_lexical_families=2))
 # Independent six-family panel: one fixed train/evaluation split, plus cross-voice.
 p=ROOT/'research/graded_relational_routing_2026-10-18';inputs={ (r['model'],r['id']):r for r in json.loads((p/'INPUTS.json').read_text())}
 for model in ['qwen7b','llama3b']:
  raw=[json.loads(s) for s in (p/model/'RAW.jsonl').read_text().splitlines()];rows=[r for r in raw if r['kind']=='scientific' and r['mode']=='baseline'];vectors={}
  for r in rows:
   with np.load(p/model/r['payload']) as z:vectors[r['id']]=z['trace'].astype(np.float64)
  for site,j in [('period',0),('answer',1)]:
   rr=[r for r in rows if site=='answer' or r['query']==0]
   for transfer,tv,ev in [('both',None,None),('active_to_passive','active','passive'),('passive_to_active','passive','active')]:
    train=[r for r in rr if r['split']=='development' and (tv is None or r['voice']==tv)];test=[r for r in rr if r['split']=='evaluation' and (ev is None or r['voice']==ev)]
    for layer in range(29):
     X=np.stack([vectors[r['id']][layer,j] for r in train]);Y=[r['role'] for r in train];T=np.stack([vectors[r['id']][layer,j] for r in test]);yy=[r['role'] for r in test]
     g,ww,cl=fit(X,Y);fits+=1;warn+=ww;prob=g(T);m=metrics(prob,yy,cl)
     detail.append(dict(study='graded',model=model,site=site,transfer=transfer,layer=layer,train_n=len(train),prior_ce=float(np.log(2)),gain=float(np.log(2))-m['ce'],**m,probabilities=prob.tolist(),labels=yy,ids=[r['id'] for r in test]))
 save(HERE/'RESULTS.json',dict(all=detail,summary=brief,probe_fits=fits,warnings=warn,new_transformer_forwards=0,source_lock_sha256=sha(HERE/'LOCK.json')))
 out=[]
 for r in detail:
  if r['study']=='atlas' and r['boundary']=='outgoing' and r['decoder_fit']=='outgoing_shared':out.append({k:r[k] for k in ['model','training','test','site','layer','ce','gain','correct','n']})
 with (HERE/'ATLAS_LEDGER.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
 print(json.dumps(dict(probe_fits=fits,warnings=len(warn),new_transformer_forwards=0,records=len(detail))))
if __name__=='__main__':main()
