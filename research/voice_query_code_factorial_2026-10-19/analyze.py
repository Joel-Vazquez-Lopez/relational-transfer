"""Frozen aggregation, held evaluation and independent payload/score audit."""
import json,collections,csv
from pathlib import Path
import numpy as np
from core import HERE,ROOT,save,sha,unit,patterns,choose,digest
from storage import load
KEYS=['family','split','panel','voice','question_voice','lexicon','mapping','position','mode','cut']
def grouped(raw):
 buckets=collections.defaultdict(list)
 for r in raw:buckets[tuple(r[k] for k in KEYS)].append(r)
 return [{**dict(zip(KEYS,k)),**unit(v)} for k,v in sorted(buckets.items())]
def routing(groups,fam):
 cells=[]
 for v in ['active','passive']:
  for qv in ['active','passive']:
   get=lambda mode:next(u for u in groups if (u['family'],u['panel'],u['voice'],u['question_voice'],u['lexicon'],u['mapping'],u['position'],u['mode'])==(fam,'factorial',v,qv,'AB',0,0,mode))
   native=get('baseline');effects={}
   for mode in ['frozen','closed','irrelevant']:
    u=get(mode);effects[mode]=dict(D=u['D'],retention=u['D']/native['D'] if native['D']>=.5 else None,native_correct=u['native_correct'])
   cells.append(dict(voice=v,question_voice=qv,native=native,effects=effects))
 valid=all(c['native']['D']>=.5 and c['native']['all_ordinal_positive'] for c in cells);clean=valid and all(abs(c['effects']['irrelevant']['retention']-1)<=.25 for c in cells)
 gaps={qv:next(c['effects']['frozen']['retention'] for c in cells if c['voice']=='active' and c['question_voice']==qv)-next(c['effects']['frozen']['retention'] for c in cells if c['voice']=='passive' and c['question_voice']==qv) if valid else None for qv in ['active','passive']}
 source=clean and all(x>=.25 for x in gaps.values());match=clean and gaps['active']>=.25 and gaps['passive']<=-.25
 return dict(family=fam,cells=cells,valid_positive_ordinal_baselines=valid,neutral_drift_gate=bool(clean),retention_gaps=gaps,patterns=dict(source_voice_dependence=bool(source),query_match_dependence=bool(match)))
def analyze(root=HERE):
 root=Path(root);lock=json.loads((root/'LOCK.json').read_text());checks=[]
 def check(name,ok,**kw):
  checks.append(dict(check=name,passed=bool(ok),**kw));assert ok,(name,kw)
 for n,h in lock['files'].items():check('frozen_source',sha(root/n)==h,path=n)
 inputs=json.loads((root/'INPUTS.json').read_text());control=json.loads((root/'CONTROL_INPUT.json').read_text());lookup={r['id']:r for r in inputs+[control]};raw=[json.loads(s) for s in (root/'qwen7b/RAW.jsonl').read_text().splitlines()];budget=[json.loads(s) for s in (root/'BUDGET.jsonl').read_text().splitlines()];check('complete_call_cap',len(raw)==len(budget)==1119);check('all_unique_calls',[r['call'] for r in raw]==list(range(1,1120)))
 cfg=lock['output_shapes'];base_traces={};base_records={}
 for r in raw:
  row=lookup[r['id']]
  path=root/r['payload']
  if path.exists():check('payload_SHA256',sha(path)==r['payload_sha256'],call=r['call'])
  with load(root,r) as z:
   logits=z['logits'].astype(np.float64);trace=z['trace'].astype(np.float32);depths=z['depths'].tolist();check('full_output_shape_finite',logits.shape==(cfg['vocab_size'],) and np.isfinite(logits).all(),call=r['call'])
   cls={k:float(np.logaddexp.reduce(logits[ids])) for k,ids in row['aliases'].items()};s=cls['entailed']-cls['not_entailed'];best=int(np.argmax(logits));actual=next((k for k,ids in row['aliases'].items() if best in ids),None)
   check('independent_score',abs(s-r['s_entail'])<=1e-10 and actual==r['actual'] and best==r['argmax_id'],call=r['call']);check('native_endpoint',r['native_correct']==(actual==row['expected']) and r['format_compliant']==(actual is not None),call=r['call']);check('journal_join',all(r[k]==budget[r['call']-1][k] for k in ['id','mode','cut','kind','physical_length']),call=r['call'])
   if r['wrapped']:
    expected_depths=list(range(29)) if r['mode']=='baseline' else [0,7,14,21,28];check('saved_trace_scope',depths==expected_depths and trace.shape==(len(depths),3,cfg['hidden_size']) and np.isfinite(trace).all(),call=r['call'])
    if r['mode']=='baseline':check('native_full_trace_hash',digest(trace)==r['full_trace_sha256'],call=r['call'])
   else:check('unwrapped_empty_trace',trace.size==0,call=r['call'])
   if r['kind']=='scientific' and r['mode']=='baseline':base_traces[r['id']]=trace.copy();base_records[r['id']]=r
   if r['kind']=='scientific' and r['mode']!='baseline':
    ref=base_traces[r['id']];err=max(float(np.max(np.abs(trace[i]-ref[d]))) for i,d in enumerate(depths) if d<=r['cut']);check('saved_before_cut_anchors',err<=1e-6,call=r['call'],error=err);check('untouched_prefix_hashes',r['hashes']['source_prefix']==base_records[r['id']]['hashes']['source_prefix'],call=r['call'])
 production=json.loads((root/'qwen7b/CONTROLS.json').read_text());check('all_production_controls',all(c['passed'] for c in production),n=len(production))
 scientific=[r for r in raw if r['kind']=='scientific'];check('scientific_count',len(scientific)==1104)
 groups=grouped(scientific);profiles=[patterns(groups,fam) for fam in sorted({r['family'] for r in inputs})];routes=[routing(groups,fam) for fam in sorted({r['family'] for r in inputs})]
 from run import choose_routes
 dev=[p for p in profiles if p['family'] in ['j01','j02','j03']];held=[p for p in profiles if p['family'] in ['j04','j05','j06']];devroute=[p for p in routes if p['family'] in ['j01','j02','j03']];heldroute=[p for p in routes if p['family'] in ['j04','j05','j06']]
 prediction=json.loads((root/'qwen7b/PREDICTION_BEFORE_EVALUATION.json').read_text());check('development_prediction_replay',prediction['behavior']==choose(dev) and prediction['routing']==choose_routes(devroute));candidate=prediction['behavior']['candidate'];rc=prediction['routing']['candidate'];hc=sum(p['patterns'].get(candidate,False) for p in held) if candidate else None;hrc=sum(p['patterns'].get(rc,False) for p in heldroute) if rc else None
 prefix_raw=[r for r in raw if r['split']!='evaluation'];serialized=''.join(json.dumps(r,allow_nan=False)+'\n' for r in prefix_raw);import hashlib
 check('prediction_before_held_raw_hash',hashlib.sha256(serialized.encode()).hexdigest()==prediction['raw_sha256'])
 tables=[]
 for key in sorted({(r['voice'],r['question_voice'],r['lexicon'],r['mapping'],r['position']) for r in scientific if r['panel']=='factorial' and r['mode']=='baseline'}):
  rr=[r for r in scientific if r['panel']=='factorial' and r['mode']=='baseline' and tuple(r[k] for k in ['voice','question_voice','lexicon','mapping','position'])==key];g=[u for u in groups if u['panel']=='factorial' and u['mode']=='baseline' and tuple(u[k] for k in ['voice','question_voice','lexicon','mapping','position'])==key]
  tables.append(dict(zip(['voice','question_voice','lexicon','mapping','position'],key))|dict(n_answers=len(rr),native_correct=sum(r['native_correct'] for r in rr),format_compliant=sum(r['format_compliant'] for r in rr),n_units=len(g),native_complete=sum(u['native_complete'] for u in g),mean_D=float(np.mean([u['D'] for u in g])),positive_mean=float(np.mean([u['positive_mean'] for u in g])),negative_mean=float(np.mean([u['negative_mean'] for u in g]))))
 result=dict(scope='Small fixed factorial diagnostic, not a population significance or semantic-uniqueness claim',forwards=len(raw),scientific=len(scientific),prediction=prediction,held_behavior_count=hc,held_behavior_replicated=candidate is not None and hc>=2,held_routing_count=hrc,held_routing_replicated=rc is not None and hrc>=2,held_behavior_counts=choose(held)['counts'],held_routing_counts=choose_routes(heldroute)['counts'],profiles=profiles,routing=routes,groups=groups,native_tables=tables,production_controls=len(production),independent_checks=len(checks),all_checks_passed=all(c['passed'] for c in checks),no_p_values=True)
 save(root/'RESULTS.json',result);save(root/'INDEPENDENT_AUDIT.json',dict(passed=True,n_checks=len(checks),checks=checks))
 with (root/'ANSWER_LEDGER.csv').open('w') as f:
  cols=['call','id','family','split','panel','voice','question_voice','lexicon','mapping','position','role','query','mode','kind','expected','expected_symbol','actual','argmax_token','native_correct','format_compliant','s_entail','payload','payload_sha256'];w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(raw)
 save(root/'UNEXPECTED_ANSWERS.json',[r for r in scientific if not r['native_correct']]);save(root/'COMPLETE_FAMILY_OUTCOMES.json',[dict(family=f,panel=panel,n_answers=len(rr),native_correct=sum(r['native_correct'] for r in rr),all_correct=all(r['native_correct'] for r in rr)) for f in sorted({r['family'] for r in inputs}) for panel in ['old_reference','factorial'] for rr in [[r for r in scientific if r['family']==f and r['panel']==panel and r['mode']=='baseline']]])
 print(json.dumps({k:result[k] for k in ['forwards','prediction','held_behavior_count','held_behavior_replicated','held_behavior_counts','held_routing_count','held_routing_replicated','held_routing_counts','production_controls','independent_checks']},indent=2));return result
if __name__=='__main__':analyze()
