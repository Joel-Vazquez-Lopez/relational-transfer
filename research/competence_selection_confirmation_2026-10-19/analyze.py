"""Frozen arithmetic audit/analysis, independent full-vocabulary score reconstruction."""
import json,hashlib,csv,math
from pathlib import Path
import numpy as np
from core import HERE,ROOT,save,sha
MODELS=['qwen7b','llama3b'];VOICES=['active','passive'];MODES=['baseline','frozen','closed','irrelevant']
def sign_test(values):
 nonzero=[x for x in values if x!=0];n=len(nonzero);k=sum(x>0 for x in nonzero)
 return dict(positive=k,negative=n-k,ties=len(values)-n,n_nonzero=n,one_sided_p=sum(math.comb(n,j) for j in range(k,n+1))/2**n if n else 1.,assumption='Independent/exchangeable family signs under sign-symmetry null; authored panel/shared templates limit inference')
def contrast(rows):
 x={(r['role'],r['query']):r['yes_minus_no'] for r in rows};assert len(x)==4
 return sum((-1)**(r+q)*s for (r,q),s in x.items())/4

def main():
 lock=json.loads((HERE/'LOCK.json').read_text());checks=[]
 def check(name,ok,**kw):
  checks.append(dict(check=name,passed=bool(ok),**kw));assert ok,(name,kw)
 for n,h in lock['files'].items():check('frozen_file',sha(HERE/n)==h,path=n)
 rows=json.loads((HERE/'INPUTS.json').read_text());lookup={(r['model'],r['id']):r for r in rows};raw=[]
 budget=[json.loads(s) for s in (HERE/'BUDGET.jsonl').read_text().splitlines()];check('attempt_cap',len(budget)==1562)
 for m in MODELS:
  rr=[json.loads(s) for s in (HERE/m/'RAW.jsonl').read_text().splitlines()];check('model_call_count',len(rr)==781,model=m)
  controls=json.loads((HERE/m/'CONTROLS.json').read_text());check('production_controls',all(c['passed'] for c in controls),model=m,n=len(controls))
  for r in rr:
   row=lookup[m,r['id']];p=HERE/m/r['payload'];check('payload_sha256',sha(p)==r['payload_sha256'],call=r['call'])
   with np.load(p) as a:
    z=a['logits'].astype(np.float64);cls={k:float(np.logaddexp.reduce(z[ids])) for k,ids in row['aliases'].items()};s=cls['Yes']-cls['No'];best=int(z.argmax());native=next((k for k,ids in row['aliases'].items() if best in ids),None)
    check('score_replay',abs(s-r['yes_minus_no'])<1e-10 and native==r['actual'] and best==r['argmax_id'],call=r['call'])
    cal='Yes' if s>r['threshold'] else 'No' if s<r['threshold'] else None
    check('labels_replay',r['correct']==(native==row['expected'] and (cls[row['expected']]>cls['No' if row['expected']=='Yes' else 'Yes'])) and r['calibrated_class']==cal and r['calibrated_correct']==(cal==row['expected']),call=r['call'])
    if r['mode']=='baseline' and r['wrapped']:
     check('native_complete_trace',a['trace'].shape[0:2]==(29,2) and a['depths'].tolist()==list(range(29)),call=r['call'])
   check('journal_replay',all(r[k]==budget[r['call']-1][k] for k in ['model','id','mode','cut','kind']),call=r['call'])
  raw+=rr
 check('unique_calls',sorted(r['call'] for r in raw)==list(range(1,1563)))
 scientific=[r for r in raw if r['kind']=='scientific'];units=[]
 for m in MODELS:
  for fam in sorted({r['family'] for r in rows}):
   for voice in VOICES:
    group=[r for r in scientific if (r['model'],r['family'],r['voice'])==(m,fam,voice)];baseline=[r for r in group if r['mode']=='baseline'];check('factorial_unit',len(group)==16 and len(baseline)==4)
    d=contrast(baseline);scores={(r['role'],r['query']):r['yes_minus_no'] for r in baseline};yes=[r['yes_minus_no'] for r in baseline if r['expected']=='Yes'];no=[r['yes_minus_no'] for r in baseline if r['expected']=='No'];lo=max(no);hi=min(yes);t=baseline[0]['threshold']
    ordinary=[(-1)**q*(scores[0,q]-scores[1,q]) for q in [0,1]]+[(-1)**r*(scores[r,0]-scores[r,1]) for r in [0,1]]
    unit=dict(model=m,family=fam,voice=voice,native_complete=all(r['correct'] for r in baseline),calibrated_complete=all(r['calibrated_correct'] for r in baseline),native_correct=sum(r['correct'] for r in baseline),calibrated_correct=sum(r['calibrated_correct'] for r in baseline),zero_threshold_correct=sum(('Yes' if r['yes_minus_no']>0 else 'No' if r['yes_minus_no']<0 else None)==r['expected'] for r in baseline),off_label=sum(r['actual'] is None for r in baseline),D_native=d,offset=sum(scores.values())/4,role_term=sum((-1)**r*s for (r,q),s in scores.items())/4,question_term=sum((-1)**q*s for (r,q),s in scores.items())/4,interval=[lo,hi],separable=lo<hi,threshold_inside=lo<t<hi,all_ordinal_positive=all(x>0 for x in ordinary),ordinal_pairs=ordinary,interventions={})
    for mode in MODES[1:]:
     changed=[r for r in group if r['mode']==mode];dd=contrast(changed)
     unit['interventions'][mode]=dict(D=dd,retention=dd/d if abs(d)>=.5 else None,native_correct=sum(r['correct'] for r in changed),calibrated_correct=sum(r['calibrated_correct'] for r in changed))
    units.append(unit)
 families=[]
 for fam in sorted({u['family'] for u in units}):
  f=dict(family=fam,models={})
  for m in MODELS:
   a,p=[next(u for u in units if (u['family'],u['model'],u['voice'])==(fam,m,v)) for v in VOICES]
   f['models'][m]=dict(d=int(a['calibrated_complete'])-int(a['native_complete'])-int(p['calibrated_complete'])+int(p['native_complete']),native_whole_family=a['native_complete'] and p['native_complete'],calibrated_whole_family=a['calibrated_complete'] and p['calibrated_complete'])
  families.append(f)
 primary=sign_test([f['models']['qwen7b']['d'] for f in families]);primary['alpha']=.05;primary['supported']=primary['one_sided_p']<=.05 and primary['positive']>primary['negative'];primary['mean_d']=sum(f['models']['qwen7b']['d'] for f in families)/24
 totals=[]
 for m in MODELS:
  for v in VOICES:
   us=[u for u in units if u['model']==m and u['voice']==v]
   totals.append(dict(model=m,voice=v,n_answers=96,n_units=24,**{k:sum(u[k] for u in us) for k in ['native_correct','calibrated_correct','zero_threshold_correct','off_label','native_complete','calibrated_complete','separable','all_ordinal_positive']},mean_D=float(np.mean([u['D_native'] for u in us])),mean_offset=float(np.mean([u['offset'] for u in us]))))
 profiles=[];gates=['all','native_complete','calibrated_complete','native_whole_family','calibrated_whole_family']
 for mode in MODES[1:]:
  for gate in gates:
   selections={}
   for m in MODELS:
    def eligible(u):
     if gate=='all':return True
     if gate.endswith('whole_family'):return next(f for f in families if f['family']==u['family'])['models'][m][gate]
     return u[gate]
    selections[m]={(u['family'],u['voice']) for u in units if u['model']==m and eligible(u)}
   for policy in ['own_selected','common_selected']:
    common=selections[MODELS[0]]&selections[MODELS[1]];record=dict(mode=mode,gate=gate,policy=policy,models={})
    for m in MODELS:
     selected=selections[m] if policy=='own_selected' else common;us=[u for u in units if u['model']==m and (u['family'],u['voice']) in selected];valid=[u for u in us if u['interventions'][mode]['retention'] is not None]
     record['models'][m]=dict(n_selected=len(us),n_ratio=len(valid),units=[list(x) for x in sorted(selected)],ratio_units=[[u['family'],u['voice']] for u in valid],negative_native_D=sum(u['D_native']<0 for u in us),mean_retention=float(np.mean([u['interventions'][mode]['retention'] for u in valid])) if valid else None,median_retention=float(np.median([u['interventions'][mode]['retention'] for u in valid])) if valid else None,mean_raw_D=float(np.mean([u['interventions'][mode]['D'] for u in us])) if us else None)
    x=record['models'];record['gap']=x['llama3b']['mean_retention']-x['qwen7b']['mean_retention'] if all(x[m]['mean_retention'] is not None for m in MODELS) else None;record['raw_D_gap']=x['llama3b']['mean_raw_D']-x['qwen7b']['mean_raw_D'] if all(x[m]['mean_raw_D'] is not None for m in MODELS) else None;profiles.append(record)
 def prof(gate,policy):return next(x for x in profiles if (x['mode'],x['gate'],x['policy'])==('frozen',gate,policy))
 base=prof('all','own_selected');native=prof('native_complete','own_selected');common=prof('native_complete','common_selected');cal=prof('calibrated_complete','own_selected')
 delta=native['gap']-base['gap'] if native['gap'] is not None and base['gap'] is not None else None;cd=common['gap']-base['gap'] if common['gap'] is not None and base['gap'] is not None else None
 secondary=dict(full_gap=base['gap'],native_gate_gap=native['gap'],calibrated_gate_gap=cal['gap'],delta_gap=delta,common_native_gap=common['gap'],common_delta_gap=cd,descriptive_replication=delta is not None and delta>=.10,matched_support_direction=cd is not None and cd>0,no_secondary_significance_test=True)
 result=dict(scope='One frozen prospective panel; primary family sign test and descriptive secondary, no adaptive inference',primary=primary,totals=totals,families=families,units=units,profiles=profiles,secondary=secondary,n_forwards=len(raw),n_independent_checks=len(checks),production_controls={m:len(json.loads((HERE/m/'CONTROLS.json').read_text())) for m in MODELS})
 save(HERE/'RESULTS.json',result);save(HERE/'INDEPENDENT_AUDIT.json',dict(passed=all(c['passed'] for c in checks),n_checks=len(checks),checks=checks))
 columns=['call','model','family','voice','role','query','id','kind','mode','cut','expected','actual','argmax_token','correct','yes_minus_no','calibrated_class','calibrated_correct','payload','payload_sha256']
 with (HERE/'ANSWER_LEDGER.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(raw)
 save(HERE/'UNEXPECTED_ANSWERS.json',[r for r in scientific if not r['correct'] or not r['calibrated_correct']])
 print(json.dumps(dict(primary=primary,totals=totals,secondary=secondary,forwards=len(raw),audit_checks=len(checks)),indent=2))
if __name__=='__main__':main()
