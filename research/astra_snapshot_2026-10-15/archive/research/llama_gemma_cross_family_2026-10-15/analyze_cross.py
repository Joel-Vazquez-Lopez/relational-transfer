"""Generation-first cross-family analysis; preserves original study scoring and archives."""
import collections,hashlib,json,random
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def groups(rows,keys):
 d=collections.defaultdict(list)
 for r in rows:d[tuple(r[k] for k in keys)].append(r)
 return d

def ci(values):
 assert len(values)==6
 rng=random.Random(20261015);draws=sorted(sum(values[rng.randrange(6)] for _ in range(6))/6 for _ in range(10000))
 return {'estimate':sum(values)/6,'interval95':[draws[249],draws[9749]],'clusters':6,'cluster_values':values}

def joint(rows):
 result=[]
 for _,g in groups(rows,['id','form','wording','order']).items():
  assert len(g)==2 and {x['role'] for x in g}=={'actor','patient'}
  a=next(x for x in g if x['role']=='actor');p=next(x for x in g if x['role']=='patient')
  x={k:a[k] for k in ['id','pair_index','form','wording','order']};x.update(joint=int(a['correct'] and p['correct']),both_wrong=int(not a['correct'] and not p['correct']))
  if 'actor_margin' in a:x['contrast_reversed']=(a['actor_margin']-p['actor_margin'])<0
  result.append(x)
 return result

def transitions(old,new):
 a={x['id']:x for x in old};b={x['id']:x for x in new};assert a.keys()==b.keys()
 gain=[k for k in a if not a[k]['joint'] and b[k]['joint']];loss=[k for k in a if a[k]['joint'] and not b[k]['joint']]
 vals=[sum(b[k]['joint']-a[k]['joint'] for k in a if a[k]['pair_index']==i)/sum(a[k]['pair_index']==i for k in a) for i in range(6)]
 return {'newly_correct':gain,'newly_incorrect':loss,'paired_difference':ci(vals)}

def main():
 cfg=json.loads((ROOT/'LOCAL_PROTOCOL.json').read_text())
 for name in ['LOCAL','BRIDGE','GEMMA']:
  for f,h in json.loads((ROOT/(name+'_SOURCE_LOCK.json')).read_text()).items():assert sha(ROOT/f)==h,f
 qwen=ROOT.parent/'stronger_model_development_2026-10-15'
 for f,h in cfg['qwen_frozen_checksums'].items():
  if f not in {'NEW_REVIEW.csv','NEW_PROPOSED_LABELS.json','REVIEW_INSTRUCTIONS.txt'}:assert sha(qwen/f)==h,f
 closed=json.loads((ROOT.parent/'FRONTIER_DIAGNOSTICS_CLOSED_2026-10-15.json').read_text())
 for f,h in closed['sha256'].items():assert sha(ROOT.parents[1]/f)==h,f
 paths={'llama':ROOT/'llama.jsonl','llama_no_prefill':ROOT/'llama_no_prefill.jsonl','gemma':ROOT/'gemma.jsonl','phi3':ROOT.parent/'phi3_cross_family_development_2026-10-15'/'phi3.jsonl'}
 paths.update({n:qwen/(n+'.jsonl') for n in ['qwen05','qwen15','qwen7b']})
 data={};report={'status':'Exploratory reused constructed examples, six name-pair clusters; not independent natural-language confirmation','uncertainty_caveat':'Only six reused name-pair clusters. Bootstrap intervals are unstable and degenerate for all-zero/all-one cells; they do not bound unseen-population error.','models':{},'interface_transitions':{},'checksums':{}}
 for n,p in paths.items():
  if not p.exists():continue
  rows=[json.loads(l) for l in p.read_text().splitlines()];assert len(rows)==len({r['key'] for r in rows})
  expected=192 if n=='llama_no_prefill' else 1152 if n.startswith('qwen') else 1728
  if len(rows)!=expected:continue
  if n in ['llama','llama_no_prefill','gemma']:
   complete={'llama':'LOCAL_COMPLETE','llama_no_prefill':'BRIDGE_COMPLETE','gemma':'GEMMA_COMPLETE'}[n]
   assert sha(p)==json.loads((ROOT/(complete+'.json')).read_text())['sha256']
  assert all(r['correct']==(r['answer']==r['gold']) and r['valid']==(r['answer'] in [r['actor'],r['patient']]) for r in rows)
  data[n]=rows;report['checksums'][n]=sha(p)
  free=[r for r in rows if r['panel']=='strong_free'];jj=joint(free);cells={}
  for (form,w),g in groups(jj,['form','wording']).items():
   raw=[r for r in free if r['form']==form and r['wording']==w]
   vals=[sum(x['joint'] for x in g if x['pair_index']==i)/sum(x['pair_index']==i for x in g) for i in range(6)]
   cell={'events':len(g),'joint_correct':sum(x['joint'] for x in g),'both_wrong':sum(x['both_wrong'] for x in g),'accuracy_uncertainty':ci(vals),'actor_correct':sum(r['correct'] for r in raw if r['role']=='actor'),'patient_correct':sum(r['correct'] for r in raw if r['role']=='patient'),'compliant':sum(r['valid'] for r in raw),'answers':len(raw),'errors':[{k:r[k] for k in ['id','role','answer','gold']} for r in raw if not r['correct']]}
   if 'contrast_reversed' in g[0]:cell['contrast_reversed']=sum(x['contrast_reversed'] for x in g)
   cells[form+'/'+w]=cell
  wording={form:transitions([x for x in jj if x['form']==form and x['wording']=='lexical'],[x for x in jj if x['form']==form and x['wording']=='explicit']) for form in cfg['forms']}
  order={}
  for panel in ['strong_list','legacy']:
   gg=groups([r for r in rows if r['panel']==panel],['id','form','wording','role'])
   if not gg or not all(len(g)==2 for g in gg.values()):continue
   order[panel]={form:{'comparisons':sum(k[1]==form for k in gg),'answer_flips':sum(g[0]['answer']!=g[1]['answer'] for k,g in gg.items() if k[1]==form)} for form in cfg['forms']}
  heuristics={}
  for (form,wording),g in groups(free,['form','wording']).items():
   counts=collections.Counter()
   for r in g:
    first=r['patient'] if form in ['passive','object_cleft'] else r['actor'];last=r['actor'] if first==r['patient'] else r['patient'];subj=r['patient'] if form=='passive' else r['actor'];other=r['actor'] if subj==r['patient'] else r['patient']
    pred={'always_actor':r['actor'],'always_patient':r['patient'],'always_first':first,'always_last':last,'first_as_actor':first if r['role']=='actor' else last,'last_as_actor':last if r['role']=='actor' else first,'event_subject_as_actor':subj if r['role']=='actor' else other,'event_subject_as_patient':other if r['role']=='actor' else subj}
    for k,v in pred.items():counts[k]+=v==r['answer']
   heuristics[form+'/'+wording]={'n':len(g),'agreement':dict(counts)}
  report['models'][n]={'records':len(rows),'strong_free':cells,'wording_transitions':wording,'candidate_order':order,'heuristic_prediction_agreement':heuristics}
 if 'llama_no_prefill' in data:
  a=joint([r for r in data['llama'] if r['panel']=='strong_free']);b=joint(data['llama_no_prefill'])
  for form in cfg['forms']:
   for w in cfg['strong_questions']:
    report['interface_transitions'][form+'/'+w]=transitions([x for x in a if x['form']==form and x['wording']==w],[x for x in b if x['form']==form and x['wording']==w])
 if 'gemma' in data:
  from budget import CreditGuard
  guard=CreditGuard(ROOT/'SPEND_LEDGER.jsonl');pending,settled,total=guard.state()
  allowed=json.loads((ROOT/'GEMMA_RECOVERY.json').read_text())['allowed_pending_request']
  assert set(pending)=={allowed} and len(settled)==1729
  report['billing']={'successful_requests_including_smoke':len(settled),'conservative_accounted_eur':str(total),'cap_eur':'1.00','unresolved_rejected_request_upper_eur':str(sum(pending.values()))}
 (ROOT/'CROSS_ANALYSIS.json').write_text(json.dumps(report,indent=2)+'\n')
 for n,m in report['models'].items():
  print(n,{k:v['joint_correct'] for k,v in m['strong_free'].items()})
 if 'billing' in report:print(report['billing'])
if __name__=='__main__':main()
