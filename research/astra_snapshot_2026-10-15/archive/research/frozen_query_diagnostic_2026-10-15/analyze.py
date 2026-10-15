"""Descriptive analysis of frozen development protocol; no model inference."""
import collections, hashlib, json, random, re
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def norm(s):return re.sub(r'\s+',' ',re.sub(r'[^\w\s]',' ',(s or '').casefold())).strip()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def exact(s,spans):return s is not None and norm(s) in {norm(x) for x in spans}
def f1(s,spans):
 p=norm(s).split();scores=[]
 for g in spans:
  q=norm(g).split();n=sum((collections.Counter(p)&collections.Counter(q)).values());scores.append(2*n/(len(p)+len(q)) if p or q else 0.)
 return max(scores,default=0.)
def ci(v):
 rng=random.Random(910215);means=sorted(sum(rng.choices(v,k=len(v)))/len(v) for _ in range(10000));return [means[249],means[9749]]
def main():
 cfg=json.loads((ROOT/'PROTOCOL.json').read_text());items=json.loads((ROOT/'ITEMS.json').read_text())['items'];out={};lines=['# Frozen pretrained query diagnostic','', 'Exploratory development; 24 active subject/direct-object events, one pretrained family, two sizes. No training or semantic agent/patient confirmation.','']
 for model in cfg['models']:
  p=ROOT/f'PREDICTIONS_{model}.jsonl';complete=ROOT/f'COMPLETE_{model}.json'
  if not complete.exists():continue
  meta=json.loads(complete.read_text());assert meta['protocol_sha256']==sha(ROOT/'PROTOCOL.json');assert meta['predictions_sha256']==sha(p)
  rs=[json.loads(x) for x in p.read_text().splitlines()];assert len(rs)==120
  by={(r['id'],r['arm'],r['role']):r for r in rs};assert len(by)==120
  assert all(r['protocol_sha256']==sha(ROOT/'PROTOCOL.json') for r in rs)
  metrics={};vec={}
  for arm in cfg['arms']:
   m=collections.Counter();both=[]
   for item in items:
    event=[];tuples=[]
    for role in ['subject','object']:
     r=by[item['id'],arm,role if arm!='query_blind' else 'both'];fields=r['fields'];pred=r['direct_answer'] if arm=='direct' else fields.get('answer') if arm=='query_visible' else fields.get(role)
     gold=item[role+'_accepted'];ok=exact(pred,gold);event.append(ok);m['answers_correct']+=ok;m['answer_f1_sum']+=f1(pred,gold);m['missing_answers']+=pred is None;m['length_limit_responses']+=r['hit_length_limit']
     if arm!='direct':
      field=fields.get(role);fc=exact(field,gold);m['requested_field_correct']+=fc;m['correct_field_wrong_answer']+=fc and not ok;m['answer_matches_field']+=pred is not None and field is not None and norm(pred)==norm(field);tuples.append(tuple(norm(fields.get(k)) for k in ['subject','object']))
    both.append(int(all(event)));m['both_answers_correct']+=all(event)
    if arm=='query_visible':
     a=by[item['id'],arm,'subject'];b=by[item['id'],arm,'object'];parsable=all(all(k in r['fields'] for k in ['subject','object']) for r in [a,b]);m['both_role_tuples_parse']+=parsable;m['role_tuple_changes']+=parsable and tuples[0]!=tuples[1]
   m['answer_denominator']=48;m['event_denominator']=24;m['mean_answer_f1']=m.pop('answer_f1_sum')/48;metrics[arm]=dict(m);vec[arm]=both
  contrasts={a+'_minus_direct':{'both_correct_difference_pp':100*sum(x-y for x,y in zip(vec[a],vec['direct']))/24,'paired_event_bootstrap_95_pp':[100*x for x in ci([x-y for x,y in zip(vec[a],vec['direct'])])]} for a in ['query_visible','query_blind']}
  out[model]={'metrics':metrics,'contrasts':contrasts};lines+=['## Qwen '+model,'','| Condition | Individual exact /48 | Both exact /24 | Mean span F1 |','|---|---:|---:|---:|']
  for a,m in metrics.items():lines.append(f"| {a} | {m['answers_correct']} | {m['both_answers_correct']} | {m['mean_answer_f1']:.3f} |")
  lines+=['',json.dumps(contrasts,indent=2),'']
 result={'status':'exploratory_development_only','protocol_sha256':sha(ROOT/'PROTOCOL.json'),'models':out,'caveats':cfg['limits']};(ROOT/'ANALYSIS.json').write_text(json.dumps(result,indent=2)+'\n');(ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
