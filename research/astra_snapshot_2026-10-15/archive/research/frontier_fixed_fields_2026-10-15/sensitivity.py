"""Explicitly post hoc comparison with ALL already available interface controls."""
from pathlib import Path
import json,random,re,hashlib
R=Path(__file__).resolve().parent;P=R.parent/'frontier_answer_contract_2026-10-15'
def norm(s):return re.sub(r'\s+',' ',re.sub(r'[^\w\s]',' ',(s or '').casefold())).strip()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
items=json.loads((R/'ITEMS.json').read_text());idx={i['id']:i for i in items};prior=[json.loads(x) for x in (P/'PREDICTIONS.jsonl').read_text().splitlines()];now=[json.loads(x) for x in (R/'SCORED_ROWS.jsonl').read_text().splitlines()];cfg=json.loads((R/'COMPUTE_PROXY.json').read_text());d=cfg['hidden'];W=cfg['dense_layer_weights'];N=cfg['layers'];V=cfg['vocab']
def work(L,g):return 2*W*(L+g-1)+2*d*V*g+4*N*d*(L*L+(g-1)*L+g*(g-1)/2)
def interval(v):
 rng=random.Random(20261015);v=list(v);b=sorted(100*sum(rng.choices(v,k=len(v)))/len(v) for _ in range(10000));return {'difference_pp':100*sum(v)/len(v),'paired_event_bootstrap_95_pp':[b[250],b[9750]]}
pg=[x for x in now if x['kind']=='generated' and x['sentence'] and not x['history'] and not x['swap']];event={i['id']:sum(x['scores']['exact'] for x in pg if x['id']==i['id'])/4 for i in items};out={}
for ins,pre in [('original',''),('original','Answer:'),('explicit_span_contract',''),('explicit_span_contract','Answer:')]:
 rr=[x for x in prior if x['id'] in idx and x['instruction']==ins and x['prefix']==pre];sc=lambda x:int(norm(x['raw']) in {norm(a) for a in idx[x['id']][x['role']+'_accepted']});by={i['id']:sum(sc(x) for x in rr if x['id']==i['id'])/2 for i in items};out[ins+'/'+(pre or 'none')]={'correct':sum(sc(x) for x in rr),'n':24,'mean_forward_work_proxy':sum(work(x['input_tokens'],x['generated_tokens']) for x in rr)/len(rr),'generated_fields_reader_minus_prior_direct':interval([event[e]-by[e] for e in idx])}
ref=[x for x in now if x['kind']=='reference' and not x['swap']];wrong_compliant=[x for x in ref if x['scores']['compliant'] and not x['scores']['exact']]
result={'scope':'Post hoc sensitivity; all prior interface cells shown, not a new selected confirmatory comparator','prior_direct_cells':out,'reference_fields_wrong_but_compliant':{'n':len(wrong_compliant),'all_choose_opposite_field':all(x['scores']['follows_opposite'] for x in wrong_compliant),'cases':[{'id':x['id'],'role':x['role'],'order':x['order'],'raw':x['raw'],'supplied_fields':x['supplied_fields']} for x in wrong_compliant]},'source_hashes':{'previous_predictions':sha(P/'PREDICTIONS.jsonl'),'current_scored_rows':sha(R/'SCORED_ROWS.jsonl')}}
(R/'SENSITIVITY.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['prior_direct_cells'],indent=2));print('Wrong compliant answers with reference fields:',len(wrong_compliant))
