"""Frozen development follow-up analysis; no inference or model selection."""
import json
from pathlib import Path
from analyze import exact,norm,sha,ci
ROOT=Path(__file__).resolve().parent
items=json.loads((ROOT/'ITEMS.json').read_text())['items'];cfg=json.loads((ROOT/'FIELD_ONLY_PROTOCOL.json').read_text());out={}
for key in cfg['models']:
 path=ROOT/f'FIELD_ONLY_{key}.jsonl';complete=ROOT/f'FIELD_ONLY_COMPLETE_{key}.json'
 if not complete.exists():continue
 meta=json.loads(complete.read_text());assert meta['protocol_sha256']==sha(ROOT/'FIELD_ONLY_PROTOCOL.json');assert meta['predictions_sha256']==sha(path)
 rows=[json.loads(l) for l in path.read_text().splitlines()];assert len(rows)==48
 by={(r['id'],r['role']):r for r in rows};assert len(by)==48
 base=[json.loads(l) for l in (ROOT/f'PREDICTIONS_{key}.jsonl').read_text().splitlines()];blind={r['id']:r for r in base if r['arm']=='query_blind'}
 vc=bc=vb=bb=changes=both_parse=missing=0;diffs=[];gains=losses=0
 for i in items:
  vs=[];bs=[]
  for role in ['subject','object']:
   r=by[i['id'],role];a=exact(r['fields'].get(role),i[role+'_accepted']);b=exact(blind[i['id']]['fields'].get(role),i[role+'_accepted']);vc+=a;bc+=b;vs.append(a);bs.append(b);missing+=role not in r['fields']
  v=int(all(vs));b=int(all(bs));vb+=v;bb+=b;diffs.append(b-v);gains+=b>v;losses+=b<v
  a,p=[by[i['id'],role]['fields'] for role in ['subject','object']];parse=all(all(k in f for k in ['subject','object']) for f in [a,p]);both_parse+=parse;changes+=parse and any(norm(a[k])!=norm(p[k]) for k in ['subject','object'])
 out[key]={'question_visible_fields_only':{'requested_correct':vc,'both_correct':vb,'missing_requested_fields':missing,'role_tuple_changes':changes,'both_tuples_parse':both_parse},'question_blind':{'requested_correct':bc,'both_correct':bb},'answer_total':48,'event_total':24,'blind_minus_visible_both_pp':100*sum(diffs)/24,'paired_event_bootstrap_95_pp':[100*x for x in ci(diffs)],'blind_gains':gains,'blind_losses':losses}
result={'status':'exploratory_followup_same_development_items','protocol_sha256':sha(ROOT/'FIELD_ONLY_PROTOCOL.json'),'models':out,'limits':'Output schema and system instruction match; query presence, requested role, wording and input length differ. One family, active subject-first events, grammatical role labels. Printed output differences do not establish hidden representations or a new training principle.'}
(ROOT/'FIELD_ONLY_ANALYSIS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
