"""Post hoc join to exposed prior knockouts, requiring exact reproduction."""
from pathlib import Path
import json,numpy as np,hashlib
P=Path(__file__).resolve().parent;OLD=P.parent/'graded_relational_routing_2026-10-18'
def main():
 oldrows=[json.loads(x) for x in (OLD/'qwen7b/RAW.jsonl').read_text().splitlines()];raw=[json.loads(x) for x in (P/'RAW.jsonl').read_text().splitlines()];out=[]
 oldinputs={r['id']:r for r in json.loads((OLD/'INPUTS.json').read_text()) if r['model']=='qwen7b'};newinputs=json.loads((P/'INPUTS.json').read_text())
 for row in newinputs:
  if row['split']=='development':assert row['token_ids']==oldinputs[row['id']]['token_ids'] and row['prompt']==oldinputs[row['id']]['prompt']
 for r in raw:
  if r['kind']!='scientific' or r['split']!='development' or r['condition'] not in ['native','F','N']:continue
  mode={'native':'baseline','F':'frozen','N':'neighbor'}[r['condition']];cut=28 if mode=='baseline' else 14
  prior=next(x for x in oldrows if x['id']==r['id'] and x['kind']=='scientific' and x['mode']==mode and x['cut']==cut)
  assert hashlib.sha256((OLD/'qwen7b'/prior['payload']).read_bytes()).hexdigest()==prior['payload_sha256']
  with np.load(P/r['payload']) as a,np.load(OLD/'qwen7b'/prior['payload']) as b:err=float(np.max(abs(a['logits'].astype(np.float32)-b['logits'].astype(np.float32))))
  out.append(dict(id=r['id'],condition=r['condition'],max_logit_error=err,score_error=r['yes_minus_no']-prior['yes_minus_no']))
 maximum=max(r['max_logit_error'] for r in out);assert maximum==0,'Join prohibited without exact baseline reproduction'
 units=[]
 for fam in ['g01','g02']:
  for voice in ['active','passive']:
   subset=[r for r in oldrows if r['kind']=='scientific' and r['family']==fam and r['voice']==voice and r['mode']=='closed' and r['cut']==14]
   assert len(subset)==4
   for r in subset:assert hashlib.sha256((OLD/'qwen7b'/r['payload']).read_bytes()).hexdigest()==r['payload_sha256']
   vals={(r['role'],r['query']):r['yes_minus_no'] for r in subset};d=(vals[0,0]-vals[0,1]-vals[1,0]+vals[1,1])/4
   u=next(u for u in json.loads((P/'DEVELOPMENT_SUMMARY.json').read_text())['units'] if u['family']==fam and u['voice']==voice)
   units.append(dict(family=fam,voice=voice,closed_D=d,F_D=u['D']['F'],donated_opposite_role_KV_D=u['D']['P_KV'],route_benefit=u['D']['F']-d,role_donation_E=u['E']['P_KV']))
 result=dict(scope='Post hoc descriptive join of exposed prior route tests; not a new held prediction',comparisons=out,maximum=maximum,units=units,old_raw_sha256=hashlib.sha256((OLD/'qwen7b/RAW.jsonl').read_bytes()).hexdigest(),new_raw_sha256=hashlib.sha256((P/'RAW.jsonl').read_bytes()).hexdigest())
 (P/'CROSS_STUDY_REPRODUCTION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(maximum=maximum,comparisons=len(out),units=units),indent=2))
if __name__=='__main__':main()
