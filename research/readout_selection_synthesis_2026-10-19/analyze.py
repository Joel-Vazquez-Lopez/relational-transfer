"""Post hoc complete-panel synthesis; no significance tests or model calls."""
from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent;ROOT=P.parent.parent

def main():
 sources=[];rows=[]
 for study in ['graded_relational_routing_2026-10-18','period_message_interchange_2026-10-19']:
  p=ROOT/'research'/study
  for model in (['qwen7b','llama3b'] if study.startswith('graded') else ['qwen7b']):
   path=p/model/'RAW.jsonl' if study.startswith('graded') else p/'RAW.jsonl';rr=[json.loads(x) for x in path.read_text().splitlines()]
   sources.append(dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
   for r in rr:
    if r['kind']=='scientific' and r['split']=='evaluation' and (r.get('mode')=='baseline' or r.get('condition')=='native'):
     rows.append(dict(model=model,study=study,id=r['id'],family=r['family'],voice=r['voice'],role=r['role'],query=r['query'],native_correct=r['correct'],calibrated_correct=r['calibrated_correct'],threshold=r['threshold'],actual=r['actual'],calibrated_class=r['calibrated_class'],yes_minus_no=r['yes_minus_no'],expected=r['expected']))
 assert len(rows)==80 and len({(r['model'],r['id']) for r in rows})==80
 groups=[];summary=[]
 for model in ['qwen7b','llama3b']:
  mr=[r for r in rows if r['model']==model];assert len({r['threshold'] for r in mr})==1
  for fam in sorted({r['family'] for r in mr}):
   for voice in ['active','passive']:
    rr=[r for r in mr if r['family']==fam and r['voice']==voice];assert len(rr)==4
    scores={(r['role'],r['query']):r['yes_minus_no'] for r in rr};D=(scores[0,0]-scores[0,1]-scores[1,0]+scores[1,1])/4
    groups.append(dict(model=model,family=fam,voice=voice,n=4,native_correct=sum(r['native_correct'] for r in rr),calibrated_correct=sum(r['calibrated_correct'] for r in rr),D=D,native_complete=all(r['native_correct'] for r in rr),calibrated_complete=all(r['calibrated_correct'] for r in rr)))
  for voice in ['active','passive']:
   rr=[r for r in mr if r['voice']==voice];gg=[g for g in groups if g['model']==model and g['voice']==voice]
   summary.append(dict(model=model,voice=voice,n=len(rr),native_correct=sum(r['native_correct'] for r in rr),calibrated_correct=sum(r['calibrated_correct'] for r in rr),native_complete_units=sum(g['native_complete'] for g in gg),calibrated_complete_units=sum(g['calibrated_complete'] for g in gg),units=len(gg)))
 result=dict(scope='Post hoc descriptive synthesis of every available native evaluation cell; Qwen6 families, Llama4. Not a new preregistered endpoint or significance test',sources=sources,summary=summary,groups=groups,rows=rows,new_model_forwards=0,new_probe_fits=0,p_values_computed=False)
 (P/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary))
if __name__=='__main__':main()
