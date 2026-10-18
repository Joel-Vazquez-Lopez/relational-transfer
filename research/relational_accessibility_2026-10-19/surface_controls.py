"""Prespecified syntax/position competitors evaluated from text only."""
from pathlib import Path
import re,json,numpy as np
P=Path(__file__).resolve().parent;R=P.parent.parent;results=[]
for model in ['qwen7b','llama3b']:
 raw=json.loads((R/'research/linguistic_development_atlas_continuation_2026-10-18'/model/'RAW.json').read_text())['rows']
 for fam in ['a01','a02']:
  rows=[r for r in raw if r['question_key']=='agent' and r['family']==fam];original=next(r for r in rows if r['frame']=='within' and r['swap']==0);canonical=original['annotations']['theme']
  records=[]
  for r in rows:
   first,mark,second,mark2,third=re.fullmatch(r'\w+ \w+ (\w+) (and|to) (\w+) (and|to) (\w+)\.',r['sentence']).groups();assert mark!=mark2
   actual=0 if len(r['annotations']['theme'])==2 else 1 if canonical[0] in r['annotations']['theme'] else 2
   parser=0 if mark=='and' else 1 if first==canonical[0] else 2
   frame=0 if mark=='and' else 1;prob=[1,0,0] if mark=='and' else [0,.5,.5]
   assert parser==actual
   records.append(dict(id=r['id'],target=actual,parser=parser,frame_only=frame,frame_probability=prob,gold_probability=prob[actual]))
  results.append(dict(model=model,family=fam,n=4,parser_correct=4,frame_only_correct=sum(x['frame_only']==x['target'] for x in records),frame_ce=float(-np.log([x['gold_probability'] for x in records]).mean()),rows=records))
(P/'SURFACE_CONTROLS.json').write_text(json.dumps(dict(probe_fits=0,new_model_forwards=0,scope='Required predefined surface competitors; post-probe evaluation, not a new endpoint',groups=results),indent=2)+'\n')
print(json.dumps([(r['model'],r['family'],r['frame_only_correct'],r['frame_ce'],r['parser_correct']) for r in results]))
