"""Independent regex grammar check of AI-authored templates; no model inference."""
import json,re
from pathlib import Path
from study import dataset
P=Path(__file__).resolve().parent
rows=[]
for split in ['train','dev','held']:
    examples,_,_=dataset(split)
    for a in examples:
        s=a['sentence']
        s=re.sub(r'^(Today, |At noon, )','',s)
        s=re.sub(r' (yesterday|at noon)\.$','.',s)
        s=s[0].upper()+s[1:]
        patterns=[(r'([A-Z][a-z]+) was the person whom ([A-Z][a-z]+) (praised|thanked)\.',1),
                  (r'By ([A-Z][a-z]+), ([A-Z][a-z]+) was (praised|thanked)\.',-1),
                  (r'([A-Z][a-z]+) was (praised|thanked) by ([A-Z][a-z]+)\.',-1),
                  (r'([A-Z][a-z]+) (praised|thanked) ([A-Z][a-z]+)\.',1)]
        parsed=None
        for index,(pat,voice) in enumerate(patterns):
            m=re.fullmatch(pat,s)
            if m:
                g=m.groups()
                if index==0:agent,patient=g[1],g[0]
                elif index==1:agent,patient=g[0],g[1]
                elif index==2:agent,patient=g[2],g[0]
                else:agent,patient=g[0],g[2]
                parsed=(agent,patient,voice);break
        assert parsed is not None,(a['id'],s)
        assert parsed[:2]==(a['agent'],a['patient'])
        canonical=a['options'] if a['L']>0 else list(reversed(a['options']))
        first_a=a['sentence'].index(canonical[0])<a['sentence'].index(canonical[1])
        assert (a['features'][0]>0)==first_a
        assert a['features'][1]==parsed[2]
        assert a['R']==a['features'][0]*a['features'][1]*a['features'][2]
        assert a['answer']==(agent if a['Q']>0 else patient)
        assert a['options'][0 if a['Y']>0 else 1]==a['answer']
    rows.append({'split':split,'checked':len(examples),'all_pass':True})
(P/'CASE_VALIDATION.json').write_text(json.dumps({'checks':rows,'method':'Independent template regex parser; AI-authored code, not human adjudication or an independent natural-language annotator.'},indent=2)+'\n')
print(json.dumps(rows))
