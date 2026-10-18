"""Post hoc full-profile selection audit. No fitted model or new forwards."""
from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent;ROOT=P.parent.parent;source=ROOT/'research/graded_relational_routing_2026-10-18';data=json.loads((P/'RESULTS.json').read_text());groups=[g for g in data['groups'] if g['family'] in ['g03','g04','g05','g06']];records=[]
sets={}
for model in ['qwen7b','llama3b']:
 for gate in ['all','native_complete','calibrated_complete']:
  sets[model,gate]={(g['family'],g['voice']) for g in groups if g['model']==model and (gate=='all' or g[gate])}
common={gate:sets['qwen7b',gate]&sets['llama3b',gate] for gate in ['all','native_complete','calibrated_complete']}
for model in ['qwen7b','llama3b']:
 raw=json.loads((source/model/'EVALUATION_SUMMARY.json').read_text())
 for gate in ['all','native_complete','calibrated_complete']:
  for policy,selection in [('own_selected',sets[model,gate]),('common_selected',common[gate])]:
   for mode,cut in sorted({(r['mode'],r['cut']) for r in raw}):
    rr=[r for r in raw if r['mode']==mode and r['cut']==cut and (r['family'],r['voice']) in selection]
    assert len(rr)==len(selection) and all(r['ratio'] is not None for r in rr)
    records.append(dict(model=model,gate=gate,policy=policy,mode=mode,cut=cut,n_units=len(rr),units=[list(x) for x in sorted(selection)],mean_retention=sum(r['ratio'] for r in rr)/len(rr) if rr else None,mean_raw_D=sum(r['intervention']['D'] for r in rr)/len(rr) if rr else None))
selected=[r for r in records if r['mode']=='frozen' and r['cut']==14 and r['policy']=='own_selected']
(P/'FILTER_PROFILE_RESULTS.json').write_text(json.dumps(dict(scope='Post hoc conditional profile audit; not a preregistered endpoint, no significance tests. Every fixed source-access profile is retained. Means under different gates estimate different selected samples.',sources=[dict(path=str((source/m/'EVALUATION_SUMMARY.json').relative_to(ROOT)),sha256=hashlib.sha256((source/m/'EVALUATION_SUMMARY.json').read_bytes()).hexdigest()) for m in ['qwen7b','llama3b']],selection_source_sha256=hashlib.sha256((P/'RESULTS.json').read_bytes()).hexdigest(),records=records,illustration_fixed_cut14=selected),indent=2)+'\n')
print(json.dumps(selected,indent=2))
