"""Offline audit of pinned public inventories; no model download/inference."""
from pathlib import Path
import json,csv
P=Path(__file__).resolve().parent
ALGORITHMS=[[59,112],[59,48,124],[59,74,99,124],[59,125],[108,99,124],[63,108]]
repo=json.loads((P/'interpbench_pinned_metadata.json').read_text())
assert repo['status']==200 and not repo['link']
files={a['path'] for a in repo['data'] if a['type']=='file'}
rows=[]
for i,chain in enumerate(ALGORITHMS,1):
 present=[v for v in chain if f'{v}/ll_model.pth' in files and f'{v}/ll_model_cfg.pkl' in files]
 rows.append({'algorithm':i,'components':chain,'available_component_ids':present,'missing_component_ids':[v for v in chain if v not in present],'complete':len(present)==len(chain)})
required=sorted(set(v for c in ALGORITHMS for v in c));present=[v for v in required if f'{v}/ll_model.pth' in files]
result={'status':'Published trained-bank comparison NOT RUN: prerequisite not met in inspected public artifacts',
 'repository':repo['repo'],'revision':repo['revision'],'manifest_items':len(repo['data']),'pagination_complete':True,
 'required_component_ids':required,'available_component_ids':present,'complete_chains':sum(r['complete'] for r in rows),'algorithms':rows,
 'scope':'Original public InterpBench bank is not the authors modified trained-variant/correspondence bank. Presence alone would not prove compatibility. Checked public releases/searches do not prove global nonavailability.',
 'reference_fallback':'Separate compiled-reference protocol, declared restricted implementation sampler and unchanged practical released functions; not a trained-bank reproduction.'}
(P/'PUBLIC_ASSET_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
