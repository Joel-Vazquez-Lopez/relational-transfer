"""Descriptive pair-cluster uncertainty for archived margin contrasts only."""
from pathlib import Path
import json,random,statistics
R=Path(__file__).resolve().parent;d=json.loads((R/'MARGIN_DECOMPOSITIONS.json').read_text())
def cluster(values):
 rng=random.Random(20261015);keys=sorted(values);b=[]
 for _ in range(10000):
  chosen=[x for k in rng.choices(keys,k=len(keys)) for x in values[k]];b.append(statistics.mean(chosen))
 b.sort();return {'mean':statistics.mean([x for vv in values.values() for x in vv]),'cluster_bootstrap95':[b[250],b[9750]],'clusters':len(keys),'scope':'post hoc descriptive name-pair bootstrap, six groups; all events constructed and reused'}
out={}
for name in ['untouched_qwen15','untouched_qwen05']:
 out[name]={}
 for pair,v in d[name].items():
  a={x['id']:x for x in v['active']};c={x['id']:x for x in v['object_cleft']};vals={};cont={}
  for eid,x in a.items():
   g=x['pair_index'];vals.setdefault(g,[]).append(100*(int(c[eid]['positive_contrast'])-int(x['positive_contrast'])));cont.setdefault(g,[]).append(2*(c[eid]['role_contrast']-x['role_contrast']))
  out[name][pair]={'object_minus_active_positive_contrast_pp':cluster(vals),'object_minus_active_query_difference_logit':cluster(cont)}
(R/'UNCERTAINTY.json').write_text(json.dumps(out,indent=2)+'\n');print('Saved descriptive uncertainty; no model inference.')
