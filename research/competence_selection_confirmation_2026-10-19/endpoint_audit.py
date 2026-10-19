"""Supplementary semantic/endpoint transparency. No extra p values or forwards."""
import json
from core import HERE,save
raw=[json.loads(s) for m in ['qwen7b','llama3b'] for s in (HERE/m/'RAW.jsonl').read_text().splitlines()]
base=[r for r in raw if r['kind']=='scientific' and r['mode']=='baseline'];assert len(base)==384
conflicts=[dict(call=r['call'],model=r['model'],id=r['id'],actual=r['actual'],expected=r['expected'],frozen_correct=r['correct'],argmax_only_correct=r['actual']==r['expected'],gold_margin=r['gold_margin']) for r in base if r['correct']!=(r['actual']==r['expected'])]
counts=[]
for m in ['qwen7b','llama3b']:
 for v in ['active','passive']:
  for target in ['Yes','No']:
   rr=[r for r in base if (r['model'],r['voice'],r['expected'])==(m,v,target)]
   counts.append(dict(model=m,voice=v,target=target,n=len(rr),native_correct=sum(r['correct'] for r in rr),calibrated_correct=sum(r['calibrated_correct'] for r in rr),interpretation='Explicitly asserted directional proposition' if target=='Yes' else 'Converse not entailed; assigned No by directed-event convention, not open-world disproof'))
scopes=[]
for excluded in [[],['c15'],['c09','c15']]:
 rr=[r for r in base if r['family'] not in excluded];models={}
 for m in ['qwen7b','llama3b']:
  models[m]={}
  for v in ['active','passive']:
   families=sorted({r['family'] for r in rr if r['model']==m});units=[[r for r in rr if (r['model'],r['voice'],r['family'])==(m,v,f)] for f in families]
   models[m][v]=dict(n_units=len(units),native_complete=sum(all(r['correct'] for r in u) for u in units),calibrated_complete=sum(all(r['calibrated_correct'] for r in u) for u in units))
 scopes.append(dict(excluded_annotation_risk_families=excluded,models=models))
save(HERE/'ENDPOINT_AUDIT.json',dict(scope='Supplementary descriptive transparency only; not a replacement test or outcome-based exclusion. Semantic-risk flags documented during collection before fresh answer inspection. The primary24-family analysis remains intact.',native_definition_note='Inherited scorer additionally requires positive aggregate alias margin. This audit checks whether that changes the native endpoint relative to the frozen prose definition (vocabulary-wide argmax). Any discrepancy must be disclosed, not silently repaired.',native_definition_conflicts=conflicts,primary_native_definition_equivalent_on_observed_baselines=not conflicts,yes_no_partition=counts,annotation_risk_descriptive_sensitivity=scopes,no_new_p_values=True))
print(json.dumps(dict(native_definition_conflicts=conflicts,counts=counts),indent=2))
