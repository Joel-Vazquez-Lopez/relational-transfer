"""Descriptive diagnostics for denominator, inverse-contrast, lexical-risk and outlier scope.
No added hypothesis tests, refits, thresholds, layer searches or model calls.
"""
import json
import numpy as np
from core import HERE,save
r=json.loads((HERE/'RESULTS.json').read_text());units=r['units'];models=['qwen7b','llama3b'];lookup={(u['model'],u['family'],u['voice']):u for u in units};pairs=sorted({(u['family'],u['voice']) for u in units})
def gap(selected,gate='all'):
 vals={}
 for m in models:
  us=[u for u in units if u['model']==m and (u['family'],u['voice']) in selected and (gate=='all' or u[gate])];us=[u for u in us if u['interventions']['frozen']['retention'] is not None]
  vals[m]=float(np.mean([u['interventions']['frozen']['retention'] for u in us])) if us else None
 return vals[models[1]]-vals[models[0]] if all(vals[m] is not None for m in models) else None
joint={x for x in pairs if all(lookup[m,*x]['interventions']['frozen']['retention'] is not None for m in models)}
positive={x for x in joint if all(lookup[m,*x]['D_native']>=.5 and lookup[m,*x]['all_ordinal_positive'] for m in models)}
matched=[]
for name,support in [('joint_valid_denominators',joint),('joint_positive_and_ordinal',positive)]:
 selected={x for x in support if all(lookup[m,*x]['native_complete'] for m in models)};a=gap(support);b=gap(selected);matched.append(dict(support=name,n_full=len(support),n_native_common=len(selected),full_gap=a,native_common_gap=b,shift=b-a if a is not None and b is not None else None,full_units=[list(x) for x in sorted(support)],selected_units=[list(x) for x in sorted(selected)]))
loo=[]
for fam in sorted({x[0] for x in pairs}):
 support={x for x in pairs if x[0]!=fam};a=gap(support);b=gap(support,'native_complete');loo.append(dict(omitted_family=fam,full_gap=a,native_gate_gap=b,shift=b-a if a is not None and b is not None else None))
annotation=[]
for excluded in [['c15'],['c09','c15']]:
 support={x for x in pairs if x[0] not in excluded};a=gap(support);b=gap(support,'native_complete');annotation.append(dict(excluded=excluded,full_gap=a,native_gap=b,shift=b-a if a is not None and b is not None else None))
summary=[]
for mode in ['frozen','closed','irrelevant']:
 for gate in ['all','native_complete','calibrated_complete']:
  p=next(p for p in r['profiles'] if (p['mode'],p['gate'],p['policy'])==(mode,gate,'own_selected'));q,l=[p['models'][m] for m in models];summary.append(dict(mode=mode,gate=gate,mean_ratio_gap=p['gap'],difference_of_model_medians=l['median_retention']-q['median_retention'] if all(x['median_retention'] is not None for x in [q,l]) else None,raw_D_gap=p['raw_D_gap'],n_ratio={m:p['models'][m]['n_ratio'] for m in models}))
rescue=[]
for m in models:
 for v in ['active','passive']:
  us=[u for u in units if (u['model'],u['voice'])==(m,v)];rescue.append(dict(model=m,voice=v,rescued=sum(not u['native_complete'] and u['calibrated_complete'] for u in us),lost=sum(u['native_complete'] and not u['calibrated_complete'] for u in us),unchanged=sum(u['native_complete']==u['calibrated_complete'] for u in us)))
out=dict(scope='Supplementary descriptive diagnostics after frozen results. All original results retained. No extra p values, changed endpoint or new model inference.',matched_support=matched,leave_one_family_out=loo,leave_one_family_out_shift_range=[min(x['shift'] for x in loo),max(x['shift'] for x in loo)],annotation_sensitivity=annotation,all_control_metrics=summary,rescue_and_loss=rescue)
save(HERE/'DESCRIPTIVE_ROBUSTNESS.json',out);print(json.dumps(out,indent=2))
