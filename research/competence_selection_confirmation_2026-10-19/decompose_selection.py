"""Supplementary exact descriptive decomposition, not a confirmatory endpoint.
The frozen primary test remains unchanged. No extra p values or model calls.
"""
import json
import numpy as np
from core import HERE,save

def decomposition(all_units,selected,mode):
 def value(u):return u['interventions'][mode]['retention']
 full=[u for u in all_units if value(u) is not None];sub=[u for u in full if (u['family'],u['voice']) in selected]
 if not full or not sub:return dict(defined=False,n_full=len(full),n_selected=len(sub))
 composition=0.;within=0.;terms=[]
 for v in ['active','passive']:
  a=[u for u in full if u['voice']==v];b=[u for u in sub if u['voice']==v];wa=len(a)/len(full);wb=len(b)/len(sub);ma=float(np.mean([value(u) for u in a])) if a else None;mb=float(np.mean([value(u) for u in b])) if b else None
  # If an entire stratum is absent from the full support, both weights are zero.
  c=(wb-wa)*ma if ma is not None else 0.;w=wb*(mb-ma) if b else 0.
  composition+=c;within+=w;terms.append(dict(voice=v,n_full=len(a),n_selected=len(b),weight_full=wa,weight_selected=wb,mean_full=ma,mean_selected=mb,construction_composition=c,within_construction_selection=w))
 change=float(np.mean([value(u) for u in sub])-np.mean([value(u) for u in full]));assert abs(change-composition-within)<1e-10
 return dict(defined=True,n_full=len(full),n_selected=len(sub),change=change,construction_composition=composition,within_construction_selection=within,terms=terms)

def main():
 results=json.loads((HERE/'RESULTS.json').read_text());out=[]
 for mode in ['frozen','closed','irrelevant']:
  for gate in ['native_complete','calibrated_complete']:
   sets={m:{(u['family'],u['voice']) for u in results['units'] if u['model']==m and u[gate]} for m in ['qwen7b','llama3b']}
   for policy in ['own_selected','common_selected']:
    models={m:decomposition([u for u in results['units'] if u['model']==m],sets[m] if policy=='own_selected' else sets['qwen7b']&sets['llama3b'],mode) for m in sets}
    gap={k:models['llama3b'][k]-models['qwen7b'][k] for k in ['change','construction_composition','within_construction_selection']} if all(m['defined'] for m in models.values()) else None
    if gap:assert abs(gap['change']-gap['construction_composition']-gap['within_construction_selection'])<1e-10
    out.append(dict(mode=mode,gate=gate,policy=policy,models=models,gap_change_decomposition=gap))
 save(HERE/'SELECTION_DECOMPOSITION.json',dict(scope='Supplementary descriptive algebra; not a frozen confirmatory endpoint, no significance tests. Baseline full-sample stratum means define the reference decomposition; another reference would allocate the interaction differently. Terms describe composition of the report, not biological/model causal mediation. Within-construction term includes lexical-family selection and any remaining differences.',records=out))
 print(json.dumps(next(r for r in out if (r['mode'],r['gate'],r['policy'])==('frozen','native_complete','own_selected')),indent=2))
if __name__=='__main__':main()
