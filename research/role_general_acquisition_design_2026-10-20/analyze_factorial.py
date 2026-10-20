"""Frozen endpoints and decisions; model-level units, no fitting to hidden states."""
from pathlib import Path
import csv,itertools,json
import numpy as np
HERE=Path(__file__).resolve().parent
def dump(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def signflip(v):
    v=np.asarray(v,float);obs=v.mean()
    return sum(np.mean(v*np.asarray(s))>=obs-1e-12 for s in itertools.product([-1,1],repeat=len(v)))/(2**len(v))
def ci(v):
    v=np.asarray(v,float);rng=np.random.default_rng(7284991)
    samples=v[rng.integers(0,len(v),size=(10000,len(v)))].mean(1)
    return [float(x) for x in np.quantile(samples,[.05,.95])]
def main():
    gate=json.loads((HERE/'DEVELOPMENT_GATE.json').read_text()) if (HERE/'DEVELOPMENT_GATE.json').exists() else None
    paths=sorted((HERE/'factorial_runs').rglob('RESULT.json'))
    completed=[json.loads(p.read_text()) for p in paths]
    rows=[]
    for r in completed:
        m=r['metrics'][-1]
        for role in ['source','target']:
            h=m['held'][role]
            rows.append(dict(phase=r['phase'],seed=r['seed'],arm=r['arm'],role=role,paired_rate=h['paired_rate'],paired_correct=h['paired_correct'],pairs=h['pairs'],correct_answers=h['correct_answers'],answers=h['answers'],familiar_accuracy=m['familiar_accuracy'],required_accuracy=m['required_accuracy'][role],required_n=m['required_n'][role],object_permutation_agreement=m['object_permutation_agreement'],seconds=r['seconds']))
    with (HERE/'MODEL_ENDPOINTS.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['phase','seed','arm','role']);w.writeheader();w.writerows(rows)
    result=dict(status='INCOMPLETE_OR_STOPPED',runs_completed=len(completed),development_gate=gate,unit='paired initialization and identity relabeling, one fixed abstract graph',scope='Synthetic role slots; no LLM/native semantic-role mechanism or novelty claim.')
    confirm=[r for r in completed if r['phase']=='confirmation']
    if gate and gate['status']!='PASS_CONTINUE_CONFIRMATION':
        result['status']='STOP_DEVELOPMENT_GATE';result['confirmation_runs']=len(confirm)
    elif len(confirm)==48:
        seeds=sorted({r['seed'] for r in confirm});lookup={(r['seed'],r['arm']):r['metrics'][-1] for r in confirm}
        def vector(arm,role):return np.asarray([lookup[(s,arm)]['held'][role]['paired_rate'] for s in seeds])
        def avg(arm):return (vector(arm,'source')+vector(arm,'target'))/2
        effect=.5*((vector('source','source')-vector('source','target'))+(vector('target','target')-vector('target','source')))
        gains={a:{role:vector(a,role)-vector('neither',role) for role in ['source','target']} for a in ['source','target','both','source_double','target_double']}
        p=signflip(effect);interval=ci(effect)
        valid=all(m['familiar_accuracy']>=.95 and m['identity_max_error']<=1e-5 and all(v is None or v>=.95 for v in m['required_accuracy'].values()) for m in lookup.values())
        role_selective=effect.mean()>=.15 and p<=.05 and gains['source']['source'].mean()>=.15 and gains['target']['target'].mean()>=.15
        broad=all(gains[a][r].mean()>=.15 for a in ['source','target'] for r in ['source','target']) and interval[0]>=-.05 and interval[1]<=.05
        dose={a:avg('both')-avg(a) for a in ['source_double','target_double']};rawp={a:signflip(v) for a,v in dose.items()}
        ordered=sorted(rawp,key=rawp.get);adjusted={ordered[0]:min(1,2*rawp[ordered[0]]),ordered[1]:min(1,max(2*rawp[ordered[0]],rawp[ordered[1]]))}
        single_intervals={a:{r:ci(gains[a][r]) for r in ['source','target']} for a in ['source','target','source_double','target_double']}
        singles_equiv=all(lo>=-.05 and hi<=.05 for d in single_intervals.values() for lo,hi in d.values())
        joint=singles_equiv and all(gains['both'][r].mean()>=.15 for r in ['source','target']) and all(dose[a].mean()>=.15 and adjusted[a]<=.05 for a in dose)
        flags=dict(role_selective=bool(role_selective),broad_identity_compatible=bool(broad),both_demand_compatible=bool(joint))
        chosen=[k for k,v in flags.items() if v]
        status='INCONCLUSIVE_TASK_LEARNING' if not valid else (chosen[0].upper() if len(chosen)==1 else ('MIXED_PATTERNS' if chosen else 'NO_SUPPORT_FOR_FROZEN_PROTOTYPES'))
        result.update(status=status,confirmation_runs=48,training_validity=valid,prototype_flags=flags,role_selectivity=dict(per_seed=effect.tolist(),mean=float(effect.mean()),one_sided_signflip_p=p,bootstrap90=interval),means={a:{r:float(vector(a,r).mean()) for r in ['source','target']} for a in ['neither','source','target','both','source_double','target_double']},gains={a:{r:v.tolist() for r,v in dd.items()} for a,dd in gains.items()},single_gain_bootstrap90=single_intervals,dose_comparison={a:dict(per_seed=dose[a].tolist(),mean=float(dose[a].mean()),raw_p=rawp[a],holm_p=adjusted[a]) for a in dose},statistical_limitations=['Eight model-seed/identity-map units, not independent graph topologies.','Sign-flip p-values assume signed exchangeability under the null.','Paired scenes/object slots/rotations are repeated measurements, not independent sample units.','Development selection is separated from the eight confirmation seeds.','Restricted prediction patterns do not identify unique neural mechanisms.'])
    dump(HERE/'FACTORIAL_RESULTS.json',result)
    lines=['# Role-demand factorial: outcome','','**Status: '+result['status']+'**','','No prompts, hypotheses, sites or training settings were changed after the frozen model run began.','','## Completed models','', '| Phase | Seed | Arm | Source paired | Target paired | Familiar |','|---|---:|---|---:|---:|---:|']
    for r in completed:
        m=r['metrics'][-1];lines.append(f"| {r['phase']} | {r['seed']} | {r['arm']} | {m['held']['source']['paired_rate']:.3f} | {m['held']['target']['paired_rate']:.3f} | {m['familiar_accuracy']:.3f} |")
    if gate:
        lines+=['','## Development gate','', '```json',json.dumps(gate['checks'],indent=2),'```']
    lines+=['','## Interpretation boundary','','This is one synthetic ordered-argument task. A failed prerequisite is not evidence that role-general transfer is impossible. A positive prototype is not a unique mechanism, a linguistic invariance claim, or evidence of novelty. All stopped/failed arms are retained. See FACTORIAL_RESULTS.json and MODEL_ENDPOINTS.csv for full results.']
    (HERE/'FACTORIAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in ['status','runs_completed','confirmation_runs','means','role_selectivity','prototype_flags']},indent=2))
if __name__=='__main__':main()
