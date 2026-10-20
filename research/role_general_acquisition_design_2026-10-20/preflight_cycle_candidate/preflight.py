"""Exhaustive design checks before any model is initialized."""
from pathlib import Path
import hashlib,json,shutil,time
from itertools import permutations
import numpy as np
import factorial_data as data

HERE=Path(__file__).resolve().parent
def dump(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def main():
    start=time.time();assert shutil.disk_usage(HERE).free>=5*1024**3
    symbolic=[data.symbolic_audit(a) for a in data.ARMS]
    for k in ['positive_counts','comparison_spectrum','families']:
        assert all(s[k]==symbolic[0][k] for s in symbolic),k
    for r in symbolic:
        expected_s={'neither':0,'source':6,'target':0,'both':6,'source_double':12,'target_double':0}[r['arm']]
        expected_t={'neither':0,'source':0,'target':6,'both':6,'source_double':0,'target_double':12}[r['arm']]
        assert r['focal_comparisons']['source']==r['focal_comparisons']['source_event']==expected_s
        assert r['focal_comparisons']['target']==r['focal_comparisons']['target_event']==expected_t
    # Every renaming preserves legality, disjoint held relations and spectra.
    for p in permutations(range(6)):
        inverse={v:i for i,v in enumerate(p)}
        observed={tuple([p[s],p[t],e]) for s,t,e in data.meanings()}
        held={(p[s],p[t],e) for s in data.H for t in data.H if s!=t for e in (0,1)}
        assert len(observed)==48 and len(held)==12 and not observed&held
        assert all(inverse[a]!=inverse[b] for a,b,e in observed)
    maps=[(0,1,2,3,4,5),(3,5,1,4,0,2),(5,0,4,1,2,3)]
    actual=[]
    for imap in maps:
        reference=None
        for arm in data.ARMS:
            r=data.full_input_audit(arm,imap,7283001)
            if reference is None:reference=r
            for k in ['positive_signature','positive_scene_signature','candidate_signature']:
                assert r[k]==reference[k],(arm,k)
            for role in ('source','target'):
                count={'neither':0,'source':3 if role=='source' else 0,'target':3 if role=='target' else 0,'both':3,'source_double':6 if role=='source' else 0,'target_double':6 if role=='target' else 0}[arm]
                expected=1-.5*count/84
                assert abs(r['ceilings'][role+'_query']-expected)<1e-12,(arm,role,r)
                assert abs(r['ceilings'][role+'_complete']-expected)<1e-12,(arm,role,r)
            assert abs(r['ceilings']['unordered_query']-6/7)<1e-12
            assert r['ceilings']['no_event_query']==.5
            actual.append(r)
        print('FLOAT32_INPUT_GATES_PASS',imap,flush=True)
    # Materialize the exact compact training tables, verify paired batching and
    # all scene/distractor/slot marginals. These tables are preflight materials.
    positive_rows=None;positive_scene_rows=None;base=None
    row_counts={};array_bytes={}
    for arm in data.ARMS:
        ds=data.build_training(arm,maps[0],7283001)
        key=ds['rows'][:,:2]
        positive_scene=ds['rows'][np.arange(len(key)),2+ds['rows'][:,1]]
        if positive_rows is None:positive_rows=key;positive_scene_rows=positive_scene;base=ds['base']
        else:
            assert np.array_equal(key,positive_rows)
            assert np.array_equal(positive_scene,positive_scene_rows)
            assert np.array_equal(ds['base'],base)
        assert len(ds['rows'])==96768
        assert np.ptp(np.bincount(ds['rows'][:,1],minlength=4))==0
        row_counts[arm]=len(ds['rows']);array_bytes[arm]=sum(x.nbytes for x in ds.values())
    result=dict(status='PASS_PRETRAINING_DESIGN',symbolic=symbolic,full_float32_input_checks=actual,identity_renamings_enumerated=720,rendered_identity_maps=len(maps),training_rows_per_arm=row_counts,compact_array_bytes_per_arm=array_bytes,positive_labels_and_queries_coupled_per_index=True,exact_positive_scene_and_candidate_marginals_matched=True,candidate_only_ceiling=.25,query_only_ceiling=.25,unordered_query_ceiling=6/7,no_event_query_ceiling=.5,notes=['The ceiling is for the specified complete encoded finite inputs and role-conditioned coarsening, not a theorem about every possible representation.','Within-block identical-input/incompatible-label witnesses hold for every nuisance rendering because each candidate tensor is reused for all four target queries.','The role slots are fixed; natural-language semantic role invariance is not tested.','All identity maps are relabelings of one abstract graph, not nonisomorphic graph replications.','Source/target-only off-factor comparisons still provide indirect information; absence of an explicit focal comparison does not imply absence of all role information.'],seconds=time.time()-start,model_initializations=0,model_forwards=0,training_runs=0)
    dump(HERE/'PREFLIGHT.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['symbolic','full_float32_input_checks']},indent=2))
if __name__=='__main__':
    try:main()
    except Exception as exc:
        dump(HERE/'PREFLIGHT_FAILURE.json',dict(status='STOP_PRETRAINING',error=repr(exc),model_initializations=0))
        raise
