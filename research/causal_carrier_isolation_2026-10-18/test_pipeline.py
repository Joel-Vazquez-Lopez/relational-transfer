"""Offline graph/scorer/persistence/selection checks, not LLM evidence."""
import tempfile
from pathlib import Path
import numpy as np
from core import score, mask, CUTS, MODES, calibration, select_prediction, qualifies, save

def check(rows):
    from run import persist
    checks={}
    calibration(); checks['known_routes_have_distinct_signatures']=True
    for r in rows:
        n=max(i for v in r['aliases'].values() for i in v)+3
        logits=np.full(n,-20.,dtype=np.float32)
        logits[r['aliases'][r['expected']][0]]=3
        decode=lambda ids: next((k for k,v in r['aliases'].items() if ids[0] in v),'unexpected')
        result=score(logits,r,decode)
        assert result['correct'] and result['gold_margin']>0
        logits[-1]=10
        assert not score(logits,r,decode)['correct'],'Unexpected vocab argmax must fail'
    checks['all_actual_rows_score_and_reject_unexpected_argmax']=True
    r=rows[0]; n=max(i for v in r['aliases'].values() for i in v)+1
    logits=np.full(n,-5.,dtype=np.float32); logits[r['aliases'][r['expected']][0]]=3
    with tempfile.TemporaryDirectory() as d:
        raw=[]; out=Path(d)
        entry=dict(call=1,id=r['id'],mode='baseline',cut=28,kind='scientific')
        result=persist(out,r,entry,logits,{'trace':np.zeros((28,2,3),dtype=np.float32)},lambda ids:'Yes',raw)
        assert result['expected']==r['expected'] and len(raw)==1
        assert np.array_equal(np.load(out/'CALL_0001_logits.npy'),logits)
        with np.load(out/'CALL_0001_states.npz') as z: assert z['trace'].shape==(28,2,3)
        try: persist(out,r,entry,logits,{'trace':np.zeros((28,2,3))},lambda ids:'Yes',raw)
        except AssertionError: pass
        else: raise AssertionError('Duplicate call not rejected')
    checks['production_persistence_round_trip_and_duplicate_rejection']=True
    base=dict(family='synthetic',cut=7,n=8,correct=dict(closed=0,frozen=8,live=8,neighbor=0,irrelevant=8),
              frozen_rescue=2.,refresh_benefit=0.,neutral_absolute_drift=.1)
    assert qualifies(base,'cached') and not qualifies(base,'refresh')
    assert select_prediction([base])['signature']=='cached'
    refresh={**base,'correct':{**base['correct'],'frozen':0},'frozen_rescue':0.,'refresh_benefit':2.}
    assert select_prediction([refresh])['signature']=='refresh'
    null={**base,'correct':{**base['correct'],'closed':8}}
    assert select_prediction([null]) is None
    assert select_prediction([{**base,'cut':14},base])['cut']==7
    checks['selection_priority_ties_and_null_gate']=True
    # A graph reachability certificate for cut0: no R-varying original token can
    # influence P or any later token under frozen/closed, including via intermediates.
    for r in rows:
        s,p=r['source_start'],r['period']
        for mode in ['closed','frozen']:
            dep=np.zeros(128,dtype=bool); dep[s:p]=True
            for b in range(28):
                dep=dep | (mask(r,mode,0,b) @ dep)
                assert not dep[p:].any()
    checks['all_rows_all_layers_no_information_path_cut0']=True
    return dict(kind='offline_synthetic_only_zero_model_calls',checks=checks,rows=len(rows))

if __name__=='__main__':
    import json
    from core import HERE
    print(json.dumps(check(json.loads((HERE/'INPUTS.json').read_text())),indent=2))
