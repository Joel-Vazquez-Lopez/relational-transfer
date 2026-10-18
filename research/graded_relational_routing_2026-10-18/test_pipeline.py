"""Synthetic full-controller rehearsal and real-metadata tests. No MLX imports."""
import contextlib
import io
import json
import tempfile
from pathlib import Path
import numpy as np
from core import mask,score,sensitivity,select_prediction,save

class FakeEngine:
    def __init__(self,model,lock):
        self.tok=type('Tok',(),{'decode':staticmethod(lambda ids:'Yes' if ids[0] in [1,2] else 'No' if ids[0] in [3,4] else 'unexpected')})()
    def compute(self,row,mode,cut,wrapped=True,native=False):
        truth=(-1)**(row['role']+row['query'])
        keep=1 if mode in ['baseline','irrelevant','live','neighbor'] or cut==28 or (mode=='frozen' and cut>0) else 0
        logits=np.full(16,-50.,dtype=np.float32);logits[3]=0;logits[4]=-20
        logits[1]=-8+4*truth*keep;logits[2]=logits[1]-20
        trace=np.zeros((29,2,2),dtype=np.float32);trace[1:,1]=truth
        if keep==0:trace[cut+1:,1]=0
        arrays={'trace':trace if wrapped else np.empty((0,),dtype=np.float32)}
        hashes={'prefix':['unchanged']*28,'prefix_without_neighbor':['unchanged']*28} if wrapped else {}
        if cut==0 and mode in ['closed','frozen']:arrays['null_states']=np.zeros((28,2,2),dtype=np.float32)
        return logits,arrays,hashes

def check(rows,calibrators):
    from run import execute,lossless,persist
    findings={}
    for r in rows:
        size=max(i for v in r['aliases'].values() for i in v)+2
        logits=np.full(size,-20.,dtype=np.float32);logits[r['aliases'][r['expected']][0]]=5
        assert score(logits,r,lambda ids:'test')['correct']
        logits[-1]=10;assert not score(logits,r,lambda ids:'unexpected')['correct']
        s,p=r['source_start'],r['period']
        for mode in ['closed','frozen']:
            dep=np.zeros(128,dtype=bool);dep[s:p]=True
            for b in range(28):
                dep=dep | (mask(r,mode,0,b) @ dep)
                assert not dep[p:].any()
    findings['all96_actual_rows_scoring_unexpected_argmax_and_disconnection_certificate']=True
    for a in [np.array([.5,-3,100],dtype=np.float32),np.array([.1234567],dtype=np.float32)]:
        assert np.array_equal(lossless(a).astype(a.dtype),a)
    findings['lossless_storage_and_float32_fallback']=True
    # Constant answer bias changes arbitrary raw choices but cancels D exactly.
    rr=[]
    for r in [0,1]:
        for q in [0,1]:
            rr.append(dict(role=r,query=q,class_logits={'Yes':-20+3*((-1)**(r+q)),'No':0},correct=False))
    assert sensitivity(rr)['D']==3 and sensitivity(rr)['all_ordinal_pairs_positive']
    findings['constant_bias_cancellation_without_correct_label_gate']=True
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);save(root/'LOCK.json',dict(max_forwards=1800,max_per_model=900))
        save(root/'READOUT_CALIBRATION.json',calibrators)
        results={}
        # One development and one evaluation family per synthetic model covers
        # every production collection, selection, persistence and control branch.
        for model in ['qwen7b','llama3b']:
            originals=[r for r in rows if r['model']==model and r['family'] in ['g01','g03']]
            small=[{**r,'aliases':{'Yes':[1,2],'No':[3,4]}} for r in originals]
            with contextlib.redirect_stdout(io.StringIO()):
                result=execute(model,FakeEngine,root,small,False)
            assert result['status']=='PROSPECTIVE_SIGNATURE_SUPPORTED',result
            assert result['recorded_forwards']==295
            # Fake native choices deliberately fail; the causal controller must
            # still complete, instead of repeating the closed pilot's gate.
            normal=json.loads((root/model/'EVALUATION_NORMAL_ANSWERS.json').read_text())
            assert normal[0]['native_correct']==4
            raw=[json.loads(s) for s in (root/model/'RAW.jsonl').read_text().splitlines()]
            assert len(raw)==295 and len({r['call'] for r in raw})==295
            for result in raw:
                with np.load(root/model/result['payload']) as z:
                    assert np.isfinite(z['logits']).all() and np.isfinite(z['trace']).all()
            results[model]=dict(synthetic_calls=295,status='full_controller_pass',bad_native_accuracy_did_not_exclude_cases=True)
        assert len((root/'BUDGET.jsonl').read_text().splitlines())==590
    findings['complete_production_controller_rehearsed_both_models']=results
    return dict(kind='entirely_synthetic_rehearsal_no_LLM_calls',findings=findings)
