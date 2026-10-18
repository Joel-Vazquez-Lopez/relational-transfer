"""Complete bounded factorial collection. Bad labels are results, not an exclusion gate."""
import argparse
import hashlib
import json
import shutil
import time
import traceback
from pathlib import Path
import numpy as np
from core import HERE,ROOT,CUTS,jobs,score,save,sha,summarize,select_prediction,prediction_cells
from engine import Engine

def append(p,obj):
    with Path(p).open('a') as f:
        f.write(json.dumps(obj,allow_nan=False)+'\n');f.flush()

def lossless(a):
    with np.errstate(over='ignore',invalid='ignore'): b=a.astype(np.float16)
    return b if np.isfinite(b).all() and np.array_equal(b.astype(a.dtype),a) else a

def persist(out,row,entry,logits,arrays,hashes,decode,raw,threshold):
    name=f"CALL_{entry['call']:04d}.npz";path=out/name
    assert not path.exists(),'Never overwrite a payload'
    packed=dict(logits=lossless(logits),trace=lossless(arrays['trace']))
    null=arrays.get('null_states')
    if null is not None: hashes={**hashes,'null_states':hashlib.sha256(np.ascontiguousarray(null).tobytes()).hexdigest()}
    np.savez_compressed(path,**packed)
    scored=score(logits,row,decode);s=scored['class_logits']['Yes']-scored['class_logits']['No']
    calibrated='Yes' if s>threshold else 'No' if s<threshold else None
    result={**entry,**scored,'family':row['family'],'split':row['split'],'voice':row['voice'],
        'role':row['role'],'query':row['query'],'yes_minus_no':s,'calibrated_class':calibrated,
        'calibrated_correct':calibrated==row['expected'],'threshold':threshold,
        'payload':name,'payload_sha256':sha(path),'prefix_hashes':hashes,
        'stored_dtypes':{k:str(v.dtype) for k,v in packed.items()}}
    raw.append(result);append(out/'RAW.jsonl',result)
    return result

def execute(model_name,engine_factory=Engine,study_root=HERE,override_rows=None,validate_lock=True):
    root=Path(study_root);lock=json.loads((root/'LOCK.json').read_text())
    if validate_lock:
        for n,h in lock['files'].items():assert sha(root/n)==h,('Changed frozen source',n)
        for n,h in lock['native_sources'].items():assert sha(ROOT/n)==h,('Changed runtime',n)
        assert (root/'FREEZE_RECEIPT.json').exists()
    out=root/model_name;out.mkdir(exist_ok=True)
    assert not (out/'STARTED.json').exists(),'No automatic resume/retry'
    save(out/'STARTED.json',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),lock_sha256=sha(root/'LOCK.json')))
    engine=engine_factory(model_name,lock)
    rows=override_rows if override_rows is not None else [r for r in json.loads((root/'INPUTS.json').read_text()) if r['model']==model_name]
    threshold=json.loads((root/'READOUT_CALIBRATION.json').read_text())[model_name]['threshold']
    raw=[];baselines={};findings=[];prediction=None;budget=[json.loads(s) for s in (root/'BUDGET.jsonl').read_text().splitlines()] if (root/'BUDGET.jsonl').exists() else []
    def forward(row,mode='baseline',cut=28,kind='scientific',wrapped=True,native=False):
        assert len(budget)<lock['max_forwards'] and sum(r['model']==model_name for r in budget)<lock['max_per_model']
        assert shutil.disk_usage(root).free>1024**3,'Storage below1GiB reserve'
        entry=dict(call=len(budget)+1,model=model_name,id=row['id'],mode=mode,cut=cut,kind=kind,wrapped=wrapped,native=native,
                   utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        budget.append(entry);append(root/'BUDGET.jsonl',entry)
        logits,arrays,hashes=engine.compute(row,mode,cut,wrapped,native)
        result=persist(out,row,entry,logits,arrays,hashes,engine.tok.decode,raw,threshold)
        if kind=='scientific' and mode!='baseline':
            ref=baselines[row['id']];key='prefix_without_neighbor' if mode=='neighbor' else 'prefix'
            findings.append(dict(call=entry['call'],check='untouched_prefix',passed=hashes[key]==ref['hashes'][key]))
            err=float(np.max(np.abs(arrays['trace'][:cut+1]-ref['arrays']['trace'][:cut+1])))
            findings.append(dict(call=entry['call'],check='before_cut',error=err,passed=err<=1e-6))
            if mode=='live':
                err=float(np.max(np.abs(arrays['trace'][:,0]-ref['arrays']['trace'][:,0])))
                findings.append(dict(call=entry['call'],check='live_period',error=err,passed=err<=1e-6))
            if cut==28:
                err=float(np.max(np.abs(logits-ref['logits'])))
                findings.append(dict(call=entry['call'],check='sham',error=err,passed=err<=.05))
        if len(raw)%20==0 or mode=='baseline':print('CASE',entry['call'],model_name,row['id'],mode,cut,result['actual'],result['correct'],round(result['yes_minus_no'],4),flush=True)
        return dict(logits=logits,arrays=arrays,hashes=hashes,result=result)
    def finish(status):
        save(out/'CONTROLS.json',findings)
        decision=dict(model=model_name,status=status,prediction=prediction,recorded_forwards=len(raw),
            attempted_forwards=sum(r['model']==model_name for r in budget),
            technical_failures=[f for f in findings if not f['passed']])
        save(out/'DECISION.json',decision);print('DECISION',model_name,status,flush=True);return decision
    for phase in ['development','evaluation']:
        rr=[r for r in rows if r['split']==phase]
        if phase=='development':
            unwrapped=forward(rr[0],kind='validation_unwrapped',wrapped=False)
            native=forward(rr[0],kind='validation_wrapped_native',native=True)
        for row in rr:baselines[row['id']]=forward(row)
        if phase=='development':
            repeat=forward(rr[0],kind='validation_repeat');ref=baselines[rr[0]['id']]
            for name,a,b in [('wrapper',unwrapped,native),('dense',native,ref),('repeat',repeat,ref)]:
                e=float(np.max(np.abs(a['logits']-b['logits'])))
                findings.append(dict(check=name,error=e,passed=e<=.05))
            e=float(np.max(np.abs(repeat['arrays']['trace']-ref['arrays']['trace'])))
            findings.append(dict(check='repeat_trace',error=e,passed=e<=1e-6))
        for fam in sorted({r['family'] for r in rr}):
            for role in [0,1]:
                for voice in ['active','passive']:
                    pair=[r for r in rr if r['family']==fam and r['role']==role and r['voice']==voice]
                    a,b=[baselines[r['id']]['arrays']['trace'][:,0] for r in pair]
                    e=float(np.max(np.abs(a-b)));findings.append(dict(check='question_prefix',family=fam,role=role,voice=voice,error=e,passed=e<=1e-6))
        report=[]
        for fam in sorted({r['family'] for r in rr}):
            rs=[baselines[r['id']]['result'] for r in rr if r['family']==fam]
            report.append(dict(family=fam,n=8,native_correct=sum(r['correct'] for r in rs),
                               calibrated_correct=sum(r['calibrated_correct'] for r in rs)))
        save(out/(phase.upper()+'_NORMAL_ANSWERS.json'),report)
        if not all(f['passed'] for f in findings):return finish('STOP_'+phase.upper()+'_NUMERICAL')
        # Collect the complete frozen panel, regardless of raw/calibrated accuracy.
        for row in rr:
            for mode,k in jobs():forward(row,mode,k)
            if row['role']==0 and row['query']==0:forward(row,'closed',28)
            if any(not f['passed'] for f in findings):return finish('STOP_'+phase.upper()+'_TECHNICAL_INTERVENTION')
        lookup={(r['id'],r['mode'],r['cut']):r for r in raw if r['kind']=='scientific'}
        for fam in sorted({r['family'] for r in rr}):
            for voice in ['active','passive']:
                for q in [0,1]:
                    pair=[r for r in rr if r['family']==fam and r['voice']==voice and r['query']==q]
                    for mode in ['closed','frozen']:
                        a,b=[lookup[r['id'],mode,0] for r in pair]
                        la,lb=[np.load(out/r['payload'])['logits'].astype(np.float32) for r in [a,b]]
                        le=float(np.max(np.abs(la-lb)));same=a['prefix_hashes']['null_states']==b['prefix_hashes']['null_states']
                        findings.append(dict(check='cut0_disconnection',family=fam,voice=voice,query=q,mode=mode,
                                             logit_error=le,suffix_hash_equal=same,passed=le<=.05 and same))
        summary=summarize(rr,raw);save(out/(phase.upper()+'_SUMMARY.json'),summary)
        save(out/'CONTROLS.json',findings)
        if not all(f['passed'] for f in findings):return finish('STOP_'+phase.upper()+'_DISCONNECTION_CONTROL')
        if phase=='development':
            prediction=select_prediction(summary)
            save(out/'PREDICTION_BEFORE_EVALUATION.json',dict(prediction=prediction,raw_sha256=sha(out/'RAW.jsonl'),
                summary_sha256=sha(out/'DEVELOPMENT_SUMMARY.json'),evaluation_outputs_observed=False,
                no_candidate_action='Complete the same fixed evaluation panel; no post-hoc positive site selection'))
        else:
            evaluated=prediction_cells(summary,prediction);save(out/'PROSPECTIVE_EVALUATION.json',evaluated)
            return finish('PROSPECTIVE_SIGNATURE_SUPPORTED' if evaluated and all(r['passed'] for r in evaluated)
                          else 'PROSPECTIVE_SIGNATURE_FAILED' if prediction else 'COMPLETE_NO_DEVELOPMENT_CANDIDATE')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model',choices=['qwen7b','llama3b']);a=p.parse_args()
    try:execute(a.model)
    except Exception:
        save(HERE/a.model/'ERROR.json',dict(error=traceback.format_exc(),no_automatic_resume=True));raise
