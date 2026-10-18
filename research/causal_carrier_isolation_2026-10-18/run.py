"""One bounded attempt per pretrained model. Never retry a failed run."""
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path
import numpy as np
from core import HERE, ROOT, LENGTH, BLOCKS, CUTS, MODES, sha, save, score, summarize, select_prediction, qualifies

def digest(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def persist(out, row, entry, logits, arrays, decode, raw):
    # Persist payload first, before semantic scoring/dictionary assembly.
    assert not any(r['call']==entry['call'] for r in raw),'Duplicate recorded call'
    name=f"CALL_{entry['call']:04d}"
    np.save(out/(name+'_logits.npy'),logits)
    np.savez_compressed(out/(name+'_states.npz'),**arrays)
    result={**entry,**score(logits,row,decode),
        'family':row['family'],'split':row['split'],'role':row['role'],
        'voice':row['voice'],'query':row['query'],
        'payload':name,'logits_sha256':sha(out/(name+'_logits.npy')),
        'states_sha256':sha(out/(name+'_states.npz'))}
    raw.append(result); save(out/'RAW.json',raw)
    return result

class Engine:
    def __init__(self,model_name,lock):
        import mlx.core as mx
        from mlx_lm import load
        self.mx=mx
        self.model,self.tok=load(str(ROOT/lock['models'][model_name]['path']),tokenizer_config={'trust_remote_code':False})
        self.model.eval(); mx.set_cache_limit(256*1024**2)
    def compute(self,row,mode,cut,wrapped=True,native=False):
        from instrument import instrument
        mx=self.mx; ids=row['token_ids']; p=row['period']; e=row['prompt_end']
        x=mx.array([ids+[row['pad']]*(LENGTH-len(ids))])
        arrays={}; hashes={}
        if wrapped:
            with instrument(self.model,row,mode,cut,native=native) as records:
                y=self.model(x); logits=y[0,e].astype(mx.float32)
                selected=[records[b]['outgoing'][0,[p,e]].astype(mx.float32) for b in range(BLOCKS)]
                # Verify unmodified prefixes at every block. Store hashes of exact arrays.
                before=[records[b]['outgoing'][0,:p].astype(mx.float32) for b in range(BLOCKS)]
                null=[records[b]['outgoing'][0,p:e+1].astype(mx.float32) for b in range(BLOCKS)] if cut==0 and mode in ['closed','frozen'] else []
                mx.eval(logits,*selected,*before,*null)
                arrays['trace']=np.stack([np.asarray(a).copy() for a in selected])
                hashes['prefix']=[digest(np.asarray(a)) for a in before]
                hashes['prefix_without_neighbor']=[digest(np.asarray(a)[:-1]) for a in before]
                if null: arrays['null_states']=np.stack([np.asarray(a).copy() for a in null])
                result=np.asarray(logits).copy()
        else:
            y=self.model(x); logits=y[0,e].astype(mx.float32); mx.eval(logits)
            result=np.asarray(logits).copy(); arrays['trace']=np.empty((0,),dtype=np.float32)
        assert np.isfinite(result).all() and all(np.isfinite(a).all() for a in arrays.values())
        mx.clear_cache()
        return result,arrays,hashes

def execute(model_name):
    lock=json.loads((HERE/'LOCK.json').read_text())
    for n,h in lock['files'].items(): assert sha(HERE/n)==h,('Changed frozen source',n)
    for n,h in lock['native_sources'].items(): assert sha(ROOT/n)==h,('Changed native runtime',n)
    assert (HERE/'FREEZE_RECEIPT.json').exists(),'Remote protocol commit required'
    out=HERE/model_name; out.mkdir(exist_ok=True)
    assert not (out/'STARTED.json').exists(),'No automatic retry/resume'
    save(out/'STARTED.json',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),lock_sha256=sha(HERE/'LOCK.json')))
    engine=Engine(model_name,lock); rows=[r for r in json.loads((HERE/'INPUTS.json').read_text()) if r['model']==model_name]
    raw=[]; findings=[]; baselines={}; prediction=None

    def forward(row,mode='baseline',cut=28,kind='scientific',wrapped=True,native=False):
        budget=json.loads((HERE/'BUDGET.json').read_text()) if (HERE/'BUDGET.json').exists() else []
        assert len(budget)<lock['max_forwards'] and sum(r['model']==model_name for r in budget)<lock['max_per_model']
        entry=dict(call=len(budget)+1,model=model_name,id=row['id'],mode=mode,cut=cut,kind=kind,wrapped=wrapped,native=native)
        budget.append(entry); save(HERE/'BUDGET.json',budget)
        logits,arrays,hashes=engine.compute(row,mode,cut,wrapped,native)
        result=persist(out,row,entry,logits,arrays,engine.tok.decode,raw)
        if hashes:
            result.update(prefix_hashes=hashes)
            save(out/'RAW.json',raw)
        if kind=='scientific' and mode!='baseline':
            ref=baselines[row['id']]
            key='prefix_without_neighbor' if mode=='neighbor' else 'prefix'
            findings.append(dict(call=entry['call'],check='untouched_prefix',passed=hashes[key]==ref['hashes'][key]))
            err=float(np.max(np.abs(arrays['trace'][:cut]-ref['arrays']['trace'][:cut]))) if cut else 0.
            findings.append(dict(call=entry['call'],check='before_cut',error=err,passed=err<=1e-6))
            if mode=='live':
                err=float(np.max(np.abs(arrays['trace'][:,0]-ref['arrays']['trace'][:,0])))
                findings.append(dict(call=entry['call'],check='live_carrier_native',error=err,passed=err<=1e-6))
            if cut==28:
                err=float(np.max(np.abs(logits-ref['logits'])))
                findings.append(dict(call=entry['call'],check='sham_logits',error=err,passed=err<=.05))
        print('CASE',entry['call'],model_name,row['id'],mode,cut,result['actual'],result['correct'],round(result['gold_margin'],4),flush=True)
        return dict(logits=logits,arrays=arrays,hashes=hashes,result=result)

    def finish(status):
        save(out/'CONTROLS.json',findings)
        decision=dict(model=model_name,status=status,prediction=prediction,
                      attempted_forwards=sum(r['model']==model_name for r in json.loads((HERE/'BUDGET.json').read_text())),
                      recorded_forwards=len(raw),control_failures=[f for f in findings if not f['passed']])
        save(out/'DECISION.json',decision); print('DECISION',json.dumps(decision),flush=True); return decision

    def phase_controls(rr):
        lookup={(r['id'],r['mode'],r['cut']):r for r in raw if r['kind']=='scientific'}
        for family in sorted({r['family'] for r in rr}):
            for voice in ['active','passive']:
                for q in [0,1]:
                    pair=[r for r in rr if r['family']==family and r['voice']==voice and r['query']==q]
                    for mode in ['closed','frozen']:
                        a,b=[lookup[r['id'],mode,0] for r in pair]
                        la,lb=[np.load(out/(r['payload']+'_logits.npy')) for r in [a,b]]
                        with np.load(out/(a['payload']+'_states.npz')) as z: sa=z['null_states'].copy()
                        with np.load(out/(b['payload']+'_states.npz')) as z: sb=z['null_states'].copy()
                        le=float(np.max(np.abs(la-lb))); se=float(np.max(np.abs(sa-sb)))
                        findings.append(dict(family=family,voice=voice,query=q,mode=mode,check='cut0_disconnection',logit_error=le,state_error=se,passed=le<=.05 and se<=1e-6))
            for row in [r for r in rr if r['family']==family]:
                for k in CUTS:
                    r=lookup[row['id'],'irrelevant',k]
                    findings.append(dict(call=r['call'],check='irrelevant_correct',passed=r['correct']))
        save(out/'CONTROLS.json',findings)
        return all(f['passed'] for f in findings)

    for phase in ['development','evaluation']:
        rr=[r for r in rows if r['split']==phase]
        if phase=='development':
            unwrapped=forward(rr[0],kind='validation_unwrapped',wrapped=False)
            native=forward(rr[0],kind='validation_wrapped_native',native=True)
        for row in rr: baselines[row['id']]=forward(row)
        if phase=='development':
            repeat=forward(rr[0],kind='validation_repeat')
            ref=baselines[rr[0]['id']]
            for name,a,b in [('wrapper',unwrapped,native),('dense_mask',native,ref),('repeat',repeat,ref)]:
                error=float(np.max(np.abs(a['logits']-b['logits'])))
                findings.append(dict(check=name,error=error,passed=error<=.05))
            error=float(np.max(np.abs(repeat['arrays']['trace']-ref['arrays']['trace'])))
            findings.append(dict(check='repeat_states',error=error,passed=error<=1e-6))
        for family in sorted({r['family'] for r in rr}):
            for role in [0,1]:
                for voice in ['active','passive']:
                    pair=[r for r in rr if r['family']==family and r['role']==role and r['voice']==voice]
                    a,b=[baselines[r['id']]['arrays']['trace'][:,0] for r in pair]
                    err=float(np.max(np.abs(a-b)))
                    findings.append(dict(check='later_question_prefix_invariance',family=family,role=role,voice=voice,error=err,passed=err<=1e-6))
        competence={f:dict(correct=sum(baselines[r['id']]['result']['correct'] for r in rr if r['family']==f),total=8) for f in sorted({r['family'] for r in rr})}
        save(out/(phase.upper()+'_COMPETENCE.json'),competence)
        if not all(f['passed'] for f in findings): return finish('STOP_'+phase.upper()+'_NUMERICAL')
        if any(x['correct']!=8 for x in competence.values()): return finish('STOP_'+phase.upper()+'_COMPETENCE')
        for row in rr:
            for k in CUTS:
                for mode in MODES: forward(row,mode,k)
            if row['query']==0 and row['role']==0: forward(row,'closed',28)
        summary=summarize(rr,raw); save(out/(phase.upper()+'_SUMMARY.json'),summary)
        if not phase_controls(rr): return finish('STOP_'+phase.upper()+'_INTERVENTION_CONTROLS')
        if phase=='development':
            prediction=select_prediction(summary)
            save(out/'PREDICTION_BEFORE_EVALUATION.json',dict(prediction=prediction,
                raw_sha256=sha(out/'RAW.json'),summary_sha256=sha(out/'DEVELOPMENT_SUMMARY.json'),
                evaluation_outputs_observed=False))
            if prediction is None: return finish('STOP_NO_DISCRIMINATING_DEVELOPMENT_SIGNATURE')
        else:
            selected=[s for s in summary if s['cut']==prediction['cut']]
            passed=all(qualifies(s,prediction['signature']) for s in selected)
            return finish('PROSPECTIVE_SIGNATURE_SUPPORTED' if passed else 'PROSPECTIVE_SIGNATURE_FAILED')

def main():
    p=argparse.ArgumentParser(); p.add_argument('model',choices=['qwen7b','llama3b']); args=p.parse_args()
    try: execute(args.model)
    except Exception:
        save(HERE/args.model/'ERROR.json',dict(error=traceback.format_exc(),no_automatic_resume=True))
        raise

if __name__=='__main__': main()
