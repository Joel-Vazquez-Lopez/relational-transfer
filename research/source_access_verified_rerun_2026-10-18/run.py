"""One attempt per model. All calls journaled before execution; no automatic resume."""
import argparse
import json
from pathlib import Path
import time
import traceback
import numpy as np
from core import BLOCKS,LENGTH,CONDITIONS,sha,save,matched_summary,select_prediction
from scoring import full_score

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent.parent
BASE=dict(name='baseline',cut=28,source='event',restore=[])

def assert_lock():
    lock=json.loads((HERE/'LOCK.json').read_text())
    for n,h in lock['files'].items(): assert sha(HERE/n)==h,('changed frozen file',n)
    for p,h in lock['native_sources'].items(): assert sha(ROOT/p)==h,('changed runtime',p)
    assert (HERE/'FREEZE_RECEIPT.json').exists(), 'Commit frozen sources before inference'
    return lock

def execute(model_name,engine_factory=None):
    lock=assert_lock(); out=HERE/model_name; out.mkdir(exist_ok=True)
    assert not (out/'STARTED.json').exists(), 'One attempt only; no resume'
    prediction_from_qwen=None
    if model_name=='llama3b':
        prior=json.loads((HERE/'qwen7b/DECISION.json').read_text())
        assert prior['evaluation_supported'], 'No cross-model execution after failed Qwen prediction'
        prediction_from_qwen=prior['prediction']
    save(out/'STARTED.json',dict(lock_sha256=sha(HERE/'LOCK.json'),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    if engine_factory is None:
        from engine import Engine
        engine_factory=Engine
    engine=engine_factory(model_name,lock)
    rows=[r for r in json.loads((HERE/'INPUTS.json').read_text()) if r['model']==model_name]
    raw=[]; baseline_arrays={}; validations={}; technical_errors=[]

    forward_records=[]
    def dispatch(row,condition,prefix=(),wrapped=True,native=False,capture=True):
        budget=json.loads((HERE/'BUDGET.json').read_text()) if (HERE/'BUDGET.json').exists() else []
        assert len(budget)<lock['max_forwards']
        assert sum(x['model']==model_name for x in budget)<lock['max_per_model']
        call=len(budget)+1
        entry=dict(call=call,model=model_name,id=row['id'],condition=condition['name'],
            prefix_token_ids=list(prefix),wrapped=wrapped,native=native,capture=capture)
        budget.append(entry);save(HERE/'BUDGET.json',budget)
        logits,arrays,original=engine.compute(row,condition,prefix,wrapped,native,capture)
        # Save every forward payload before scoring or constructing semantic rows.
        np.save(out/f'CALL_{call:04d}_logits.npy',logits)
        if arrays:
            retained={'last':arrays['last']}
            if condition['name']=='baseline':retained['prefix']=arrays['prefix']
            if 'suffix' in arrays:retained['suffix']=arrays['suffix']
            np.savez_compressed(out/f'CALL_{call:04d}_states.npz',**retained)
        entry['argmax_id']=int(np.argmax(logits));entry['argmax_token']=engine.decode([entry['argmax_id']])
        forward_records.append(entry);save(out/'FORWARD_RECORDS.json',forward_records)
        return logits,arrays,original,entry

    def forward(row,condition,wrapped=True,native=False):
        logits,arrays,_,entry=dispatch(row,condition,wrapped=wrapped,native=native,capture=wrapped)
        cache={():logits}; child_calls=[]
        def get_logits(prefix):
            prefix=tuple(prefix)
            if prefix not in cache:
                value,_,original,child=dispatch(row,condition,prefix,wrapped,native,capture=False)
                e=float(np.max(np.abs(original-logits)))
                child_calls.append(dict(call=child['call'],prefix=list(prefix),original_prompt_logit_error=e))
                if e>.05:technical_errors.append(dict(call=child['call'],kind='continuation_causal_prefix',value=e))
                cache[prefix]=value
            return cache[prefix]
        scientific=condition['name'] not in ['native_validation','wrapped_native_validation','dense_repeat_validation']
        if scientific:
            scored=full_score(get_logits,row,engine.decode)
        else:
            best=int(np.argmax(logits))
            scored=dict(expected=row['expected'],argmax_id=best,argmax_token=engine.decode([best]),
                actual=None,correct=None,margin=None,assessment='technical_only_not_answer_scored')
        result={**entry,**scored,'split':row['split'],'family':row['family'],'query':row['query'],
            'orientation':row['orientation'],'voice':row['voice'],'child_calls':child_calls,
            'source_span':row['source_span'],'irrelevant_span':row['irrelevant_span'],
            'suffix_start':row['suffix_start'],'prompt_end':row['prompt_end']}
        if wrapped:
            states=arrays['last'].astype(np.float64)
            err=states[:,3]-states[:,0]-states[:,1]-states[:,2]
            recon=float(np.max(np.linalg.norm(err,axis=-1)/np.maximum(np.linalg.norm(states[:,3],axis=-1),1e-12)))
            result['reconstruction_relative_error']=recon
            if recon>.002:technical_errors.append(dict(call=entry['call'],kind='reconstruction',value=recon))
            if row['id'] in baseline_arrays and condition['name']!='baseline':
                ref=baseline_arrays[row['id']]
                error=float(np.max(np.linalg.norm((arrays['prefix']-ref['prefix']).astype(np.float64),axis=-1))/max(np.max(np.linalg.norm(ref['prefix'].astype(np.float64),axis=-1)),1e-12))
                result['prefix_relative_error']=error
                if error>1e-6:technical_errors.append(dict(call=entry['call'],kind='prefix_invariance',value=error))
                k=condition['cut']
                if 0<k<28:
                    e=float(np.max(np.abs(arrays['last'][:k]-ref['last'][:k])))
                    result['before_cut_max_absolute_error']=e
                    if e>1e-6:technical_errors.append(dict(call=entry['call'],kind='before_cut_invariance',value=e))
        assert not any(v['call']==entry['call'] for v in raw)
        raw.append(result);save(out/'RAW.json',dict(model=model_name,rows=raw))
        margin=round(result['margin'],4) if result['margin'] is not None else None
        print('CASE',entry['call'],model_name,row['id'],condition['name'],repr(result.get('generated_text',result['argmax_token'])),result['correct'],margin,flush=True)
        return logits,arrays,result

    def stop(reason,prediction=None,summary=None):
        decision=dict(model=model_name,status=reason,evaluation_supported=False,prediction=prediction,
            technical_errors=technical_errors,development_summary=summary,completed_rows=len(raw))
        save(out/'DECISION.json',decision); print('STOP',model_name,reason,flush=True); return decision

    def check_phase(phase_rows,phase):
        phase_ids={r['id'] for r in phase_rows}; selected=[r for r in raw if r['id'] in phase_ids]
        lookup={(r['id'],r['condition']):r for r in selected}
        findings=[]
        for row in phase_rows:
            ref=np.load(out/f"CALL_{lookup[row['id'],'baseline']['call']:04d}_logits.npy")
            for condition in ['cut_28','restore_all']:
                call=lookup[row['id'],condition]['call']
                e=float(np.max(np.abs(np.load(out/f'CALL_{call:04d}_logits.npy')-ref)))
                findings.append(dict(kind=condition,id=row['id'],error=e,passed=e<=.05))
        for fam in sorted({r['family'] for r in phase_rows}):
            for q in ['agent','first']:
                pair=[r for r in phase_rows if r['family']==fam and r['query']==q]
                saved=[]; logits=[]
                for row in pair:
                    call=lookup[row['id'],'cut_0']['call']
                    with np.load(out/f'CALL_{call:04d}_states.npz') as z: saved.append(z['suffix'].copy())
                    logits.append(np.load(out/f'CALL_{call:04d}_logits.npy'))
                e=float(np.max(np.abs(saved[0]-saved[1])))
                le=float(np.max(np.abs(logits[0]-logits[1])))
                findings.append(dict(kind='zero_access_orientation_invariance',family=fam,query=q,state_error=e,logit_error=le,passed=e<=1e-6 and le<=.05))
        negatives=[lookup[r['id'],f'irrelevant_{k}'] for r in phase_rows for k in [7,14,21]]
        count=sum(x['correct'] for x in negatives)
        findings.append(dict(kind='irrelevant_accuracy',correct=count,total=len(negatives),passed=count>=23))
        save(out/f'{phase.upper()}_CONTROLS.json',findings)
        return all(x['passed'] for x in findings) and not technical_errors

    prediction=None; dev_summary=None
    for phase in ['development','evaluation']:
        phase_rows=[r for r in rows if r['split']==phase]
        if phase=='development':
            first=phase_rows[0]
            native,_,_=forward(first,dict(BASE,name='native_validation'),wrapped=False,native=True)
            wrapped,_,_=forward(first,dict(BASE,name='wrapped_native_validation'),native=True)
        for row in phase_rows:
            logits,arrays,result=forward(row,BASE)
            baseline_arrays[row['id']]=arrays
            if phase=='development' and row['id']==phase_rows[0]['id']:
                dense=logits.copy(); initial=arrays['last'].copy()
        if phase=='development':
            repeat,rep,_=forward(phase_rows[0],dict(BASE,name='dense_repeat_validation'))
            validations=dict(wrapper_max_logit_error=float(np.max(np.abs(native-wrapped))),
                native_dense_max_logit_error=float(np.max(np.abs(native-dense))),
                repeat_max_logit_error=float(np.max(np.abs(repeat-dense))),
                repeat_max_state_error=float(np.max(np.abs(rep['last']-initial))))
            save(out/'VALIDATION.json',validations)
            if (max(validations[k] for k in ['wrapper_max_logit_error','native_dense_max_logit_error','repeat_max_logit_error'])>.05
                or validations['repeat_max_state_error']>1e-6):
                return stop('STOP_NUMERICAL_VALIDATION')
        baseline_rows=[r for r in raw if r['condition']=='baseline' and r['split']==phase]
        # All four answers in both families are required; no successful-family substitution.
        competence=dict(phase=phase,correct=sum(r['correct'] for r in baseline_rows),total=len(baseline_rows),
            families={f:all(r['correct'] for r in baseline_rows if r['family']==f) for f in sorted({r['family'] for r in phase_rows})})
        save(out/f'{phase.upper()}_COMPETENCE.json',competence)
        if not all(r['correct'] for r in baseline_rows): return stop('STOP_'+phase.upper()+'_COMPETENCE',prediction,dev_summary)
        # Exact same causal prefix under later question demands.
        for f in sorted({r['family'] for r in phase_rows}):
            for ori in [0,1]:
                pair=[r for r in phase_rows if r['family']==f and r['orientation']==ori]
                a,b=[baseline_arrays[r['id']]['prefix'] for r in pair]
                if not np.array_equal(a,b): technical_errors.append(dict(kind='baseline_question_prefix_invariance',family=f,orientation=ori))
        if technical_errors: return stop('STOP_'+phase.upper()+'_BASELINE_TECHNICAL',prediction,dev_summary)
        for row in phase_rows:
            for condition in CONDITIONS: forward(row,condition)
        controls_pass=check_phase(phase_rows,phase)
        summary=matched_summary(phase_rows,raw)
        save(out/f'{phase.upper()}_SUMMARY.json',summary)
        if not controls_pass: return stop('STOP_'+phase.upper()+'_INTERVENTION_CONTROLS',prediction,summary)
        if phase=='development':
            dev_summary=summary
            if prediction_from_qwen is None: prediction=select_prediction(summary)
            else:
                prediction=prediction_from_qwen
                item=next(s for s in summary if s['cut']==prediction['cut'])
                if not (abs(item['difference'])>=.5 and item['difference']*prediction['sign']>0
                        and all(x['difference']*prediction['sign']>0 for x in item['families'].values())
                        and abs(item['difference'])>=2*item['irrelevant_absolute_margin_drift']):
                    return stop('STOP_CROSS_MODEL_DEVELOPMENT_PREDICTION',prediction,summary)
            save(out/'PREDICTION_BEFORE_EVALUATION.json',dict(prediction=prediction,raw_sha256=sha(out/'RAW.json'),
                rule='frozen development selection; unchanged depth/sign for Llama',baseline_evaluation_observed=False))
            if prediction is None: return stop('STOP_NO_DISCRIMINATING_DEVELOPMENT_PATTERN',None,summary)
            baseline_arrays.clear()
        else:
            item=next(s for s in summary if s['cut']==prediction['cut'])
            support=(abs(item['difference'])>=.5 and item['difference']*prediction['sign']>0
                and all(x['difference']*prediction['sign']>0 for x in item['families'].values())
                and abs(item['difference'])>=2*item['irrelevant_absolute_margin_drift'])
            decision=dict(model=model_name,status='EVALUATION_SUPPORTED' if support else 'EVALUATION_PREDICTION_FAILED',
                evaluation_supported=bool(support),prediction=prediction,development_summary=dev_summary,
                evaluation_summary=summary,technical_errors=technical_errors,completed_rows=len(raw))
            save(out/'DECISION.json',decision); print('DECISION',model_name,decision['status'],flush=True)
            return decision

def main():
    parser=argparse.ArgumentParser();parser.add_argument('model',choices=['qwen7b','llama3b']);args=parser.parse_args()
    try: execute(args.model)
    except Exception:
        out=HERE/args.model;out.mkdir(exist_ok=True)
        save(out/'ERROR.json',dict(error=traceback.format_exc(),no_automatic_resume=True))
        raise

if __name__=='__main__': main()
