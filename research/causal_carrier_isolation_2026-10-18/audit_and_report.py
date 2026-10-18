"""Post-run saved-data audit/report. No MLX, model loads, inference or fitted repair."""
import csv
import hashlib
import json
import re
import tarfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,o): Path(p).write_text(json.dumps(o,indent=2,allow_nan=False)+'\n')
def logsum(x): return float(np.logaddexp.reduce(np.asarray(x,dtype=np.float64)))

def rank_audit(rows,field):
    yes=[float(field(r)) for r in rows if r['expected']=='Yes']
    no=[float(field(r)) for r in rows if r['expected']=='No']
    if not yes or not no:
        return dict(n=len(rows),yes=len(yes),no=len(no),auroc=None,separation_gap=None,
                    reason='A class is absent; between-family pooled rank can be confounded')
    comparisons=[dict(yes=a,no=b,score=float(a>b)+.5*float(a==b)) for a in yes for b in no]
    return dict(n=len(rows),yes=len(yes),no=len(no),auroc=float(np.mean([x['score'] for x in comparisons])),
                separation_gap=min(yes)-max(no),yes_scores=yes,no_scores=no,
                all_pair_comparisons=comparisons,
                caution='Exposed post hoc descriptive scores; no calibrated threshold evaluated on fresh cases')

def main():
    lock=json.loads((HERE/'LOCK.json').read_text())
    checks=[]; ledger=[]; new={}; newrows={}; decisions={}; technical={}
    inputs=json.loads((HERE/'INPUTS.json').read_text())
    byid={(r['model'],r['id']):r for r in inputs}
    for n,h in lock['files'].items():
        assert sha(HERE/n)==h,('Frozen file changed',n)
        checks.append(dict(kind='frozen_source_hash',file=n,passed=True))
    budget=json.loads((HERE/'BUDGET.json').read_text())
    assert len(budget)==22 and [r['call'] for r in budget]==list(range(1,23))
    assert not any(r['mode']!='baseline' for r in budget)
    assert not any(byid[r['model'],r['id']]['split']=='evaluation' for r in budget)
    recorded=[]
    for model in ['qwen7b','llama3b']:
        out=HERE/model; raw=json.loads((out/'RAW.json').read_text())
        decisions[model]=json.loads((out/'DECISION.json').read_text())
        assert decisions[model]['status']=='STOP_DEVELOPMENT_COMPETENCE'
        assert decisions[model]['prediction'] is None
        assert not (out/'PREDICTION_BEFORE_EVALUATION.json').exists()
        controls=json.loads((out/'CONTROLS.json').read_text()); assert all(r['passed'] for r in controls)
        technical[model]=controls
        checks.append(dict(kind='budget_and_stop',model=model,recorded=len(raw),passed=len(raw)==11))
        for r in raw:
            row=byid[model,r['id']]
            # Independent grammar parse, not the input generator's role flag.
            if ' was ' in row['sentence']:
                pat,verb,act=re.fullmatch(r'(\w+) was (\w+) by (\w+)\.',row['sentence']).groups()
            else: act,verb,pat=re.fullmatch(r'(\w+) (\w+) (\w+)\.',row['sentence']).groups()
            subj,base,obj=re.fullmatch(r'Did (\w+) (\w+) (\w+)\?',row['question']).groups()
            assert base==row['base'] and verb==row['past']
            expected='Yes' if (subj,obj)==(act,pat) else 'No'
            assert expected==row['expected']==r['expected']
            lp=out/(r['payload']+'_logits.npy'); sp=out/(r['payload']+'_states.npz')
            assert sha(lp)==r['logits_sha256'] and sha(sp)==r['states_sha256']
            logits=np.load(lp); best=int(logits.argmax())
            scores={k:logsum(logits[v]) for k,v in row['aliases'].items()}
            actual=next((k for k,v in row['aliases'].items() if best in v),None)
            other='No' if expected=='Yes' else 'Yes'; margin=scores[expected]-scores[other]
            err=max(abs(scores[k]-r['class_logits'][k]) for k in scores)
            assert err<1e-10 and abs(margin-r['gold_margin'])<1e-10
            assert best==r['argmax_id'] and actual==r['actual']
            assert bool(actual==expected and margin>0)==r['correct']
            checks.append(dict(kind='independent_full_logit_scoring_and_annotation',model=model,call=r['call'],max_error=err,passed=True))
            recorded.append(r['call'])
            if r['kind']=='scientific':
                ledger.append(dict(model=model,family=row['family'],id=row['id'],voice=row['voice'],role=row['role'],query=row['query'],
                    sentence=row['sentence'],question=row['question'],expected=expected,actual=actual,
                    correct=r['correct'],yes_minus_no=scores['Yes']-scores['No'],gold_margin=margin,call=r['call']))
        science=[r for r in raw if r['kind']=='scientific']; newrows[model]=science
        assert len(science)==8
        new[model]=rank_audit(science,lambda r:r['class_logits']['Yes']-r['class_logits']['No'])
        # Algebraic 2×2 contrasts, no optimization/threshold fit. R×Q is truth
        # orientation here; a syntax/position rule can still account for it.
        new[model]['factorial_decomposition']={}
        for voice in ['active','passive']:
            rr=[r for r in science if r['voice']==voice]
            scores={(r['role'],r['query']):r['class_logits']['Yes']-r['class_logits']['No'] for r in rr}
            coeff={}
            for name,dr,dq in [('offset',0,0),('role_main',1,0),('query_main',0,1),('role_x_query',1,1)]:
                coeff[name]=sum(v*((-1)**(dr*role+dq*q)) for (role,q),v in scores.items())/4
            for (role,q),value in scores.items():
                reconstructed=coeff['offset']+coeff['role_main']*((-1)**role)+coeff['query_main']*((-1)**q)+coeff['role_x_query']*((-1)**(role+q))
                assert abs(value-reconstructed)<1e-10
            new[model]['factorial_decomposition'][voice]=coeff
        # Independently reconstruct every reported native/dense/repeat null.
        def load_logits(kind):
            r=next(r for r in raw if r['kind']==kind); return np.load(out/(r['payload']+'_logits.npy'))
        first=next(r for r in raw if r['kind']=='scientific')
        firstlog=np.load(out/(first['payload']+'_logits.npy'))
        validation=dict(wrapper=float(np.max(np.abs(load_logits('validation_unwrapped')-load_logits('validation_wrapped_native')))),
                        dense_mask=float(np.max(np.abs(load_logits('validation_wrapped_native')-firstlog))),
                        repeat=float(np.max(np.abs(load_logits('validation_repeat')-firstlog))))
        assert all(x==0 for x in validation.values())
        repeat=next(r for r in raw if r['kind']=='validation_repeat')
        with np.load(out/(first['payload']+'_states.npz')) as z: first_trace=z['trace'].copy()
        with np.load(out/(repeat['payload']+'_states.npz')) as z: repeat_trace=z['trace'].copy()
        assert np.array_equal(first_trace,repeat_trace)
        for role in [0,1]:
            for voice in ['active','passive']:
                pair=[r for r in science if r['role']==role and r['voice']==voice]
                traces=[]
                for r in pair:
                    with np.load(out/(r['payload']+'_states.npz')) as z: traces.append(z['trace'][:,0].copy())
                assert np.array_equal(*traces)
        checks.append(dict(kind='independent_native_dense_repeat_prefix_nulls',model=model,errors=validation,passed=True))
    assert sorted(recorded)==list(range(1,23))
    # An exploratory comparison with existing exposed data prevents an unjustified
    # claim that *all* historical errors were caused by a single label offset.
    legacy={}; legacy_sources={}
    old=ROOT/'research/linguistic_development_atlas_continuation_2026-10-18'
    for model in ['qwen7b','llama3b']:
        path=old/model/'RAW.json'; oldrows=json.loads(path.read_text())['rows']
        legacy_sources[str(path.relative_to(ROOT))]=sha(path); legacy[model]={}
        for demand in ['agent','theme','goal','order']:
            rr=[r for r in oldrows if r['question_key']==demand]
            value=rank_audit(rr,lambda r:r['class_logit_logsumexp']['Yes']-r['class_logit_logsumexp']['No'])
            value['original_correct']=sum(r['pass_competence'] for r in rr)
            value['within_family']={f:rank_audit([r for r in rr if r['family']==f],
                lambda r:r['class_logit_logsumexp']['Yes']-r['class_logit_logsumexp']['No']) for f in sorted({r['family'] for r in rr})}
            legacy[model][demand]=value
    save(HERE/'INDEPENDENT_AUDIT.json',dict(kind='saved_data_only_no_new_inference',checks=checks,
        all_passed=all(c['passed'] for c in checks),total_checks=len(checks),full_logit_records=22,
        primary_records=16,technical_forward_records=6,attempted_forwards=22,
        causal_interventions=0,evaluation_forwards=0,control_details=technical))
    save(HERE/'EXPLORATORY_READOUT_AUDIT.json',dict(status='POST_HOC_DIAGNOSTIC_ORIGINAL_GATES_UNCHANGED',
        new=new,legacy=legacy,legacy_source_sha256=legacy_sources,
        caution='No threshold fitted, no generated-answer repair, no held-out inference, no new causal evidence'))
    with (HERE/'ANSWER_LEDGER.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(ledger[0])); w.writeheader(); w.writerows(ledger)
    save(HERE/'FAILURES.json',[r for r in ledger if not r['correct']])
    # Plot the actual unmodified decision boundary, never a post-hoc fitted one.
    fig,axes=plt.subplots(1,3,figsize=(14.3,4.9),gridspec_kw={'width_ratios':[1,1,1.25]})
    for ax,model,title in zip(axes[:2],['qwen7b','llama3b'],['Qwen 7B','Llama 3B']):
        rr=[r for r in ledger if r['model']==model]
        for i,r in enumerate(rr):
            color='#2563eb' if r['expected']=='Yes' else '#dc2626'
            ax.scatter(i,r['yes_minus_no'],c=color,s=65,marker='o' if r['correct'] else 'x',linewidths=2)
        ax.axhline(0,color='black',lw=1,label='Unchanged Yes/No boundary')
        ax.axvline(3.5,color='#bbbbbb',ls=':',lw=1)
        ax.set_xticks(range(8),[f"r{r['role']} {r['voice'][0]} q{r['query']}" for r in rr],rotation=70,fontsize=8)
        ax.set_ylabel('Yes − No alias log score (nats)')
        ax.set_title(f"{title}: {sum(r['correct'] for r in rr)}/8 original labels correct\nTrue/false ranking AUC {new[model]['auroc']:.2f}",fontsize=11)
        ax.grid(axis='y',alpha=.2)
    ax=axes[2]; labels=[]; offsets=[]; interactions=[]
    for model,short in [('qwen7b','Qwen'),('llama3b','Llama')]:
        for voice in ['active','passive']:
            c=new[model]['factorial_decomposition'][voice]
            labels.append(short+' '+voice); offsets.append(c['offset']); interactions.append(c['role_x_query'])
    x=np.arange(4); ax.bar(x-.18,offsets,.36,label='Mean Yes/No offset',color='#64748b')
    ax.bar(x+.18,interactions,.36,label='Role × question contrast',color='#059669')
    ax.axhline(0,color='black',lw=1); ax.set_xticks(x,labels,rotation=35,ha='right',fontsize=9)
    ax.set_ylabel('Factorial coefficient (nats)'); ax.set_title('Answer bias and truth-sensitive variation coexist',fontsize=11)
    ax.legend(fontsize=8); ax.grid(axis='y',alpha=.2)
    fig.suptitle('A failed answer gate can conceal a graded response to role and question changes',fontsize=14,y=1.01)
    fig.text(.5,-.07,'Exploratory saved-data diagnostic · one lexical family per model · frozen competence gates still FAIL\nBlue = true; red = false. Crosses = wrong original label. No calibration threshold fitted; no causal intervention executed.',ha='center',fontsize=10)
    fig.tight_layout()
    for suffix in ['png','svg','pdf']: fig.savefig(HERE/('READOUT_DIAGNOSTIC.'+suffix),dpi=170,bbox_inches='tight')
    plt.close(fig)
    # Complete payloads, not just selected scores: preserve every original-end
    # full-vocabulary vector and every measured state archive in a single file.
    with tarfile.open(HERE/'COMPLETE_RAW_ARRAYS.tar.gz','w:gz') as tar:
        for model in ['qwen7b','llama3b']:
            for p in sorted((HERE/model).glob('CALL_*')): tar.add(p,arcname=str(p.relative_to(HERE)))
    save(HERE/'RAW_ARCHIVE.json',dict(path='COMPLETE_RAW_ARRAYS.tar.gz',sha256=sha(HERE/'COMPLETE_RAW_ARRAYS.tar.gz'),
        bytes=(HERE/'COMPLETE_RAW_ARRAYS.tar.gz').stat().st_size,
        members=44,description='All22 complete full-vocabulary original-end logit NPYs and22 measured-state NPZs. Two unwrapped state archives have empty traces by design.',
        local_original_files_retained=True,model_weights_included=False))
    print(json.dumps(dict(checks=len(checks),all_passed=True,new={m:dict(accuracy=sum(r['correct'] for r in ledger if r['model']==m),
        auroc=new[m]['auroc'],separation_gap=new[m]['separation_gap']) for m in new},raw_bytes=(HERE/'COMPLETE_RAW_ARRAYS.tar.gz').stat().st_size),indent=2))

if __name__=='__main__': main()
