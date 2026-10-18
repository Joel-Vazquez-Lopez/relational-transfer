"""Saved-data independent replay, figures and ledgers. No model inference."""
import csv
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
MODES=['closed','frozen','live','irrelevant'];CUTS=[0,7,14,21]
def save(p,o):Path(p).write_text(json.dumps(o,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda:f.read(2**20),b''):h.update(c)
    return h.hexdigest()
def lse(x):return float(np.logaddexp.reduce(np.asarray(x,dtype=np.float64)))
def writecsv(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def independent_D(rows):
    d={(r['role'],r['query']):r['yes_minus_no'] for r in rows};assert len(d)==4
    value=(d[0,0]+d[1,1]-d[0,1]-d[1,0])/4
    pairs=[d[0,0]-d[1,0],d[1,1]-d[0,1],d[0,0]-d[0,1],d[1,1]-d[1,0]]
    return value,pairs
def independent_selection(summary):
    lookup={(r['family'],r['voice'],r['mode'],r['cut']):r for r in summary}
    cells=sorted({(r['family'],r['voice']) for r in summary})
    def valid(g):return g['baseline']['D']>=.5 and g['baseline']['all_ordinal_pairs_positive']
    for signature,target,rival in [('cached','frozen','closed'),('refresh','live','frozen')]:
        eligible=[]
        for k in [7,14,21]:
            effects=[]
            for fam,voice in cells:
                g={m:lookup[fam,voice,m,k] for m in MODES};r={m:g[m]['ratio'] for m in MODES}
                if (not valid(g['closed']) or any(v is None for v in r.values()) or abs(r['irrelevant']-1)>.25
                    or r[target]<.5 or r[rival]>.75 or r[target]-r[rival]<.25):break
                effects.append(r[target]-r[rival])
            if len(effects)==len(cells):eligible.append(dict(signature=signature,cut=k,min_effect=min(effects)))
        if eligible:return sorted(eligible,key=lambda r:(-r['min_effect'],r['cut']))[0]
    for lo,hi in [(0,7),(7,14),(14,21)]:
        effects=[]
        for fam,voice in cells:
            a,b,na,nb=[lookup[fam,voice,m,k] for m,k in [('closed',lo),('closed',hi),('irrelevant',lo),('irrelevant',hi)]]
            if (not valid(b) or any(r['ratio'] is None for r in [a,b,na,nb]) or a['ratio']>.25 or b['ratio']<.75
                or abs(na['ratio']-1)>.25 or abs(nb['ratio']-1)>.25):break
            effects.append(b['ratio']-a['ratio'])
        if len(effects)==len(cells):return dict(signature='suffix',lower_cut=lo,cut=hi,min_effect=min(effects))
    return None

def main():
    lock=json.loads((HERE/'LOCK.json').read_text());checks=[];ledger=[];profile=[];normal=[];allraw={};allsummary={};decisions={}
    for n,h in lock['files'].items():assert sha(HERE/n)==h,('Changed frozen source',n)
    checks.append(dict(check='all_frozen_source_hashes',passed=True,n=len(lock['files'])))
    inputs=json.loads((HERE/'INPUTS.json').read_text());byid={(r['model'],r['id']):r for r in inputs}
    budget=[json.loads(s) for s in (HERE/'BUDGET.jsonl').read_text().splitlines()]
    for model in ['qwen7b','llama3b']:
        out=HERE/model;decision=json.loads((out/'DECISION.json').read_text());decisions[model]=decision
        lines=(out/'RAW.jsonl').read_bytes().splitlines(keepends=True)
        raw=[json.loads(s) for s in lines];allraw[model]=raw
        assert len(raw)==decision['recorded_forwards']
        assert len({r['call'] for r in raw})==len(raw)
        for r in raw:
            row=byid[model,r['id']];path=out/r['payload'];assert sha(path)==r['payload_sha256']
            with np.load(path) as z:logits=z['logits'].astype(np.float32);trace=z['trace'].astype(np.float32)
            cls={k:lse(logits[v]) for k,v in row['aliases'].items()};s=cls['Yes']-cls['No'];best=int(logits.argmax())
            actual=next((k for k,v in row['aliases'].items() if best in v),None)
            err=max(abs(cls[k]-r['class_logits'][k]) for k in cls)
            assert err<1e-10 and abs(s-r['yes_minus_no'])<1e-10 and best==r['argmax_id'] and actual==r['actual']
            if ' was ' in row['sentence']:
                patient,verb,agent=re.fullmatch(r'(\w+) was (\w+) by (\w+)\.',row['sentence']).groups()
            else:agent,verb,patient=re.fullmatch(r'(\w+) (\w+) (\w+)\.',row['sentence']).groups()
            subject,base,obj=re.fullmatch(r'Did (\w+) (\w+) (\w+)\?',row['question']).groups()
            truth='Yes' if (subject,obj)==(agent,patient) else 'No';assert truth==r['expected']
            margin=s if truth=='Yes' else -s;assert abs(margin-r['gold_margin'])<1e-10
            assert bool(actual==truth and margin>0)==r['correct']
            cal='Yes' if s>r['threshold'] else 'No' if s<r['threshold'] else None
            assert cal==r['calibrated_class'] and (cal==truth)==r['calibrated_correct']
            checks.append(dict(check='full_logit_replay_and_grammar',model=model,call=r['call'],max_error=err,passed=True))
            if r['kind']=='scientific':ledger.append(dict(model=model,id=r['id'],family=r['family'],split=r['split'],voice=r['voice'],role=r['role'],query=r['query'],mode=r['mode'],cut=r['cut'],expected=truth,actual=actual,
                native_correct=r['correct'],calibrated_class=cal,calibrated_correct=r['calibrated_correct'],yes_minus_no=s,gold_margin=margin,call=r['call']))
        sci=[r for r in raw if r['kind']=='scientific'];lookup={(r['id'],r['mode'],r['cut']):r for r in sci}
        def payload(r):
            with np.load(out/r['payload']) as z:return z['logits'].astype(np.float32),z['trace'].astype(np.float32)
        for r in sci:
            if r['mode']=='baseline':continue
            baseline=lookup[r['id'],'baseline',28];bl,bt=payload(baseline);il,it=payload(r)
            k=r['cut'];assert np.array_equal(bt[:k+1],it[:k+1])
            key='prefix_without_neighbor' if r['mode']=='neighbor' else 'prefix'
            assert r['prefix_hashes'][key]==baseline['prefix_hashes'][key]
            if r['mode']=='live':assert np.array_equal(bt[:,0],it[:,0])
            if k==28:assert np.max(np.abs(il-bl))<=.05 and np.array_equal(bt,it)
            checks.append(dict(check='before_cut_prefix_live_or_sham',model=model,call=r['call'],passed=True))
        for fam in sorted({r['family'] for r in sci}):
            for voice in ['active','passive']:
                for role in [0,1]:
                    qpair=[lookup[r['id'],'baseline',28] for r in inputs if r['model']==model and r['family']==fam and r['voice']==voice and r['role']==role]
                    _,a=payload(qpair[0]);_,b=payload(qpair[1]);assert np.array_equal(a[:,0],b[:,0])
                for q in [0,1]:
                    for mode in ['closed','frozen']:
                        rp=[lookup[r['id'],mode,0] for r in inputs if r['model']==model and r['family']==fam and r['voice']==voice and r['query']==q]
                        a,_=payload(rp[0]);b,_=payload(rp[1]);assert np.max(np.abs(a-b))<=.05
                        assert rp[0]['prefix_hashes']['null_states']==rp[1]['prefix_hashes']['null_states']
                baseline=[r for r in sci if r['family']==fam and r['voice']==voice and r['mode']=='baseline']
                d,pairs=independent_D(baseline);score=[r['yes_minus_no'] for r in baseline]
                yes=[r['yes_minus_no'] for r in baseline if r['expected']=='Yes'];no=[r['yes_minus_no'] for r in baseline if r['expected']=='No']
                auc=float(np.mean([float(a>b)+.5*float(a==b) for a in yes for b in no]))
                normal.append(dict(model=model,family=fam,split=baseline[0]['split'],voice=voice,D=d,
                    all_ordinal_pairs_positive=all(v>0 for v in pairs),native_correct=sum(r['correct'] for r in baseline),calibrated_correct=sum(r['calibrated_correct'] for r in baseline),n=4,within_voice_auc=auc))
        summaries=[]
        for phase in ['development','evaluation']:
            path=out/(phase.upper()+'_SUMMARY.json')
            summary=json.loads(path.read_text());summaries+=summary
            for s in summary:
                subset=[r for r in sci if r['family']==s['family'] and r['voice']==s['voice'] and r['mode']==s['mode'] and r['cut']==s['cut']]
                base=[r for r in sci if r['family']==s['family'] and r['voice']==s['voice'] and r['mode']=='baseline']
                d,_=independent_D(subset);b,_=independent_D(base)
                assert abs(d-s['intervention']['D'])<1e-10 and abs(b-s['baseline']['D'])<1e-10
                ratio=d/b if abs(b)>=.5 else None
                assert (ratio is None and s['ratio'] is None) or (ratio is not None and abs(ratio-s['ratio'])<1e-10)
                profile.append(dict(model=model,split=phase,family=s['family'],voice=s['voice'],mode=s['mode'],cut=s['cut'],D_native=b,D_intervened=d,retention=ratio))
        allsummary[model]=summaries
        predfile=json.loads((out/'PREDICTION_BEFORE_EVALUATION.json').read_text())
        dev=[s for s in summaries if s['family'] in ['g01','g02']];ind=independent_selection(dev)
        assert ind==predfile['prediction']==decision['prediction']
        devlines=[line for line,r in zip(lines,raw) if r['split']=='development']
        assert hashlib.sha256(b''.join(devlines)).hexdigest()==predfile['raw_sha256']
        assert max(r['call'] for r in raw if r['split']=='development')<min(r['call'] for r in raw if r['split']=='evaluation')
        checks.append(dict(check='independent_dev_selection_and_unexposed_prediction_hash',model=model,prediction=ind,passed=True))
        held=[s for s in summaries if s['family'] not in ['g01','g02']]
        expected_eval=[]
        if ind:
            lookup_h={(s['family'],s['voice'],s['mode'],s['cut']):s for s in held}
            for fam,voice in sorted({(s['family'],s['voice']) for s in held}):
                k=ind['cut'];b=lookup_h[fam,voice,'closed',k]['baseline']
                valid=b['D']>=.5 and b['all_ordinal_pairs_positive']
                if ind['signature']=='suffix':
                    lo=ind['lower_cut'];a,z,na,nz=[lookup_h[fam,voice,m,j]['ratio'] for m,j in [('closed',lo),('closed',k),('irrelevant',lo),('irrelevant',k)]]
                    passed=bool(valid and all(v is not None for v in [a,z,na,nz]) and a<=.25 and z>=.75 and abs(na-1)<=.25 and abs(nz-1)<=.25)
                else:
                    r={m:lookup_h[fam,voice,m,k]['ratio'] for m in MODES}
                    target,rival=('frozen','closed') if ind['signature']=='cached' else ('live','frozen')
                    passed=bool(valid and all(v is not None for v in r.values()) and abs(r['irrelevant']-1)<=.25 and r[target]>=.5 and r[rival]<=.75 and r[target]-r[rival]>=.25)
                expected_eval.append(dict(family=fam,voice=voice,passed=passed))
        reported=json.loads((out/'PROSPECTIVE_EVALUATION.json').read_text())
        assert [{k:r[k] for k in ['family','voice','passed']} for r in reported]==expected_eval
        expected_status=('PROSPECTIVE_SIGNATURE_SUPPORTED' if expected_eval and all(r['passed'] for r in expected_eval)
            else 'PROSPECTIVE_SIGNATURE_FAILED' if ind else 'COMPLETE_NO_DEVELOPMENT_CANDIDATE')
        assert decision['status']==expected_status
        checks.append(dict(check='independent_held_prediction_and_final_status',model=model,status=expected_status,passed=True))
        assert not decision['technical_failures']
    assert len(budget)==sum(len(v) for v in allraw.values())==1758
    assert sorted(r['call'] for rr in allraw.values() for r in rr)==list(range(1,1759))
    writecsv(HERE/'ANSWER_LEDGER.csv',ledger);writecsv(HERE/'NORMAL_RESPONSE_LEDGER.csv',normal);writecsv(HERE/'INTERVENTION_LEDGER.csv',profile)
    save(HERE/'FAILURES.json',[r for r in ledger if not r['native_correct']])
    save(HERE/'INDEPENDENT_AUDIT.json',dict(all_passed=True,checks=checks,total_checks=len(checks),full_logit_replays=1758,
        no_new_model_inference=True,control_prefix_arrays_represented_by_frozen_hashes=True))
    summary={}
    for model in allraw:
        normals=[r for r in normal if r['model']==model];held=[r for r in normals if r['split']=='evaluation']
        evaluated=json.loads((HERE/model/'PROSPECTIVE_EVALUATION.json').read_text())
        family={f:all(r['passed'] for r in evaluated if r['family']==f) for f in sorted({r['family'] for r in evaluated})}
        summary[model]=dict(decision=decisions[model],normal_native_correct=sum(r['native_correct'] for r in normals),normal_calibrated_correct=sum(r['calibrated_correct'] for r in normals),normal_answers=48,
            held_native_correct=sum(r['native_correct'] for r in held),held_calibrated_correct=sum(r['calibrated_correct'] for r in held),held_answers=32,
            held_sensitive_cells=sum(r['D']>=.5 and r['all_ordinal_pairs_positive'] for r in held),held_cells=8,
            primary_family_results=family,prospective_cell_results=evaluated)
    save(HERE/'SUMMARY.json',summary)
    colors=dict(closed='#b91c1c',frozen='#2563eb',live='#059669',irrelevant='#64748b')
    labels=dict(closed='Sentence closed',frozen='Period source-isolated',live='Period can reread',irrelevant='Neutral control')
    fig,axes=plt.subplots(2,2,figsize=(12.5,8),sharex=True)
    for i,model in enumerate(['qwen7b','llama3b']):
        for j,voice in enumerate(['active','passive']):
            ax=axes[i,j]
            for mode in MODES:
                curves=[]
                for fam in ['g03','g04','g05','g06']:
                    rs=[next(s for s in profile if s['model']==model and s['family']==fam and s['voice']==voice and s['mode']==mode and s['cut']==k)['retention'] for k in CUTS]
                    vals=np.array([np.nan if r is None else r for r in rs]);curves.append(vals)
                    ax.plot(CUTS,vals,color=colors[mode],alpha=.22,lw=1)
                a=np.asarray(curves);mean=np.mean(a,axis=0)
                ax.plot(CUTS,mean,color=colors[mode],marker='o',lw=2.4,label=labels[mode])
            ax.axhline(0,color='black',lw=.7);ax.axhline(1,color='black',lw=.7,ls=':')
            ax.set_title(model+' · '+voice+' · four evaluation families');ax.set_xticks(CUTS);ax.grid(alpha=.15)
            ax.set_xlabel('First block with access removed (zero-based)');ax.set_ylabel('D under intervention / native D')
    axes[0,0].legend(fontsize=8);fig.suptitle('Causal source dependence of the graded role × question response',fontsize=15)
    fig.text(.5,-.01,'Thin lines: individual fresh families. Thick: unweighted mean. Negative/amplified effects retained.\nRatios below the frozen native-sensitivity floor are undefined; no interpolation or hidden time alignment.',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.04,1,.96])
    for ext in ['png','svg','pdf']:fig.savefig(HERE/('CAUSAL_PROFILES.'+ext),dpi=170,bbox_inches='tight')
    plt.close(fig)
    # Full-development vector interactions are descriptive and cannot identify
    # semantic computation; retain their magnitudes, including zero period nulls.
    trajectory=[];fig,axes=plt.subplots(1,2,figsize=(12,4.3))
    for ax,model in zip(axes,['qwen7b','llama3b']):
        raw=allraw[model];modelcurves=[]
        for fam in ['g03','g04','g05','g06']:
            for voice in ['active','passive']:
                rr=[r for r in raw if r['kind']=='scientific' and r['mode']=='baseline' and r['family']==fam and r['voice']==voice]
                h={}
                for r in rr:
                    with np.load(HERE/model/r['payload']) as z:h[r['role'],r['query']]=z['trace'].astype(np.float64)
                v=(h[0,0]-h[0,1]-h[1,0]+h[1,1])/4
                norms=np.linalg.norm(v,axis=-1);assert np.max(norms[:,0])==0
                modelcurves.append(norms[:,1]);ax.plot(range(29),norms[:,1],alpha=.3,lw=1)
                trajectory.append(dict(model=model,family=fam,voice=voice,period_norm=norms[:,0].tolist(),answer_norm=norms[:,1].tolist()))
        ax.plot(range(29),np.mean(modelcurves,axis=0),color='black',lw=2,label='Mean of eight family/voice curves')
        ax.set_title(model);ax.set_xlabel('Embedding (0) then residual block outputs');ax.set_ylabel('Native norm of role × question contrast');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Full residual development at the answer site — descriptive, not a causal mechanism',fontsize=12)
    fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(HERE/('NATIVE_DEVELOPMENT.'+ext),dpi=170,bbox_inches='tight')
    plt.close(fig);save(HERE/'NATIVE_DEVELOPMENT.json',trajectory)
    print(json.dumps(dict(summary=summary,independent_checks=len(checks)),indent=2))

if __name__=='__main__':main()
