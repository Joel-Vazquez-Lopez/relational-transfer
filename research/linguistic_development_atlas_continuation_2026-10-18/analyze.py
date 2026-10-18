"""Frozen descriptive analyses; reads captures only, never runs a model."""
import csv
import itertools
import json
import os
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
PARTS=['residual','attention','mlp']
CELLS=[('within',0),('within',1),('across',0),('across',1)]


def write(p,value):
    Path(p).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def interaction(a):
    a=np.asarray(a,dtype=np.float64)
    return (a[3]-a[2])-(a[1]-a[0])


def relative_null(a):
    return float(np.max(np.linalg.norm(interaction(a),axis=-1))/max(np.max(np.linalg.norm(a.astype(np.float64),axis=-1)),1e-12))


def read(out,r):
    with np.load(out/(r['id']+'.npz'),allow_pickle=False) as z:
        return {p:z[p] for p in PARTS+['reconstruction_error_norm']}


def technical_controls(model,rows,out,tokens):
    by={(r['family'],r['frame'],r['swap'],r['question_key']):r for r in rows}
    families=sorted({r['family'] for r in rows}); checks=[];qchecks=[]
    embed_null=0.0
    for f,q in itertools.product(families,['agent','theme','goal','order']):
        rs=[by[f,k,p,q] for k,p in CELLS];ds=[tokens[model,r['id']] for r in rs]
        positions=ds[0]['retained_positions'];assert all(d['retained_positions']==positions for d in ds)
        cue=min(i for i,(a,b) in enumerate(zip(ds[0]['token_ids'],ds[2]['token_ids'])) if a!=b)
        before=[j for j,pos in enumerate(positions) if pos<cue]
        arrays=[read(out,r) for r in rs]
        for part in PARTS:
            a=np.stack([x[part][:,before] for x in arrays])
            checks.append(dict(family=f,question=q,part=part,cue_position=cue,
                               relative_interaction=relative_null(a)))
        emb=np.stack([x['residual'][0] for x in arrays])
        embed_null=max(embed_null,relative_null(emb))
    for f,k,p in itertools.product(families,['within','across'],[0,1]):
        rs=[by[f,k,p,q] for q in ['agent','theme','goal','order']]
        arrays=[read(out,r) for r in rs]
        for part in PARTS:
            a=[]
            for r,x in zip(rs,arrays):
                d=tokens[model,r['id']];idx=[d['retained_positions'].index(pos) for pos in d['sentence_positions']]
                a.append(x[part][:,idx].astype(np.float64))
            den=max(max(float(np.max(np.linalg.norm(x,axis=-1))) for x in a),1e-12)
            rel=max(float(np.max(np.linalg.norm(x-a[0],axis=-1)))/den for x in a[1:])
            qchecks.append(dict(family=f,frame=k,swap=p,part=part,relative_discrepancy=rel))
    raw=json.loads((out/'RAW.json').read_text())['rows'];chosen={r['id'] for r in rows}
    recon=max(r['max_reconstruction_relative_error'] for r in raw if r['id'] in chosen)
    a=max(r['relative_interaction'] for r in checks);b=max(r['relative_discrepancy'] for r in qchecks)
    return dict(prefix_pass=bool(max(a,b,embed_null)<=1e-6),components_pass=bool(recon<=.002),
        max_precue_relative_interaction=a,max_question_prefix_relative_discrepancy=b,
        max_embedding_relative_interaction=embed_null,max_native_reconstruction_relative_error=recon,
        precue=checks,question_prefix=qchecks)


def features(a):
    a=np.asarray(a,dtype=np.float64)
    within=a[1]-a[0];across=a[3]-a[2];I=across-within
    nw=np.linalg.norm(within,axis=-1);na=np.linalg.norm(across,axis=-1);ni=np.linalg.norm(I,axis=-1)
    ratio=np.divide(ni,nw+na,out=np.full_like(ni,np.nan),where=nw+na>0)
    cosine=np.divide(np.sum(within*across,axis=-1),nw*na,out=np.full_like(ni,np.nan),where=nw*na>0)
    return dict(within_norm=nw,across_norm=na,interaction_norm=ni,ratio=ratio,cosine=cosine),I


def self_check():
    # Exact additive, interactive and rotated known systems; undefined cosine stays undefined.
    additive=np.array([[0,0],[1,0],[0,1],[1,1]],dtype=float)[:,None,None,:]
    assert np.array_equal(interaction(additive),np.zeros((1,1,2)))
    changed=additive.copy();changed[3,0,0,1]+=2
    f,I=features(changed);assert np.array_equal(I,[[[0,2]]])
    assert 0<=f['ratio'].item()<=1
    q=np.array([[0.,1.],[-1.,0.]])
    assert np.allclose(features(changed@q)[0]['interaction_norm'],f['interaction_norm'])
    assert np.isnan(features(np.zeros_like(additive))[0]['cosine']).all()
    return dict(additive_zero=True,known_interaction=True,rotation_invariant_norm=True,undefined_not_zero=True)


def main():
    from run import inputs,assert_lock,sha
    lock=assert_lock();rows,tokens=inputs();summaries={};ledger=[];desc={};maps={}
    for model in ['qwen7b','llama3b']:
        out=HERE/model
        if not (out/'RAW.json').exists():continue
        raw=json.loads((out/'RAW.json').read_text())['rows'];available={r['id'] for r in raw}
        family_out=[];group_out=[]
        controls=json.loads((out/'CONTROLS.json').read_text())
        for family in ['a01','a02','a03','a04']:
            fs=[r for r in raw if r['family']==family]
            if not fs:continue
            family_out.append(dict(family=family,split=fs[0]['split'],answers=len(fs),
                correct=sum(r['pass_competence'] for r in fs),
                fully_competent=bool(len(fs)==16 and all(r['pass_competence'] for r in fs)),
                unexpected_answers=[r['id'] for r in fs if r['actual_class'] is None]))
            for q in ['agent','theme','goal','order']:
                rs=[next(r for r in rows if r['id']==f'{family}_{k}_{p}_{q}') for k,p in CELLS]
                if not all(r['id'] in available for r in rs):continue
                data=[tokens[model,r['id']] for r in rs];positions=data[0]['retained_positions']
                arrays=[read(out,r) for r in rs];base=f'{model}__{family}__{q}'
                metrics={};interactions={}
                for part in PARTS:
                    metrics[part],interactions[part]=features(np.stack([x[part] for x in arrays]))
                    for key,v in metrics[part].items():desc[f'{base}__{part}__{key}']=v
                # Every residual block's I change is accounted by writes plus native rounding.
                gap=interactions['residual'][1:]-interactions['residual'][:-1]-interactions['attention']-interactions['mlp']
                gap_norm=np.linalg.norm(gap,axis=-1)
                desc[f'{base}__component_accounting_gap_norm']=gap_norm
                desc[f'{base}__sum_four_native_rounding_error_norm_bounds']=sum(x['reconstruction_error_norm'] for x in arrays)
                period_pos=data[0]['sentence_positions'][-1];period=positions.index(period_pos)
                answer=positions.index(data[0]['original_prompt_end'])
                cue=min(i for i,(a,b) in enumerate(zip(data[0]['token_ids'],data[2]['token_ids'])) if a!=b)
                actual=[next(x for x in raw if x['id']==r['id']) for r in rs]
                margins=np.array([r['class_logit_logsumexp']['Yes']-r['class_logit_logsumexp']['No'] for r in actual])
                # Relation Grams at the two predeclared anchors, preserving operation labels.
                residual=np.stack([x['residual'] for x in arrays]).astype(np.float64)
                labeled=np.stack([residual[1]-residual[0],residual[3]-residual[2],
                                  residual[2]-residual[0],residual[3]-residual[1],interactions['residual']],axis=-2)
                for anchor,j in [('period',period),('readout',answer)]:
                    x=labeled[:,j];n=np.linalg.norm(x,axis=-1,keepdims=True)
                    unit=np.divide(x,n,out=np.full_like(x,np.nan),where=n>0)
                    desc[f'{base}__{anchor}__response_cosine_gram']=unit@np.swapaxes(unit,-1,-2)
                m=metrics['residual']
                group_out.append(dict(family=family,split=rs[0]['split'],question=q,
                    four_answers_correct=all(r['pass_competence'] for r in actual),
                    actual_classes=[r['actual_class'] for r in actual],
                    expected_classes=[r['expected'] for r in actual],
                    yes_minus_no_margin_interaction=float(interaction(margins)),
                    period_mean_ratio=float(np.nanmean(m['ratio'][1:,period])),
                    readout_mean_ratio=float(np.nanmean(m['ratio'][1:,answer])),
                    period_final_interaction_norm=float(m['interaction_norm'][-1,period]),
                    readout_final_interaction_norm=float(m['interaction_norm'][-1,answer]),
                    max_component_accounting_gap_norm=float(np.max(gap_norm))))
                maps[model,family,q]=dict(ratio=m['ratio'],raw=m['interaction_norm'],positions=positions,
                    period=period,answer=answer,cue=cue,split=rs[0]['split'],correct=all(r['pass_competence'] for r in actual))
                for stage,j in itertools.product(range(29),range(len(positions))):
                    ledger.append(dict(model=model,family=family,split=rs[0]['split'],question=q,
                        stage=stage,absolute_position=positions[j],site_index=j,
                        role='sentence' if positions[j] in data[0]['sentence_positions'] else 'question_or_prompt_end',
                        within_norm=float(m['within_norm'][stage,j]),across_norm=float(m['across_norm'][stage,j]),
                        interaction_norm=float(m['interaction_norm'][stage,j]),
                        ratio=None if np.isnan(m['ratio'][stage,j]) else float(m['ratio'][stage,j]),
                        cosine=None if np.isnan(m['cosine'][stage,j]) else float(m['cosine'][stage,j])))
        complete=json.loads((out/'COMPLETE.json').read_text()) if (out/'COMPLETE.json').exists() else {'status':'incomplete'}
        summaries[model]=dict(run=complete,controls=controls,families=family_out,groups=group_out,
            total_answers=len(raw),correct_answers=sum(r['pass_competence'] for r in raw),
            question_counts={q:dict(correct=sum(r['pass_competence'] for r in raw if r['question_key']==q),total=sum(r['question_key']==q for r in raw)) for q in ['agent','theme','goal','order']})
    np.savez(HERE/'DESCRIPTORS.npz',**desc)
    if ledger:
        with (HERE/'DEVELOPMENT_MAP.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(ledger[0]));w.writeheader();w.writerows(ledger)
    write(HERE/'SUMMARY.json',dict(status='descriptive_pilot_no_unique_mechanism_claim',models=summaries,
        lock_sha256=sha(HERE/'LOCK.json'),model_forwards=sum(s['run'].get('total_forwards',0) for s in summaries.values()),
        grammar_review=json.loads((HERE/'ANNOTATION_DECISION.json').read_text()),self_check=self_check(),
        gram_labels=['P_within','P_across','K_original','K_swap','interaction'],
        units='Four lexical families, two held-out names per model; layers/tokens/answers not independent replications.'))
    if maps:plot(maps)
    # Local source and array manifest; full tensors/logits remain local with exact checksums.
    manifest={str(p.relative_to(HERE)):dict(bytes=p.stat().st_size,sha256=sha(p))
              for p in sorted(HERE.rglob('*')) if p.is_file() and not any(x in p.parts for x in ['__pycache__','.mpl']) and p.name not in ['RESULT_MANIFEST.json','GITHUB_PRESERVATION.json']}
    write(HERE/'RESULT_MANIFEST.json',manifest)
    print('ANALYSIS_COMPLETE',[(m,s['correct_answers'],s['total_answers']) for m,s in summaries.items()],flush=True)


def plot(maps):
    os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.mpl'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for model in sorted({k[0] for k in maps}):
        families=sorted({k[1] for k in maps if k[0]==model});qs=['agent','theme','goal','order']
        fig,axes=plt.subplots(len(families),4,figsize=(18,3.3*len(families)),squeeze=False,constrained_layout=True)
        for i,f in enumerate(families):
            for j,q in enumerate(qs):
                ax=axes[i,j];d=maps.get((model,f,q))
                if d is None:ax.set_visible(False);continue
                im=ax.imshow(d['ratio'],origin='lower',aspect='auto',vmin=0,vmax=1,cmap='magma')
                eligible=next((k for k,p in enumerate(d['positions']) if p>=d['cue']),len(d['positions']))
                ax.axvline(eligible-.5,color='cyan',linewidth=1)
                ax.set_title(f'{f} {d["split"]} / {q} / 4 correct: {d["correct"]}',fontsize=9)
                ax.set_xlabel('Retained token index (sentence → question → readout)');ax.set_ylabel('Residual stage')
                ax.set_xticks(range(len(d['positions'])));ax.tick_params(axis='x',labelsize=6)
        fig.colorbar(im,ax=axes,shrink=.7,label='||I|| / (||P within|| + ||P across||); undefined = blank')
        fig.suptitle(model+': grouping-conditioned permutation response (descriptive)')
        fig.savefig(HERE/f'ATLAS_{model}.png',dpi=150);fig.savefig(HERE/f'ATLAS_{model}.svg');plt.close(fig)
        for measure,label in [('ratio','Normalized interaction'),('raw','Interaction norm (native units)')]:
            fig,axes=plt.subplots(4,2,figsize=(11,11),constrained_layout=True)
            for i,q in enumerate(qs):
                for j,anchor in enumerate(['period','answer']):
                    ax=axes[i,j]
                    for f in families:
                        d=maps.get((model,f,q))
                        if d is None:continue
                        ax.plot(np.arange(29),d[measure][:,d[anchor]],label=f+' '+d['split'],linestyle='--' if d['split']=='held_out' else '-')
                    ax.set_title(q+' / '+('sentence period' if anchor=='period' else 'prompt end'))
                    ax.set_xlabel('Residual stage (0 = embedding)');ax.set_ylabel(label);ax.grid(alpha=.2)
                    if i==0 and j==0:ax.legend(fontsize=7)
            fig.suptitle(model+' / '+label+' / all exposed families; no fitted alignment')
            fig.savefig(HERE/f'LINES_{measure}_{model}.png',dpi=150);fig.savefig(HERE/f'LINES_{measure}_{model}.svg');plt.close(fig)


if __name__=='__main__':main()
