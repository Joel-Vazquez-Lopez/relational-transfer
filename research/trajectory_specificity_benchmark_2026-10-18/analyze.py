"""All prespecified descriptors and comparisons; no inference or fitted metric."""
import csv
import gzip
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
LOCK=json.loads((HERE/'PROTOCOL.json').read_text())
FIDS=[f['id'] for f in LOCK['families']]
CATS=['role_active','role_passive','voice_agent_a','voice_agent_b','unrelated','artifact']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(v):
    if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v
def save(name,v):(HERE/name).write_text(json.dumps(clean(v),indent=2,allow_nan=False)+'\n')
def table(name,rows):
    with (HERE/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def average(vals):
    vals=[x for x in vals if x is not None and np.isfinite(x)]
    return float(np.mean(vals)) if vals else None
def corr(a,b):
    a,b=np.asarray(a).ravel(),np.asarray(b).ravel()
    mask=np.isfinite(a)&np.isfinite(b)
    if mask.sum()<3:return None
    a,b=a[mask],b[mask];a=a-a.mean();b=b-b.mean()
    den=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.clip(a@b/den,-1,1)) if den>0 else None
def cos(a,b):
    den=np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)
    out=np.full(den.shape,np.nan)
    np.divide(np.sum(a*b,axis=-1),den,out=out,where=den>0)
    return np.clip(out,-1,1)
def describe(path):
    v=np.diff(path,axis=0);speed=np.linalg.norm(v,axis=-1)
    unit=np.full(v.shape,np.nan)
    np.divide(v,speed[:,None],out=unit,where=speed[:,None]>0)
    return {'gram':unit@unit.T,'speed':speed,'norm':np.linalg.norm(path,axis=-1)}
def compare(a,b):
    return {**{f'gram_lag{lag}':corr(a['gram'][np.triu_indices(28,lag)],b['gram'][np.triu_indices(28,lag)]) for lag in [1,8]},
            'speed':corr(a['speed'],b['speed'])}
def cka(a,b):
    def centered_gram(x):
        k=x@x.T
        return k-k.mean(0,keepdims=True)-k.mean(1,keepdims=True)+k.mean()
    k,l=centered_gram(a),centered_gram(b)
    den=np.linalg.norm(k)*np.linalg.norm(l)
    return float(np.clip(np.sum(k*l)/den,0,1)) if den>0 else None


def main():
    metadata={};states={};paths={};descriptors={};saved={};path_rows=[];angles=[];null_rows=[]
    for model in LOCK['models']:
        d=json.loads((HERE/f'RAW_{model}.json').read_text());metadata[model]=d
        assert sha(HERE/f'STATES_{model}.npz')==d['states_sha256']
        z=np.load(HERE/f'STATES_{model}.npz');states[model]=z
        for mode,fid in itertools.product(['native','fixed'],FIDS):
            for cat in CATS:
                if cat=='artifact':
                    if f'{fid}__null_64' not in z:continue
                    delta=z[f'{fid}__null_65'].astype(np.float64)-z[f'{fid}__null_64'].astype(np.float64)
                elif cat=='unrelated':
                    nxt=FIDS[(FIDS.index(fid)+1)%len(FIDS)]
                    delta=z[f'{nxt}__active_a__{mode}'].astype(np.float64)-z[f'{fid}__active_a__{mode}'].astype(np.float64)
                else:
                    a,b=LOCK['contrasts'][cat]
                    delta=z[f'{fid}__{b}__{mode}'].astype(np.float64)-z[f'{fid}__{a}__{mode}'].astype(np.float64)
                paths[(model,mode,cat,fid)]=delta
                desc=describe(delta[:,3]);descriptors[(model,mode,cat,fid)]=desc
                prefix=f'{model}__{mode}__{cat}__{fid}'
                for key,value in desc.items():saved[f'{prefix}__{key}']=value
                speed=desc['speed'];total=speed.sum()
                path_rows.append({'model':model,'mode':mode,'category':cat,'family':fid,
                                  'endpoint_norm':float(desc['norm'][-1]),'travel':float(total),
                                  'top_five_travel_fraction':float(np.sort(speed)[-5:].sum()/total) if total else None,
                                  'zero_steps':int(np.sum(speed==0))})
            a=paths[(model,mode,'role_active',fid)];b=paths[(model,mode,'role_passive',fid)]
            saved[f'{model}__{mode}__{fid}__role_angle']=cos(a,b)
        for fid in FIDS:
            native=saved[f'{model}__native__{fid}__role_angle'][:,3]
            fixed=saved[f'{model}__fixed__{fid}__role_angle'][:,3]
            for l in range(29):
                angles.append({'model':model,'family':fid,'stage':l,'native_cosine':clean(float(native[l])),
                               'fixed_cosine':clean(float(fixed[l])),
                               'absolute_change':clean(float(abs(native[l]-fixed[l])))})
            fa=next(f for f in d['families'] if f['family']['id']==fid)
            early={}
            for mode in ['native','fixed']:
                delta=z[f'{fid}__passive_b__{mode}'].astype(np.float64)-z[f'{fid}__active_a__{mode}'].astype(np.float64)
                early[mode]=float(np.linalg.norm(delta[:,0]))
            ratio=None
            if (model,'native','artifact',fid) in paths:
                artifact=paths[(model,'native','artifact',fid)]
                role=paths[(model,'native','role_active',fid)]
                ratio=float(np.linalg.norm(artifact[-1,3])/max(np.linalg.norm(role[-1,3]),1e-30))
            null_rows.append({'model':model,'family':fid,'same_shape_content_max_relative':fa['same_shape_content_max_relative'],
                              'prefix_identical':fa['first_person_active_passive_prefix_identical'],
                              'first_person_native_total_contrast':early['native'],'first_person_fixed_total_contrast':early['fixed'],
                              'artifact_available':fa['null_available'],'artifact_to_native_role_endpoint_ratio':ratio})
    pairs=[]
    for m1,m2 in [(m,m) for m in LOCK['models']]+[tuple(LOCK['models'])]:
        for mode,c1,c2 in itertools.product(['native','fixed'],CATS,CATS):
            for f1,f2 in itertools.product(FIDS,FIDS):
                if m1==m2 and c1==c2 and FIDS.index(f1)>=FIDS.index(f2):continue
                k1,k2=(m1,mode,c1,f1),(m2,mode,c2,f2)
                if k1 not in descriptors or k2 not in descriptors:continue
                pairs.append({'model_a':m1,'model_b':m2,'mode':mode,'category_a':c1,'category_b':c2,
                              'family_a':f1,'family_b':f2,**compare(descriptors[k1],descriptors[k2])})
    table('PAIRWISE.csv',pairs);table('PATHS.csv',path_rows);table('ANGLES.csv',angles);table('NULLS.csv',null_rows)
    np.savez_compressed(HERE/'DESCRIPTORS.npz',**saved)
    summaries={}
    def group(model,mode,a,b,lag,leave_out=None):
        rows=[r for r in pairs if r['model_a']==r['model_b']==model and r['mode']==mode and r['category_a']==a and r['category_b']==b]
        if leave_out:
            def allowed(cat,fid):
                return fid!=leave_out and (cat!='unrelated' or FIDS[(FIDS.index(fid)+1)%8]!=leave_out)
            rows=[r for r in rows if allowed(a,r['family_a']) and allowed(b,r['family_b'])]
        return average([r[f'gram_lag{lag}'] for r in rows])
    for model in LOCK['models']:
        detail=[]
        for mode,lag,omit in itertools.product(['native','fixed'],[1,8],[None]+FIDS):
            within=group(model,mode,'role_active','role_active',lag,omit)
            unrelated=group(model,mode,'role_active','unrelated',lag,omit)
            artifact=group(model,mode,'role_active','artifact',lag,omit)
            detail.append({'mode':mode,'lag':lag,'leave_out':omit,'role_within':within,'role_to_unrelated':unrelated,
                           'role_to_artifact':artifact,'specificity_over_unrelated':within-unrelated if None not in [within,unrelated] else None,
                           'specificity_over_artifact':within-artifact if None not in [within,artifact] else None})
        changes=[]
        for fid in FIDS:
            v=[r['absolute_change'] for r in angles if r['model']==model and r['family']==fid and r['absolute_change'] is not None]
            changes.append({'family':fid,'maximum_angle_change':max(v) if v else None})
        reversals=[]
        for lag,field in itertools.product([1,8],['specificity_over_unrelated','specificity_over_artifact']):
            n=next(r for r in detail if r['mode']=='native' and r['lag']==lag and r['leave_out'] is None)[field]
            f=next(r for r in detail if r['mode']=='fixed' and r['lag']==lag and r['leave_out'] is None)[field]
            reversals.append({'lag':lag,'criterion':field,'native':n,'fixed':f,'sign_reversed':n*f<0 if None not in [n,f] else None})
        summaries[model]={'technical_pass':metadata[model]['technical_pass'],'forwards':metadata[model]['model_forwards'],
                          'primary':detail,'case_angle_changes':changes,'rank_reversals':reversals,
                          'families_angle_change_over_0_05':sum(r['maximum_angle_change'] is not None and r['maximum_angle_change']>0.05 for r in changes),
                          'artifact_within_gram_lag1':group(model,'native','artifact','artifact',1),
                          'artifact_within_gram_lag8':group(model,'native','artifact','artifact',8)}
    cka_rows=[]
    for mode,cat,l in itertools.product(['native','fixed'],CATS,range(29)):
        common=[f for f in FIDS if all((m,mode,cat,f) in paths for m in LOCK['models'])]
        mats=[np.stack([paths[(m,mode,cat,f)][l,3] for f in common]) for m in LOCK['models']] if common else []
        cka_rows.append({'mode':mode,'category':cat,'stage':l,'families':len(common),'linear_cka':cka(*mats) if common else None})
    table('CKA.csv',cka_rows)
    decision={'technical_pass':all(v['technical_pass'] for v in summaries.values()),
              'angle_material_in_both_models':all(v['families_angle_change_over_0_05']>=2 for v in summaries.values()),
              'primary_gap_sign_reversal_any':any(r['sign_reversed'] for v in summaries.values() for r in v['rank_reversals']),
              'answers_scored':0,'causal_patches':0,'population_or_mechanism_claim':False}
    save('SUMMARY.json',{'decision':decision,'models':summaries,'case_count':8,'pairwise_rows':len(pairs)})
    # Figure 1: absolute sensitivity, plus all selected primary comparison groups.
    fig,axs=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for col,model in enumerate(LOCK['models']):
        values=[r['maximum_angle_change'] for r in summaries[model]['case_angle_changes']]
        axs[0,col].bar(FIDS,values,color='#387d98')
        axs[0,col].axhline(.05,color='#bd5528',linestyle='--',label='Frozen 0.05 threshold')
        axs[0,col].set(title=model+' · period role-angle sensitivity',ylabel='Maximum native/fixed cosine change',ylim=(0,max(.055,max(values)*1.1)))
        axs[0,col].tick_params(axis='x',rotation=45);axs[0,col].legend(fontsize=8)
        labels=[('role_active','role_active'),('unrelated','unrelated'),('artifact','artifact'),('role_active','artifact'),('role_active','unrelated')]
        for j,mode in enumerate(['native','fixed']):
            vals=[group(model,mode,a,b,8) for a,b in labels]
            axs[1,col].bar(np.arange(5)+(j-.5)*.36,vals,width=.36,label=mode,color=['#b46b3b','#397e98'][j])
        axs[1,col].set(xticks=range(5),xticklabels=['Role/role','Unrelated/\nunrelated','Null/null','Role/null','Role/unrelated'],
                       ylabel='Mean step-Gram correlation',title='Lag ≥ 8 · whole-dimensional path shapes',ylim=(-1,1))
        axs[1,col].axhline(0,color='#555',lw=.7);axs[1,col].legend(fontsize=8)
    fig.suptitle('Does correcting execution shape change the linguistic comparison?\nEight constructed families; no semantic mechanism inferred',fontsize=13)
    for ext in ['png','pdf']:fig.savefig(HERE/f'BENCHMARK.{ext}',dpi=170)
    plt.close(fig)
    # Figure 2 keeps the scale visible before showing normalized line shapes.
    fig,axs=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for col,model in enumerate(LOCK['models']):
        for cat,label,color in [('role_active','Role reversal','#327d99'),('artifact','Identical-prefix numerical null','#c57538')]:
            vals=np.array([descriptors[(model,'native',cat,f)]['norm'] for f in FIDS if (model,'native',cat,f) in descriptors])
            for v in vals:axs[0,col].plot(range(29),v,color=color,alpha=.12)
            axs[0,col].plot(range(29),vals.mean(0),color=color,label=label)
            for v in vals:
                if v[-1]>0:axs[1,col].plot(range(29),v/v[-1],color=color,alpha=.15)
            good=vals[:,-1]>0
            if good.any():axs[1,col].plot(range(29),(vals[good]/vals[good,-1,None]).mean(0),color=color,label=label)
        axs[0,col].set(title=model,ylabel='Contrast L2 norm · log scale',yscale='log')
        axs[1,col].set(xlabel='Stage (embedding = 0)',ylabel='Contrast norm / its endpoint norm')
        axs[0,col].legend(fontsize=8);axs[1,col].legend(fontsize=8)
    fig.suptitle('Keep magnitude separate from normalized trajectory shape\nThin lines: each family · thick lines: family mean',fontsize=13)
    for ext in ['png','pdf']:fig.savefig(HERE/f'LINES.{ext}',dpi=170)
    plt.close(fig)
    report=['# Trajectory specificity benchmark — 18 October 2026','',
            '## Frozen question and outcome','',
            'Can numerical contrast paths look coherently structured, and does fixed-shape extraction materially change the primary sentence-period linguistic comparison? Eight newly constructed lexical families were declared before inference. No model answer was evaluated, so this is a measurement benchmark rather than a semantic competence or causal mechanism test.',
            '',f'Frozen decision fields: `{json.dumps(decision)}`.','',
            '| Model | Forwards | Content null passed | Cases with angle change >0.05 | Null within-Gram, lag 1 | Null within-Gram, lag 8 |','|---|---:|---|---:|---:|---:|']
    for m,s in summaries.items():report.append(f"| {m} | {s['forwards']} | {s['technical_pass']} | {s['families_angle_change_over_0_05']}/8 | {s['artifact_within_gram_lag1']:.6f} | {s['artifact_within_gram_lag8']:.6f} |")
    report+=['','Every angle, rank-gap sign, leave-one-family-out sensitivity, cross-model pairing and CKA value is retained in SUMMARY.json and the ledgers. Pairs share families, cyclic unrelated contrasts share sentences, and layers share a network; none is an independent replication. Leave-one-out summaries remove any unrelated path involving the omitted family, without redefining that path. They are sensitivity checks, not confidence intervals.',
             '', '## Interpret the scale and specificity separately','',
             'A nonzero numerical null can have a well-defined normalized Gram even though its mathematical input prefix is unchanged. A high correlation describes normalized numerical-error geometry, not shared linguistic computation. Conversely, a small absolute error does not imply that a linguistic conclusion changes. The fixed primary materiality criterion and sign comparisons distinguish these questions; if they remain stable, this study supplies evidence against blaming the main period-level linguistic comparison on this artifact.',
             '', '![Benchmark](BENCHMARK.png)','', '![Lines with absolute and normalized scale](LINES.png)',
             '', '## Complete results and controls','',
             'RAW_<model>.json preserves every prompt, token ID, token site, native length, exclusion and content-null result. STATES_<model>.npz contains embedding plus every block output at four sites. DESCRIPTORS.npz contains every step Gram, norm and speed. PAIRWISE.csv includes all prespecified within/cross-model category pairs; PATHS.csv, ANGLES.csv, NULLS.csv and CKA.csv provide separate views. The primary site is the sentence period; the first-person identical-prefix channel is explicitly checked and saved as a negative control. Undefined zero-vector directions remain missing rather than receiving a fabricated cosine.',
             '', 'The artifact reference is the native 65-minus-64 numerical contrast in both extraction-mode comparison tables. This intentionally asks whether native/fixed linguistic paths resemble the same numerical reference. It is not a claim that the corrected same-shape null stays nonzero; that separate null is reported in NULLS.csv.',
             '', 'Cells reverse roles and active/passive voice but also change lexical order and syntax. They are controlled language contrasts, not isolated semantic variables. CKA has only eight family samples and is descriptive. Native/fixed comparisons share one MLX backend/device; two architectures are not two independent runtimes. No independent evaluator or corpus annotations are claimed. No target-model coordinate alignment, PCA, learned probe or smoothing was fitted.',
             '', '## Prior-work boundary','',
             '[Same Probe, Different Numbers (September 2026)](https://arxiv.org/abs/2609.31796) directly studies activation-monitor robustness to serving numerics and reports that high global representational similarity can coexist with local changes. It is a close precedent; discovering numerical activation differences alone is not a contribution. Our narrower benchmark concerns normalized contrast-path descriptors and exact causal-prefix nulls, without probe training or generated-text divergence. That difference does not establish priority.',
             '', '[Thinking Machines Lab (2025)](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) establishes shape-dependent reductions. [Ding et al. (2021)](https://arxiv.org/abs/2108.01661) and [ReSi (2025)](https://arxiv.org/abs/2408.00531) already ground similarity measures in function and nuisance controls. [DSA](https://arxiv.org/abs/2306.10168) already compares dynamics rather than static geometry; distinct feedforward blocks are not automatically a stationary recurrent system. [Dai et al. (2024)](https://aclanthology.org/2024.emnlp-main.967/) already demonstrates causal binding edits. This benchmark does not identify a new semantic mechanism or validate a novel similarity measure.',
             '', '## Reproduction and stopping','',
             'Use the existing local MLX environment: run_benchmark.py qwen7b and run_benchmark.py llama3b, then analyze.py. The runner refuses a second start. Protocol and source hashes were saved before loading either model. Existing local weights only; no training, model download, paid API or adaptive follow-up. Complete verbose JSON/CSV files are preserved losslessly in gzip for the research branch; full state archives remain local with committed checksums.','']
    (HERE/'REPORT.md').write_text('\n'.join(report))
    for p in sorted(HERE.iterdir()):
        if p.is_file() and (p.suffix=='.csv' or (p.name.startswith('RAW_') and p.suffix=='.json')):
            out=p.with_suffix(p.suffix+'.gz');out.write_bytes(gzip.compress(p.read_bytes(),compresslevel=9,mtime=0))
            assert gzip.decompress(out.read_bytes())==p.read_bytes()
    manifest=[]
    for p in sorted(HERE.iterdir()):
        if p.is_file() and p.name not in ['MANIFEST.json','GITHUB_PRESERVATION.json']:
            b=p.read_bytes();storage='local_tensor_archive' if p.name.startswith('STATES_') else 'research_branch'
            if p.suffix=='.csv' or (p.name.startswith('RAW_') and p.suffix=='.json'):storage='local_expanded_and_gzip_on_branch'
            manifest.append({'path':p.name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),
                             'git_blob_sha':hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),'storage':storage})
    save('MANIFEST.json',{'artifacts':manifest,'source_runtime_model_manifest':'../prefix_shape_audit_2026-10-18/SOURCE_MANIFEST.json',
                          'source_manifest_sha256':sha(HERE.parent/'prefix_shape_audit_2026-10-18/SOURCE_MANIFEST.json')})
    print(json.dumps({'decision':decision,'models':{m:{k:v for k,v in s.items() if k!='primary'} for m,s in summaries.items()}},indent=2))


if __name__=='__main__':main()
