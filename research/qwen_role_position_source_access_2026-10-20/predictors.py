"""Fixed out-of-quartet/out-of-construction predictors and native gates."""
from core import *

def features(row):
    return np.array([row['aa'],row['bb'],row.get('target_aa',row['aa']),row.get('target_bb',row['bb']),row['entropy_a'],row['entropy_b'],row['target_token_index'],row['period']-row['target_token_index'],row['readout_index']-row['period']]+[float(row['position']==p) for p in range(3)],dtype=np.float64)

def crossfit(rows):
    rows=[r for r in rows if r['eligible']]
    predictions=[];fits=[]
    for family in ['p01','p02']:
        for form in FORMS:
            for cut in [7,14,21]:
                train=[r for r in rows if r['family']!=family and r['form']!=form and r['cut']==cut]
                test=[r for r in rows if r['family']==family and r['form']==form and r['cut']==cut]
                assert train and test
                y=np.array([r['authority'] for r in train]);x=np.stack([features(r) for r in train])
                mu=x.mean(axis=0);sd=x.std(axis=0);sd[sd<1e-8]=1
                xx=np.column_stack([np.ones(len(x)),(x-mu)/sd]);penalty=np.eye(xx.shape[1]);penalty[0,0]=0
                weights=np.linalg.solve(xx.T@xx+penalty,xx.T@y)
                role={k:float(np.mean([r['authority'] for r in train if r['role_index']==k])) for k in range(3)}
                position={k:float(np.mean([r['authority'] for r in train if r['position']==k])) for k in range(3)}
                generic=float(y.mean())
                fits.append(dict(family=family,form=form,cut=cut,training_units=[r['unit'] for r in train],test_units=[r['unit'] for r in test],mu=mu.tolist(),sd=sd.tolist(),weights=weights.tolist(),role=role,position=position,generic=generic))
                for r in test:
                    p=dict(role=role[r['role_index']],position=position[r['position']],generic=generic,nuisance=float(np.r_[1.,(features(r)-mu)/sd]@weights))
                    predictions.append(dict(unit=r['unit'],family=family,form=form,cut=cut,observed=r['authority'],predicted=p,squared_errors={k:float((v-r['authority'])**2) for k,v in p.items()}))
    pooled={k:float(np.mean([r['squared_errors'][k] for r in predictions])) for k in ['role','position','generic','nuisance']}
    folds=[]
    for family in ['p01','p02']:
        for form in FORMS:
            group=[r for r in predictions if r['family']==family and r['form']==form]
            loss={k:float(np.mean([r['squared_errors'][k] for r in group])) for k in pooled}
            folds.append(dict(family=family,form=form,points=len(group),mse=loss,role_beats_every_rival=all(loss['role']<loss[k] for k in ['position','generic','nuisance'])))
    residuals=[]
    for cut in [7,14,21]:
        group=[r['authority'] for r in rows if r['cut']==cut]
        residuals.extend(np.array(group)-np.mean(group))
    variation=float(np.std(residuals));best=min(pooled[k] for k in ['position','generic','nuisance'])
    return dict(pooled_mse=pooled,folds=folds,predictions=predictions,fits=fits,within_cut_residual_sd=variation,role_criterion=bool(variation>=.15 and pooled['role']<=.75*best and all(r['role_beats_every_rival'] for r in folds)))

def query_contrast(values,role):
    return float(values[role]-.5*sum(v for i,v in enumerate(values) if i!=role))

def endpoint_contrasts(unit,native):
    q=unit['queries'][unit['role_index']]
    old,new=native[q['a']]['gold'],native[q['b']]['gold']
    aa=[];bb=[]
    for q in unit['queries']:
        a,b=native[q['a']],native[q['b']]
        aa.append(a['name_logits'][new]-a['name_logits'][old])
        bb.append(b['name_logits'][new]-b['name_logits'][old])
    return aa,bb

def native_gate(native,inputs,units):
    index={r['id']:r for r in inputs}
    ordinary=[r for r in native.values() if index[r['id']]['variant']!='sham']
    def valid(r):
        row=index[r['id']]
        return bool(r['terminated'] and r['actual'] is not None and r['generated_tokens']==[row['candidate_ids'][r['actual']]])
    strata=[]
    for family in ['p01','p02']:
        for form in FORMS:
            group=[r for r in ordinary if index[r['id']]['family']==family and index[r['id']]['form']==form]
            strata.append(dict(family=family,form=form,format_valid=sum(valid(r) for r in group),total=len(group),passed=sum(valid(r) for r in group)>=.8*len(group)))
    eligibility={};cells=[];effects=[]
    for unit in units:
        aa,bb=endpoint_contrasts(unit,native);ri=unit['role_index']
        effect=query_contrast(bb,ri)-query_contrast(aa,ri);target_delta=bb[ri]-aa[ri]
        formats=all(valid(native[q[side]]) for q in unit['queries'] for side in ['a','b'])
        eligibility[unit['id']]=bool(formats and effect>=1 and target_delta>=1)
        effects.append(dict(unit=unit['id'],family=unit['family'],form=unit['form'],role_index=ri,aa=aa,bb=bb,interaction=effect,target_delta=target_delta,positive=bool(effect>0 and target_delta>0),format_valid=formats))
    for family in ['p01','p02']:
        for form in FORMS:
            for ri in range(3):
                group=[u for u in units if u['family']==family and u['form']==form and u['role_index']==ri]
                n=sum(eligibility[u['id']] for u in group)
                cells.append(dict(family=family,form=form,role_index=ri,eligible=n,total=len(group),passed=n>=3))
    n=sum(valid(r) for r in ordinary);positive=sum(r['positive'] for r in effects)
    return dict(passed=bool(len(ordinary)==288 and n>=.9*288 and positive>=.9*72 and all(s['passed'] for s in strata+cells)),correct=sum(r['native_correct'] and r['first_token_correct'] for r in ordinary),format_valid=n,total=288,positive_interactions=positive,total_interactions=72,strata=strata,cells=cells,eligibility=eligibility,effects=effects,criterion='Format validity and direct plus query-selective role response; hard native accuracy separately reported and not a gate.')
