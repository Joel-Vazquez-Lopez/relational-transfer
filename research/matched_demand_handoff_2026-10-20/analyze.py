"""Frozen primary and complete outcome tables; saved data only."""
import csv
from core import *

def read(name):return [json.loads(l) for l in (HERE/name).read_text().splitlines()]

def main():
    native={r['id']:r for r in read('NATIVE.jsonl')}
    hybrids=read('HYBRIDS.jsonl')
    units=json.loads((HERE/'UNITS.json').read_text())
    assert len(native)==144 and len(hybrids)==432
    indexed={(r['unit'],r['cut'],r['demand'],r['query_index'],r['hybrid']):r for r in hybrids}
    contrasts=[]
    for cut in [7,14,21]:
        for u in units:
            for q in [q for q in u['queries'] if q['demand']=='role' and q['changed']]:
                oq=next(z for z in u['queries'] if z['demand']=='ordinal' and z['role_index']==q['role_index'])
                result=dict(unit=u['id'],family=u['family'],form=u['form'],role_index=q['role_index'],cut=cut)
                eligible=True
                for demand,question in [('role',q),('ordinal',oq)]:
                    a,b=native[question['a']],native[question['b']]
                    pair=[indexed[u['id'],cut,demand,question['query_index'],h] for h in ['BA','AB']]
                    def margin(r):return r['name_logits'][b['gold']]-r['name_logits'][a['gold']]
                    aa,bb=margin(a),margin(b);ba,ab=map(margin,pair)
                    authority=source_authority(aa,bb,ba,ab)
                    good=a['native_correct'] and b['native_correct'] and authority is not None
                    eligible=eligible and good
                    result.update({demand+'_native_aa':aa,demand+'_native_bb':bb,demand+'_hybrid_ba':ba,demand+'_hybrid_ab':ab,demand+'_denominator':bb-aa,demand+'_authority':authority,demand+'_native_complete':a['native_correct'] and b['native_correct']})
                result['eligible']=eligible
                result['delta']=result['role_authority']-result['ordinal_authority'] if eligible else None
                contrasts.append(result)
    strata=[]
    for cut in [7,14,21]:
        for family in ['u01','u02']:
            for form in FORMS:
                group=[r for r in contrasts if r['cut']==cut and r['family']==family and r['form']==form and r['eligible']]
                strata.append(dict(cut=cut,family=family,form=form,eligible=len(group),mean_delta=float(np.mean([r['delta'] for r in group])) if group else None,positive=sum(r['delta']>0 for r in group),mean_role_authority=float(np.mean([r['role_authority'] for r in group])) if group else None,mean_ordinal_authority=float(np.mean([r['ordinal_authority'] for r in group])) if group else None))
    primary=[r for r in contrasts if r['cut']==14 and r['eligible']]
    ps=[s for s in strata if s['cut']==14]
    correct=sum(r['native_correct'] for r in native.values());first=sum(r['first_token_correct'] for r in native.values())
    competence=correct/144>=.90 and first/144>=.95 and len(primary)>=20 and all(s['eligible']>=4 for s in ps)
    support=competence and all(s['mean_delta']>=.20 for s in ps) and sum(r['delta']>0 for r in primary)>=.80*len(primary)
    decision='INCONCLUSIVE_COMPETENCE' if not competence else 'DEMAND_DEPENDENT_PILOT' if support else 'NO_SUPPORT_FOR_PROLONGED_ROLE_DEPENDENCE'
    groups=[]
    for cut in [7,14,21]:
        for u in units:
            for demand in ['role','ordinal']:
                questions=[q for q in u['queries'] if q['demand']==demand]
                for h in ['BA','AB']:
                    rows=[indexed[u['id'],cut,demand,q['query_index'],h] for q in questions]
                    groups.append(dict(unit=u['id'],family=u['family'],form=u['form'],cut=cut,demand=demand,hybrid=h,answers=[r['generated_text'] for r in rows],terminated=all(r['terminated'] for r in rows),complete_a=all(r['terminated'] and r['actual']==r['gold_a'] for r in rows),complete_b=all(r['terminated'] and r['actual']==r['gold_b'] for r in rows),known_all=all(r['terminated'] and r['actual'] is not None for r in rows)))
    counts=[]
    for cut in [7,14,21]:
        for demand in ['role','ordinal']:
            r=[r for r in hybrids if r['cut']==cut and r['demand']==demand]
            g=[g for g in groups if g['cut']==cut and g['demand']==demand]
            counts.append(dict(cut=cut,demand=demand,answers=len(r),terminated=sum(x['terminated'] for x in r),known_names=sum(x['terminated'] and x['actual'] is not None for x in r),a_matches=sum(x['terminated'] and x['actual']==x['gold_a'] for x in r),b_matches=sum(x['terminated'] and x['actual']==x['gold_b'] for x in r),changed_answers=sum(x['changed'] for x in r),changed_source_matches=sum(x['changed'] and x['terminated'] and x['actual']==(x['gold_b'] if x['hybrid']=='BA' else x['gold_a']) for x in r),invariant_preserved=sum(not x['changed'] and x['terminated'] and x['actual']==x['gold_a'] for x in r),complete_a=sum(x['complete_a'] for x in g),complete_b=sum(x['complete_b'] for x in g),complete_source=sum(x['complete_b'] if x['hybrid']=='BA' else x['complete_a'] for x in g),complete_suffix=sum(x['complete_a'] if x['hybrid']=='BA' else x['complete_b'] for x in g),coherent_source_edges=sum(all(next(x for x in g if x['unit']==u['id'] and x['hybrid']==h)['complete_b' if h=='BA' else 'complete_a'] for h in ['BA','AB']) for u in units),coherent_suffix_edges=sum(all(next(x for x in g if x['unit']==u['id'] and x['hybrid']==h)['complete_a' if h=='BA' else 'complete_b'] for h in ['BA','AB']) for u in units)))
    source_groups={}
    for r in native.values():source_groups.setdefault((r['family'],r['form'],tuple(r['assignment'])),[]).append(r)
    native_by_form=[dict(form=f,correct=sum(r['native_correct'] for r in native.values() if r['form']==f),total=72,role_correct=sum(r['native_correct'] for r in native.values() if r['form']==f and r['demand']=='role'),ordinal_correct=sum(r['native_correct'] for r in native.values() if r['form']==f and r['demand']=='ordinal')) for f in FORMS]
    results=dict(decision=decision,competence_gate=competence,native=dict(correct=correct,total=144,first_token_correct=first,complete_six_question_sources=sum(all(r['native_correct'] for r in g) for g in source_groups.values()),source_cases=24,by_form=native_by_form,failures=[r for r in native.values() if not r['native_correct']]),primary=dict(cut=14,eligible=len(primary),total=24,positive=sum(r['delta']>0 for r in primary),mean_delta=float(np.mean([r['delta'] for r in primary])) if primary else None,strata=ps),profiles=strata,hybrid_counts=counts,three_question_groups=groups,contrasts=contrasts,unexpected_answers=[r for r in hybrids if not r['terminated'] or r['actual'] is None],interpretation='Descriptive two-family pilot; no population p-value, unique algorithm, independent cross-model replication or novelty claim.')
    save(HERE/'RESULTS.json',results)
    with (HERE/'CONTRASTS.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(contrasts[0]));writer.writeheader();writer.writerows(contrasts)
    with (HERE/'ANSWER_LEDGER.csv').open('w') as f:
        fields=['unit','family','form','cut','demand','query_index','role_index','hybrid','changed','gold_a','gold_b','generated_text','terminated','actual','base_call']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(hybrids)
    print(json.dumps({k:results[k] for k in ['decision','competence_gate','primary','hybrid_counts']},indent=2))

if __name__=='__main__':main()
