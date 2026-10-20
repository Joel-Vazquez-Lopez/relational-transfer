"""Frozen causal contrast and cross-fitted prediction; saved outputs only."""
import csv
from core import *
from predictors import crossfit

def read(name):return [json.loads(l) for l in (HERE/name).read_text().splitlines()]

def native_summary():
    rows=json.loads((HERE/'INPUTS.json').read_text());index={r['id']:r for r in rows};native=read('NATIVE.jsonl')
    ordinary=[r for r in native if index[r['id']]['variant']!='sham']
    def good(r):return bool(r['native_correct'] and r['first_token_correct'])
    sources=[]
    for key in sorted({(r['family'],r['rotation'],r['form'],r['variant']) for r in ordinary}):
        group=[r for r in ordinary if (r['family'],r['rotation'],r['form'],r['variant'])==key]
        sources.append(dict(family=key[0],rotation=key[1],form=key[2],variant=key[3],complete_correct=len(group)==3 and all(good(r) for r in group),correct=sum(good(r) for r in group)))
    summary=dict(primary_answers=len(ordinary),primary_correct=sum(good(r) for r in ordinary),primary_full_generation_correct=sum(r['native_correct'] for r in ordinary),primary_source_cases=len(sources),primary_complete_sources=sum(s['complete_correct'] for s in sources),sham_native_answers=len(native)-len(ordinary),sham_native_correct=sum(good(r) for r in native if index[r['id']]['variant']=='sham'),by_family_form_role=[dict(family=f,form=form,role_index=q,total=sum(r['family']==f and r['form']==form and r['query_index']==q for r in ordinary),correct=sum(good(r) for r in ordinary if r['family']==f and r['form']==form and r['query_index']==q)) for f in ['p01','p02'] for form in FORMS for q in range(3)],source_cases=sources,failures=[r for r in native if not good(r)])
    save(HERE/'NATIVE_SUMMARY.json',summary);return summary

def main():
    ns=native_summary();native={r['id']:r for r in read('NATIVE.jsonl')}
    gate=json.loads((HERE/'NATIVE_GATE.json').read_text());assert gate['passed']
    inputs={r['id']:r for r in json.loads((HERE/'INPUTS.json').read_text())}
    units=json.loads((HERE/'UNITS.json').read_text());hybrids=read('HYBRIDS.jsonl');shams=read('SHAM_RESULTS.jsonl')
    assert len(hybrids)==720 and len(shams)==72
    lookup={(r['unit'],r['cut'],r['query_index'],r['hybrid']):r for r in hybrids}
    def entropy(r):
        x=np.array(list(r['name_logits'].values()),dtype=np.float64);p=np.exp(x-x.max());p/=p.sum()
        return float(-np.sum(p*np.log(p+1e-300)))
    profiles=[]
    for u in units:
        qi=u['role_index'];q=u['queries'][qi];a,b=native[q['a']],native[q['b']];row=inputs[q['a']]
        def margin(r):return r['name_logits'][b['gold']]-r['name_logits'][a['gold']]
        for cut in [7,14,21]:
            ba,ab=[lookup[u['id'],cut,qi,h] for h in ['BA','AB']]
            aa,bb=margin(a),margin(b);mba,mab=margin(ba),margin(ab)
            s=source_authority(aa,bb,mba,mab)
            profiles.append(dict(unit=u['id'],family=u['family'],rotation=u['rotation'],form=u['form'],role_index=qi,position=u['position'],cut=cut,eligible=gate['eligibility'][u['id']],authority=s,aa=aa,bb=bb,ba=mba,ab=mab,denominator=bb-aa,entropy_a=entropy(a),entropy_b=entropy(b),target_token_index=row['target_token_index'],period=row['period'],readout_index=len(row['token_ids'])-1))
    prediction=crossfit(profiles)
    q14=[r for r in hybrids if r['cut']==14]
    known=sum(r['terminated'] and r['actual'] is not None for r in q14)
    invariant=[]
    for r in q14:
        if r['changed']:continue
        a,b=native[r['a']],native[r['b']]
        competent=all(x['native_correct'] and x['first_token_correct'] for x in [a,b])
        invariant.append(dict(unit=r['unit'],hybrid=r['hybrid'],query_index=r['query_index'],native_competent=competent,correct=bool(r['terminated'] and r['actual']==r['gold_a'])))
    inv=[r for r in invariant if r['native_competent']]
    sham=[]
    for r in shams:
        a,b=native[r['a']],native[r['b']]
        competent=all(x['native_correct'] and x['first_token_correct'] for x in [a,b])
        sham.append(dict(a=r['a'],b=r['b'],native_competent=competent,correct=bool(r['terminated'] and r['actual']==a['gold'])))
    ss=[r for r in sham if r['native_competent']]
    effect_gate=bool(known>=.9*432 and len(inv)>=.75*288 and len(ss)>=.75*72 and sum(r['correct'] for r in inv)>=.9*len(inv) and sum(r['correct'] for r in ss)>=.9*len(ss))
    coherent=[]
    for u in units:
        for h in ['BA','AB']:
            r=[lookup[u['id'],14,q,h] for q in range(3)]
            coherent.append(dict(unit=u['id'],family=u['family'],form=u['form'],target_role=u['role_index'],hybrid=h,answers=[x['generated_text'] for x in r],source_correct=all(x['terminated'] and x['actual']==(x['gold_b'] if h=='BA' else x['gold_a']) for x in r),suffix_correct=all(x['terminated'] and x['actual']==(x['gold_a'] if h=='BA' else x['gold_b']) for x in r),a_correct=all(x['terminated'] and x['actual']==x['gold_a'] for x in r),b_correct=all(x['terminated'] and x['actual']==x['gold_b'] for x in r)))
    changed=[r for r in q14 if r['changed']]
    outcome='INCONCLUSIVE_INTERVENTION' if not effect_gate else 'ROLE_ACCESS_PILOT' if prediction['role_criterion'] else 'NO_SUPPORT_FOR_ROLE_INDEXED_ACCESS'
    result=dict(decision=outcome,native_gate=gate,native_summary=ns,measurement_gate=dict(passed=effect_gate,known_terminated=known,total=432,invariant_native_competent=len(inv),invariant_correct=sum(r['correct'] for r in inv),invariant_possible=288,sham_native_competent=len(ss),sham_correct=sum(r['correct'] for r in ss),sham_possible=72),primary_prediction=prediction,profiles=profiles,invariants=invariant,shams=sham,complete_three_question_hybrids=coherent,cut14_changed_role=dict(total=len(changed),source_correct=sum(r['terminated'] and r['actual']==(r['gold_b'] if r['hybrid']=='BA' else r['gold_a']) for r in changed),suffix_correct=sum(r['terminated'] and r['actual']==(r['gold_a'] if r['hybrid']=='BA' else r['gold_b']) for r in changed)),unexpected_cut14=[r for r in q14 if not r['terminated'] or r['actual'] is None],unexpected_shams=[r for r in shams if not r['terminated'] or r['actual'] is None],interpretation='Out-of-quartet and out-of-form task-dependent source-access prediction under fixed full-state interventions. No population p-value, internal algorithm, semantic-variable construction time, cross-model family law or novelty claim.')
    save(HERE/'RESULTS.json',result)
    with (HERE/'PROFILES.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(profiles[0]));w.writeheader();w.writerows(profiles)
    with (HERE/'ANSWER_LEDGER.csv').open('w') as f:
        fields=['unit','family','rotation','form','target_role','position','query_index','changed','cut','hybrid','measurement','gold_a','gold_b','first_token','generated_text','terminated','actual','base_call']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(hybrids)
    print(json.dumps(dict(decision=outcome,native_correct=ns['primary_correct'],measurement_gate=result['measurement_gate'],pooled_mse=prediction['pooled_mse'],within_cut_sd=prediction['within_cut_residual_sd'],folds=prediction['folds']),indent=2))

if __name__=='__main__':main()
