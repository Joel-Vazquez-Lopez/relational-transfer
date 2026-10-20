"""CPU structural tests; no language-model initialization or forwards."""
import numpy as np
from replacement import replace

def check(rows,jobs,units,native):
    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    checks=[]
    def test(name,ok):assert ok,name;checks.append(name)
    x=np.arange(60,dtype=np.float32).reshape(1,12,5);v=np.arange(15,dtype=np.float32).reshape(3,5)+100
    actual=np.asarray(replace(mx,mx.array(x),[8,2,5],v));expected=x.copy();expected[:,[8,2,5]]=v
    test('exact_unsorted_three_site_replacement',np.array_equal(actual,expected))
    test('identity_replacement',np.array_equal(np.asarray(replace(mx,mx.array(x),[8,2,5],x[0,[8,2,5]])),x))
    try:replace(mx,mx.array(x),[2,2,5],v);rejected=False
    except AssertionError:rejected=True
    test('duplicate_sites_rejected',rejected)
    test('development_only',all(r['family'] in ['t01','t02'] and r['split']=='development' for r in rows))
    test('all_six_native_baselines_per_unit',all(all(native[i]['native_correct'] for i in u['answer_ids']) for u in units))
    main=[j for j in jobs if j['condition']=='main']
    test('role_vs_mention_predictions_disjoint',all(j['predictions']['role_table']!=j['predictions']['mention_order'] for j in main))
    test('unexcluded_strong_rivals_explicit',all(j['predictions']['role_table']==j['predictions']['syntax_aware_parser']==j['predictions']['complete_future_answer_table'] for j in jobs))
    test('neighbors_outside_every_name',all(len(set(r['neighbors']))==3 and not set(r['neighbors'])&set(r['person_span_tokens']) for r in rows))
    test('balanced_conditions',all(sum(j['condition']==c for j in jobs)==108 for c in ['main','same_role_voice','neighbor','cross_lexical']))
    test('query_independent_donor_context',all(len({j['donor_context'] for j in jobs if (j['unit'],j['direction'],j['condition'])==(u['unit'],d,c)})==1 for u in units for d in ['forward','reverse'] for c in ['main','same_role_voice','neighbor','cross_lexical']))
    test('fixed_depth_panel',all(sum(j['block']==b for j in jobs)==144 for b in [7,14,21]))
    test('same_role_expected_preservation',all(j['predictions']['role_table']==next(r['gold'] for r in rows if r['id']==j['recipient_id']) for j in jobs if j['condition']=='same_role_voice'))
    from analyze import gates,HYP
    def fixture(account):
        return [dict(j,hypothesis_matches={h:bool((h in ['role_table','syntax_aware_parser','complete_future_answer_table'] if account=='role_table' else h==account) and j['condition']!='neighbor') for h in HYP}) for j in jobs]
    role=gates(fixture('role_table'),True)
    test('synthetic_role_gate',role['qualifying_blocks']==[7,14,21] and role['selected_block']==7)
    test('synthetic_mention_gate',gates(fixture('mention_order'),True)['decision']=='NARROW_MENTION_ORDER_STEERING')
    bad=fixture('role_table');bad[0]['hypothesis_matches']['role_table']=False
    test('one_failed_answer_prevents_that_depth',gates(bad,True)['qualifying_blocks']==[14,21])
    test('technical_failure_prevents_qualification',not gates(fixture('role_table'),False)['qualifying_blocks'])
    return dict(passed=True,checks=checks,n=len(checks),language_model_forwards=0)
