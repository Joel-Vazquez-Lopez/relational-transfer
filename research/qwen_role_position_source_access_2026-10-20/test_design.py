"""Independent known-generating-rule fixtures and linguistic balance checks."""
import itertools
from core import *
from predictors import crossfit,native_gate,query_contrast

def main():
    rows=json.loads((HERE/'INPUTS.json').read_text());units=json.loads((HERE/'UNITS.json').read_text())
    assert len(rows)==360 and len(units)==72
    for r in rows:
        assert isinstance(r['eos_ids'],list) and r['eos_ids'] and all(isinstance(t,int) for t in r['eos_ids'])
        name=r['gold'];token=r['candidate_ids'][name];base=np.zeros(152064);base[token]=1;end=np.zeros(152064);end[r['eos_ids'][0]]=1
        result=score_generation(r,lambda prefix:end if prefix else base,lambda tokens:' '+name if tokens else '')
        assert result['terminated'] and result['actual']==name and result['generated_tokens']==[token]
    assert len({r['id'] for r in rows})==360
    index={r['id']:r for r in rows}
    for u in units:
        for q in u['queries']:
            a,b=index[q['a']],index[q['b']]
            assert (a['gold']!=b['gold'])==q['changed']
        assert sum(q['changed'] for q in u['queries'])==1
    for f,form in itertools.product(['p01','p02'],FORMS):
        for ri in range(4):
            rr=[r for r in rows if r['family']==f and r['form']==form and r['variant']=='base' and r['query_index']==0]
            assert {r['assignment'][ri] for r in rr}=={0,1,2,3}
    for f in ['p01','p02']:
        for ri in range(3):assert {u['position'] for u in units if u['family']==f and u['role_index']==ri}=={0,1,2}
    assert source_authority(-3,4,4,-3)==1 and source_authority(-3,4,-3,4)==-1
    def fixture(rule):
        result=[]
        profiles={0:[.9,.3,-.8],1:[.5,-.4,-.95],2:[.95,.8,-.2]}
        for family,rotation,form,ri in itertools.product(['p01','p02'],range(4),FORMS,range(3)):
            position=surface_roles(form).index(ri)
            for ci,cut in enumerate([7,14,21]):
                authority=profiles[ri if rule=='role' else position][ci]
                result.append(dict(unit=f'{family}_{rotation}_{form}_{ri}',family=family,rotation=rotation,form=form,role_index=ri,position=position,cut=cut,eligible=True,authority=authority,aa=-10.,bb=10.,entropy_a=.1,entropy_b=.1,target_token_index=10+position,period=20,readout_index=40))
        return result
    role=crossfit(fixture('role'));position=crossfit(fixture('position'))
    assert role['pooled_mse']['role']<1e-25 and role['role_criterion']
    assert position['pooled_mse']['position']<1e-25 and not position['role_criterion']
    for fit in role['fits']:
        assert all(not n.startswith(fit['family']) and f'_{fit["form"]}_' not in n for n in fit['training_units'])
    fake={};copier={}
    for row in rows:
        scores={n:float(10*(n==row['gold'])+100*(n==row['names'][0])) for n in row['names']}
        actual=max(scores,key=scores.get)
        fake[row['id']]=dict(id=row['id'],gold=row['gold'],name_logits=scores,actual=actual,terminated=True,generated_tokens=[row['candidate_ids'][actual]],native_correct=actual==row['gold'],first_token_correct=actual==row['gold'])
        scores={n:float(100*(n==row['names'][0])) for n in row['names']}
        actual=max(scores,key=scores.get)
        copier[row['id']]=dict(id=row['id'],gold=row['gold'],name_logits=scores,actual=actual,terminated=True,generated_tokens=[row['candidate_ids'][actual]],native_correct=actual==row['gold'],first_token_correct=actual==row['gold'])
    evidence=native_gate(fake,rows,units);no_evidence=native_gate(copier,rows,units)
    assert evidence['passed'] and evidence['correct']<.9*288
    assert not no_evidence['passed'] and no_evidence['positive_interactions']==0
    rng=np.random.default_rng(20261020)
    for _ in range(100):
        s=rng.normal(size=(2,3,4));ri=int(rng.integers(3));old,new=0,3
        def effect(t):return query_contrast(t[1,:,new]-t[1,:,old],ri)-query_contrast(t[0,:,new]-t[0,:,old],ri)
        nuisance=rng.normal(size=(2,1,4))*50+rng.normal(size=(1,3,4))*50+rng.normal(size=(2,3,1))*50
        assert abs(effect(s)-effect(s+nuisance))<1e-10
    save(HERE/'CPU_PREFLIGHT.json',dict(passed=True,native_prompts=360,isolated_edges=72,balanced_four_name_rotations=True,role_position_latin_square=True,role_fixture_pooled_mse=role['pooled_mse'],position_fixture_pooled_mse=position['pooled_mse'],predictor_does_not_automatically_favor_roles=True,graded_gate_passes_known_relations_despite_biased_wrong_answers=True,graded_gate_rejects_query_independent_copier=True,interaction_nuisance_invariance_fixtures=100,new_model_forwards=0))

if __name__=='__main__':main()
