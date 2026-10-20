"""Post hoc relational-response audit. No model calls or claim of confirmation."""
from core import *

def interaction(values,role):
    return float(values[role]-.5*sum(v for i,v in enumerate(values) if i!=role))

def main():
    native={r['id']:r for r in map(json.loads,(HERE/'NATIVE.jsonl').read_text().splitlines())}
    units=json.loads((HERE/'UNITS.json').read_text());rows=[]
    for u in units:
        q=u['queries'][u['role_index']];old,new=native[q['a']]['gold'],native[q['b']]['gold']
        aa=[];bb=[]
        for q in u['queries']:
            a,b=native[q['a']],native[q['b']]
            aa.append(a['name_logits'][new]-a['name_logits'][old]);bb.append(b['name_logits'][new]-b['name_logits'][old])
        effect=interaction(bb,u['role_index'])-interaction(aa,u['role_index'])
        rows.append(dict(unit=u['id'],family=u['family'],form=u['form'],role_index=u['role_index'],aa=aa,bb=bb,effect=effect))
    rng=np.random.default_rng(20261020);checks=0
    for _ in range(100):
        scores=rng.normal(size=(2,3,4));r=int(rng.integers(3));old,new=0,3
        def effect(s):return interaction(s[1,:,new]-s[1,:,old],r)-interaction(s[0,:,new]-s[0,:,old],r)
        # Arbitrary source×candidate bias, question×candidate fixed bias and
        # source×question offsets all cancel in the triple contrast.
        nuisance=rng.normal(size=(2,1,4))*20+rng.normal(size=(1,3,4))*20+rng.normal(size=(2,3,1))*20
        assert abs(effect(scores)-effect(scores+nuisance))<1e-10;checks+=1
    out=dict(status='EXPLORATORY_AFTER_NATIVE_GATE_FAILURE',model_forwards=0,edges=72,positive=sum(r['effect']>0 for r in rows),at_least_one_logit=sum(r['effect']>=1 for r in rows),mean_effect=sum(r['effect'] for r in rows)/72,nuisance_invariance_checks=checks,rows=rows,interpretation='Query-specific response to one changed role under fixed other answers. Not native accuracy, a new internal variable, causal module evidence, or proof against a syntax-aware copier. The originally frozen native gate remains failed.')
    save(HERE/'EXPLORATORY_INTERACTIONS.json',out);print(json.dumps({k:v for k,v in out.items() if k!='rows'}))

if __name__=='__main__':main()
