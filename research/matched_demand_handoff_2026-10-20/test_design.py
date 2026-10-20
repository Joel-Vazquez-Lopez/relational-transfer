"""Meaningful analytical fixtures for authority, matching and forced endpoints."""
from core import *

def main():
    assert source_authority(-4,6,6,-4)==1
    assert source_authority(-4,6,-4,6)==-1
    assert source_authority(-4,6,1,1)==0
    assert source_authority(-.1,.1,6,-4) is None
    assert source_authority(-4,6,16,-4)==2  # do not clip nonlinearity
    assert surface_roles('goal_fronted')==[2,0,1]
    rows=json.loads((HERE/'INPUTS.json').read_text())
    units=json.loads((HERE/'UNITS.json').read_text())
    assert len({r['id'] for r in rows})==144
    assert len(units)==12
    assert sum(q['changed'] for u in units for q in u['queries'])==48
    assert sum(q['changed'] and q['demand']=='role' for u in units for q in u['queries'])==24
    # Counterbalanced same answer/different demand across constructions.
    index={r['id']:r for r in rows}
    for u in units:
        for q in [q for q in u['queries'] if q['demand']=='role']:
            other=next(z for z in u['queries'] if z['demand']=='ordinal' and z['role_index']==q['role_index'])
            for side in ['a','b']:
                a,b=index[q[side]],index[other[side]]
                assert a['gold']==b['gold'] and a['source']==b['source']
                assert len(a['token_ids'])==len(b['token_ids'])
    save(HERE/'CPU_PREFLIGHT.json',dict(passed=True,analytical_authority_fixtures=5,matched_changed_contrasts=24,matched_invariant_contrasts=12,independent_lexical_families=2,independent_population_replication=False))

if __name__=='__main__': main()
