"""Exact finite identifiability certificate; synthetic arithmetic, no model."""
import itertools
from core import *

def winners(s):
    assignments=list(itertools.permutations(range(s.shape[1]),s.shape[0]))
    scores=np.array([sum(s[q,p[q]] for q in range(s.shape[0])) for p in assignments])
    best=scores.max()
    return [list(p) for p,v in zip(assignments,scores) if abs(v-best)<1e-8]

def main():
    # The first query supplies literally no candidate information.
    three=np.array([[0.,0.,0.],[0.,10.,0.],[0.,0.,10.]])
    four=np.column_stack([three,np.zeros(3)])
    assert winners(three)==[[0,1,2]]
    assert winners(four)==[[0,1,2],[3,1,2]]
    rng=np.random.default_rng(20261020);checked=0
    for _ in range(100):
        s=rng.normal(size=(3,4))
        for q in range(3):
            lesioned=s.copy();lesioned[q]=0
            assert len(winners(lesioned))>=2
            checked+=1
    proof='For k queried roles and n>k candidates, fix any maximizing assignment of the other k−1 rows after setting one row to zero. At least n−k+1≥2 names remain available for the zero row. All such extensions have identical scores. Thus no unique complete assignment can be inferred without that row. For n=k, only one name remains, so a zero-information query can be filled in by elimination. This is a property of this external assignment objective, not a neural mechanism.'
    save(HERE/'NONEXHAUSTIVE_CERTIFICATE.json',dict(passed=True,model_forwards=0,three_candidate_zero_information_row_winners=winners(three),four_candidate_zero_information_row_winners=winners(four),random_zero_row_checks=checked,proof=proof,limitation='Row necessity is insufficient to prove positive knowledge of a relation. Candidate exclusion, shared salience, syntax-aware copying and external constraints remain competitors. A role-isolated counterfactual must change its target while the other queried answers stay fixed.'))
    print(json.dumps(dict(passed=True,random_zero_row_checks=checked,three_winners=winners(three),four_winners=winners(four))))

if __name__=='__main__':main()
