"""Authored CPU fixtures for full-token scoring, EOS and pair admission."""
import math
import numpy as np
from core import full_score,eligible_pairs


def check(real_rows):
    checks=[]
    def test(name,ok):
        checks.append(dict(check=name,passed=bool(ok)))
        assert ok,name
    row=dict(names=['Agnes','Basil','Celia'],gold='Agnes',eos_ids=[9],candidate_sequences={'Agnes':[[1,2],[1,2]],'Basil':[[1,3]],'Celia':[[4]]})
    def get(prefix):
        z=np.zeros(10)
        if not prefix:z[1]=5;z[4]=-2
        elif tuple(prefix)==(1,):z[3]=6;z[2]=-2
        else:z[9]=10
        return z
    decode=lambda tokens:{(1,2):'Agnes',(1,3):'Basil',(4,):'Celia'}.get(tuple(tokens),'unexpected')
    out=full_score(row,get,decode)
    test('shared_first_token_does_not_make_wrong_full_name_correct',out['actual']=='Basil' and not out['native_correct'] and out['terminated'])
    test('duplicate_token_path_counted_once',len(out['candidate_paths']['Agnes'])==1)
    def lp(prefix,token):
        z=get(prefix);return float(z[token])-math.log(sum(math.exp(float(v)) for v in z))
    test('independent_full_name_probability_sum',abs(out['candidate_log_probabilities']['Agnes']-(lp((),1)+lp((1,),2)))<1e-12)
    right=full_score({**row,'gold':'Basil'},get,decode)
    test('complete_correct_name_and_eos_pass',right['native_correct'] and right['score_correct'])
    def never_eos(prefix):
        z=np.zeros(10);z[1]=5;return z
    truncated=full_score(row,never_eos,lambda tokens:'Agnes')
    test('text_match_without_termination_is_failure',not truncated['terminated'] and not truncated['native_correct'])
    def invalid(prefix):
        z=np.zeros(10);z[7 if not prefix else 9]=10;return z
    bad=full_score(row,invalid,decode)
    test('offformat_answer_retained_as_failure',bad['generated_tokens']==[7] and bad['actual'] is None and not bad['format_compliant'])
    toy=[{**r,'native_correct':True} for r in real_rows]
    pairs=eligible_pairs(toy)
    test('162_six_answer_pairs',len(pairs)==162 and all(p['eligible'] and p['native_correct']==6 for p in pairs))
    toy[0]['native_correct']=False;changed=eligible_pairs(toy)
    test('single_answer_failure_disqualifies_all_nine_affected_pairs',sum(not p['eligible'] for p in changed)==9)
    test('held_out_inputs_absent',len(real_rows)==108 and {r['family'] for r in real_rows}=={'t01','t02'})
    return dict(passed=True,n_checks=len(checks),checks=checks,scope='Synthetic scoring/controller fixtures only; zero model forwards.')
