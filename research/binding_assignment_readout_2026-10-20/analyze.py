"""Exploratory saved-output-only three-role assignment audit; no model access."""
from pathlib import Path
import argparse,csv,hashlib,itertools,json
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
ROLES=['sender','sent_person','destination_person'];PERMS=list(itertools.permutations(range(3)))
EPS=1e-8

def assignment_scores(s):return np.array([sum(s[q,p[q]] for q in range(3)) for p in PERMS])
def cycles(s):return s[:2,:2]-s[:2,2:3]-s[2:3,:2]+s[2,2]
def test():
    rng=np.random.default_rng(20261020);checks=[]
    def check(n,b):assert b,n;checks.append(n)
    s=rng.normal(size=(3,3));sh=s+rng.normal(size=(3,1))*100+rng.normal(size=(1,3))*100
    v,w=assignment_scores(s),assignment_scores(sh)
    check('all_permutation_gaps_invariant',np.max(np.abs((v-v[0])-(w-w[0])))<EPS)
    check('cycle_contrasts_invariant',np.max(np.abs(cycles(s)-cycles(sh)))<EPS)
    zero=np.array([[2,9,1]]*3);check('pure_name_preference_has_no_binding_discrimination',np.ptp(assignment_scores(zero))<EPS and np.max(np.abs(cycles(zero)))<EPS)
    s=np.eye(3)*4;s[:,1]+=20
    check('native_collapses_while_assignment_recovers',list(s.argmax(1))==[1,1,1] and PERMS[int(assignment_scores(s).argmax())]==(0,1,2))
    patient=np.zeros((3,3));patient[1]=[0,8,0];v=assignment_scores(patient)
    check('single_role_does_not_identify_full_assignment',sum(abs(v-v.max())<EPS)==2)
    return dict(passed=True,n=len(checks),checks=checks,model_forwards=0)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('model',choices=['qwen7b','llama3b']);args=ap.parse_args()
    source=ROOT/'research'/('three_role_native_development_2026-10-20' if args.model=='qwen7b' else 'three_role_llama_native_2026-10-20')
    answers=[json.loads(s) for s in (source/'ANSWERS.jsonl').read_text().splitlines()];assert len(answers)==108
    inputs={r['id']:r for r in json.loads((source/'INPUTS.json').read_text())}
    grouped={}
    for a in answers:
        r=inputs[a['id']];key=(r['family'],tuple(r['assignment']),r['form']);grouped.setdefault(key,{})[r['query_role']]=a
    assert len(grouped)==36
    out=[];rng=np.random.default_rng(20261020);maxerr=0
    for (family,gold,form),d in sorted(grouped.items()):
        assert set(d)==set(ROLES);names=inputs[d['sender']['id']]['names']
        s=np.array([[d[q]['candidate_log_probabilities'][n] for n in names] for q in ROLES],dtype=np.float64)
        scores=assignment_scores(s);best=int(scores.argmax());others=np.delete(scores,best);unique=bool(scores[best]-others.max()>EPS)
        goldidx=PERMS.index(gold);margin=float(scores[goldidx]-np.delete(scores,goldidx).max())
        shifted=s+rng.normal(size=(3,1))*100+rng.normal(size=(1,3))*100
        shifted_scores=assignment_scores(shifted);err=float(np.max(np.abs((scores-scores[0])-(shifted_scores-shifted_scores[0]))));maxerr=max(maxerr,err);assert err<EPS
        patient=s.copy();patient[[0,2]]=0;ps=assignment_scores(patient);patient_ties=int(sum(abs(ps-ps.max())<=EPS))
        residual=s-s.mean(0,keepdims=True)-s.mean(1,keepdims=True)+s.mean()
        row_null=[]
        for order in PERMS:
            sv=assignment_scores(s[list(order)]);unique_null=np.partition(sv,-2)[-1]-np.partition(sv,-2)[-2]>EPS
            row_null.append(bool(unique_null and PERMS[int(sv.argmax())]==gold))
        out.append(dict(family=family,form=form,gold_assignment=list(gold),names=names,source=inputs[d['sender']['id']]['source'],answer_ids=[d[q]['id'] for q in ROLES],generated=[d[q]['generated_text'] for q in ROLES],native_correct=sum(d[q]['native_correct'] for q in ROLES),native_complete=all(d[q]['native_correct'] for q in ROLES),score_matrix=s.tolist(),permutation_scores=[dict(assignment=list(p),score=float(v)) for p,v in zip(PERMS,scores)],decoded_assignment=list(PERMS[best]) if unique else None,unique=unique,structured_correct=bool(unique and PERMS[best]==gold),gold_assignment_margin=margin,cycle_contrasts=cycles(s).tolist(),double_centered_interaction_energy=float(np.sum(residual**2)),single_sent_person_row_best_ties=patient_ties,row_permutation_null_correct_count=sum(row_null)))
    results=dict(model=args.model,scope='Exploratory analysis of exposed fixed development outputs, not preregistered confirmation or causal evidence. One-to-one role constraint is supplied by the authored task. Six assignment scores combine three separate question-conditioned runs.',n_source_units=36,native_correct=sum(a['native_correct'] for a in answers),n_answers=108,native_complete=sum(r['native_complete'] for r in out),structured_complete=sum(r['structured_correct'] for r in out),ambiguous=sum(not r['unique'] for r in out),rescued=sum(r['structured_correct'] and not r['native_complete'] for r in out),lost=sum(r['native_complete'] and not r['structured_correct'] for r in out),max_bias_invariance_error=maxerr,source_files={str((source/f).relative_to(ROOT)):hashlib.sha256((source/f).read_bytes()).hexdigest() for f in ['INPUTS.json','ANSWERS.jsonl','RESULTS.json']},new_model_forwards=0,unit_outcomes=out,cpu_fixtures=test())
    (HERE/f'{args.model}_RESULTS.json').write_text(json.dumps(results,indent=2)+'\n')
    fields=['family','form','source','gold_assignment','decoded_assignment','native_correct','native_complete','structured_correct','gold_assignment_margin','single_sent_person_row_best_ties']
    with (HERE/f'{args.model}_LEDGER.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in out)
    print(json.dumps({k:v for k,v in results.items() if k not in ['unit_outcomes','source_files']},indent=2))
if __name__=='__main__':main()
