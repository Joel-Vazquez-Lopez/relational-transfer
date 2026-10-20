"""Descriptive joint-score audit; never changes the frozen native gate or selection."""
import itertools,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
ROLES=['sender','sent_person','destination_person'];PERMS=list(itertools.permutations(range(3)));EPS=1e-8

def score(s,p):return float(sum(s[q,p[q]] for q in range(3)))
def main():
    a=[json.loads(s) for s in (HERE/'ANSWERS.jsonl').read_text().splitlines()];assert len(a)==432
    rows={r['id']:r for r in json.loads((HERE/'INPUTS.json').read_text())}
    native={r['id']:r for r in map(json.loads,(ROOT/'research/three_role_llama_native_2026-10-20/ANSWERS.jsonl').read_text().splitlines())}
    grouped={}
    for r in a:grouped.setdefault((r['block'],r['unit'],r['direction'],r['condition']),{})[r['query_role']]=r
    assert len(grouped)==144;ledger=[];rng=np.random.default_rng(20261020);maxerror=0
    for (block,unit,direction,condition),d in sorted(grouped.items()):
        assert set(d)==set(ROLES);base=rows[d['sender']['recipient_id']];names=base['names']
        s=np.array([[d[q]['candidate_log_probabilities'][n] for n in names] for q in ROLES],np.float64)
        before=np.array([[native[d[q]['recipient_id']]['candidate_log_probabilities'][n] for n in names] for q in ROLES],np.float64)
        def pred(h):
            vals=[d[q]['predictions'][h] for q in ROLES]
            return tuple(names.index(v) for v in vals) if all(v in names for v in vals) and len(set(vals))==3 else None
        rp,mp=pred('role_table'),pred('mention_order');recipient=tuple(base['assignment'])
        values=np.array([score(s,p) for p in PERMS]);best=int(values.argmax());unique=bool(values[best]-np.delete(values,best).max()>EPS)
        recovered=PERMS[best] if unique else None
        shifted=s+rng.normal(size=(3,1))*100+rng.normal(size=(1,3))*100
        vv=np.array([score(shifted,p) for p in PERMS]);err=float(np.max(np.abs(values-values[0]-(vv-vv[0]))));assert err<EPS;maxerror=max(maxerror,err)
        gap=score(s,rp)-score(s,mp) if rp is not None and mp is not None else None
        donor_recipient=score(s,rp)-score(s,recipient) if rp is not None else None
        before_dr=score(before,rp)-score(before,recipient) if rp is not None else None
        ledger.append(dict(block=block,unit=unit,direction=direction,condition=condition,recipient_ids=[d[q]['recipient_id'] for q in ROLES],names=names,native_three_answer_role=all(d[q]['hypothesis_matches']['role_table'] for q in ROLES),native_three_answer_mention=all(d[q]['hypothesis_matches']['mention_order'] for q in ROLES),native_three_answer_recipient=all(d[q]['native_correct'] for q in ROLES),decoded_assignment=list(recovered) if recovered else None,unique=unique,role_assignment=list(rp) if rp is not None else None,mention_assignment=list(mp) if mp is not None else None,recipient_assignment=list(recipient),structured_role=bool(rp is not None and recovered==rp),structured_mention=bool(mp is not None and recovered==mp),structured_recipient=bool(recovered==recipient),role_minus_mention_assignment_score=gap,role_minus_recipient_assignment_score=donor_recipient,untreated_role_minus_recipient_score=before_dr,change_in_role_minus_recipient_score=donor_recipient-before_dr if donor_recipient is not None else None,score_matrix=s.tolist(),all_permutation_scores=[dict(assignment=list(p),score=float(v)) for p,v in zip(PERMS,values)]))
    summary=[]
    frozen=json.loads((HERE/'RESULTS.json').read_text())
    for block in [7,14,21]:
        for cond in ['main','same_role_voice','neighbor','cross_lexical']:
            group=[r for r in ledger if r['block']==block and r['condition']==cond]
            gaps=[r['role_minus_mention_assignment_score'] for r in group if r['role_minus_mention_assignment_score'] is not None]
            changes=[r['change_in_role_minus_recipient_score'] for r in group if r['change_in_role_minus_recipient_score'] is not None]
            old=next(s for s in frozen['summary'] if s['block']==block and s['condition']==cond)
            if gaps:assert abs(np.mean(gaps)/3-old['mean_role_minus_mention_logprob'])<EPS
            summary.append(dict(block=block,condition=cond,n_three_question_directions=len(group),structured_role=sum(r['structured_role'] for r in group),structured_mention=sum(r['structured_mention'] for r in group),structured_recipient=sum(r['structured_recipient'] for r in group),ambiguous=sum(not r['unique'] for r in group),native_three_answer_role=sum(r['native_three_answer_role'] for r in group),native_three_answer_mention=sum(r['native_three_answer_mention'] for r in group),native_three_answer_recipient=sum(r['native_three_answer_recipient'] for r in group),mean_role_minus_mention_assignment_score=float(np.mean(gaps)) if gaps else None,mean_change_role_minus_recipient=float(np.mean(changes)) if changes else None))
    out=dict(scope='Descriptive saved-score structured readout, not frozen candidate qualification or independent confirmation. Fixed assignment rule motivated by exposed native scores; no new inference or p-value. Combining three future question runs with an external bijection constraint does not show one model-internal solver. Foreign-name donor assignment scores are unavailable under recipient-only candidate scoring.',summary=summary,max_bias_invariance_error=maxerror,ledger=ledger,model_forwards=0)
    (HERE/'STRUCTURED_READOUT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(summary=summary,max_bias_invariance_error=maxerror),indent=2))
if __name__=='__main__':main()
