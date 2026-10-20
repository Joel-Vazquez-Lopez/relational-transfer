"""Post hoc saved-score lesion: is every query needed for full assignment recovery?

The known bijection can fill the third answer from two questions. Therefore
perfect composite assignment recovery alone does not establish three separate
independently accessible role answers. No model calls and no tuned thresholds.
"""
import itertools
from core import *

PERMS=list(itertools.permutations(range(3)))

def main():
    result={}
    for model,folder in [('qwen7b','three_role_native_development_2026-10-20'),('llama3b','three_role_llama_native_2026-10-20')]:
        path=ROOT/'research'/folder/'ANSWERS.jsonl'
        rows=[json.loads(l) for l in path.read_text().splitlines()]
        groups=[]
        for key in sorted({(r['family'],r['form'],r['source']) for r in rows}):
            group=sorted([r for r in rows if (r['family'],r['form'],r['source'])==key],key=lambda r:ROLES.index(r['query_role']))
            names=sorted(group[0]['candidate_log_probabilities'])
            s=np.array([[r['candidate_log_probabilities'][n] for n in names] for r in group])
            outcomes=[]
            for removed in range(3):
                kept=[i for i in range(3) if i!=removed]
                scores=np.array([sum(s[q,p[q]] for q in kept) for p in PERMS])
                ranks=np.argsort(scores);p=PERMS[int(ranks[-1])];margin=float(scores[ranks[-1]]-scores[ranks[-2]])
                outcomes.append(dict(removed=ROLES[removed],correct=margin>1e-8 and all(names[p[q]]==r['gold'] for q,r in enumerate(group)),margin=margin,prediction=[names[j] for j in p]))
            groups.append(dict(family=key[0],form=key[1],source=key[2],outcomes=outcomes))
        result[model]=dict(input_sha256=sha(path),sources=36,by_removed_role=[dict(removed=role,complete_correct=sum(next(x for x in g['outcomes'] if x['removed']==role)['correct'] for g in groups)) for role in ROLES],every_row_removal_still_correct=sum(all(x['correct'] for x in g['outcomes']) for g in groups),groups=groups)
    # Apply the already-defined score operations descriptively to the complete
    # observed new source cases; recovered response is kept distinct in journals.
    rows=json.loads((HERE/'INPUTS.json').read_text());index={r['id']:r for r in rows}
    native=[json.loads(l) for l in (HERE/'NATIVE.jsonl').read_text().splitlines()]
    recovery=json.loads((HERE/'RECOVERED_FORWARD.json').read_text());row=index[recovery['id']]
    with np.load(HERE/'payloads'/f'CALL_{recovery["parent_call"]:04d}.npz',allow_pickle=False) as z:
        scores=z['logits'].astype(np.float32)
    native.append({**row,**recovery,'name_logits':{n:float(scores[t]) for n,t in row['candidate_ids'].items()}})
    new=[]
    for key in sorted({(r['family'],r['form'],tuple(r['assignment'])) for r in native}):
        for demand in ['role','ordinal']:
            group=sorted([r for r in native if (r['family'],r['form'],tuple(r['assignment']))==key and r['demand']==demand],key=lambda r:r['query_index'])
            assert len(group)==3
            names=index[group[0]['id']]['names'];s=np.array([[r['name_logits'][n] for n in names] for r in group])
            cs=s-s.mean(axis=0);pred=cs.argmax(axis=1)
            margins=[cs[q,w]-max(cs[q,j] for j in range(3) if j!=w) for q,w in enumerate(pred)]
            centered=sum(names[w]==r['gold'] and m>1e-8 for w,m,r in zip(pred,margins,group))
            values=np.array([sum(s[q,p[q]] for q in range(3)) for p in PERMS]);rank=values.argsort();p=PERMS[int(rank[-1])]
            assignment=bool(values[rank[-1]]-values[rank[-2]]>1e-8 and all(names[p[q]]==r['gold'] for q,r in enumerate(group)))
            new.append(dict(family=key[0],form=key[1],assignment=list(key[2]),demand=demand,native_correct=sum(r['native_correct'] for r in group),centered_correct=int(centered),assignment_complete=assignment))
    new_summary=[dict(demand=d,source_cases=15,answers=45,native_correct=sum(r['native_correct'] for r in new if r['demand']==d),centered_correct=sum(r['centered_correct'] for r in new if r['demand']==d),assignment_complete=sum(r['assignment_complete'] for r in new if r['demand']==d)) for d in ['role','ordinal']]
    save(HERE/'ROW_NECESSITY.json',dict(status='EXPLORATORY_SAVED_DATA_ONLY',model_forwards=0,threshold=1e-8,old_panels=result,new_observed_summary=new_summary,new_observed_cases=new,interpretation='A correct composite assignment can be inferred by elimination without a separately correct third-role answer. Lesioning a question removes its score row, not a model component. New-panel recalibration is explicitly post hoc and does not repair its failed frozen native gate.'))
    print(json.dumps(dict(old={m:{k:v[k] for k in ['by_removed_role','every_row_removal_still_correct']} for m,v in result.items()},new_observed_summary=new_summary),indent=2))

if __name__=='__main__':main()
