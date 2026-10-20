"""Exploratory prior-data audit; written after the new native screen failed.

No model calls, no tuning weights, and no held-out discovery claim. Tests whether
the previously perfect constrained readout needs global assignment decoding.
"""
import itertools
import re
from core import *

PERMS=list(itertools.permutations(range(3)))

def evaluate(groups, matrices):
    entries=[]
    for g,scores in zip(groups,matrices):
        winners=np.argmax(scores,axis=1)
        margins=[float(scores[q,winners[q]]-max(scores[q,j] for j in range(3) if j!=winners[q])) for q in range(3)]
        row_correct=[m>1e-8 and g['names'][int(w)]==r['gold'] for w,m,r in zip(winners,margins,g['rows'])]
        entries.append(dict(family=g['family'],form=g['form'],source=g['source'],answers=[g['names'][int(w)] for w in winners],margins=margins,correct=row_correct,complete=all(row_correct)))
    return dict(correct_answers=sum(sum(x['correct']) for x in entries),answers=len(entries)*3,complete_sources=sum(x['complete'] for x in entries),sources=len(entries),entries=entries)

def main():
    out={}
    datasets={'qwen7b':'three_role_native_development_2026-10-20','llama3b':'three_role_llama_native_2026-10-20'}
    for model,folder in datasets.items():
        path=ROOT/'research'/folder/'ANSWERS.jsonl'
        rows=[json.loads(l) for l in path.read_text().splitlines()]
        keys=sorted({(r['family'],r['form'],r['source']) for r in rows})
        groups=[]
        for family,form,sentence in keys:
            rr=sorted([r for r in rows if (r['family'],r['form'],r['source'])==(family,form,sentence)],key=lambda r:ROLES.index(r['query_role']))
            assert len(rr)==3
            names=sorted(rr[0]['candidate_log_probabilities'])
            mentions=[m.group() for m in re.finditer(r'\b(?:'+'|'.join(names)+r')\b',sentence)]
            assert len(mentions)==3 and len(set(mentions))==3
            s=np.array([[r['candidate_log_probabilities'][n] for n in names] for r in rr],dtype=np.float64)
            groups.append(dict(family=family,form=form,source=sentence,names=names,mentions=mentions,rows=rr,s=s))
        raw=[g['s'] for g in groups]
        lexical=[];position=[];demeaned=[];training=[]
        for g in groups:
            # Out-of-form lexical prior: same names, other two constructions.
            lex_train=[z for z in groups if z['family']==g['family'] and z['form']!=g['form']]
            assert len(lex_train)==12
            lex=np.mean([z['s'].mean(axis=0) for z in lex_train],axis=0)
            lexical.append(g['s']-lex)
            # Out-of-family surface prior, pooled over all three questions.
            pos_train=[z for z in groups if z['family']!=g['family'] and z['form']==g['form']]
            assert len(pos_train)==6
            pos=np.mean([z['s'][:,[z['names'].index(n) for n in z['mentions']]].mean(axis=0) for z in pos_train],axis=0)
            bias=np.array([pos[g['mentions'].index(n)] for n in g['names']])
            position.append(g['s']-bias)
            demeaned.append(g['s']-g['s'].mean(axis=0,keepdims=True))
            training.append(dict(source=g['source'],lexical_training_sources=[z['source'] for z in lex_train],lexical_bias=lex.tolist(),surface_training_sources=[z['source'] for z in pos_train],surface_bias=pos.tolist()))
        methods={name:evaluate(groups,data) for name,data in [('raw_independent',raw),('other_form_lexical_prior',lexical),('other_family_surface_prior',position),('within_source_three_question_column_centering',demeaned)]}
        assignment=[]
        for g in groups:
            values=[sum(g['s'][q,p[q]] for q in range(3)) for p in PERMS]
            rank=np.argsort(values)
            winner=PERMS[int(rank[-1])]
            gap=float(values[int(rank[-1])]-values[int(rank[-2])])
            assignment.append(dict(source=g['source'],correct=gap>1e-8 and all(g['names'][winner[q]]==r['gold'] for q,r in enumerate(g['rows'])),margin=gap))
        out[model]=dict(input=str(path.relative_to(ROOT)),input_sha256=sha(path),native_correct=sum(r['native_correct'] for r in rows),native_complete=sum(all(r['native_correct'] for r in g['rows']) for g in groups),methods=methods,assignment_correct=sum(x['correct'] for x in assignment),assignment_cases=len(assignment),assignment=assignment,training=training)
    result=dict(status='EXPLORATORY_SAVED_DATA_ONLY',model_forwards=0,posthoc=True,threshold=1e-8,fixed_calibration_weight=1,interpretation='Standard calibration and constrained decoding competitors, not new mechanisms. The source-specific centering uses all three test questions and cannot be called independent single-question competence. The other-family prior is transferred, but every underlying score was previously exposed. Only two families and three fixed templates.',models=out)
    save(HERE/'CALIBRATION_RIVALS.json',result)
    print(json.dumps({m:dict(native_correct=v['native_correct'],native_complete=v['native_complete'],assignment_correct=v['assignment_correct'],methods={k:{x:d[x] for x in ['correct_answers','complete_sources']} for k,d in v['methods'].items()}) for m,v in out.items()},indent=2))

if __name__=='__main__':main()
