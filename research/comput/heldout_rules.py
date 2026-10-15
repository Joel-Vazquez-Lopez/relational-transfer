#!/usr/bin/env python3
"""Leave-one-name-pair-group-out fit to simple rules; stdlib only."""
import json,sys
from identifiability import rules
NEEDED=('group_id','event_id','participant_first','participant_last','grammatical_subject','question_role','observed_answer')
def evaluate(rows):
    if not rows: raise ValueError('empty')
    for i,r in enumerate(rows):
        miss=[x for x in NEEDED if x not in r]
        if miss: raise ValueError(f'row {i}: missing {miss}')
        if r['observed_answer'] not in (r['participant_first'],r['participant_last']):
            raise ValueError(f'row {i}: invalid candidate response; keep format failures in separate audit')
    groups=sorted({r['group_id'] for r in rows})
    if len(groups)<2: raise ValueError('requires multiple groups')
    names=list(rules(rows[0]))
    folds=[]
    for group in groups:
        train=[r for r in rows if r['group_id']!=group]
        test=[r for r in rows if r['group_id']==group]
        scores={n:sum(rules(r)[n]==r['observed_answer'] for r in train) for n in names}
        peak=max(scores.values())
        winners=[n for n in names if scores[n]==peak]
        detail=[]
        for r in test:
            predictions=set(rules(r)[n] for n in winners)
            detail.append({'event_id':r['event_id'],'all_best_agree':len(predictions)==1,
                           'unanimous_correct':len(predictions)==1 and r['observed_answer'] in predictions,
                           'optimistic_correct':r['observed_answer'] in predictions})
        folds.append({'group':group,'tied_winners':winners,'best_training_count':peak,
                      'heldout_count':len(test),'unanimous_correct':sum(d['unanimous_correct'] for d in detail),
                      'optimistic_correct':sum(d['optimistic_correct'] for d in detail),
                      'events':detail})
    return {'groups':len(groups),'rows':len(rows),'folds':folds,
            'unanimous_correct':sum(f['unanimous_correct'] for f in folds),
            'optimistic_correct':sum(f['optimistic_correct'] for f in folds),
            'warning':'optimistic tally is a bound across tied rules, not realizable held-out accuracy'}
if __name__=='__main__':
    with open(sys.argv[1],encoding='utf8') as f: data=[json.loads(s) for s in f if s.strip()]
    print(json.dumps(evaluate(data),indent=2))
