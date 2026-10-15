#!/usr/bin/env python3
"""Comput: strict unsealed event-level model-vs-shortcut audit, Python stdlib."""
import json,re,sys,collections
from pathlib import Path
NEED={'event_id','group_id','predicate','sentence','participant_a','participant_b','question_role','construction','model_answer','gold_answer'}
RULES=('first','last','first_actor','last_actor','construction_position')
def predict(r,policy):
    a,b=r['participant_a'],r['participant_b']; s=r['sentence']
    hits=lambda n: [m.start() for m in re.finditer(r'(?<!\w)'+re.escape(n)+r'(?!\w)',s)]
    if a==b or not hits(a) or not hits(b):raise ValueError('invalid participants or missing surface mentions')
    f,l=(a,b) if hits(a)[0]<hits(b)[0] else (b,a)
    role=r['question_role']
    if policy=='first':return f
    if policy=='last':return l
    if policy=='first_actor':return f if role=='actor' else l
    if policy=='last_actor':return l if role=='actor' else f
    first_actor=r['construction'] in ('active','subject_cleft')
    return (f if first_actor else l) if role=='actor' else (l if first_actor else f)
def evaluate(rows):
    if not rows:raise ValueError('empty')
    seen=set(); records=[]
    for r in rows:
        if NEED-set(r):raise ValueError('missing: '+str(NEED-set(r)))
        if r['question_role'] not in ('actor','patient') or r['construction'] not in ('active','passive','subject_cleft','object_cleft'):raise ValueError('unsupported role/form')
        if r['gold_answer'] not in (r['participant_a'],r['participant_b']):raise ValueError('invalid gold')
        key=(r['event_id'],r['question_role'],r['construction'])
        if key in seen:raise ValueError('duplicate')
        seen.add(key)
        if not isinstance(r['model_answer'],str):raise ValueError('invalid answer')
        valid=r['model_answer'] in (r['participant_a'],r['participant_b'])
        rules={name:predict(r,name) for name in RULES}
        records.append({'event_id':r['event_id'],'group_id':r['group_id'],'construction':r['construction'],
          'format_valid':valid,'model_correct':r['model_answer']==r['gold_answer'],
          'agreement':{name:valid and val==r['model_answer'] for name,val in rules.items()},
          'rule_correct':{name:val==r['gold_answer'] for name,val in rules.items()}})
    def metrics(rr):
        return {'n':len(rr),'format_valid':sum(r['format_valid'] for r in rr),'model_correct':sum(r['model_correct'] for r in rr),
         'model_agreement':{name:sum(r['agreement'][name] for r in rr) for name in RULES},
         'baseline_correct':{name:sum(r['rule_correct'][name] for r in rr) for name in RULES}}
    groups=set(r['group_id'] for r in records)
    if len(groups)<2:raise ValueError('requires two groups')
    folds=[]
    for g in sorted(groups):
        train=[r for r in records if r['group_id']!=g];test=[r for r in records if r['group_id']==g]
        tr={name:sum(r['agreement'][name] for r in train) for name in RULES}
        win=[name for name in RULES if tr[name]==max(tr.values())]
        ts={name:sum(r['agreement'][name] for r in test) for name in win}
        folds.append({'heldout_group':g,'training_ties':win,'heldout_counts':ts,'lower':min(ts.values()),'upper':max(ts.values()),'n':len(test)})
    return {'overall':metrics(records),'by_construction':{f:metrics([r for r in records if r['construction']==f]) for f in sorted(set(r['construction'] for r in records))},'folds':folds,'event_results':records}
if __name__=='__main__':
    data=[json.loads(s) for s in Path(sys.argv[1]).read_text().splitlines() if s.strip()]
    output=json.dumps(evaluate(data),indent=2)+'\n'
    if len(sys.argv)>2:
        if len(sys.argv)!=4 or sys.argv[2]!='--output':raise ValueError('usage: comput010.py source.jsonl [--output output.json]')
        Path(sys.argv[3]).write_text(output)
    else:print(output)
