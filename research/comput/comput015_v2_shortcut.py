#!/usr/bin/env python3
"""COMPUT-015 generator-aware lexical-position shortcut. No reference labels in predictor."""
import argparse,collections,json,re
from pathlib import Path
VERBS=('nudged','helped','praised','warned','guided','followed','rescued','thanked')
NAME_RE=re.compile(r'\b(?:Ari|Bea|Cai|Dee)\d{3}\b')
QUERY_RE=re.compile(r"For the event '([a-z]+)', who is the (actor|patient)\?")
def predict(sentence,question):
    m=QUERY_RE.fullmatch(question)
    if not m or m.group(1) not in VERBS:raise ValueError('Unsupported question')
    verb,role=m.groups()
    names=sorted(set(NAME_RE.findall(sentence)))
    if len(names)!=4:raise ValueError('Must have four unique names')
    voice=int(' was ' in sentence)
    clauses=sentence.rstrip('.').split('. ')
    hits=[i for i,c in enumerate(clauses) if re.search(r'\b'+re.escape(verb)+r'\b',c)]
    if len(clauses)!=2 or len(hits)!=1:raise ValueError('Bad event count / query verb')
    return names[(VERBS.index(verb)+2*voice+hits[0]+2*int(role=='patient'))%4]
def audit(rows):
    by=collections.defaultdict(collections.Counter)
    splits=collections.defaultdict(collections.Counter)
    ids=set()
    for r in rows:
        k=r['pair_id']
        if k in ids:raise ValueError('Duplicate')
        ids.add(k)
        for side in ('base','donor'):
            guess=predict(r[side+'_context'],r['query'])
            gold=r[side+'_gold']
            if gold not in NAME_RE.findall(r[side+'_context']):raise ValueError('Gold name absent from text')
            hit=int(guess==gold)
            by[r['condition']][side+'_correct']+=hit
            by[r['condition']][side+'_n']+=1
            splits[r['split']][side+'_correct']+=hit
            splits[r['split']][side+'_n']+=1
    return {'n_pairs':len(rows),'n_families':len({r['family_id'] for r in rows}),
            'total_correct':sum(v['base_correct']+v['donor_correct'] for v in by.values()),
            'by_condition':{k:dict(v) for k,v in sorted(by.items())},
            'by_split':{k:dict(v) for k,v in sorted(splits.items())},
            'scope':'open development, generator-aware, no model inference; counterfactual failures remain'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('v2_jsonl',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args()
    rows=[json.loads(line) for line in a.v2_jsonl.read_text().splitlines() if line.strip()]
    out=json.dumps(audit(rows),indent=2)+'\n'
    if a.output:a.output.write_text(out)
    else:print(out)
