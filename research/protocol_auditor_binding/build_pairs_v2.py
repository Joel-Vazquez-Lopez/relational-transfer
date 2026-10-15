#!/usr/bin/env python3
"""Open-development same-query binding contrast with family-balanced names.

Never use as sealed confirmation. The semantic-lookup parser is a shortcut control.
"""
import argparse
from collections import Counter,defaultdict
import json, re
from pathlib import Path

VERBS = ('nudged', 'helped', 'praised', 'warned', 'guided', 'followed', 'rescued', 'thanked')
CONDITIONS = ('identity', 'target_role_swap', 'distractor_role_swap', 'predicate_swap')
NAME_RE = re.compile(r'\b[A-Z][a-z]+\d{3}\b')


def render(events, voice, clause_order):
    ordered = events if clause_order == 'first' else list(reversed(events))
    if voice == 'active':
        return '. '.join(f'{actor} {verb} {patient}' for verb,actor,patient in ordered)+'.'
    return '. '.join(f'{patient} was {verb} by {actor}' for verb,actor,patient in ordered)+'.'


def bag(text):
    return Counter(re.findall(r'[A-Za-z0-9]+',text.lower()))


def gold(events, verb, role):
    e = [e for e in events if e[0] == verb]
    assert len(e)==1
    return e[0][1 if role=='actor' else 2]


def make_rows(families=16):
    rows=[]
    for i in range(families):
        names = tuple(f'{prefix}{i:03d}' for prefix in ('Ari','Bea','Cai','Dee'))
        v,w=VERBS[i%len(VERBS)],VERBS[(i+3)%len(VERBS)]
        for vi,voice in enumerate(('active','passive')):
            for oi,order in enumerate(('first','second')):
                for ri,role in enumerate(('actor','patient')):
                    # Equal exposure across family/voice/order/role cells: sorted-name
                    # heuristics cannot solve all clean base worlds as in the v1 design.
                    s=(i+2*vi+oi+ri)%4
                    a,b,c,d=(names[(s+j)%4] for j in range(4))
                    base=[(v,a,b),(w,c,d)]
                    donors={
                        'identity': list(base),
                        'target_role_swap':[(v,b,a),(w,c,d)],
                        'distractor_role_swap':[(v,a,b),(w,d,c)],
                        'predicate_swap':[(w,a,b),(v,c,d)]
                    }
                    for cond,donor in donors.items():
                        b_text=render(base,voice,order)
                        d_text=render(donor,voice,order)
                        assert bag(b_text)==bag(d_text)
                        q=f"For the event '{v}', who is the {role}?"
                        bg=gold(base,v,role); dg=gold(donor,v,role)
                        assert (bg!=dg)==(cond in ('target_role_swap','predicate_swap'))
                        rows.append(dict(pair_id=f'F{i:03d}-{voice}-{order}-{role}-{cond}',
                                         family_id=f'F{i:03d}',
                                         split='open_development' if i<(families*3//4) else 'open_stress',
                                         voice=voice,clause_order=order,role=role,condition=cond,
                                         base_context=b_text,donor_context=d_text,query=q,
                                         base_gold=bg,donor_gold=dg,informative=bg!=dg,
                                         identical_word_multiset=True))
    return rows


def surface_lookup(text,question):
    """Local syntax/verb-string lookup, with no reusable stored relation."""
    q=re.fullmatch(r"For the event '([a-z]+)', who is the (actor|patient)\?",question)
    if q is None: raise ValueError('unsupported question')
    v,role=q.groups()
    for clause in text.rstrip('.').split('. '):
        p=re.fullmatch(r'([A-Z][a-z]+\d{3}) was ([a-z]+) by ([A-Z][a-z]+\d{3})',clause)
        if p:
            patient,verb,actor=p.groups()
        else:
            a=re.fullmatch(r'([A-Z][a-z]+\d{3}) ([a-z]+) ([A-Z][a-z]+\d{3})',clause)
            if a is None: raise ValueError('unsupported clause')
            actor,verb,patient=a.groups()
        if verb==v:
            return actor if role=='actor' else patient
    raise ValueError('verb not found')


def sorted_name_policy(text,question):
    names=sorted(set(NAME_RE.findall(text)))
    if len(names)!=4: raise ValueError('four names expected')
    role='actor' if question.endswith('actor?') else 'patient'
    return names[0 if role=='actor' else 1]


def bag_oracle_ceiling(rows):
    # In-sample fitted gold-aware bound, NOT a deployable predictor or CV score.
    groups=defaultdict(Counter)
    for r in rows:
        for pref in ('base','donor'):
            text=r[f'{pref}_context']
            key=(tuple(sorted(bag(text).items())),r['query'])
            groups[key][r[f'{pref}_gold']]+=1
    return sum(max(c.values()) for c in groups.values()),sum(sum(c.values()) for c in groups.values()),len(groups)


def audit(rows):
    assert len({r['pair_id'] for r in rows})==len(rows)
    for r in rows:
        assert bag(r['base_context'])==bag(r['donor_context'])
        assert (r['base_gold'] != r['donor_gold'])==r['informative']
        assert r['informative'] == (r['condition'] in ('target_role_swap','predicate_swap'))
    counts=Counter()
    per=defaultdict(Counter)
    for r in rows:
        for pref in ('base','donor'):
            text=r[f'{pref}_context']; truth=r[f'{pref}_gold']
            counts['total']+=1
            counts['surface_correct']+=surface_lookup(text,r['query'])==truth
            counts['sorted_correct']+=sorted_name_policy(text,r['query'])==truth
            per[r['condition']]['total']+=1
            per[r['condition']]['sorted_correct']+=sorted_name_policy(text,r['query'])==truth
            per[r['condition']]['surface_correct']+=surface_lookup(text,r['query'])==truth
    bc,bt,ng=bag_oracle_ceiling(rows)
    return dict(pairs=len(rows),families=len({r['family_id'] for r in rows}),
                total_answers=counts['total'],surface_lookup_correct=counts['surface_correct'],
                sorted_name_correct=counts['sorted_correct'],
                bag_plus_question_oracle_ceiling=dict(correct=bc,total=bt,groups=ng,privileged_fitted=True),
                by_condition={k:dict(v) for k,v in sorted(per.items())})


def main():
    p=argparse.ArgumentParser();p.add_argument('--families',type=int,default=16)
    p.add_argument('--output',type=Path,default=Path('open_binding_pairs_v2.jsonl'))
    a=p.parse_args()
    if a.families<4: p.error('families must be >=4')
    rows=make_rows(a.families)
    summary=audit(rows)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('w') as f:
        for r in rows: f.write(json.dumps(r,sort_keys=True)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
