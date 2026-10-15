#!/usr/bin/env python3
"""Open synthetic paired contexts for testing query-token patch specificity.

Development-only design fixtures; no model inference, no causal claims.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

VERBS = ('nudged', 'helped', 'praised', 'warned', 'guided', 'followed', 'rescued', 'thanked')
CONDITIONS = ('identity', 'target_role_swap', 'distractor_role_swap', 'predicate_swap')


def answer(events, verb, role):
    hits = [e for e in events if e[0] == verb]
    assert len(hits) == 1
    return hits[0][1 if role == 'actor' else 2]


def render(events, voice, order):
    ordered = events if order == 'first' else list(reversed(events))
    clauses = []
    for verb, actor, patient in ordered:
        if voice == 'active':
            clauses.append(f'{actor} {verb} {patient}')
        else:
            clauses.append(f'{patient} was {verb} by {actor}')
    return '. '.join(clauses) + '.'


def bag(s):
    return Counter(re.findall(r'[A-Za-z0-9]+', s.lower()))


def make_rows(families=16):
    rows = []
    for i in range(families):
        names = tuple(f'{p}{i:03d}' for p in ('Ari', 'Bea', 'Cai', 'Dee'))
        verbs = (VERBS[i % len(VERBS)], VERBS[(i + 3) % len(VERBS)])
        assert verbs[0] != verbs[1]
        v, w = verbs
        a, b, c, d = names
        base = [(v, a, b), (w, c, d)]
        donors = {
            'identity': list(base),
            'target_role_swap': [(v, b, a), (w, c, d)],
            'distractor_role_swap': [(v, a, b), (w, d, c)],
            'predicate_swap': [(w, a, b), (v, c, d)],
        }
        for voice in ('active', 'passive'):
            for order in ('first', 'second'):
                for role in ('actor', 'patient'):
                    for condition, donor in donors.items():
                        base_text = render(base, voice, order)
                        donor_text = render(donor, voice, order)
                        query = f"For the event '{v}', who is the {role}?"
                        gold_base = answer(base, v, role)
                        gold_donor = answer(donor, v, role)
                        assert bag(base_text) == bag(donor_text)
                        assert (gold_base != gold_donor) == (condition in ('target_role_swap', 'predicate_swap'))
                        rows.append({
                            'pair_id': f'F{i:03d}-{voice}-{order}-{role}-{condition}',
                            'family_id': f'F{i:03d}',
                            'split': 'open_development' if i < (families * 3 // 4) else 'open_stress',
                            'voice': voice, 'clause_order': order, 'role': role,
                            'condition': condition,
                            'base_context': base_text,
                            'donor_context': donor_text,
                            'query': query,
                            'base_gold': gold_base,
                            'donor_gold': gold_donor,
                            'informative': gold_base != gold_donor,
                            'identical_word_multiset': True,
                        })
    return rows


def check(rows):
    ids = [r['pair_id'] for r in rows]
    assert len(ids) == len(set(ids))
    assert set(r['condition'] for r in rows) == set(CONDITIONS)
    split_by_family = {}
    for row in rows:
        assert bag(row['base_context']) == bag(row['donor_context'])
        assert (row['base_gold'] != row['donor_gold']) == row['informative']
        assert row['informative'] == (row['condition'] in ('target_role_swap', 'predicate_swap'))
        split_by_family.setdefault(row['family_id'], set()).add(row['split'])
    assert all(len(v) == 1 for v in split_by_family.values())
    return len(split_by_family)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--families', type=int, default=16)
    parser.add_argument('--output', type=Path, default=Path('open_binding_pairs.jsonl'))
    args = parser.parse_args()
    if args.families < 4:
        parser.error('Need at least 4 families')
    rows = make_rows(args.families)
    n = check(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + '\n')
    print(f'Generated {len(rows)} synthetic development pairs in {n} families, 4 conditions, 2 voices, 2 orders, 2 query roles')


if __name__ == '__main__':
    main()
