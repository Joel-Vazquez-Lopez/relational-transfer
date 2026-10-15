"""Counterexample to claiming that the V2 dataset is shortcut-resistant.

The family number is visibly encoded into participant names and rotations are
functions of family, voice, target-event clause index and question role.
This rule never extracts actor/patient positions.
"""
import re
from collections import Counter
from build_pairs_v2 import NAME_RE, make_rows


def family_index_rule(context, question):
    names = sorted(set(NAME_RE.findall(context)))
    if len(names) != 4:
        raise ValueError('four names expected')
    family_number = int(names[0][-3:])
    passive = int(' was ' in context)
    verb_match = re.search(r"event '([a-z]+)'", question)
    if verb_match is None:
        raise ValueError('verb not found in question')
    query_verb = verb_match.group(1)
    clauses = context.rstrip('.').split('. ')
    first_clause = int(re.search(r'\b' + re.escape(query_verb) + r'\b', clauses[0]) is None)
    patient = int('patient?' in question)
    return names[(family_number + 2*passive + first_clause + 2*patient) % 4]


def audit(rows):
    out = Counter()
    for row in rows:
        for pref in ('base', 'donor'):
            out[(pref,row['condition'],'total')] += 1
            out[(pref,row['condition'],'correct')] += (
                family_index_rule(row[f'{pref}_context'],row['query']) == row[f'{pref}_gold'])
    return out


if __name__ == '__main__':
    counts = audit(make_rows(16))
    for pref in ('base', 'donor'):
        for condition in ('identity','target_role_swap','distractor_role_swap','predicate_swap'):
            print(pref,condition,counts[pref,condition,'correct'],counts[pref,condition,'total'])
    print('total',sum(v for (pref,cond,typ),v in counts.items() if typ=='correct'),
          '/',sum(v for (pref,cond,typ),v in counts.items() if typ=='total'))
