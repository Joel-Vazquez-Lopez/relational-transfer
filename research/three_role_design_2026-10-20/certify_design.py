"""Exact finite-design certificate and candidate stimuli, never model inference.

All predictions are authored algorithmic accounts, not measurements of LLMs.
Uniform conditional entropies describe the enumerated stimulus design only.
"""
import csv
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PERMS = list(itertools.permutations(range(3)))
ORDERS = {'active': (0, 1, 2), 'passive': (1, 2, 0), 'goal_fronted': (2, 0, 1)}
ROLES = ['sender', 'sent_person', 'destination_person']
QUESTIONS = ['Who sent someone to another person?', 'Who was sent to another person?', 'Who was someone sent to?']
FAMILIES = [('t01', 'development', ['Agnes', 'Basil', 'Celia']), ('t02', 'development', ['Dorian', 'Esther', 'Florian']), ('t03', 'evaluation', ['Greta', 'Hugo', 'Imogen']), ('t04', 'evaluation', ['Jonah', 'Kezia', 'Laurence'])]


def ambiguity(name, assignments, queries):
    buckets = defaultdict(list)
    for state in assignments:
        for q in queries:
            buckets[q, state[q]].append(state)
    hist = Counter(len(v) for v in buckets.values())
    total = len(assignments) * len(queries)
    entropy = sum(len(v)/total * math.log2(len(v)) for v in buckets.values())
    return dict(design=name, n_assignments=len(assignments), n_queries=len(queries), conditioning='question role and its correct answer; fixed lexical set and surface template; uniform enumerated assignments/questions', compatible_assignment_count_histogram=dict(sorted(hist.items())), H_assignment_given_question_answer_bits=entropy, buckets=[dict(query=q, answer=a, compatible_assignments=v) for (q,a),v in sorted(buckets.items())])


def sentence(names, perm, form):
    sender, theme, goal = [names[i] for i in perm]
    if form == 'active':
        return f'{sender} sent {theme} to {goal}.'
    if form == 'passive':
        return f'{theme} was sent to {goal} by {sender}.'
    return f'To {goal}, {sender} sent {theme}.'


def main():
    checks = []

    def check(name, ok):
        checks.append(dict(name=name, passed=bool(ok)))
        if not ok:
            raise AssertionError(name)

    audits = [ambiguity('binary_people', [(0,1),(1,0)], range(2)), ambiguity('two_people_fixed_object', [(0,1,2),(1,0,2)], range(3)), ambiguity('three_people_cyclic_only', [(0,1,2),(1,2,0),(2,0,1)], range(3)), ambiguity('three_people_full_permutation', PERMS, range(3))]
    check('binary_identified_by_answer_plus_query', audits[0]['compatible_assignment_count_histogram'] == {1:4})
    check('fixed_object_only_ambiguous_on_object_question', audits[1]['compatible_assignment_count_histogram'] == {1:4,2:1})
    check('cyclic_three_role_design_still_answer_identified', audits[2]['compatible_assignment_count_histogram'] == {1:9})
    check('full_three_role_design_ambiguous_for_every_query_answer', audits[3]['compatible_assignment_count_histogram'] == {2:9})
    check('one_bit_of_unqueried_assignment_in_design', abs(audits[3]['H_assignment_given_question_answer_bits'] - 1.) < 1e-12)
    rows = []
    # Every donor is a transposition of two roles, preserving the third.
    for recipient in PERMS:
        for swapped in itertools.combinations(range(3),2):
            donor = list(recipient)
            i,j = swapped
            donor[i],donor[j] = donor[j],donor[i]
            donor = tuple(donor)
            invariant = next(k for k in range(3) if k not in swapped)
            for donor_form, recipient_form in itertools.product(ORDERS, repeat=2):
                donor_mentions = [donor[k] for k in ORDERS[donor_form]]
                for query in range(3):
                    slot = ORDERS[recipient_form].index(query)
                    rows.append(dict(recipient=''.join(map(str,recipient)), donor=''.join(map(str,donor)), swapped_roles=''.join(map(str,swapped)), invariant_role=invariant, donor_form=donor_form, recipient_form=recipient_form, query_role=query, native_recipient=recipient[query], reusable_role_table=donor[query], mention_order_table=donor_mentions[slot], fixed_one_answer=donor[invariant], syntax_aware_role_parser=donor[query], complete_future_answer_table=donor[query]))
    check('486_symbolic_prediction_rows', len(rows) == 486)
    pairs = []
    accounts = ['reusable_role_table','mention_order_table','fixed_one_answer','syntax_aware_role_parser','complete_future_answer_table']
    for a,b in itertools.combinations(accounts,2):
        same = [r for r in rows if r['donor_form'] == r['recipient_form']]
        cross = [r for r in rows if r['donor_form'] != r['recipient_form']]
        pairs.append(dict(a=a,b=b,n=len(rows),disagreements=sum(r[a]!=r[b] for r in rows),same_form_n=len(same),same_form_disagreements=sum(r[a]!=r[b] for r in same),cross_form_n=len(cross),cross_form_disagreements=sum(r[a]!=r[b] for r in cross)))
    role_position = next(x for x in pairs if (x['a'],x['b'])==('reusable_role_table','mention_order_table'))
    check('role_vs_mention_order_separates_only_cross_form', role_position['same_form_disagreements']==0 and role_position['cross_form_disagreements']==324)
    role_answer = next(x for x in pairs if (x['a'],x['b'])==('reusable_role_table','fixed_one_answer'))
    check('role_table_vs_fixed_one_answer_324_disagreements', role_answer['disagreements']==324)
    parser = next(x for x in pairs if (x['a'],x['b'])==('reusable_role_table','syntax_aware_role_parser'))
    check('strong_parser_remains_equivalent', parser['disagreements']==0)
    future = next(x for x in pairs if (x['a'],x['b'])==('reusable_role_table','complete_future_answer_table'))
    check('complete_future_answer_table_remains_equivalent', future['disagreements']==0)
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in ['recipient','donor','swapped_roles','donor_form','recipient_form'])].append(r)
    check('162_three_query_units', len(groups)==162 and all(len(v)==3 for v in groups.values()))
    check('two_changed_one_unchanged_in_every_unit', all(sum(r['reusable_role_table']!=r['native_recipient'] for r in v)==2 and sum(r['query_role']==r['invariant_role'] and r['reusable_role_table']==r['native_recipient'] for r in v)==1 for v in groups.values()))
    prompts = []
    for fid,split,names in FAMILIES:
        for perm in PERMS:
            for form in ORDERS:
                text = sentence(names,perm,form)
                for q in range(3):
                    prompts.append(dict(id=f'{fid}_{"".join(map(str,perm))}_{form}_q{q}',family=fid,split=split,assignment=list(perm),form=form,source=text,question=QUESTIONS[q],query_role=ROLES[q],gold=names[perm[q]],system="Read the sentence and answer the question about that event. Return only the person's name, copied exactly from the sentence. Do not explain.",annotation='AI-authored exact template; grammatical sender/theme/destination labels, not human-validated thematic-role judgments; hypothetical future evaluation, no LLM outcomes collected'))
    check('216_candidate_prompts_72_sources', len(prompts)==216 and len({r['source'] for r in prompts})==72)
    check('unique_candidate_ids', len({r['id'] for r in prompts})==216)
    check('balanced_query_answer_identity', all(Counter(r['gold'] for r in prompts if r['family']==fid and r['query_role']==role)==Counter({n:6 for n in names}) for fid,_,names in FAMILIES for role in ROLES))
    with (HERE/'RIVAL_PREDICTIONS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (HERE/'CANDIDATE_INPUTS.json').write_text(json.dumps(prompts,indent=2)+'\n')
    result=dict(scope='Finite design audit and symbolic rival predictions only; no model observations, no estimated internal entropy, no significance test or model inference.',passed=True,n_checks=len(checks),checks=checks,ambiguity_audits=audits,pairwise_predictions=pairs,independent_lexical_families=4,candidate_prompts=216,family_level_power='Exploratory only; four authored lexical families do not support a broad population/family claim.',strong_claim_gate='NO-GO for uniquely semantic/algorithmic interpretation: role table, syntax-aware parser and complete future-answer table have identical predictions.',operational_claim_gate='Logical witnesses exist against the declared fixed one-answer and recipient-grammar mention-order accounts; native competence, valid interventions and generalization remain untested.')
    (HERE/'DESIGN_CERTIFICATE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,checks=len(checks),ambiguity=[{k:v for k,v in x.items() if k!='buckets'} for x in audits],pairwise_predictions=pairs,candidate_prompts=len(prompts)),indent=2))

if __name__ == '__main__':
    main()
