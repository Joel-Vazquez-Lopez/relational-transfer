#!/usr/bin/env python3
"""COMPUT-027: read-only audit of archived Qwen/Llama factorial JSON (no inference).
Accepts the two owner-published raw JSON inputs; emits only aggregate metadata.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

def parse_surface_roles(sentence):
    patterns = [
        (r'^It was by ([A-Za-z]+) that ([A-Za-z]+) was ([A-Za-z]+)\.$', 'cleft'),
        (r'^([A-Za-z]+) was ([A-Za-z]+) by ([A-Za-z]+)\.$', 'passive'),
        (r'^([A-Za-z]+) ([A-Za-z]+) ([A-Za-z]+)\.$', 'active'),
    ]
    for pattern, kind in patterns:
        m = re.fullmatch(pattern, sentence)
        if not m:
            continue
        a,b,c = m.groups()
        if kind == 'cleft':
            return {'agent':a,'patient':b,'verb':c,'surface':kind}
        if kind == 'passive':
            return {'agent':c,'patient':a,'verb':b,'surface':kind}
        return {'agent':a,'patient':c,'verb':b,'surface':kind}
    raise ValueError('Unknown sentence grammar; abstain instead of guessing')

def audit_one(data):
    if not isinstance(data.get('families'), list):
        raise ValueError('Missing frozen model families')
    rows = {}; recorded = 0; repeated = 0
    for group in data['families']:
        f = group['family']
        if f['a'] == f['b']:
            raise ValueError('Repeated participant')
        for item in group['rows']:
            recorded += 1
            if any(item[name] not in (f['a'],f['b']) for name in ('generated','expected','rival')):
                raise ValueError('Out-of-family name')
            if item['expected'] == item['rival']:
                raise ValueError('Nonunique answer')
            surface = parse_surface_roles(item['sentence'])
            if surface[item['role']] != item['expected'] or surface['verb'] != f['verb']:
                raise ValueError('Source-visible grammar does not match private gold')
            if (item['generated'] == item['expected']) != item['greedy_correct']:
                raise ValueError('Incorrect supplied generation score')
            k=(f['id'],item['r'],item['v'],item['role'],
               'active' if item['v']==0 else item['frame'])
            if k in rows:
                prev = rows[k]
                if (prev['_prompt'],prev['_generated'],prev['_expected'],prev['_margin']) != (
                    item['prompt'],item['generated'],item['expected'],item['score_margin']
                ):
                    raise ValueError('Recorded active-prompt duplication is nonidentical')
                repeated += 1
                continue
            if (item['v']==0) != (surface['surface']=='active'):
                raise ValueError('Voice metadata mismatch')
            first=min((f['a'],f['b']),key=lambda name:item['sentence'].index(name))
            correct=item['generated']==item['expected']
            rows[k]=dict(
                key='|'.join(map(str,k)),family=f['id'],verb=f['verb'],
                frame=k[-1],voice=item['v'],role=item['role'],
                correct=correct,gold_first=item['expected']==first,
                answer_first=item['generated']==first,
                wrong_A=(not correct and item['generated']==f['a']),
                wrong_opposite_role=(not correct and item['generated']==item['rival']),
                grammar_correct=surface[item['role']]==item['expected'],
                score_correct=(item['score_margin']>0)==correct,
                _prompt=item['prompt'],_generated=item['generated'],
                _expected=item['expected'],_margin=item['score_margin'])
    usable=list(rows.values())
    def bucket(predicate=lambda r:True):
        selected=[r for r in usable if predicate(r)]
        return dict(n=len(selected),correct=sum(r['correct'] for r in selected),
                    wrong=sum(not r['correct'] for r in selected))
    def breakdown(field):
        return {str(k):bucket(lambda r:r[field]==k)
                for k in sorted({r[field] for r in usable},key=str)}
    wrong=[r for r in usable if not r['correct']]
    return dict(
        n_recorded=recorded,n_duplicate_active_rows=repeated,
        distinct=bucket(),gold_first=bucket(lambda r:r['gold_first']),
        gold_second=bucket(lambda r:not r['gold_first']),
        by_role=breakdown('role'),by_verb=breakdown('verb'),
        by_frame=breakdown('frame'),by_voice=breakdown('voice'),
        by_family=breakdown('family'),
        errors_choose_first=sum(r['answer_first'] for r in wrong),
        errors_choose_second=sum(not r['answer_first'] for r in wrong),
        errors_choose_family_A=sum(r['wrong_A'] for r in wrong),
        errors_choose_other_role=sum(r['wrong_opposite_role'] for r in wrong),
        template_parser_correct=sum(r['grammar_correct'] for r in usable),
        score_margin_vs_greedy_disagreements=sum(not r['score_correct'] for r in usable),
        _keys={r['key']:r['correct'] for r in usable}
    )

def audit_pair(qwen,llama):
    q=audit_one(qwen)
    l=audit_one(llama)
    a=q.pop('_keys');b=l.pop('_keys')
    if a.keys()!=b.keys():
        raise ValueError('Source conditions differ; cannot compare model errors')
    overlap=Counter((a[k],b[k]) for k in a)
    return dict(qwen=q,llama=l,
                matched_semantic_cells=len(a),
                agreement=dict(both_correct=overlap[True,True],
                               qwen_only_wrong=overlap[False,True],
                               llama_only_wrong=overlap[True,False],
                               both_wrong=overlap[False,False]),
                scope='Correlated synthetic development, saved historical model outputs; no new model inference or mechanism inference')
def main():
    p=argparse.ArgumentParser()
    p.add_argument('qwen_json',type=Path)
    p.add_argument('llama_json',type=Path)
    p.add_argument('--out',type=Path)
    args=p.parse_args()
    result=audit_pair(json.loads(args.qwen_json.read_text()),json.loads(args.llama_json.read_text()))
    payload=json.dumps(result,sort_keys=True,indent=2)+'\n'
    if args.out:args.out.write_text(payload)
    else:print(payload)
if __name__=='__main__':main()
