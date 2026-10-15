#!/usr/bin/env python3
"""Recreate the exact CAL-01 prompts and separate scorer-only gold. No inference."""
from pathlib import Path
from hashlib import sha256
import argparse, json

CHAINS = [('Alice','Evan','Iris'), ('Bob','Faye','Jude'), ('Cara','Gina','Kira'), ('Dan','Hugo','Liam')]
ROOT = ' '.join(f'{a} was pushed by {h}. {h} helped {t}.' for a,h,t in CHAINS)
EXPECTED_INPUT_SHA = 'c38c348dacadd1c3939a53ab12a6ec5bdbaf54bf44819f26f6aa93c791e87d49'
EXPECTED_GOLD_SHA = '5c54584606f08aad2ac3b0a9a63f215d9fc7ac0d1cede1fb4a8750d9b610fbbc'

def build():
    inputs, gold = [], []
    for q, (a,h,t) in enumerate(CHAINS):
        for context in ('minimal', 'full'):
            events = f'{a} was pushed by {h}. {h} helped {t}.' if context == 'minimal' else ROOT
            for task in ('helper_lookup', 'direct_upstream', 'two_hop'):
                question = {'helper_lookup':f'Who helped {t}?',
                            'direct_upstream':f'Who was pushed by {h}?',
                            'two_hop':f'Who was pushed by the person who helped {t}?'}[task]
                identifier = f'cal01.q{q}.{context}.{task}'
                inputs.append(dict(context=context, document_id='cal01-synthetic-world-0',
                                   family_id='cal01-grammar-0', id=identifier,
                                   prompt=f"Answer with exactly one person's name.\nEvents: {events}\nQuestion: {question}\nAnswer:",
                                   source_kind='synthetic_development', task=task))
                gold.append(dict(context=context, gold=h if task == 'helper_lookup' else a,
                                 id=identifier, q=q, task=task))
    def raw(rows):
        return ''.join(json.dumps(x, sort_keys=True) + '\n' for x in rows).encode()
    return raw(inputs), raw(gold)

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=Path('generated'))
    args=p.parse_args()
    inputs, gold=build()
    for name,data,expected in [('model_input_only.jsonl',inputs,EXPECTED_INPUT_SHA),
                               ('privileged_gold_for_scorer_ONLY.jsonl',gold,EXPECTED_GOLD_SHA)]:
        observed=sha256(data).hexdigest()
        if observed != expected: raise RuntimeError(f'{name} checksum mismatch: {observed}')
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f'Refusing to overwrite files in {args.out}')
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'model_input_only.jsonl').write_bytes(inputs)
    (args.out/'privileged_gold_for_scorer_ONLY.jsonl').write_bytes(gold)
    print('PASS: 24 prompts and separate 24 gold labels; original export SHA-256s match')
