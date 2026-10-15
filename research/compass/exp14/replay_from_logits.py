#!/usr/bin/env python3
"""Recompute Compass 14 whole-world results from saved logits (stdlib only).
Input world_predictions.jsonl is in the verified Compass 14 conversation archive.
This tests a post-hoc output constraint, NOT a native neural mechanism.
"""
import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

PERMS = tuple(itertools.permutations(range(4)))

def map_decode(logits):
    return max(PERMS, key=lambda p: sum(logits[q][p[q]] for q in range(4)))

def hard_expectation(hard, gold):
    best = max(sum(p[q] == hard[q] for q in range(4)) for p in PERMS)
    tied = [p for p in PERMS if sum(p[q] == hard[q] for q in range(4)) == best]
    return sum(p == gold for p in tied) / len(tied)

def audit(rows):
    by_run = defaultdict(list)
    for x in rows:
        by_run[(x['arm'], x['seed'])].append(x)
    out = {}
    for key, items in sorted(by_run.items()):
        items.sort(key=lambda x: x['context_id'])
        assert len(items) == 216
        raw_ok = map_ok = 0
        hard_expected = 0.0
        four_raw = four_map = 0
        for x in items:
            gold = tuple(x['gold'])
            logits = x['logits']
            hard = tuple(max(range(4), key=lambda k: logits[q][k]) for q in range(4))
            mapped = map_decode(logits)
            assert hard == tuple(x['independent_argmax'])
            assert mapped == tuple(x['map_all_permutations'])
            raw_ok += (hard == gold)
            map_ok += (mapped == gold)
            hard_expected += hard_expectation(hard, gold)
            if x['cycle_type'] == '4':
                four_raw += (hard == gold)
                four_map += (mapped == gold)
        out[f'{key[0]}-{key[1]}'] = {
            'raw_whole': raw_ok,
            'map_whole': map_ok,
            'hard_expected_whole': round(hard_expected, 9),
            'four_raw': four_raw,
            'four_map': four_map,
        }
    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('jsonl', type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.jsonl.read_text().splitlines()]
    print(json.dumps(audit(rows), indent=2))
