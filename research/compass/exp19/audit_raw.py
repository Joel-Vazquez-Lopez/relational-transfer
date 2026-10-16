#!/usr/bin/env python3
"""Compass 19: stdlib-only replay from archived paired_predictions.jsonl.

Usage: python research/compass/exp19/audit_raw.py /path/to/paired_predictions.jsonl
The full PyTorch readback and synthetic raw outputs are in the standalone archive.
"""
import argparse, collections, json, random
from pathlib import Path

def audit(rows):
    by=collections.defaultdict(list);worlds=collections.defaultdict(list)
    for r in rows:
        key=(r['seed'],r['layout'])
        by[key].append(r);worlds[r['world']].append(r)
        assert r['op_changed']==int(r['base']!=r['op'])
        assert r['cell_changed']==int(r['base']!=r['cell'])
        assert r['op_correct']==int(r['op']==r['new_gold'])
        assert r['cell_correct']==int(r['cell']==r['new_gold'])
        assert r['gold']!=r['new_gold']
    assert len(rows)==1128 and len(worlds)==95
    assert len(by)==6 and all(len(x)==188 for x in by.values())
    metrics={
      'n':len(rows),'world_clusters':len(worlds),
      'instruction_changed':sum(r['op_changed'] for r in rows),
      'operand_changed':sum(r['cell_changed'] for r in rows),
      'instruction_correct':sum(r['op_correct'] for r in rows),
      'operand_correct':sum(r['cell_correct'] for r in rows),
      'both_correct':sum(r['op_correct'] and r['cell_correct'] for r in rows),
      'base_correct':sum(r['base_correct'] for r in rows)
    }
    assert [metrics[k] for k in ('instruction_changed','operand_changed','instruction_correct','operand_correct','both_correct','base_correct')]==[690,302,270,457,201,445]
    null=[r for r in rows if 'null_changed' in r]
    metrics['matched_token_null']={
      'n':len(null),'changed':sum(r['null_changed'] for r in null),
      'relevant_operand_changed':sum(r['cell_changed'] for r in null),
      'worlds':len(set(r['world'] for r in null))}
    assert metrics['matched_token_null']=={'n':546,'changed':104,'relevant_operand_changed':166,'worlds':66}
    rng=random.Random(20261019);keys=list(sorted(worlds));boots=[]
    for _ in range(2000):
        sample=[r for w in (rng.choice(keys) for _ in keys) for r in worlds[w]]
        boots.append(100*sum(r['op_correct']-r['cell_correct'] for r in sample)/len(sample))
    boots.sort()
    metrics['world_bootstrap_correctness_gap_95pp']=[round(boots[50],2),round(boots[1950],2)]
    metrics['warning']='Post-hoc descriptive, seeded models reused on same 95 development worlds; not evidence of a native relational mechanism.'
    return metrics

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('jsonl',type=Path)
    a=p.parse_args()
    rows=[json.loads(line) for line in a.jsonl.read_text().splitlines()]
    print(json.dumps(audit(rows),indent=2))
