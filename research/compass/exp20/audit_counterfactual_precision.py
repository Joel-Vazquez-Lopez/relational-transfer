#!/usr/bin/env python3
"""Compass20: verify target-acquisition and off-path counterfactual scores.

Read-only stdlib audit. Requires the two raw JSONL files in the separately
SHA256-verified Compass Investigation 20 archive. No model weights are loaded.
"""
import argparse,collections,json,random
from pathlib import Path
def read(p):
    return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def clustered(rows, fn, B=4000, seed=20261020):
    by=collections.defaultdict(list)
    for r in rows: by[r["world"]].append(r)
    keys=sorted(by); rng=random.Random(seed)
    def mean(selected):
        vals=[fn(x) for w in selected for x in by[w]]
        return sum(vals)/len(vals)
    samples=sorted(mean([rng.choice(keys) for _ in keys]) for _ in range(B))
    return {"mean":round(mean(keys),6),"interval95":[round(samples[int(.025*B)],6),round(samples[int(.975*B)],6)],"worlds":len(keys)}
def audit(original,prob,null):
    assert len(original)==len(prob)==1128 and len(null)==546
    key=lambda r:(r["seed"],r["layout"],r["world"],r["start"],tuple(r["ops"]))
    saved={key(x):x for x in original}
    assert len(saved)==1128
    for x in prob:
        y=saved[key(x)]
        assert all(x[k]==y[k] for k in ("base","op","cell","gold","new_gold"))
    n={
      "baseline_already_new_gold":sum(r["base"]==r["new_gold"] for r in original),
      "op_change_to_gold":sum(r["base"]!=r["op"]==r["new_gold"] for r in original),
      "operand_change_to_gold":sum(r["base"]!=r["cell"]==r["new_gold"] for r in original),
      "op_wrong_change":sum(r["base"]!=r["op"] and r["op"]!=r["new_gold"] for r in original),
      "operand_wrong_change":sum(r["base"]!=r["cell"] and r["cell"]!=r["new_gold"] for r in original),
      "op_correct_unchanged":sum(r["base"]==r["op"]==r["new_gold"] for r in original),
      "operand_correct_unchanged":sum(r["base"]==r["cell"]==r["new_gold"] for r in original)
    }
    assert n==dict(baseline_already_new_gold=240,op_change_to_gold=189,
       operand_change_to_gold=231,op_wrong_change=501,operand_wrong_change=71,
       op_correct_unchanged=81,operand_correct_unchanged=226)
    x={
     "counts":n,
     "op_minus_operand_true_acquisition":clustered(original,lambda r:int(r["base"]!=r["op"]==r["new_gold"])-int(r["base"]!=r["cell"]==r["new_gold"])),
     "op_minus_operand_wrong_change":clustered(original,lambda r:int(r["base"]!=r["op"] and r["op"]!=r["new_gold"])-int(r["base"]!=r["cell"] and r["cell"]!=r["new_gold"])),
     "op_minus_operand_p_target":clustered(prob,lambda r:r["op_prob_target"]-r["cell_prob_target"]),
     "op_minus_base_p_target":clustered(prob,lambda r:r["op_prob_target"]-r["base_prob_target"]),
     "cell_minus_base_p_target":clustered(prob,lambda r:r["cell_prob_target"]-r["base_prob_target"]),
     "relevant_minus_offpath_p_target":clustered(null,lambda r:r["relevant_p_new"]-r["irrelevant_p_new"]),
     "offpath_minus_base_p_target":clustered(null,lambda r:r["irrelevant_p_new"]-r["base_p_new"])
    }
    return x
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--logits",type=Path,required=True)
    p.add_argument("--offpath",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(audit(read(a.original),read(a.logits),read(a.offpath)),indent=2))
