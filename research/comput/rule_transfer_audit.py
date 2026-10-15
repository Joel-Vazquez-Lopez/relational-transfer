#!/usr/bin/env python3
"""Retrospective transfer audit of label-free positional heuristics.
Only input: historical CROSS_ANALYSIS.json (archival per-rule aggregate counts).
No gold actor/patient as candidate predictor features. Rules use sentence order and
question role; mapping question text to role is assumed, NOT autonomously learned.
Construction transfer uses *the same 12 events* with other templates, not fresh data.
"""
import argparse,json
RULES=("always_first","always_last","first_as_actor","last_as_actor")
def audit(data):
 output={}
 for model,rec in data["models"].items():
  cells=rec["heuristic_prediction_agreement"]
  keys=sorted(cells)
  assert len(keys)==8
  assert all(all(rule in cells[k]["agreement"] for rule in RULES) for k in keys)
  overall={r:sum(cells[k]["agreement"][r] for k in keys) for r in RULES}
  transfers={}
  for axis in ("wording","construction"):
   ix=1 if axis=="wording" else 0
   groups=sorted(set(k.split("/")[ix] for k in keys))
   folds=[];minimum=maximum=total=0
   for g in groups:
    held=[k for k in keys if k.split("/")[ix]==g]
    train=[k for k in keys if k not in held]
    scores={r:sum(cells[k]["agreement"][r] for k in train) for r in RULES}
    top=max(scores.values());winners=[r for r in RULES if scores[r]==top]
    test={r:sum(cells[k]["agreement"][r] for k in held) for r in winners}
    low,high=min(test.values()),max(test.values())
    n=sum(cells[k]["n"] for k in held)
    folds.append({"heldout":g,"selected_from_train":winners,"heldout_agreement_by_selected_rule":test,"low":low,"high":high,"n":n})
    minimum+=low;maximum+=high;total+=n
   transfers[axis]={"min_agreement":minimum,"max_agreement":maximum,"n":total,"folds":folds}
  output[model]={"overall_in_sample":overall,"best_single_rule_in_sample":max(overall.values()),"transfer":transfers}
 return output
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("archived_cross_analysis");args=p.parse_args()
 with open(args.archived_cross_analysis,encoding="utf8") as f: data=json.load(f)
 print(json.dumps(audit(data),indent=2))
