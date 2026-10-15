#!/usr/bin/env python3
"""Audit existing cross-family rule agreement table. No inference or label access.
Input: Astra's archived CROSS_ANALYSIS.json, from historical snapshot.
Select rules on full evaluated panel ONLY for descriptive retrospective ceilings.
"""
import argparse,json
def analyze(data):
 out={}
 for model,record in data["models"].items():
  cells=record["heuristic_prediction_agreement"]
  keys=sorted(cells)
  rules=sorted(next(iter(cells.values()))["agreement"])
  assert len(keys)==8, "Expected eight construction/wording cells"
  totals={rule:sum(cells[k]["agreement"][rule] for k in keys) for rule in rules}
  n=sum(cells[k]["n"] for k in keys)
  cell_oracle=sum(max(cells[k]["agreement"].values()) for k in keys)
  out[model]={"answers":n,"single_rule_in_sample_best":max(totals.values()),
              "best_rules":[r for r in rules if totals[r]==max(totals.values())],
              "single_rule_agreements":totals,
              "cellwise_oracle_in_sample":cell_oracle,
              "cellwise_oracle_fraction":cell_oracle/n,
              "single_rule_fraction":max(totals.values())/n,
              "cellwise_selected":{k:{"count":max(cells[k]["agreement"].values()),
                         "rules":[r for r in rules if cells[k]["agreement"][r]==max(cells[k]["agreement"].values())]}
                         for k in keys}}
 return out
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("cross_analysis_json");a=p.parse_args()
 with open(a.cross_analysis_json,encoding="utf8") as f: data=json.load(f)
 print(json.dumps(analyze(data),indent=2))
