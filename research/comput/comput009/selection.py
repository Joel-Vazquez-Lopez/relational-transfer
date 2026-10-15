#!/usr/bin/env python3
"""Training-tie sensitivity; all data synthetic and generator-known."""
import json
from comput009 import generate,PREDICTORS
STAGES=(
 (("canonical",),("as_for_actor","as_for_patient")),
 (("canonical","as_for_actor","as_for_patient"),("regarding_actor","regarding_patient")),
 (("canonical","as_for_actor","as_for_patient","regarding_actor","regarding_patient"),("parenthetical",)),
)
def run(rows):
 results=[]
 for train_variants,test_variants in STAGES:
  train=[r for r in rows if r["variant"] in train_variants]
  test=[r for r in rows if r["variant"] in test_variants]
  scores={name:sum(f(r["sentence"],r["question"])==r["gold"] for r in train) for name,f in PREDICTORS.items()}
  max_score=max(scores.values())
  winners=sorted(k for k,v in scores.items() if v==max_score)
  held={name:sum(PREDICTORS[name](r["sentence"],r["question"])==r["gold"] for r in test) for name in winners}
  results.append({"train_variants":train_variants,"test_variants":test_variants,"train_n":len(train),"test_n":len(test),"training_scores":scores,"train_tied_winners":winners,"heldout_scores_of_tied_winners":held,"heldout_agreement_range":[min(held.values()),max(held.values())]})
 return {"scope":"synthetic; heldout variants known to generator authors", "stages":results}
if __name__=="__main__":print(json.dumps(run(generate()),indent=2))
