#!/usr/bin/env python3
"""COMPUT-013: grouped CV of archived first-minus-last margin, stdlib + numpy/pandas.
Original source file required; no inference. Historical panel previously inspected."""
import hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd
SHA="653a33a0a56f34a84c5c39649dda7bdb100990462daea0390cfcb2fcb8d7ea27"
def run(path):
 b=Path(path).read_bytes()
 if hashlib.sha256(b).hexdigest()!=SHA:raise ValueError("historical source checksum mismatch")
 d=pd.DataFrame(json.loads(b)["records"])
 assert len(d)==384
 d["first_margin"]=np.where(d.voice.eq("active"),d.actor_margin,-d.actor_margin)
 d["first_name"]=np.where(d.voice.eq("active"),d.actor,d.patient)
 d["first_choice"]=d.first_token.eq(d.first_name)
 assert np.array_equal(d.first_margin.gt(0).to_numpy(),d.first_choice.to_numpy())
 d["cell"]=d.voice+"/"+d.wording;d["cellverb"]=d.cell+"/"+d.verb
 feats={"constant":[],"cell":["cell"],"cell_plus_verb":["cell","verb"],"cell_by_verb":["cellverb"]}
 output=[]
 for heldunit in ("pair_index","verb"):
  for model,cols in feats.items():
   observed=d.first_margin.to_numpy();pred=np.empty(len(d))
   for group in d[heldunit].unique():
    tr=(d[heldunit]!=group).to_numpy();te=~tr
    if cols:
     z=pd.get_dummies(d.loc[tr,cols].astype(str),dtype=float)
     zz=pd.get_dummies(d.loc[te,cols].astype(str),dtype=float).reindex(columns=z.columns,fill_value=0)
     z=np.column_stack((np.ones(len(z)),z.to_numpy()))
     zz=np.column_stack((np.ones(len(zz)),zz.to_numpy()))
    else:z=np.ones((sum(tr),1));zz=np.ones((sum(te),1))
    coef=np.linalg.solve(z.T@z+1e-6*np.eye(z.shape[1]),z.T@observed[tr])
    pred[te]=zz@coef
   mse=float(np.mean((pred-observed)**2))
   output.append({"holdout":heldunit,"model":model,"rmse":round(mse**.5,5),
                  "q2":round(1-mse/float(np.var(observed)),6),
                  "first_choice_agreement":int(np.sum((pred>0)==(observed>0))),
                  "n":len(d)})
 return {"source_sha256":SHA,"models":output,
         "warning":"Exploratory gold-oriented score-margin analysis; candidate actor/patient margins in source use reference semantic roles. Not independent new-data confirmation."}
if __name__=="__main__":print(json.dumps(run(sys.argv[1]),indent=2))
