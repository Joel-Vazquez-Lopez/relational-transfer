"""Exploratory localization of the existing failed transfer; no forward pass or refit."""
from pathlib import Path
import hashlib,itertools,json,csv
import numpy as np
P=Path(__file__).resolve().parent
D=P.parent/'learned_intervention_transport_2026-10-18/results'
z=np.load(D/'raw_outputs.npz');allrows=json.loads((D/'cases_held.json').read_text())
rows=[a for a in allrows if a['Q']<0]
mask=np.array([a['Q']<0 for a in allrows]);truth=z['s37__cached__held__identity__effects'][mask,:,1:5,0]
records=[];case_records=[];totals=[]
for source in [11,23]:
 pred=z[f'prediction_s{source}_t37_cached_causal'];delta=pred-truth
 totals.append({'source':source,'target':37,'rmse':float(np.sqrt(np.mean(delta**2))),
  'stage_rmse':np.sqrt(np.mean(delta**2,axis=(0,2))).tolist(),
  'stage_squared_error_fraction':(np.sum(delta**2,axis=(0,2))/np.sum(delta**2)).tolist()})
 for stage in range(3):
  for r,v,o in itertools.product([-1.,1.],repeat=3):
   take=np.array([a['R']==r and a['features'][1]==v and a['features'][0]==o for a in rows])
   e=delta[take,stage]
   records.append({'source':source,'target':37,'stage':stage+1,'R':r,'V':v,'O':o,'F':r*v*o,
    'case_count':int(take.sum()),'entry_count':e.size,'rmse':float(np.sqrt(np.mean(e**2))),'bias':float(e.mean()),'max_abs':float(abs(e).max())})
 for i,row in enumerate(rows):
  for k in range(3):
   for j,dose in enumerate([.125,.375,.625,.875]):
    case_records.append({'source':source,'target':37,'case_id':row['id'],'stage':k+1,'dose':dose,
       'prediction':float(pred[i,k,j]),'truth':float(truth[i,k,j]),'error':float(delta[i,k,j])})
for name,data in [('saved_failure_groups',records),('saved_failure_individual',case_records)]:
 with (P/(name+'.csv')).open('w') as f:
  out=csv.DictWriter(f,fieldnames=list(data[0]));out.writeheader();out.writerows(data)
result={'status':'Exploratory post-outcome audit; original failure and frozen gate unchanged',
 'source_commit':'037d708a23290b4cbb7af7d5e84a441c08aab8bf',
 'raw_sha256':hashlib.sha256((D/'raw_outputs.npz').read_bytes()).hexdigest(),
 'totals':totals,'groups':len(records),'individual_records':len(case_records),
 'limitation':'These are supplied grammar features. Localization does not identify a semantic mechanism or distinguish intervention mismatch from genuine learned-computation differences.'}
(P/'SAVED_FAILURE_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
