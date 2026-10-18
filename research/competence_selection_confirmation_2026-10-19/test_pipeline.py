"""CPU-only synthetic controller rehearsal. No real weight loads or model calls."""
import io,json,tempfile,contextlib
from pathlib import Path
import numpy as np
from core import HERE,save,sha
from analyze import sign_test,contrast
from run import execute
class Fake:
 def __init__(self,model,lock):self.tok=self
 def decode(self,ids):return {1:'Yes',2:'No'}.get(ids[0],'other')
 def compute(self,row,mode,cut,wrapped=True,native=False):
  r,q=row['role'],row['query'];s=8*(-1)**(r+q)
  if cut==0 and mode=='closed':s=4*(-1)**q
  z=np.zeros(4,dtype=np.float32);z[1]=10+s/2;z[2]=10-s/2
  trace=np.zeros((29,2,4),np.float32) if wrapped else np.empty((0,),np.float32)
  hashes=dict(prefix=['known']*28,prefix_without_neighbor=['known']*28)
  arrays=dict(trace=trace)
  if cut==0 and mode=='closed':arrays['null_states']=np.zeros((28,3,4),np.float32)
  return z,arrays,hashes

def check(rows):
 assert sign_test([1]*6)['one_sided_p']==1/64
 assert sign_test([0]*24)['one_sided_p']==1
 assert not sign_test([-1]*24)['one_sided_p']<.05
 assert contrast([dict(role=r,query=q,yes_minus_no=3+7*(-1)**(r+q)) for r in [0,1] for q in [0,1]])==7
 rr=[{**r,'aliases':{'Yes':[1],'No':[2]}} for r in rows]
 with tempfile.TemporaryDirectory(prefix='selection_rehearsal_') as td:
  p=Path(td);save(p/'INPUTS.json',rr);save(p/'LOCK.json',dict(max_per_model=781,max_forwards=1562));save(p/'READOUT_CALIBRATION.json',{m:dict(threshold=0.) for m in ['qwen7b','llama3b']})
  with contextlib.redirect_stdout(io.StringIO()):
   for model in ['qwen7b','llama3b']:execute(model,Fake,p,False)
  budget=(p/'BUDGET.jsonl').read_text().splitlines();assert len(budget)==1562
  decisions={m:json.loads((p/m/'DECISION.json').read_text()) for m in ['qwen7b','llama3b']};assert all(v['all_controls_passed'] for v in decisions.values())
 return dict(kind='SYNTHETIC_ONLY_NOT_EMPIRICAL',real_LLM_forwards=0,rehearsed_controller_calls=1562,decisions=decisions,sign_test_known_cases_passed=True,contrast_known_case_passed=True)
if __name__=='__main__':print(json.dumps(check(json.loads((HERE/'INPUTS.json').read_text())),indent=2))
