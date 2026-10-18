"""N3 adapter/sampler and unrelated-array baseline API checks; no N4 main data."""
from pathlib import Path
import json,numpy as np,torch
from reference import Reference
from study import Implementation,published_functions
P=Path(__file__).resolve().parent
checks=[]
with torch.no_grad():
 for kind in ['sorting','counting']:
  c=Reference(kind,3);m=Implementation(c,9001);seqs=[[0,1,2],[0,0,2]]
  native=c.trace(seqs);aug=m.trace(seqs)
  for k,(h,u) in enumerate(zip(native,aug)):
   for dose in [0.,.3,1.]:
    a=c.resume(h+dose*(c.clamp_tokens_zero(h)-h),k)
    b=m.resume(m.clamp(u,dose),k)
    err=float((a-b).abs().max());assert err<1e-8
    checks.append({'program':kind,'stage':k,'dose':dose,'max_error':err})
point,logs=published_functions();rng=np.random.default_rng(9001)
a=[rng.normal(size=(40,6)) for _ in range(3)];b=[rng.normal(size=(40,8)) for _ in range(3)]
np.random.seed(9001);import random;random.seed(9001)
result=point(a,b,n=2);assert len(logs)==4 and isinstance(result,(bool,np.bool_))
(P/'SAMPLER_PREFLIGHT.json').write_text(json.dumps({'N3_only':True,'sampler_checks':len(checks),'max_error':max(x['max_error'] for x in checks),'released_code_API_ridge_calls':len(logs),'source_blob':'abd3db7367317c51974a265026efb1649a4ec76a'},indent=2)+'\n')
print('Sampler N3 checks',len(checks),'baseline API passed')
