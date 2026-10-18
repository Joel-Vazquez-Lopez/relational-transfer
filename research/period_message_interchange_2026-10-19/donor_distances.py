"""Post hoc descriptive donor-strength check; never an adjusted causal endpoint."""
from pathlib import Path
import json,numpy as np
P=Path(__file__).resolve().parent

def main():
 inputs=json.loads((P/'INPUTS.json').read_text());raw=[json.loads(x) for x in (P/'RAW.jsonl').read_text().splitlines()];assert len(raw)==323
 by={(r['id'],r['condition']):r for r in raw if r['kind']=='scientific'};out=[]
 def kv(r):
  with np.load(P/r['payload']) as z:return z['kv'].astype(np.float64)
 for r in inputs:
  opposite=next(x for x in inputs if x['family']==r['family'] and x['voice']==r['voice'] and x['role']!=r['role'] and x['query']==r['query'])
  voice=next(x for x in inputs if x['family']==r['family'] and x['voice']!=r['voice'] and x['role']==r['role'] and x['query']==r['query'])
  for graph,site in [('F',1),('N',0)]:
   b=kv(by[r['id'],graph])
   for typ,donor in [('role',opposite)]+([('voice',voice)] if graph=='F' else []):
    d=kv(by[donor['id'],graph]);delta=d[:,:,:,site]-b[:,:,:,site];normal=b[:,:,:,site]
    for k,kind in enumerate(['K','V']):
     diff=np.linalg.norm(delta[:,k].reshape(14,-1),axis=1);den=np.linalg.norm(normal[:,k].reshape(14,-1),axis=1)
     out.append(dict(id=r['id'],family=r['family'],split=r['split'],voice=r['voice'],query=r['query'],graph=graph,donor_type=typ,port=kind,absolute_norm_by_block=diff.tolist(),relative_norm_by_block=(diff/np.maximum(den,1e-12)).tolist(),mean_absolute_norm=float(diff.mean()),mean_relative_norm=float((diff/np.maximum(den,1e-12)).mean())))
 (P/'DONOR_STRENGTH_POSTHOC.json').write_text(json.dumps(dict(scope='Post hoc exploratory donor-distance description; not magnitude-matched causal interventions, not hypothesis refitting',rows=out),indent=2)+'\n')
 print(json.dumps(dict(rows=len(out),model_forwards=0)))
if __name__=='__main__':main()
