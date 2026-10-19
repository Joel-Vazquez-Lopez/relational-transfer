"""CPU synthetic discriminability and complete-controller rehearsal; zero model loads."""
import tempfile,contextlib,io,json
from pathlib import Path
import numpy as np
from core import *
from run import execute
from analyze import analyze,grouped
class Fake:
 account='construction_consistent'
 def __init__(self,lock):self.tok=self
 def decode(self,ids):return {1:'target',2:'other'}.get(ids[0],'outside')
 def compute(self,row,mode,cut,wrapped,native,physical_length):
  truth=row['role']==row['query'];gap=6.
  if row['panel']=='factorial':
   if self.account=='voice_match_penalty':gap=6 if row['question_voice']=='active' else -6
   elif self.account=='literal_label_linked':gap=(6 if row['mapping']==0 else -6) if row['lexicon']=='YN' else 0
   elif self.account=='null':gap=0
  s=(8+(gap if row['voice']=='passive' else 0)) if truth else -8
  if mode=='closed' and cut==0:s=8*(-1)**row['query']
  z=np.zeros(4,np.float32);z[1]=10+s/2;z[2]=10-s/2;trace=np.zeros((29,3,4),np.float32) if wrapped else np.empty((0,),np.float32);hashes=dict(source_prefix=['known']*28,question_prefix=['same']*28)
  if mode=='closed' and cut==0:hashes['disconnected_suffix']='known_null'
  return z,dict(trace=trace),hashes

def check(rows):
 # Distinct, disjoint surrogate score predictions: no hidden semantic claim.
 signatures={}
 for account in ['construction_consistent','voice_match_penalty','literal_label_linked','null']:
  Fake.account=account;engine=Fake({});raw=[]
  for r in rows:
   z,_,_=engine.compute(r,'baseline',28,True,False,LENGTH);rec={**r,**score(z,{**r,'aliases':{'entailed':[1],'not_entailed':[2]}},engine.decode),'mode':'baseline','cut':28};raw.append(rec)
  groups=grouped(raw);pp=[patterns(groups,f) for f in sorted({r['family'] for r in rows})];flags=choose(pp);assert flags['candidate']==(None if account=='null' else account);signatures[account]=flags
 Fake.account='construction_consistent';fake_rows=[{**r,'aliases':{'entailed':[1],'not_entailed':[2]}} for r in rows];control=json.loads((HERE/'CONTROL_INPUT.json').read_text());control={**control,'aliases':{'entailed':[1],'not_entailed':[2]}}
 with tempfile.TemporaryDirectory(prefix='voice_query_code_rehearsal_') as td:
  p=Path(td);save(p/'INPUTS.json',fake_rows);save(p/'CONTROL_INPUT.json',control);save(p/'LOCK.json',dict(files={},max_forwards=1117,output_shapes=dict(vocab_size=4,hidden_size=4)))
  with contextlib.redirect_stdout(io.StringIO()):execute(p,Fake,False);result=analyze(p)
  assert result['held_behavior_replicated'] and result['held_behavior_count']==3 and result['forwards']==1117 and result['all_checks_passed']
 return dict(kind='SYNTHETIC_ONLY_NOT_EMPIRICAL',real_model_forwards=0,disjoint_pattern_fixtures=signatures,controller_calls=1117,independent_replay_checks=result['independent_checks'],known_construction_profile_recovered_on3_of3_held_families=True)
if __name__=='__main__':print(json.dumps(check(json.loads((HERE/'INPUTS.json').read_text())),indent=2))
