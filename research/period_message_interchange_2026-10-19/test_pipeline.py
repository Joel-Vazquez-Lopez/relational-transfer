"""No LLM calls. Production controller rehearsal plus actual attention-hook tensor tests."""
import json,tempfile,types,sys
from pathlib import Path
import numpy as np
import run
from core import HERE,save

def rehearsal():
 realrows=json.loads((HERE/'INPUTS.json').read_text());rows=[]
 for r in realrows:rows.append({**r,'aliases':{'Yes':[0],'No':[1]}})
 class Fake:
  def __init__(self,lock):self.tok=types.SimpleNamespace(decode=lambda x:'Yes' if x[0]==0 else 'No')
  def compute(self,row,graph='baseline',donor=None,port='period',kind='',attention_hook=True,native=False):
   r=row['role'];q=row['query'];pol=1 if r==0 else -1;strength=4 if graph!='neighbor' else .4
   target=pol
   if donor is not None and 'V' in kind:target=float(donor[0,1,0,1 if port=='period' else 0,0])
   s=strength*target*((-1)**q);a=np.zeros((29,3,6),dtype=np.float32);a[:,0:2]=pol;a[15:,2]=s
   kv=np.full((14,2,2,2,3),pol,dtype=np.float32)
   return dict(logits=np.array([s/2,-s/2],dtype=np.float32),trace=a,**({'kv':kv} if attention_hook else {})),[str(r)]*28
 previous=sys.modules.get('engine');saved_here=run.HERE
 with tempfile.TemporaryDirectory() as td:
  p=Path(td);save(p/'LOCK.json',dict(files={},model=dict(files={})));save(p/'FREEZE_RECEIPT.json',dict(synthetic=True));save(p/'INPUTS.json',rows);save(p/'READOUT_CALIBRATION.json',dict(threshold=0))
  run.HERE=p;sys.modules['engine']=types.SimpleNamespace(Engine=Fake)
  try:run.main();decision=json.loads((p/'DECISION.json').read_text());assert decision['passed'] and decision['prediction']=='HV' and decision['attempts']==323
  finally:
   run.HERE=saved_here
   if previous is None:sys.modules.pop('engine',None)
   else:sys.modules['engine']=previous
 return dict(production_controller_323_synthetic_calls=True,prediction_before_evaluation=True,not_empirical_model_calls=True)

def tensor_tests():
 import mlx.core as mx
 import mlx.nn as nn
 from mlx_lm.models.base import scaled_dot_product_attention
 from instrument import Attention
 class FakeAttention(nn.Module):
  def __init__(self):super().__init__();self.n_heads=4;self.n_kv_heads=2;self.scale=2**-.5
  def q_proj(self,x):return x
  def k_proj(self,x):return x[...,:4]
  def v_proj(self,x):return x[...,4:8]
  def o_proj(self,x):return x
  def rope(self,x):return x
 x=mx.array(np.random.default_rng(13).normal(size=(1,8,8)).astype(np.float32));a=FakeAttention();row={'period':3}
 q=x.reshape(1,8,4,2).transpose(0,2,1,3);k=x[...,:4].reshape(1,8,2,2).transpose(0,2,1,3);v=x[...,4:8].reshape(1,8,2,2).transpose(0,2,1,3);mask=mx.array(np.tri(8,dtype=bool))
 ref=scaled_dot_product_attention(q,k,v,cache=None,scale=a.scale,mask=mask).transpose(0,2,1,3).reshape(1,8,8)
 rec={};native=Attention(a,rec,14,row,None,'period','')(x,mask);mx.eval(native,ref,rec['kv']);assert np.array_equal(np.asarray(ref),np.asarray(native))
 donor=np.tile(np.asarray(rec['kv'])[None],(14,1,1,1,1));identity=Attention(a,{},14,row,donor,'period','KV')(x,mask);mx.eval(identity);assert np.array_equal(np.asarray(native),np.asarray(identity))
 donor[:,1,:,1,:]*=-3;edited=Attention(a,{},14,row,donor,'period','V')(x,mask);mx.eval(edited);assert np.array_equal(np.asarray(native)[:,:4],np.asarray(edited)[:,:4]);assert np.max(np.abs(np.asarray(native)[:,4:]-np.asarray(edited)[:,4:]))>1e-4
 # Independent direct-SDPA value replacement verifies axis and projected port.
 vv=mx.concatenate([v[:,:,:3,:],mx.array(donor[0,1,:,1,:])[None,:,None,:],v[:,:,4:,:]],axis=2)
 expected=scaled_dot_product_attention(q,k,vv,cache=None,scale=a.scale,mask=mask).transpose(0,2,1,3).reshape(1,8,8);mx.eval(expected)
 assert np.array_equal(np.asarray(expected)[:,4:],np.asarray(edited)[:,4:])
 # Independent key and joint ports, including GQA dimensions.
 donor[:,0,:,1,:]+=2
 for kind in ['K','KV']:
  actual=Attention(a,{},14,row,donor,'period',kind)(x,mask)
  kk=mx.concatenate([k[:,:,:3,:],mx.array(donor[0,0,:,1,:])[None,:,None,:],k[:,:,4:,:]],axis=2)
  direct=scaled_dot_product_attention(q,kk,vv if kind=='KV' else v,cache=None,scale=a.scale,mask=mask).transpose(0,2,1,3).reshape(1,8,8)
  mx.eval(actual,direct);assert np.array_equal(np.asarray(actual)[:,:4],np.asarray(native)[:,:4]);assert np.array_equal(np.asarray(actual)[:,4:],np.asarray(direct)[:,4:])
 # A blocked neighbor edit is exactly null; opening it makes the control live.
 f=np.tri(8,dtype=bool);f[4:,2]=False
 blocked=Attention(a,{},14,row,None,'neighbor','')(x,mx.array(f));ndonor=donor.copy();ndonor[:,1,:,0,:]*=-5;blockededit=Attention(a,{},14,row,ndonor,'neighbor','V')(x,mx.array(f));openedit=Attention(a,{},14,row,ndonor,'neighbor','V')(x,mask);mx.eval(blocked,blockededit,openedit)
 assert np.array_equal(np.asarray(blocked),np.asarray(blockededit));assert np.max(np.abs(np.asarray(native)[:,4:]-np.asarray(openedit)[:,4:]))>1e-4
 return dict(actual_hook_matches_native_tensor_sdpa=True,identity_exact=True,prefix_queries_unchanged=True,independent_value_port_replacement_exact=True,independent_key_and_joint_ports_exact=True,blocked_neighbor_null_and_open_neighbor_nonnull=True,not_llm_inference=True)
if __name__=='__main__':
 result=dict(rehearsal=rehearsal(),tensor=tensor_tests());save(HERE/'PREFLIGHT_TESTS.json',result);print(json.dumps(result))
