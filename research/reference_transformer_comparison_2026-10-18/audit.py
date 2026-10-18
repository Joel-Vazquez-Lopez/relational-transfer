"""Independent NumPy replay of native compiled Transformer interventions."""
from pathlib import Path
import json,hashlib
import numpy as np
P=Path(__file__).resolve().parent;D=P/'results'
meta=json.loads((D/'MODEL_METADATA.json').read_text());inputs=json.loads((D/'INPUTS.json').read_text())
z=np.load(D/'raw_outputs.npz');weights=np.load(D/'weights.npz');checks=[]

def check(name,a,b):
 err=float(np.max(abs(a-b)));checks.append({'name':name,'max_abs_error':err,'pass':err<1e-8})

for kind,m in meta.items():
 def param(key,field):return weights[kind+'__'+key+'__'+field]
 def op(h,k):
  layer=k//2
  if k%2==0:
   p=f'transformer/layer_{layer}/attn/'
   q,kk,v=[(h@param(p+name,'w')+param(p+name,'b')).reshape(h.shape[0],h.shape[1],m['config']['heads'],m['config']['key_size']) for name in ['query','key','value']]
   a=np.einsum('bthd,bshd->bhts',q,kk)/np.sqrt(m['config']['key_size']);a=np.exp(a-a.max(-1,keepdims=True));a/=a.sum(-1,keepdims=True)
   out=np.einsum('bhts,bshd->bthd',a,v).reshape(h.shape[0],h.shape[1],-1)
   return h+out@param(p+'linear','w')+param(p+'linear','b')
  p=f'transformer/layer_{layer}/mlp/'
  return h+np.maximum(0,h@param(p+'linear_1','w')+param(p+'linear_1','b'))@param(p+'linear_2','w')+param(p+'linear_2','b')
 def score(h):return (h[:,1:,m['positive']]-h[:,1:,m['negative']]).mean(1)
 def resume(h,k):
  for j in range(k,2*m['config']['layers']):h=op(h,j)
  return score(h)
 for split,doses in [('dev',[0.,.25,.75,1.]),('held',[0.,.125,.375,.625,.875,1.])]:
  seqs=inputs[split];ids=np.array([[m['token_encoding']['compiler_bos']]+[m['token_encoding'][str(x)] for x in s] for s in seqs])
  h=param('token_embed','embeddings')[ids]+param('pos_embed','embeddings')[np.arange(5)];states=[h]
  for k in range(2*m['config']['layers']):h=op(h,k);states.append(h)
  check(kind+'/'+split+'/baseline',score(h),z[kind+'__'+split+'__native_clean'])
  for k,h in enumerate(states):
   target=h.copy();target[:,1:,m['token_columns']]=0;target[:,1:,m['zero_column']]=1
   for j,dose in enumerate(doses):check(f'{kind}/{split}/stage{k}/dose{dose}',resume(h+dose*(target-h),k),z[kind+'__'+split+'__native_logits'][:,k,j])
if (D/'CONGRUITY_CODE_BASELINE.json').exists():
 calls=json.loads((D/'RIDGE_CALLS.json').read_text());rows=json.loads((D/'CONGRUITY_CODE_BASELINE.json').read_text())
 for row in rows:
  data=calls[row['ridge_log_start']:row['ridge_log_end']];assert len(data)==40
  p=sum(data[i]['r2']>data[i+1]['r2'] for i in range(0,40,2))/20
  result=(p-1.96*np.sqrt(p*(1-p))/np.sqrt(20))>.5
  assert bool(result)==row['reject_same']
result={'checks':len(checks),'passed':sum(c['pass'] for c in checks),'max_error':max(c['max_abs_error'] for c in checks),'failed':[c for c in checks if not c['pass']],
 'baseline_decisions_reconstructed':len(rows) if (D/'CONGRUITY_CODE_BASELINE.json').exists() else 0,'raw_sha256':hashlib.sha256((D/'raw_outputs.npz').read_bytes()).hexdigest()}
(D/'INDEPENDENT_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));assert not result['failed']
