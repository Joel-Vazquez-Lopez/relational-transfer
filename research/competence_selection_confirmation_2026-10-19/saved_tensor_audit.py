"""Independent shape/anchor/hash verification of the actual saved tensor scope."""
import json,hashlib
import numpy as np
from core import HERE,ROOT,save
lock=json.loads((HERE/'LOCK.json').read_text());rows=json.loads((HERE/'INPUTS.json').read_text());checks=[]
def check(name,passed,**kw):
 checks.append(dict(check=name,passed=bool(passed),**kw));assert passed,(name,kw)
for m in ['qwen7b','llama3b']:
 config=json.loads((ROOT/lock['models'][m]['path']/'config.json').read_text());width=config['hidden_size'];vocab=config['vocab_size'];rr=[json.loads(s) for s in (HERE/m/'RAW.jsonl').read_text().splitlines()];bases={}
 for r in rr:
  with np.load(HERE/m/r['payload']) as z:
   logits=z['logits'];trace=z['trace'].astype(np.float32);depths=z['depths'].tolist()
   check('full_vocab_shape_finite',logits.shape==(vocab,) and np.isfinite(logits).all(),model=m,call=r['call'])
   check('trace_finite',np.isfinite(trace).all(),model=m,call=r['call'])
   if r['wrapped']:
    want=29 if r['mode']=='baseline' else 5
    check('saved_trace_scope',trace.shape==(want,2,width) and depths==(list(range(29)) if want==29 else [0,7,14,21,28]),model=m,call=r['call'])
    if r['mode']=='baseline':check('full_trace_hash_replay',hashlib.sha256(np.ascontiguousarray(trace).tobytes()).hexdigest()==r['full_trace_sha256'],model=m,call=r['call'])
   else:check('unwrapped_no_trace',trace.size==0,model=m,call=r['call'])
   if r['kind']=='scientific' and r['mode']=='baseline':bases[r['id']]=(r,trace.copy())
 for r in rr:
  if r['kind']=='scientific' and r['mode']!='baseline':
   ref,t=bases[r['id']]
   with np.load(HERE/m/r['payload']) as z:
    trace=z['trace'].astype(np.float32);depths=z['depths'].tolist();keep=[(i,d) for i,d in enumerate(depths) if d<=r['cut']]
    error=max(float(np.max(np.abs(trace[i]-t[d]))) for i,d in keep)
   check('saved_before_cut_anchors',error<=1e-6,model=m,call=r['call'],max_error=error)
   check('untouched_full_prefix_hashes',r['prefix_hashes']['prefix']==ref['prefix_hashes']['prefix'],model=m,call=r['call'])
 for fam in sorted({r['family'] for r in rows}):
  for role in [0,1]:
   for voice in ['active','passive']:
    pair=[bases[r['id']][1][:,0] for r in rows if (r['model'],r['family'],r['role'],r['voice'])==(m,fam,role,voice)]
    check('saved_native_period_question_invariance',np.array_equal(*pair),model=m,family=fam,role=role,voice=voice)
save(HERE/'SAVED_TENSOR_AUDIT.json',dict(passed=all(c['passed'] for c in checks),n_checks=len(checks),checks=checks,scope='Every full vocabulary vector; all saved native residual states and nonnative anchors; full-trace hashes replayed only where full trace values are retained. Prefix arrays and nonnative omitted depths cannot be recovered from hashes.'))
print('Saved-tensor audit passed',len(checks),'checks')
