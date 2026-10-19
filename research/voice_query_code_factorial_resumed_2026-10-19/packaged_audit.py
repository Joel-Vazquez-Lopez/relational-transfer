"""Read back every original NPZ from verified tar storage and replay target scores."""
import json,hashlib
import numpy as np
from core import HERE,sha,save
from storage import load
index=json.loads((HERE/'PAYLOAD_BUNDLES.json').read_text());raw=[json.loads(s) for s in (HERE/'qwen7b/RAW.jsonl').read_text().splitlines()];rows=json.loads((HERE/'INPUTS.json').read_text())+[json.loads((HERE/'CONTROL_INPUT.json').read_text())];lookup={r['id']:r for r in rows};checks=[]
for b in index['bundles']:assert sha(HERE/b['path'])==b['sha256'];checks.append(dict(check='bundle_sha256',path=b['path'],passed=True))
assert len(raw)==len(index['members'])==1117
for r in raw:
 row=lookup[r['id']];assert index['members'][r['payload']]['sha256']==r['payload_sha256']
 with load(HERE,r) as z:
  logits=z['logits'].astype(np.float64);score=float(np.logaddexp.reduce(logits[row['aliases']['entailed']])-np.logaddexp.reduce(logits[row['aliases']['not_entailed']]));best=int(np.argmax(logits));actual=next((k for k,ids in row['aliases'].items() if best in ids),None)
  assert abs(score-r['s_entail'])<=1e-10 and actual==r['actual'];checks.append(dict(check='original_member_sha_and_score',call=r['call'],passed=True))
assert not list((HERE/'qwen7b').glob('CALL_*.npz'))
save(HERE/'PACKAGED_REPLAY_AUDIT.json',dict(passed=True,original_payloads=1117,bundles=len(index['bundles']),checks=checks,no_model_calls=True,all_original_member_bytes_recoverable=True));print('Verified all1117 original members and scores in',len(index['bundles']),'bundles')
