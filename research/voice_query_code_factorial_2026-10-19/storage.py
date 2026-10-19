"""Lossless staged/bundled payload access; original NPZ member bytes are preserved."""
from pathlib import Path
import json,tarfile,io,hashlib
import numpy as np
from core import HERE,sha,save
_LIMIT=32*1024**2
_INDEX={}
def load(root,record):
 root=Path(root);path=root/record['payload']
 if path.exists():return np.load(path,allow_pickle=False)
 key=str(root)
 if key not in _INDEX:_INDEX[key]=json.loads((root/'PAYLOAD_BUNDLES.json').read_text())['members']
 item=_INDEX[key][record['payload']]
 with tarfile.open(root/item['bundle'],'r:') as tar:b=tar.extractfile(record['payload']).read()
 assert hashlib.sha256(b).hexdigest()==record['payload_sha256'];return np.load(io.BytesIO(b),allow_pickle=False)
def pack(root=HERE):
 root=Path(root);assert json.loads((root/'INDEPENDENT_AUDIT.json').read_text())['passed'];rr=[json.loads(s) for s in (root/'qwen7b/RAW.jsonl').read_text().splitlines()];assert len(rr)==1119
 dest=root/'PAYLOAD_BUNDLES.json';index=json.loads(dest.read_text()) if dest.exists() else dict(format='Uncompressed tar bundles containing original compressed NPZ bytes. Expanded staging copies removed only after member verification.',members={},bundles=[])
 pending=[r for r in rr if r['payload'] not in index['members']]
 while pending:
  group=[];size=10240
  while pending and (not group or size+(root/pending[0]['payload']).stat().st_size+1024<=_LIMIT):
   r=pending.pop(0);group.append(r);size+=(root/r['payload']).stat().st_size+1024
  name=f'PAYLOADS_{len(index["bundles"])+1:03d}.tar';path=root/name;assert not path.exists()
  with tarfile.open(path,'w') as tar:
   for r in group:
    p=root/r['payload'];assert sha(p)==r['payload_sha256'];b=p.read_bytes();info=tarfile.TarInfo(r['payload']);info.size=len(b);info.mtime=0;info.mode=0o644;tar.addfile(info,io.BytesIO(b))
  with tarfile.open(path,'r:') as tar:
   for r in group:
    b=tar.extractfile(r['payload']).read();assert hashlib.sha256(b).hexdigest()==r['payload_sha256'];index['members'][r['payload']]=dict(bundle=name,bytes=len(b),sha256=r['payload_sha256'])
  index['bundles'].append(dict(path=name,bytes=path.stat().st_size,sha256=sha(path),members=len(group)));save(dest,index)
  for r in group:(root/r['payload']).unlink()
  print('BUNDLED',len(index['members']),'of1119',flush=True)
 assert len(index['members'])==1119;index['every_member_hash_verified']=True;save(dest,index);_INDEX.pop(str(root),None)
 print('All1119 original payloads preserved in',len(index['bundles']),'verified bundles',flush=True)
if __name__=='__main__':pack()
