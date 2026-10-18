"""Losslessly shard complete raw arrays for preservation. No model calls."""
from pathlib import Path
import json,tarfile,hashlib
P=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def main():
 rows=[json.loads(s) for s in (P/'RAW.jsonl').read_text().splitlines()];assert len(rows)==323
 bundle=P/'ALL_CALL_ARRAYS.tar.gz';assert not bundle.exists()
 with tarfile.open(bundle,'w:gz',compresslevel=4) as t:
  for r in rows:
   path=P/r['payload'];assert sha(path)==r['payload_sha256'];t.add(path,arcname=r['payload'],recursive=False)
 with tarfile.open(bundle,'r:gz') as t:
  members=t.getmembers();assert len(members)==323
  for member,r in zip(members,rows):
   assert member.name==r['payload'];assert hashlib.sha256(t.extractfile(member).read()).hexdigest()==r['payload_sha256']
 shards=[]
 with bundle.open('rb') as f:
  i=1
  while True:
   b=f.read(32*1024**2)
   if not b:break
   part=P/f'ALL_CALL_ARRAYS.part{i:03d}';part.write_bytes(b);shards.append(dict(path=part.name,bytes=len(b),sha256=sha(part)));i+=1
 h=hashlib.sha256()
 for s in shards:
  with (P/s['path']).open('rb') as f:
   for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 assert h.hexdigest()==sha(bundle)
 (P/'ARRAY_ARCHIVE_MANIFEST.json').write_text(json.dumps(dict(archive=bundle.name,bytes=bundle.stat().st_size,sha256=sha(bundle),payloads=323,all_payload_hashes_verified=True,shards=shards,reconstruct='Concatenate ordered part001..partNNN into ALL_CALL_ARRAYS.tar.gz, verify archive SHA256, then extract. Archive contains qwen7b/CALL_*.npz only; no weights or credentials.',control_prefix_arrays_are_hashes_only=True),indent=2)+'\n')
 print(json.dumps(dict(bytes=bundle.stat().st_size,shards=len(shards),all323arrays_verified=True)))
if __name__=='__main__':main()
