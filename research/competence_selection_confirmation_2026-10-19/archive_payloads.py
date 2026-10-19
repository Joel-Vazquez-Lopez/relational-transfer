"""Lossless full-payload preservation in32MiB shards; no duplicate combined archive."""
from pathlib import Path
import hashlib,json,tarfile,io,gzip
from core import HERE,sha,save
SIZE=32*1024**2
class Shards:
 def __init__(self):self.index=0;self.f=None;self.in_part=0;self.total=0;self.hash=hashlib.sha256();self.parts=[]
 def next(self):
  if self.f:self.f.close();self.parts[-1]['bytes']=self.in_part;self.parts[-1]['sha256']=sha(HERE/self.parts[-1]['path'])
  self.index+=1;p=HERE/f'PAYLOADS.part{self.index:03d}';assert not p.exists();self.f=p.open('wb');self.in_part=0;self.parts.append(dict(path=p.name))
 def write(self,b):
  self.hash.update(b);self.total+=len(b);n=len(b)
  while b:
   if self.f is None or self.in_part==SIZE:self.next()
   k=min(SIZE-self.in_part,len(b));self.f.write(b[:k]);self.in_part+=k;b=b[k:]
  return n
 def flush(self):
  if self.f:self.f.flush()
 def close(self):
  self.f.close();self.parts[-1]['bytes']=self.in_part;self.parts[-1]['sha256']=sha(HERE/self.parts[-1]['path'])
class Chain:
 def __init__(self,parts):self.parts=iter(parts);self.f=None
 def read(self,n=-1):
  chunks=[];remaining=n
  while n<0 or remaining>0:
   if self.f is None:
    try:self.f=(HERE/next(self.parts)['path']).open('rb')
    except StopIteration:break
   b=self.f.read(remaining)
   if b:chunks.append(b);remaining-=len(b)
   else:self.f.close();self.f=None
  return b''.join(chunks)
def main():
 assert json.loads((HERE/'INDEPENDENT_AUDIT.json').read_text())['passed']
 raw=[json.loads(s) for m in ['qwen7b','llama3b'] for s in (HERE/m/'RAW.jsonl').read_text().splitlines()];assert len(raw)==1562
 manifest=[];writer=Shards()
 with gzip.GzipFile(fileobj=writer,mode='wb',mtime=0,compresslevel=6) as gz:
  with tarfile.open(fileobj=gz,mode='w|') as tar:
   for r in raw:
    name=r['model']+'/'+r['payload'];p=HERE/name;assert sha(p)==r['payload_sha256'];b=p.read_bytes();entry=tarfile.TarInfo(name);entry.size=len(b);entry.mtime=0;entry.mode=0o644;tar.addfile(entry,io.BytesIO(b));manifest.append(dict(path=name,bytes=len(b),sha256=r['payload_sha256']))
 writer.close();expected={r['path']:r for r in manifest};seen=[]
 with tarfile.open(fileobj=Chain(writer.parts),mode='r|gz') as tar:
  for member in tar:
   record=expected[member.name];f=tar.extractfile(member);h=hashlib.sha256();size=0
   while True:
    b=f.read(2**20)
    if not b:break
    h.update(b);size+=len(b)
   assert h.hexdigest()==record['sha256'] and size==record['bytes'];seen.append(member.name)
 assert sorted(seen)==sorted(expected)
 save(HERE/'PAYLOAD_ARCHIVE_MANIFEST.json',dict(format='Concatenate ordered shards to a gzip tar stream; extract relative study-root paths',bytes=writer.total,sha256=writer.hash.hexdigest(),shards=writer.parts,payloads=manifest,n_payloads=len(manifest),every_reconstructed_member_SHA256_verified=True,combined_archive_duplicate_created=False))
 print('Archived and reconstructed',len(manifest),'payloads,',len(writer.parts),'shards,',writer.total,'bytes')
if __name__=='__main__':main()
