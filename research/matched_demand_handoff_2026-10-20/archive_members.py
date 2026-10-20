"""Lossless exact-NPZ byte reconstruction with compressed-member deduplication."""
import base64,hashlib,io,json,struct,tarfile,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent

def digest(b):return hashlib.sha256(b).hexdigest()
def main():
    assert not (HERE/'RAW_ARCHIVE_MANIFEST.json').exists()
    fwd=[json.loads(s) for s in (HERE/'FORWARDS.jsonl').read_text().splitlines()]
    recovered=json.loads((HERE/'RECOVERED_FORWARD.json').read_text())
    fwd.append(recovered)
    assert len(fwd)==json.loads((HERE/'STOP_RESULTS.json').read_text())['completed_raw_payloads']==189
    folder=HERE/'raw_chunks';folder.mkdir(exist_ok=True);chunks={};files={}
    def piece(b):
        if len(b)<1024:return dict(literal=base64.b64encode(b).decode())
        h=digest(b);p=folder/(h+'.bin')
        if h not in chunks:
            if p.exists():assert p.read_bytes()==b
            else:p.write_bytes(b)
            chunks[h]=dict(bytes=len(b))
        return dict(chunk=h)
    for f in fwd:
        path=HERE/f['payload'];b=path.read_bytes();assert digest(b)==f['payload_sha256'];parts=[];cursor=0
        with zipfile.ZipFile(io.BytesIO(b)) as z:
            for info in sorted(z.infolist(),key=lambda i:i.header_offset):
                at=info.header_offset;assert b[at:at+4]==b'PK\x03\x04'
                if at>cursor:parts.append(piece(b[cursor:at]))
                namelen,extra=struct.unpack('<HH',b[at+26:at+30]);start=at+30+namelen+extra;end=start+info.compress_size
                parts.extend([piece(b[at:start]),piece(b[start:end])]);cursor=end
        if cursor<len(b):parts.append(piece(b[cursor:]))
        restored=b''.join((folder/(x['chunk']+'.bin')).read_bytes() if 'chunk' in x else base64.b64decode(x['literal']) for x in parts)
        assert restored==b
        files[f['payload']]=dict(bytes=len(b),sha256=digest(b),pieces=parts)
    groups=[];current=[];size=0
    for h,c in sorted(chunks.items()):
        if current and size+c['bytes']>16*1024**2:groups.append(current);current=[];size=0
        current.append(h);size+=c['bytes']
    if current:groups.append(current)
    bundles=[]
    for i,group in enumerate(groups,1):
        name=f'RAW_MEMBERS_{i:03d}.tar';dest=HERE/name;assert not dest.exists()
        with tarfile.open(dest,'w:') as t:
            for h in group:
                member=f'chunks/{h}.bin';t.add(folder/(h+'.bin'),arcname=member,recursive=False);chunks[h].update(bundle=name,member=member)
        with tarfile.open(dest,'r:') as t:
            for h in group:assert digest(t.extractfile(chunks[h]['member']).read())==h
        bundles.append(dict(path=name,bytes=dest.stat().st_size,sha256=digest(dest.read_bytes())))
    out=dict(format='exact_npz_compressed_member_pieces_v1',scope='Exact original ZIP/NPZ bytes,not reserialized arrays. Shared compressed members deduplicated;small headers/footer/data descriptors retained as base64 literals. Expanded original files remain local.',verified=True,payloads_verified=len(files),unique_chunks_verified=len(chunks),original_bytes=sum(v['bytes'] for v in files.values()),bundle_bytes=sum(v['bytes'] for v in bundles),files=files,chunks=chunks,bundles=bundles)
    (HERE/'RAW_ARCHIVE_MANIFEST.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ['files','chunks','bundles']},indent=2))
if __name__=='__main__':main()
