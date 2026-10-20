"""Preserve raw run members, verify a lossless tar, and split for GitHub. No deletion."""
from pathlib import Path
import gzip,hashlib,json,tarfile,os
HERE=Path(__file__).resolve().parent
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while chunk:=f.read(1024**2):h.update(chunk)
    return h.hexdigest()
def main():
    assert (HERE/'COLLECTION.json').exists() or (HERE/'INTERRUPTION.json').exists(),'Collection must have stopped'
    assert (HERE/'INDEPENDENT_FACTORIAL_AUDIT.json').exists(),'Audit before archival'
    out=HERE/'RAW_FACTORIAL.tar.gz';assert not out.exists(),'Refuse overwrite archive'
    files=sorted(p for p in (HERE/'factorial_runs').rglob('*') if p.is_file())
    manifest=[dict(path=str(p.relative_to(HERE)),bytes=p.stat().st_size,sha256=digest(p)) for p in files]
    with tarfile.open(out,'w:gz',compresslevel=6) as tf:
        for p in files:tf.add(p,arcname=str(p.relative_to(HERE)),recursive=False)
    expected={r['path']:r for r in manifest};verified=0
    with tarfile.open(out,'r:gz') as tf:
        for member in tf:
            if not member.isfile():continue
            r=expected[member.name];h=hashlib.sha256();stream=tf.extractfile(member);assert stream is not None
            while chunk:=stream.read(1024**2):h.update(chunk)
            assert h.hexdigest()==r['sha256'] and member.size==r['bytes'];verified+=1
    assert verified==len(manifest)
    parts=[]
    with out.open('rb') as f:
        i=0
        while chunk:=f.read(32*1024**2):
            i+=1;p=HERE/f'RAW_FACTORIAL.part{i:03}';assert not p.exists();p.write_bytes(chunk)
            parts.append(dict(path=p.name,bytes=len(chunk),sha256=digest(p)))
    result=dict(status='EXACT_ARCHIVE_VERIFIED',members=manifest,members_verified=verified,raw_bytes=sum(r['bytes'] for r in manifest),gzip_bytes=out.stat().st_size,gzip_sha256=digest(out),parts=parts,originals_retained=True,restore='Concatenate RAW_FACTORIAL.part* in numeric order to restore the gzip stream, verify gzip_sha256, then extract the tar. Verify every restored member against this manifest. The unsplit local gzip is also retained.')
    (HERE/'RAW_FACTORIAL_MANIFEST.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='members'},indent=2))
if __name__=='__main__':main()
