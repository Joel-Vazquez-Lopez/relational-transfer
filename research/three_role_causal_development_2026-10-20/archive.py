"""Lossless small tar shards of every original payload, verified member by member."""
import hashlib
import json
import tarfile
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    assert json.loads((HERE/'COLLECTION.json').read_text())['status']in ['COMPLETE_FIXED_DEVELOPMENT_CANDIDATE','INTERRUPTED_RESOURCE_GUARD']
    assert not (HERE/'PAYLOAD_BUNDLES.json').exists()
    parts=[];members={};current=[];size=0
    for p in sorted((HERE/'payloads').glob('CALL_*.npz')):
        if current and size+p.stat().st_size>16*1024**2:
            parts.append(current);current=[];size=0
        current.append(p);size+=p.stat().st_size
    if current:parts.append(current)
    bundles=[];checks=0
    for i,files in enumerate(parts,1):
        name=f'PAYLOADS_{i:03d}.tar';dest=HERE/name
        assert not dest.exists()
        with tarfile.open(dest,'w:') as tar:
            for p in files:tar.add(p,arcname=str(p.relative_to(HERE)),recursive=False)
        with tarfile.open(dest,'r:') as tar:
            for p in files:
                rel=str(p.relative_to(HERE));original=p.read_bytes();b=tar.extractfile(rel).read()
                assert original==b
                members[rel]=dict(bundle=name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest());checks+=1
        bundles.append(dict(path=name,bytes=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    forwards=[json.loads(s) for s in (HERE/'FORWARDS.jsonl').read_text().splitlines()]
    assert len(members)==len(forwards)
    assert all(members[f['payload']]['sha256']==f['payload_sha256'] for f in forwards)
    out=dict(scope='Exact original NPZ bytes, including every validation/control/continuation; tar shards are uncompressed because NPZ members already are compressed. Expanded originals retained locally.',verified=True,member_checks=checks,bundles=bundles,members=members)
    (HERE/'PAYLOAD_BUNDLES.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(verified=True,payloads=checks,bundles=len(bundles),bytes=sum(b['bytes'] for b in bundles))))

if __name__=='__main__':main()
