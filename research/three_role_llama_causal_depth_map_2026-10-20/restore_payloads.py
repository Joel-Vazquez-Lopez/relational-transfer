"""Reconstruct exact original NPZ bytes from published shards; no model packages."""
import argparse,base64,contextlib,hashlib,json,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent

def digest(b):return hashlib.sha256(b).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--destination',type=Path,required=True);p.add_argument('--limit',type=int);args=p.parse_args()
    manifest=json.loads((HERE/'RAW_ARCHIVE_MANIFEST.json').read_text());assert manifest['format']=='exact_npz_compressed_member_pieces_v1'
    root=args.destination.resolve();root.mkdir(parents=True,exist_ok=True)
    for b in manifest['bundles']:assert digest((HERE/b['path']).read_bytes())==b['sha256']
    count=0
    with contextlib.ExitStack() as stack:
        arcs={b['path']:stack.enter_context(tarfile.open(HERE/b['path'],'r:')) for b in manifest['bundles']}
        for name,f in manifest['files'].items():
            if args.limit is not None and count>=args.limit:break
            dest=(root/name).resolve();dest.relative_to(root);assert not dest.exists(),dest
            pieces=[]
            for part in f['pieces']:
                if 'literal' in part:pieces.append(base64.b64decode(part['literal']))
                else:
                    h=part['chunk'];c=manifest['chunks'][h];b=arcs[c['bundle']].extractfile(c['member']).read();assert digest(b)==h;pieces.append(b)
            b=b''.join(pieces);assert len(b)==f['bytes'] and digest(b)==f['sha256']
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b);count+=1
    print(json.dumps(dict(restored_files=count,byte_identical_hashes_verified=True,destination=str(root))))
if __name__=='__main__':main()
