from pathlib import Path
import json,tarfile,hashlib,base64,contextlib
p=Path('research/qwen_role_position_source_access_2026-10-20');m=json.loads((p/'RAW_ARCHIVE_MANIFEST.json').read_text());n=0
with contextlib.ExitStack() as stack:
 arcs={b['path']:stack.enter_context(tarfile.open(p/b['path'])) for b in m['bundles']}
 for name,rec in m['files'].items():
  pieces=[]
  for piece in rec['pieces']:
   if 'literal' in piece:pieces.append(base64.b64decode(piece['literal']))
   else:
    h=piece['chunk'];c=m['chunks'][h];b=arcs[c['bundle']].extractfile(c['member']).read();assert hashlib.sha256(b).hexdigest()==h;pieces.append(b)
  data=b''.join(pieces);assert len(data)==rec['bytes'] and hashlib.sha256(data).hexdigest()==rec['sha256'];assert (p/name).read_bytes()==data;n+=1
assert n==2865
rec=dict(exact_original_files_verified=n,bytes_reclaimed=sum(f['bytes'] for f in m['files'].values()),retained='Every original byte reconstructible from retained verified RAW_MEMBERS tar bundles',restore_command='python3 research/qwen_role_position_source_access_2026-10-20/restore_payloads.py --destination research/qwen_role_position_source_access_2026-10-20',scope='Only exact duplicated expanded NPZ files removed; no unique data, models, scripts or reports removed')
(p/'STORAGE_NOTE.json').write_text(json.dumps(rec,indent=2)+'\n')
for name in m['files']:(p/name).unlink()
print(json.dumps(rec))
