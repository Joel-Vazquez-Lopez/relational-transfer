from pathlib import Path
import json,hashlib,base64,subprocess,tempfile,concurrent.futures
root=Path(__file__).resolve().parents[2];p=Path(__file__).resolve().parent;repo='Joel-Vazquez-Lopez/relational-transfer';branch='research/causal-developmental-relations-2026-10-17'
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def gitsha(b):return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def api(args,data=None):
 cmd=['/opt/homebrew/bin/gh','api']+args
 if data is None:return json.loads(subprocess.check_output(cmd))
 with tempfile.NamedTemporaryFile(mode='w',suffix='.json',delete=False) as f:json.dump(data,f);name=f.name
 try:return json.loads(subprocess.check_output(cmd+['--input',name]))
 finally:Path(name).unlink()
assert json.loads((p/'INDEPENDENT_AUDIT.json').read_text())['passed']
assert json.loads((p/'PAYLOAD_ARCHIVE_MANIFEST.json').read_text())['n_payloads']==1562
assert (p/'REPORT.md').exists() and (p/'UNIT_LEDGER.csv').exists()
for m in ['qwen7b','llama3b']:assert json.loads((p/m/'DECISION.json').read_text())['status']=='COMPLETE_FIXED_PANEL'
files=sorted(f for f in p.iterdir() if f.is_file() and f.name not in ['PUBLICATION_MANIFEST.json','GITHUB_PRESERVATION.json','PUBLICATION.log'])
for m in ['qwen7b','llama3b']:files+=sorted(f for f in (p/m).iterdir() if f.is_file() and f.suffix!='.npz')
for f in files:assert f.suffix not in ['.tmp','.pyc']
manifest=dict(scope='All selected scripts, logs, reports, numerical checks and complete raw score journals. Every original NPZ payload is preserved in the included reconstructable shards; weights and full-prefix arrays are not included.',files=[dict(path=str(f.relative_to(root)),bytes=f.stat().st_size,sha256=sha(f),git_blob_sha1=gitsha(f.read_bytes())) for f in files])
(p/'PUBLICATION_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');files+=[p/'PUBLICATION_MANIFEST.json']
head=api(['repos/'+repo+'/git/ref/heads/'+branch])['object']['sha'];base=api(['repos/'+repo+'/git/commits/'+head])['tree']['sha'];current=api(['repos/'+repo+'/git/trees/'+base+'?recursive=1']);assert not current['truncated'];existing={x['path']:x['sha'] for x in current['tree'] if x['type']=='blob'}
for f in files:
 path=str(f.relative_to(root))
 if path in existing:assert existing[path]==gitsha(f.read_bytes()),('Refusing to overwrite a different remote file',path)
def blob(f):
 b=f.read_bytes();h=gitsha(b);remote=api(['--method','POST','repos/'+repo+'/git/blobs'],dict(content=base64.b64encode(b).decode(),encoding='base64'))['sha'];assert remote==h
 print('BLOB',str(f.relative_to(p)),len(b),flush=True)
 return dict(path=str(f.relative_to(root)),type='blob',mode='100644',sha=h)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:entries=list(pool.map(blob,files))
tree=api(['--method','POST','repos/'+repo+'/git/trees'],dict(base_tree=base,tree=entries))['sha'];Path('/private/tmp/confirmation_publication_tree.json').write_text(json.dumps(dict(head=head,tree=tree,entries=entries),indent=2));print(json.dumps(dict(head=head,tree=tree,files=len(entries))))
