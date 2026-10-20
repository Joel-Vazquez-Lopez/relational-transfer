"""Stage add-only documentary artifacts on the existing research branch.

Requires explicit user publication authorization (already supplied for this project).
Uses GitHub's documented tree.content API, in bounded requests. Does not update a ref.
The separate existing commit helper checks head equality and every selected blob.
"""
from pathlib import Path
import hashlib,json,subprocess,tempfile
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
REPO='Joel-Vazquez-Lopez/relational-transfer'
BRANCH='research/causal-developmental-relations-2026-10-17'
EXPECTED_HEAD='9ac7caf1f3e7c8e894d5c8c1dad9b6f0385df1f1'
def api(path,data=None):
    cmd=['/opt/homebrew/bin/gh','api',path]
    if data is None:return json.loads(subprocess.check_output(cmd))
    with tempfile.NamedTemporaryFile('w',suffix='.json',delete=False) as f:
        json.dump(data,f,ensure_ascii=False);name=f.name
    try:return json.loads(subprocess.check_output(cmd+['--method','POST','--input',name]))
    finally:Path(name).unlink()
def blob(b):return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def main():
    assert json.loads((HERE/'VALIDATION.json').read_text())['status']=='PASS'
    head=api(f'repos/{REPO}/git/ref/heads/{BRANCH}')['object']['sha']
    assert head==EXPECTED_HEAD,('Branch moved; audit before rebasing publication',head)
    base=api(f'repos/{REPO}/git/commits/{head}')['tree']['sha']
    remote=api(f'repos/{REPO}/git/trees/{base}?recursive=1');assert not remote.get('truncated')
    known={r['path']:r['sha'] for r in remote['tree'] if r['type']=='blob'}
    evidence=json.loads((HERE/'EVIDENCE_MANIFEST.json').read_text())['files']
    # Existing source files must match exactly. Never replace another workstream's bytes.
    for r in evidence:
        p=r['path'];b=(ROOT/p).read_bytes()
        assert hashlib.sha256(b).hexdigest()==r['sha256']
        if p in known:assert blob(b)==known[p],('Existing source conflict',p)
    missing=[r for r in evidence if r['path'] not in known and not r['path'].startswith(str(HERE.relative_to(ROOT))+'/')]
    plan=dict(expected_head=head,branch=BRANCH,method='Add-only inline Git trees based on existing tree, followed by guarded non-force commit/ref update.',source_files_missing_remotely=len(missing),missing_source_bytes=sum(r['bytes'] for r in missing),existing_evidence_conflicts=0,scope='Only audit artifacts and missing linked documentary evidence; no checkpoint, model weights, keys or unrelated checkout changes.')
    (HERE/'PUBLICATION_PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    audit=[str(p.relative_to(ROOT)) for p in sorted(HERE.rglob('*')) if p.is_file() and p.suffix not in ['.pyc'] and '__pycache__' not in p.parts and p.name not in ['REVIEW_TEXT.json','GITHUB_PRESERVATION.json','PUBLICATION.log']]
    selected=sorted(set(audit+[r['path'] for r in evidence]))
    expected=[];pending=[]
    for p in selected:
        b=(ROOT/p).read_bytes();sha=blob(b)
        expected.append(dict(path=p,type='blob',mode='100644',sha=sha))
        if p in known:
            assert sha==known[p],('Refuse overwrite',p)
        else:pending.append(dict(path=p,type='blob',mode='100644',content=b.decode('utf-8')))
    chunks=[];batch=[];size=0
    for e in pending:
        n=len(json.dumps(e,ensure_ascii=False).encode())
        if batch and size+n>2_500_000:chunks.append(batch);batch=[];size=0
        batch.append(e);size+=n
    if batch:chunks.append(batch)
    for i,chunk in enumerate(chunks):
        result=api(f'repos/{REPO}/git/trees',dict(base_tree=base,tree=chunk));base=result['sha']
        print(json.dumps(dict(staged_chunk=i+1,chunks=len(chunks),files=len(chunk),tree=base)),flush=True)
    Path('/private/tmp/research_publication_tree.json').write_text(json.dumps(dict(head=head,tree=base,entries=expected),indent=2))
    Path('/private/tmp/research_publication_selection.json').write_text(json.dumps(dict(files=selected),indent=2))
    print(json.dumps(dict(head=head,tree=base,selected_files=len(selected),new_files=len(pending),branch_ref_updated=False)),flush=True)
if __name__=='__main__':main()
