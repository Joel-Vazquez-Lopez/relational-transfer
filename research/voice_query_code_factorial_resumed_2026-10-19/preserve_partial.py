"""Audit and archive an interrupted run without inference or hypothesis selection."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile
import numpy as np
from core import HERE, sha, save, digest

def main():
    assert not (HERE/'RESULTS.json').exists()
    raw=[json.loads(s) for s in (HERE/'qwen7b/RAW.jsonl').read_text().splitlines()]
    budget=[json.loads(s) for s in (HERE/'BUDGET.jsonl').read_text().splitlines()]
    assert len(raw)==248 and len(budget)==249
    assert [r['call'] for r in raw]==list(range(1,249))
    assert [r['call'] for r in budget]==list(range(1,250))
    assert not any(r['split']=='evaluation' for r in raw)
    assert not (HERE/'qwen7b/PREDICTION_BEFORE_EVALUATION.json').exists()
    lock=json.loads((HERE/'LOCK.json').read_text())
    for n,h in lock['files'].items(): assert sha(HERE/n)==h,n
    lookup={r['id']:r for r in json.loads((HERE/'INPUTS.json').read_text())}
    checks=[]
    for r in raw:
        p=HERE/r['payload']; assert sha(p)==r['payload_sha256']
        assert all(r[k]==budget[r['call']-1][k] for k in ('id','mode','cut','kind','wrapped','native','physical_length'))
        with np.load(p,allow_pickle=False) as z:
            logits=z['logits'].astype(np.float64); trace=z['trace'].astype(np.float32)
            assert logits.shape==(152064,) and np.isfinite(logits).all()
            row=lookup[r['id']]
            score=float(np.logaddexp.reduce(logits[row['aliases']['entailed']])-np.logaddexp.reduce(logits[row['aliases']['not_entailed']]))
            actual=next((k for k,ids in row['aliases'].items() if int(np.argmax(logits)) in ids),None)
            assert abs(score-r['s_entail'])<=1e-10 and actual==r['actual']
            assert digest(trace)==r['full_trace_sha256']
            assert (trace.shape==(29,3,3584)) if r['wrapped'] else trace.size==0
        checks.append({'call':r['call'],'payload_sha256':r['payload_sha256'],'payload_and_journal_valid':True})
    orphan=HERE/'qwen7b/CALL_0249.npz'; assert orphan.exists()
    try:
        with np.load(orphan,allow_pickle=False) as z:
            for k in z.files: z[k]
        orphan_status='ZIP readable but no committed raw record; excluded from results'
    except Exception as e:
        orphan_status=f'Incomplete/unreadable NPZ: {type(e).__name__}; bytes preserved, excluded from results'
    members=[{'path':r['payload'],'sha256':r['payload_sha256'],'scientifically_committed':True} for r in raw]
    members.append({'path':str(orphan.relative_to(HERE)),'sha256':sha(orphan),'scientifically_committed':False,'status':orphan_status})
    index={'members':{},'bundles':[],'expanded_originals_retained':True,'partial_run_only':True}
    pending=list(members)
    while pending:
        group=[];size=10240
        while pending and (not group or size+(HERE/pending[0]['path']).stat().st_size+1024<32*1024**2):
            x=pending.pop(0);group.append(x);size+=(HERE/x['path']).stat().st_size+1024
        name=f'PARTIAL_PAYLOADS_{len(index["bundles"])+1:03d}.tar';target=HERE/name
        assert not target.exists()
        with tarfile.open(target,'w') as tar:
            for x in group:
                b=(HERE/x['path']).read_bytes();info=tarfile.TarInfo(x['path']);info.size=len(b);info.mtime=0;info.mode=0o644;tar.addfile(info,io.BytesIO(b))
        with tarfile.open(target,'r:') as tar:
            for x in group:
                b=tar.extractfile(x['path']).read();assert hashlib.sha256(b).hexdigest()==x['sha256']
                index['members'][x['path']]={**x,'bundle':name,'bytes':len(b)}
        index['bundles'].append({'path':name,'sha256':sha(target),'bytes':target.stat().st_size})
    save(HERE/'PARTIAL_PAYLOAD_BUNDLES.json',index)
    save(HERE/'PARTIAL_ARCHIVAL_AUDIT.json',{'passed':True,'completed_payloads_verified':len(raw),'frozen_files_verified':len(lock['files']),'uncommitted_call_249_status':orphan_status,'checks':checks,'all_archive_members_byte_verified':True,'interpretation':'Archival/score verification only. Full-panel technical gates and scientific predictions were not evaluated.','no_model_calls':True})
    print(json.dumps({'verified_payloads':len(raw),'attempts':len(budget),'bundles':len(index['bundles']),'orphan':orphan_status}))

if __name__=='__main__':main()
