"""BX-003: post-hoc uptake audit; no training/inference; synthetic open development.

Usage: python bx003_release.py --csv /path/to/BX002/paired_predictions.csv
Requires numpy. Source BX-002: research/brain-x/bx002_paired_syndromes.py
"""
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np

N=5
ED=np.array([(i,(i+1)%N) for i in range(N)],dtype=np.int64)
A=np.array([[(j>>i)&1 for i in range(N)] for j in range(1<<N)],dtype=np.int64)
FIXTURE_SHA='9f040b4a050e2d9c35c8eac2175ca13eeeb865b30131f2f6f3d7083547d813ef'
CSV_SHA='5a1d30a65a1773168b15c6651ad1c440e567f33c8db2e6bcc0223e74ef7ba3d0'

def fixture():
    rng=np.random.default_rng(802002)
    bits=rng.integers(0,2,(4000,N),dtype=np.int64)
    local=bits^(rng.random((4000,N))<.35).astype(np.int64)
    valid=bits[:,ED[:,0]]^bits[:,ED[:,1]]^(rng.random((4000,N))<.08).astype(np.int64)
    random=rng.integers(0,2,(4000,N),dtype=np.int64)
    query=rng.integers(0,N,4000,dtype=np.int64)
    labels=bits[np.arange(4000),query]
    arr=[bits,local,valid,random,query,labels]
    assert hashlib.sha256(b''.join(x.tobytes() for x in arr)).hexdigest()==FIXTURE_SHA
    return local,valid,random,query,labels

def vote(local,edges,q):
    r=np.arange(len(q));prev=(q-1)%N;nxt=(q+1)%N
    return ((local[r,q]+(local[r,prev]^edges[r,prev])+(local[r,nxt]^edges[r,q]))>=2).astype(np.int64)

def bayes(local,edges,q,p_valid=.8):
    """Global hidden 80/20 reliability indicator is marginalized, not given as oracle."""
    m=(local[:,None,:]==A[None,:,:]).sum(2)
    p_local=.65**m*.35**(N-m)
    parity=A[:,ED[:,0]]^A[:,ED[:,1]]
    t=(edges[:,None,:]==parity[None,:,:]).sum(2)
    p_edges=p_valid*.92**t*.08**(N-t)+(1-p_valid)/32
    weight=p_local*p_edges
    weight/=weight.sum(1,keepdims=True)
    marg=weight@A
    return (marg[np.arange(len(q)),q]>.5).astype(np.int64)

def scores(pred,gold):
    return {'correct':int((pred==gold).sum()),'n':len(gold),'accuracy':float((pred==gold).mean())}

def transitions(pred,ref,gold):
    return {'rescues':int(((pred==gold)&(ref!=gold)).sum()),
            'harms':int(((pred!=gold)&(ref==gold)).sum()),
            'flips':int((pred!=ref).sum())}

def audit(path):
    digest=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    with open(path,newline='') as f: rows=list(csv.DictReader(f))
    expected=['id','query','gold','local','oracle','majority','edge17_valid','edge17_random','node17_valid','node17_random','edge29_valid','edge29_random','node29_valid','node29_random']
    assert len(rows)==4000 and list(rows[0])==expected
    c={x:np.array([int(r[x]) for r in rows],dtype=np.int64) for x in expected}
    assert np.array_equal(c['id'],np.arange(4000))
    local,valid,random,q,gold=fixture()
    source=local[np.arange(4000),q]
    v=vote(local,valid,q)
    assert np.array_equal(c['query'],q) and np.array_equal(c['gold'],gold)
    assert np.array_equal(c['local'],source) and np.array_equal(c['majority'],v)
    mixed=bayes(local,valid,q)
    assert np.array_equal(bayes(local,valid,q,p_valid=0),source)
    assert np.array_equal(bayes(1-local,valid,q),1-mixed)  # label-flip symmetry
    conflict=v!=source
    out={'experiment':'BX-003','mode':'post_hoc_synthetic_audit_no_new_model_inference',
         'source_csv_sha256':digest,'source_csv_original_sha_matches':digest==CSV_SHA,
         'fixture_sha256':FIXTURE_SHA,'worlds':len(gold),'conflict_worlds':int(conflict.sum()),
         'training_mixture':{'valid_world_fraction':.8,'random_world_fraction':.2,'known_parameters':'analytic reference only'},
         'valid_predictions':{},'conflict_predictions':{},'model_pairing':{},
         'scope':'post hoc; same fixed 5-node ring, 2 seeds, no authentic language or identified mechanism'}
    allp={'local':source,'three_vote':v,'mixture_bayes':mixed,'model17':c['edge17_valid'],'model29':c['edge29_valid']}
    for name,p in allp.items():
        out['valid_predictions'][name]={'score':scores(p,gold),'versus_local':transitions(p,source,gold)}
        out['conflict_predictions'][name]={'score':scores(p[conflict],gold[conflict]),
                                           'local_overrides':int((p[conflict]!=source[conflict]).sum())}
    for seed in (17,29):
        p=c[f'edge{seed}_valid'];r=c[f'edge{seed}_random']
        out['model_pairing'][str(seed)]={'valid_vs_random':transitions(p,r,gold),
                                         'valid_accuracy':scores(p,gold),
                                         'random_accuracy':scores(r,gold)}
    out['mixture_bayes_random_accuracy']=scores(bayes(local,random,q),gold)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--csv',type=Path,required=True);p.add_argument('--output',type=Path)
    a=p.parse_args();result=audit(a.csv)
    j=json.dumps(result,indent=2)+'\n'
    if a.output: a.output.write_text(j)
    print(j)
