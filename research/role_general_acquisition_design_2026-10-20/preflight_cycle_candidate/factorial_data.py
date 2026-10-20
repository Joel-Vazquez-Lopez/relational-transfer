"""A positive-frequency matched role-demand factorial. No model imports.

Each five-vertex Hamilton cycle has degree two. Complementary cycles therefore
change comparison partners without changing any meaning's positive frequency.
"""
from itertools import permutations,combinations
from collections import Counter
import hashlib,json
import numpy as np
from legacy_renderer import render

ARMS=('neither','source','target','both','source_double','target_double')
H=(0,1,2);G=(3,4,5);FOCAL=(1,2)
CFG=dict(d_model=48,n_heads=4,n_layers=2,vocab_size=9,message_length=4)
OBJECT_ORDERS=tuple(permutations(range(3)))

def stable_seed(*parts):
    b=json.dumps(parts,separators=(',',':')).encode()
    return int.from_bytes(hashlib.sha256(b).digest()[:8],'little')

def observed(m):
    s,t,e=m
    return s!=t and not(s in H and t in H) and e in (0,1)

def meanings():return tuple((s,t,e) for s in range(6) for t in range(6) for e in (0,1) if observed((s,t,e)))

def cycles(anchor):
    vertices=tuple(i for i in range(6) if i!=anchor)
    start=vertices[0];options=[]
    for tail in permutations(vertices[1:]):
        v=(start,)+tail
        edges=tuple(sorted(tuple(sorted((v[i],v[(i+1)%5]))) for i in range(5)))
        if (1,2) not in edges:continue
        options.append(edges)
    on=min(options)
    off=tuple(e for e in combinations(vertices,2) if e not in on)
    assert len(on)==len(off)==5 and (1,2) not in off
    assert Counter(i for e in on for i in e)==Counter(i for e in off for i in e)
    return on,off

def active(arm,role,replica):
    if arm=='neither':return False
    if arm=='source_double':return role=='source'
    if arm=='target_double':return role=='target'
    return replica==0 and (arm=='both' or arm==role)

def blocks(arm):
    assert arm in ARMS
    out=[]
    for replica in range(2):
        for role in ('source','target'):
            for anchor in G:
                on,off=cycles(anchor)
                for pair in (on if active(arm,role,replica) else off):
                    ms=tuple((i,anchor,e) if role=='source' else (anchor,i,e) for i in pair for e in (0,1))
                    out.append(dict(family=role,replica=replica,anchor=anchor,pair=pair,meanings=ms))
        # A shared role-reversal family prevents unordered-pair matching from
        # solving the complete training task, in every arm.
        for a,b in combinations(range(6),2):
            if a in H and b in H:continue
            ms=((a,b,0),(b,a,0),(a,b,1),(b,a,1))
            out.append(dict(family='reversal',replica=replica,anchor=-1,pair=(a,b),meanings=ms))
    assert len(out)==84
    return out

def pair_kind(a,b):
    s,t,e=a;ss,tt,ee=b
    if s==ss and t==tt:return 'event'
    if t==tt:return 'source' if e==ee else 'source_event'
    if s==ss:return 'target' if e==ee else 'target_event'
    if s==tt and t==ss:return 'reversal' if e==ee else 'reversal_event'
    return 'other'

def symbolic_audit(arm):
    positive=Counter();spectrum=Counter();focal=Counter();families=Counter()
    for b in blocks(arm):
        ms=b['meanings'];assert len(set(ms))==4 and all(observed(m) for m in ms)
        positive.update(ms);families[b['family']]+=1
        for a,c in combinations(ms,2):
            spectrum[pair_kind(a,c)]+=1
            if a[1]==c[1] and {a[0],c[0]}==set(FOCAL):
                focal['source' if a[2]==c[2] else 'source_event']+=1
            if a[0]==c[0] and {a[1],c[1]}==set(FOCAL):
                focal['target' if a[2]==c[2] else 'target_event']+=1
    return dict(arm=arm,blocks=84,positive_counts={str(k):v for k,v in sorted(positive.items())},comparison_spectrum=dict(spectrum),focal_comparisons={k:focal[k] for k in ['source','source_event','target','target_event']},families=dict(families))

def query_ids(m,identity_map):return np.asarray([identity_map[m[0]],identity_map[m[1]],6+m[2],8],dtype=np.int64)

def scene(m,identity_map,seed,split,geometry,distractor_rank):
    d=tuple(i for i in range(6) if i not in m[:2])[distractor_rank%4]
    rng=np.random.default_rng(stable_seed(seed,split,m,geometry,distractor_rank))
    raw,latent=render(*m,d,rng)
    # Undo its random object order then apply a complete deterministic cycle.
    canonical=raw[np.argsort(latent[-1])]
    order=OBJECT_ORDERS[geometry%6]
    raw=canonical[list(order)].copy()
    remapped=np.zeros_like(raw[:,:6])
    for i in range(6):remapped[:,identity_map[i]]=raw[:,i]
    raw[:,:6]=remapped
    return np.ascontiguousarray(raw,dtype=np.float32),tuple(order)

def coarse_query(ids,identity_map,role):
    q=ids.copy();slot=0 if role=='source' else 1
    if q[slot] in (identity_map[1],identity_map[2]):q[slot]=identity_map[1]
    return q

def coarse_scene(x,m,identity_map,role):
    y=x.copy();idx=0 if role=='source' else 1;ident=m[idx]
    if ident in FOCAL:
        at=np.flatnonzero(y[:,:6].argmax(1)==identity_map[ident]);assert len(at)==1
        y[at,:6]=0;y[at,identity_map[1]]=1
    return y

def bayes_correct(groups):return sum(max(c.values()) for c in groups.values())

def full_input_audit(arm,identity_map,seed):
    """Complete finite four-query/four-order audit for one geometry per block.

    Role-merger witness uses actual float32 observations. Repeating the exact
    four-query block with any other rendered scene preserves the within-block
    identical-input/incompatible-label witness; no symbolic-scene shortcut.
    """
    groups={k:{} for k in ('full','no_query','no_candidates','unordered_query','no_event_query','source_query','target_query','source_complete','target_complete')}
    n=0
    pos=Counter();cand=Counter();answer=Counter();encoded_positive=Counter()
    for block in blocks(arm):
        ms=block['meanings']
        # Shared scene cache key is meaning, never arm/block/selected answer.
        xs=[scene(m,identity_map,seed,'preflight',0,0)[0] for m in ms]
        for rotation in range(4):
            permutation=np.roll(np.arange(4),rotation);c=np.stack(xs)[permutation]
            cb=c.tobytes();coarse={}
            for role in ('source','target'):
                coarse[role]=np.stack([coarse_scene(xs[j],ms[j],identity_map,role) for j in permutation]).tobytes()
            for selected,m in enumerate(ms):
                q=query_ids(m,identity_map);qb=q.tobytes();label=int(np.flatnonzero(permutation==selected)[0]);n+=1
                answer[label]+=1;pos[m]+=1;cand.update(ms)
                encoded_positive[(m,hashlib.sha256(xs[selected].tobytes()).hexdigest())]+=1
                payloads=dict(full=qb+cb,no_query=cb,no_candidates=qb)
                unordered=q.copy();unordered[:2]=sorted(q[:2]);no_event=q.copy();no_event[2]=6
                payloads['unordered_query']=unordered.tobytes()+cb
                payloads['no_event_query']=no_event.tobytes()+cb
                for role in ('source','target'):
                    qq=coarse_query(q,identity_map,role).tobytes()
                    payloads[role+'_query']=qq+cb;payloads[role+'_complete']=qq+coarse[role]
                for key,data in payloads.items():
                    digest=hashlib.sha256(data).digest()
                    groups[key].setdefault(digest,Counter())[label]+=1
    result=dict(arm=arm,rows=n,identity_map=list(identity_map),answers=dict(answer),ceilings={k:bayes_correct(v)/n for k,v in groups.items()},conflicting_input_groups={k:sum(len(c)>1 for c in v.values()) for k,v in groups.items()},positive_scene_signature=hashlib.sha256(repr(sorted(encoded_positive.items())).encode()).hexdigest(),positive_signature=hashlib.sha256(repr(sorted(pos.items())).encode()).hexdigest(),candidate_signature=hashlib.sha256(repr(sorted(cand.items())).encode()).hexdigest())
    assert result['ceilings']['full']==1
    assert result['ceilings']['no_query']==result['ceilings']['no_candidates']==.25
    return result

def build_training(arm,identity_map,seed,n_geometry=18,split='train'):
    """Compact base scenes + indexed episodes; no repeated large float tensors."""
    assert n_geometry%6==0
    ms=meanings();mid={m:i for i,m in enumerate(ms)}
    # Scenes identical for every arm, conditional on the positive meaning.
    base=np.stack([scene(m,identity_map,seed,split,g,d)[0] for g in range(n_geometry) for d in range(4) for m in ms])
    rows=[]
    for b in blocks(arm):
        indices=[mid[m] for m in b['meanings']]
        for g in range(n_geometry):
            for d in range(4):
                offset=(g*4+d)*len(ms)
                for rotation in range(4):
                    perm=np.roll(np.arange(4),rotation)
                    candidates=tuple(offset+indices[j] for j in perm)
                    for selected,m in enumerate(b['meanings']):
                        label=int(np.flatnonzero(perm==selected)[0])
                        rows.append((mid[m],label,*candidates))
    # Coupled batching: same query + same label at every array index in each arm.
    rows=np.asarray(rows,np.int32)
    positive_scene=rows[np.arange(len(rows)),2+rows[:,1]]
    assert np.array_equal(positive_scene%len(ms),rows[:,0])
    sort=np.lexsort((np.arange(len(rows)),positive_scene,rows[:,1],rows[:,0]));rows=rows[sort]
    return dict(base=base,rows=rows,messages=np.stack([query_ids(m,identity_map) for m in ms]))

def evaluation(identity_map,seed,n_geometry=24):
    """Untouched H-H panels; a fixed candidate panel shared by both directions."""
    assert n_geometry%6==0
    cs=[];qs=[];ys=[];records=[]
    for event in (0,1):
        ms=((1,0,event),(2,0,event),(0,1,event),(0,2,event))
        assert all(not observed(m) for m in ms)
        for g in range(n_geometry):
            for d in range(4):
                xs=[scene(m,identity_map,seed,'held',g,d)[0] for m in ms]
                for rotation in range(4):
                    perm=np.roll(np.arange(4),rotation);c=np.stack(xs)[perm]
                    group=(event,g,d,rotation)
                    for selected,m in enumerate(ms):
                        label=int(np.flatnonzero(perm==selected)[0]);cs.append(c);qs.append(query_ids(m,identity_map));ys.append(label)
                        records.append(dict(group=group,role='source' if selected<2 else 'target',direction=selected%2,meaning=m))
    return dict(candidates=np.stack(cs),queries=np.stack(qs),labels=np.asarray(ys,np.int64),records=records)

def familiar(identity_map,seed,n_geometry=6):
    cs=[];qs=[];ys=[]
    for a,b in combinations(range(6),2):
        if a in H and b in H:continue
        ms=((a,b,0),(b,a,0),(a,b,1),(b,a,1))
        for g in range(n_geometry):
            xs=[scene(m,identity_map,seed,'familiar',g,0)[0] for m in ms]
            for rotation in range(4):
                perm=np.roll(np.arange(4),rotation);c=np.stack(xs)[perm]
                for selected,m in enumerate(ms):
                    cs.append(c);qs.append(query_ids(m,identity_map));ys.append(int(np.flatnonzero(perm==selected)[0]))
    return dict(candidates=np.stack(cs),queries=np.stack(qs),labels=np.asarray(ys,np.int64))
