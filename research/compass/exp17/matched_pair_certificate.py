#!/usr/bin/env python3
"""Compass 17: stdlib-only *stimulus* certificate, not pretrained/mechanistic inference.
The complete neural checkpoints/predictions are in the separately SHA-verified
COMPASS_INVESTIGATION_17_DEPTH_OPERAND_BINDING_2026-10-16.zip.
"""
import hashlib,itertools,json,random

def decode(wid):
    return tuple((wid>>(2*i))&3 for i in range(8))

def compose(w,start,ops):
    u=start
    for op in ops:
        if op>=0:u=w[4*op+u]
    return u

def family_worlds():
    ids=list(range(4**8))
    random.Random(1717).shuffle(ids)
    family=[i for i in ids if len(set(decode(i)[:4]))==4]
    assert len(family)==6144
    return family[:96]

def matched_pairs(world_ids,seed=171704,per_world=2):
    rng=random.Random(seed);rows=[];coverage=0
    for wid in world_ids:
        w=decode(wid);eligible=[]
        for ops in itertools.product((0,1),repeat=3):
            if ops[2]!=1:continue # edit B cell only; keep A bijective
            for start in range(4):
                u=start;path=[]
                for op in ops:
                    k=4*op+u;path.append(k);u=w[k]
                if path.count(path[2])>1:continue
                other_op=ops[:2]+(0,)
                changed=compose(w,start,other_op)
                if changed==u:continue
                changed_w=list(w);changed_w[path[2]]=changed
                assert compose(changed_w,start,ops)==changed
                assert w[:4]==tuple(changed_w[:4])
                assert compose(w,start,ops)!=changed
                eligible.append((wid,start,ops,changed))
        if eligible:
            coverage+=1
            rows.extend(rng.sample(eligible,min(per_world,len(eligible))))
    return rows,coverage

def main():
    rows,coverage=matched_pairs(family_worlds())
    assert coverage==95 and len(rows)==188
    encoded=json.dumps(rows,separators=(',',':'))
    digest=hashlib.sha256(encoded.encode()).hexdigest()
    assert digest=='a66ad109c9283d920e70684cf6ec69c50f34cf103929057c7b60d4662eb15573'
    print(json.dumps({'covered_worlds':coverage,'pairs':len(rows),'pair_sha256':digest,
      'gold_property':'same target after final B->A token flip or corresponding B-cell update',
      'warning':'Gold-conditioned stimulus design only. No trained-model evaluation or neural mechanism identified.'},indent=2))

if __name__=='__main__':main()
