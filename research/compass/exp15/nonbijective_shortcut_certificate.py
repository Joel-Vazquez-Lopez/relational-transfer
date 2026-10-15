#!/usr/bin/env python3
"""Compass 15 deterministic non-bijective graph audit (stdlib only).
Model training and checkpoints are in the SHA256-verified conversation artifact.
"""
from itertools import product
from collections import Counter
from random import Random

def world_from_id(wid):
    return tuple((wid >> (2*i)) & 3 for i in range(8))

def compose(world, start, chain):
    for op in chain:
        if op != -1:
            start=world[4*op + start]
    return start

def trace(world, start, chain):
    touched=[]
    for op in chain:
        if op != -1:
            index=4*op+start
            touched.append(index)
            start=world[index]
    return touched,start

def omitted(world,start,chain,which):
    if which == 'first': seq=chain[1:]
    elif which == 'middle': seq=(chain[0],chain[2])
    elif which == 'last': seq=chain[:2]
    elif which == 'last_only': seq=chain[2:]
    elif which == 'vote':
        options=[
            compose(world,start,chain[1:]),
            compose(world,start,(chain[0],chain[2])),
            compose(world,start,chain[:2])
        ]
        return Counter(options).most_common(1)[0][0]
    else: raise ValueError(which)
    return compose(world,start,seq)

def split(seed=1515):
    pool=list(range(65536))
    Random(seed).shuffle(pool)
    test=pool[:128]
    train=pool[128:12128]
    test2=pool[12128:12256]
    assert len(set(train))==12000
    assert not (set(train)&set(test) or set(train)&set(test2) or set(test)&set(test2))
    return train,test,test2

def audit(ids):
    correct=Counter()
    nonbijective=0
    for wid in ids:
        world=world_from_id(wid)
        for chain in product((0,1),repeat=3):
            outputs=[compose(world,start,chain) for start in range(4)]
            if len(set(outputs))<4: nonbijective+=1
            for start,y in enumerate(outputs):
                for mode in ('first','middle','last','last_only','vote'):
                    correct[mode]+=omitted(world,start,chain,mode)==y
    return dict(correct),nonbijective

def tests():
    assert world_from_id(0)==(0,)*8
    assert compose(world_from_id(0),3,(0,1,0))==0
    for wid in (11,391,4057,17001):
        world=world_from_id(wid)
        for chain in product((0,1),repeat=3):
            for start in range(4):
                path,gold=trace(world,start,chain)
                for index in set(range(8))-set(path):
                    edit=list(world);edit[index]=(edit[index]+1)%4
                    assert compose(edit,start,chain)==gold
    train,test,test2=split()
    baseline,nonbijective=audit(test)
    assert baseline=={'first':2154,'middle':1876,'last':1458,'last_only':2136,'vote':2003}
    assert nonbijective==998
    print('PASS: split disjoint; 998/1024 nonbijective maps; panel1 baselines',baseline)
    baseline2,_=audit(test2)
    print('Second unsealed development panel',baseline2)

if __name__=='__main__':tests()
