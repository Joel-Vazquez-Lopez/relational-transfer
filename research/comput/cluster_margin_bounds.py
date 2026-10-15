#!/usr/bin/env python3
"""Exact sharp event-cluster sign-test bounds from binary aggregate arm totals.
Assumes 12 clusters with exactly four matched binary observations each.
No claims about actual unseen pairings or cluster independence.
"""
from math import comb
from functools import lru_cache
import json

def sign_p(pos, neg):
    n=pos+neg
    return min(1.0,2*sum(comb(n,k) for k in range(min(pos,neg)+1))/2**n) if n else 1.

def bounds(a,b,clusters=12,size=4):
    n=clusters*size
    if not (0<=a<=n and 0<=b<=n): raise ValueError("invalid totals")
    options={(aa,bb,(aa>bb)-(aa<bb)) for aa in range(size+1) for bb in range(size+1)}
    states={(0,0,0,0)}
    for idx in range(clusters):
        nxt=set()
        for ta,tb,pos,neg in states:
            for aa,bb,sgn in options:
                na,nb=ta+aa,tb+bb
                if na<=a and nb<=b and na+(clusters-idx-1)*size>=a and nb+(clusters-idx-1)*size>=b:
                    nxt.add((na,nb,pos+(sgn>0),neg+(sgn<0)))
        states=nxt
    feasible={(pos,neg) for ta,tb,pos,neg in states if ta==a and tb==b}
    assert feasible
    ps=[sign_p(p,m) for p,m in feasible]
    return {"a":a,"b":b,"clusters":clusters,"observations_per_cluster":size,
            "possible_cluster_sign_pairs":len(feasible),"p_min":min(ps),"p_max":max(ps),
            "sharp_conditional_on_assumptions":True,
            "warning":"Event outcomes, actual paired assignments and independence are unknown."}

if __name__=="__main__":
    print(json.dumps({
        "reference_swapped":bounds(34,26),
        "fresh_transcript_accuracy":bounds(26,7),
        "fresh_transcript_compliance":bounds(42,18)},indent=2))
