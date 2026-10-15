#!/usr/bin/env python3
"""Exact paired sign tests and cluster-level sensitivity from aggregate transitions.
No simulated per-event observations. stdlib only."""
import json, math
def sign_p(gains, losses):
    n=gains+losses
    if not n: return 1.0
    return min(1.0, 2*sum(math.comb(n,k) for k in range(min(gains,losses)+1))/2**n)
def feasible_clusters(gains, losses, nclusters=6, size=2):
    opts=[(g,l) for g in range(size+1) for l in range(size-g+1)]
    states={(0,0,0,0,0)}
    for _ in range(nclusters):
        nxt=set()
        for g,l,p,n,t in states:
            for a,b in opts:
                if g+a<=gains and l+b<=losses:
                    nxt.add((g+a,l+b,p+(a>b),n+(a<b),t+(a==b)))
        states=nxt
    return sorted({(p,n,t) for g,l,p,n,t in states if g==gains and l==losses})
def run():
    studies={'qwen7b':(7,2,3,3,8),'phi3':(0,8,4,9,1)}
    out={}
    for name,(g,l,t,before,after) in studies.items():
        assert g+l+t==12 and after-before==g-l
        groups=feasible_clusters(g,l)
        out[name]={'gains':g,'losses':l,'unchanged':t,'before':before,'after':after,
                   'event_sign_p_conditional_on_independence':sign_p(g,l),
                   'possible_cluster_signs':groups,
                   'possible_cluster_sign_p_range':[min(sign_p(p,n) for p,n,_ in groups),
                                                    max(sign_p(p,n) for p,n,_ in groups)]}
    return out
if __name__=='__main__': print(json.dumps(run(),indent=2))
