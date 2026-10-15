#!/usr/bin/env python3
"""Exact finite-world oracle for an optional question-time reconstruction route.
4 independent fair binary relations (16 worlds), 4 equally likely questions,
a deterministic one-bit question-independent cache, and an exact fallback.
Error costs 1; fallback costs c per query. Requires numpy.

This certifies a *finite information bottleneck*, NOT a natural LLM algorithm.
"""
import itertools
import numpy as np

W=np.asarray(list(itertools.product((0,1),repeat=4)),dtype=np.int32)
C=((np.arange(65536,dtype=np.uint32)[:,None] >>
    np.arange(16,dtype=np.uint32)[None,:])&1).astype(np.int32)
n1=C.sum(1)[:,None]
n0=16-n1
ones1=C@W
ones0=W.sum(0)[None,:]-ones1
errors=np.stack((np.minimum(ones0,n0-ones0),
                 np.minimum(ones1,n1-ones1)),axis=2)
counts=np.stack((np.broadcast_to(n0,ones0.shape),
                 np.broadcast_to(n1,ones1.shape)),axis=2)

def objective(c):
    # Cell = one cache-code and queried relation. Optimally answer with
    # a majority vote or fall back to the perfect raw-source reader.
    return np.minimum(errors,c*counts).sum((1,2))

def certificate():
    # The exact best loss is min(48c,12+16c,20), on 0<=c<=1.
    # For any fixed encoder, loss as a function of c is concave.
    # On each claimed affine segment, lower bounds at both endpoints
    # therefore extend throughout the segment. Exact integer checks:
    endpoints=((0,1,0),(3,8,144),(1,2,40),(1,1,20))
    for num,den,want in endpoints:
        actual=np.minimum(den*errors,num*counts).sum((1,2))
        assert int(actual.min())==want,(num,den,int(actual.min()))
    # On the three intervals, exhibit one encoder whose own loss
    # equals the proposed lower bound at both endpoints.
    witnesses=((255,0,3/8,0,18),
               (831,3/8,1/2,18,20),
               (279,1/2,1,20,20))
    for idx,lo,hi,lv,hv in witnesses:
        assert abs(float(objective(lo)[idx])-lv)<1e-8
        assert abs(float(objective(hi)[idx])-hv)<1e-8
    print("PASS: min loss = min(48*c, 12+16*c, 20) for 0<=c<=1")
    for c in (0.25,0.4,0.75):
        vals=objective(c)
        idx=int(vals.argmin())
        fallback=c*counts[idx]<errors[idx]  # ties use cache
        n_fallback=int(np.where(fallback,counts[idx],0).sum())
        misses=int(np.where(fallback,0,errors[idx]).sum())
        print(f"cost={c:.3f}: misses={misses}/64, fallback={n_fallback}/64")
if __name__=="__main__":
    certificate()
