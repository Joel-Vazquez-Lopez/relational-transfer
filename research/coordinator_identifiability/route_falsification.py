"""Deterministic non-identifiability witness; source-access route is not algorithm identity."""
from itertools import permutations,product
import json
N=("Ada","Bea","Cal","Dee"); V=("praised","criticized"); R=("actor","patient")
def worlds():
 for p in permutations(N):
  yield {V[0]:dict(zip(R,p[:2])),V[1]:dict(zip(R,p[2:]))}
def swap(w,v):
 x={k:dict(t) for k,t in w.items()}
 x[v]["actor"],x[v]["patient"]=x[v]["patient"],x[v]["actor"]
 return x
def answer(w,q):return w[q[0]][q[1]]
def cache(c,s,q):return answer(c,q)
def reconstruction(c,s,q):return answer(s,q)
def shortcut(c,s,q):
 # Intentionally a source-based function with identical behavior to reconstruction.
 # Its equivalence illustrates what a port cut alone cannot identify.
 return answer(s,q)
def hybrid(c,s,q):return answer(c if q[1]=="actor" else s,q)
POLICIES={"cache":cache,"reconstruction":reconstruction,"shortcut":shortcut,"hybrid":hybrid}
def evaluate():
 out={k:{"n":0,"cache_effect":0,"source_effect":0} for k in POLICIES}
 signatures={k:[] for k in POLICIES}
 for w in worlds():
  for q in product(V,R):
   d=swap(w,q[0])
   for k,f in POLICIES.items():
    a,b,c,t=f(w,w,q),f(d,w,q),f(w,d,q),f(d,d,q)
    assert a==answer(w,q) and t==answer(d,q)
    o=out[k];o["n"]+=1;o["cache_effect"]+=a!=b;o["source_effect"]+=a!=c
    signatures[k].append((a,b,c,t))
 assert signatures["shortcut"]==signatures["reconstruction"]
 return out
if __name__=="__main__":print(json.dumps(evaluate(),indent=2))
