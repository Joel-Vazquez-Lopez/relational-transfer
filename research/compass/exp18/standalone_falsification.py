#!/usr/bin/env python3
"""Compass 18: stdlib-only equal-gold stimulus and shortcut certificate.
No neural model inference. Raw Transformer outputs/checkpoints in separate ZIP.
"""
import itertools,json,random
from collections import Counter

def decode(n):return tuple((n>>(2*i))&3 for i in range(8))
def compose(w,start,ops):
 for op in ops:
  if op>=0:start=w[4*op+start]
 return start

def layouts(w,start,ops):
 plan=[6 if op<0 else 4+op for op in ops]
 return (list(w)+[start]+plan+[7], [start]+plan+list(w)+[7])

def family_ids(seed=1717, n=96):
 ids=list(range(4**8));random.Random(seed).shuffle(ids)
 return [x for x in ids if len(set(decode(x)[:4]))==4][:n]

def paired(n=2,seed=171704):
 rng=random.Random(seed);rows=[];covered=0
 for wid in family_ids():
  w=decode(wid);candidates=[]
  for ops in itertools.product((0,1),repeat=3):
   if ops[-1]!=1:continue
   for start in range(4):
    p=[];u=start
    for op in ops:
     k=4*op+u;p.append(k);u=w[k]
    if p.count(p[-1])!=1:continue
    alternate=ops[:2]+(0,)
    new_gold=compose(w,start,alternate)
    if u==new_gold:continue
    modified=list(w);modified[p[-1]]=new_gold
    assert compose(modified,start,ops)==new_gold
    assert tuple(modified[:4])==w[:4]
    candidates.append((wid,w,start,ops,alternate,tuple(modified),u,new_gold))
  if candidates:
   covered+=1;rows+=rng.sample(candidates,min(n,len(candidates)))
 assert covered==95 and len(rows)==188
 return rows

def policy(name,w,start,ops):
 if name=='full':return compose(w,start,ops)
 if name=='last_two':return compose(w,start,ops[1:])
 if name=='first_two':return compose(w,start,ops[:2])
 if name=='final_lookup':return w[4*ops[-1]+start]
 if name=='symbol_only':return (start+ops[-1]+1)%4
 if name=='start_only':return start
 raise ValueError(name)

def audit():
 rows=paired();out={}
 for name in ('full','last_two','first_two','final_lookup','symbol_only','start_only'):
  base=[policy(name,w,s,ops) for _,w,s,ops,alt,edit,old,new in rows]
  op=[policy(name,w,s,alt) for _,w,s,ops,alt,edit,old,new in rows]
  cell=[policy(name,edit,s,ops) for _,w,s,ops,alt,edit,old,new in rows]
  out[name]={'n':len(rows),'op_changes':sum(x!=y for x,y in zip(base,op)),
    'operand_changes':sum(x!=y for x,y in zip(base,cell)),
    'op_new_gold':sum(v==row[-1] for v,row in zip(op,rows)),
    'operand_new_gold':sum(v==row[-1] for v,row in zip(cell,rows))}
 assert (out['full']['op_changes'],out['full']['operand_changes'])==(188,188)
 assert (out['last_two']['op_changes'],out['last_two']['operand_changes'])==(152,68)
 assert (out['final_lookup']['op_changes'],out['final_lookup']['operand_changes'])==(155,48)
 assert (out['symbol_only']['op_changes'],out['symbol_only']['operand_changes'])==(188,0)
 for _,w,s,ops,_,_,_,_ in rows:
  a,b=layouts(w,s,ops)
  assert len(a)==len(b)==13 and Counter(a)==Counter(b)
 print(json.dumps({'pairs':188,'worlds':95,'negative_controls':out,
    'scope':'deterministic gold and shortcut audit, not neural result'},indent=2))
if __name__=='__main__':audit()
