#!/usr/bin/env python3
"""Compass 21: stdlib-only exact-position factorial and shortcut calibration.
No neural inference or model-data access; independently generates 188 source cases.
"""
import collections,itertools,json,random

def decode(wid):return tuple((wid>>(2*i)) & 3 for i in range(8))
def compose(w,start,ops):
    for op in ops:
        if op!=-1:start=w[4*op+start]
    return start

def cases():
    ids=list(range(4**8));random.Random(1717).shuffle(ids)
    family=[wid for wid in ids if len(set(decode(wid)[:4]))==4][:96]
    rng=random.Random(171704);rows=[];covered=0
    for wid in family:
        w=decode(wid);possible=[]
        for ops in itertools.product((0,1),repeat=3):
            if ops[-1]!=1:continue
            for start in range(4):
                at=start;visited=[]
                for op in ops:
                    idx=4*op+at;visited.append(idx);at=w[idx]
                if visited.count(visited[-1])>1:continue
                other=ops[:2]+(0,);new=compose(w,start,other)
                if new==at:continue
                changed=list(w);changed[visited[-1]]=new
                assert compose(changed,start,ops)==new
                assert compose(changed,start,other)==new
                assert compose(w,start,other)==new
                assert changed[:4]==list(w[:4]);assert at!=new
                possible.append((wid,tuple(w),start,ops,tuple(other),tuple(changed),at,new))
        if possible:
            covered+=1;rows.extend(rng.sample(possible,min(2,len(possible))))
    assert covered==95 and len(rows)==188
    return rows

def policy(name,w,start,ops):
    if name=='complete':seq=ops
    elif name=='suffix_two':seq=ops[1:]
    elif name=='prefix_two':seq=ops[:2]
    elif name=='last_only':seq=ops[2:]
    elif name=='final_token_only':return (start+ops[-1]+1)%4
    elif name=='last_op_single_lookup':return w[4*ops[-1]+start]
    else:raise ValueError(name)
    return compose(w,start,seq)

def main():
    rows=cases();out=[]
    for name in ('complete','suffix_two','prefix_two','last_only','final_token_only','last_op_single_lookup'):
        totals=collections.Counter()
        for wid,w,start,ops,alt,changed,old,new in rows:
            x={c:policy(name,world,start,op) for c,world,op in (
                ('B0',w,ops),('B1',changed,ops),('A0',w,alt),('A1',changed,alt))}
            db=int(x['B1']==new)-int(x['B0']==new)
            da=int(x['A1']==new)-int(x['A0']==new)
            totals['selective_interaction']+=db-da
            totals['operand_B']+=db;totals['operand_A']+=da
            totals['complete_world']+=int(x['B0']==old and all(x[k]==new for k in ('B1','A0','A1')))
            totals['A_invariance']+=int(x['A0']==x['A1'])
        out.append({'policy':name,**{k:v/len(rows) for k,v in totals.items()}})
    assert out[0]['selective_interaction']==out[0]['complete_world']==1
    assert abs(next(x for x in out if x['policy']=='suffix_two')['selective_interaction']-67/188)<1e-12
    assert abs(next(x for x in out if x['policy']=='last_only')['selective_interaction']-48/188)<1e-12
    print(json.dumps({'n':188,'n_worlds':95,'controls':out,
      'warning':'Synthetic oracle controls; selective interaction does not establish three-step computation.'},indent=2))
if __name__=='__main__':main()
