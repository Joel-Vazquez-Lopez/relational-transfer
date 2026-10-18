"""Inference-free definitions for a bounded source-access study."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

LENGTH = 128
BLOCKS = 28
CUTS = [0, 7, 14, 21, 28]
CONDITIONS = ([dict(name=f'cut_{k}', cut=k, source='event', restore=[]) for k in CUTS]
    + [dict(name=f'irrelevant_{k}', cut=k, source='irrelevant', restore=[]) for k in [7,14,21]]
    + [dict(name='restore_all',cut=0,source='event',restore=[[0,28]]),
       dict(name='restore_7_14',cut=0,source='event',restore=[[7,14]]),
       dict(name='restore_14_21',cut=0,source='event',restore=[[14,21]])])

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for c in iter(lambda:f.read(2**20),b''): h.update(c)
    return h.hexdigest()

def save(path,obj):
    p=Path(path); t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n'); os.replace(t,p)

def mask(row, condition, block):
    a=np.tri(LENGTH,dtype=bool)
    changed=(block>=condition['cut'] and not any(lo<=block<hi for lo,hi in condition['restore']))
    if changed:
        lo,hi=row['source_span'] if condition['source']=='event' else row['irrelevant_span']
        # Include every later position: earlier question tokens cannot bypass the cut.
        a[row['suffix_start']:,lo:hi]=False
    assert a.any(axis=1).all()
    return a

def lse(x):
    x=np.asarray(x,dtype=np.float64); m=float(np.max(x))
    return float(m+np.log(np.exp(x-m).sum()))

def score(logits,row,decode):
    classes={n:lse(logits[ids]) for n,ids in row['aliases'].items()}
    gold=row['expected']; other=next(n for n in row['names'] if n!=gold)
    best=int(np.argmax(logits)); top=np.argsort(logits)[-20:][::-1]
    actual=next((n for n,ids in row['aliases'].items() if best in ids),None)
    return dict(expected=gold,argmax_id=best,argmax_token=decode([best]),actual=actual,
        margin=classes[gold]-classes[other],correct=bool(actual==gold and classes[gold]>classes[other]),
        alias_logits=classes,top20=[dict(id=int(i),token=decode([int(i)]),logit=float(logits[i])) for i in top])

def matched_summary(rows,results):
    lookup={(r['id'],r['condition']):r for r in results}
    families=sorted({r['family'] for r in rows}); out=[]
    for k in [7,14,21]:
        ds=[]; fams={}; irrelevant_drift=[]
        for fam in families:
            costs={}
            for q in ['agent','first']:
                rs=[r for r in rows if r['family']==fam and r['query']==q]
                costs[q]=float(np.mean([lookup[r['id'],f'irrelevant_{k}']['margin']-lookup[r['id'],f'cut_{k}']['margin'] for r in rs]))
            d=costs['agent']-costs['first']; ds.append(d); fams[fam]=dict(costs=costs,difference=d)
        for r in rows:
            irrelevant_drift.append(abs(lookup[r['id'],f'irrelevant_{k}']['margin']-lookup[r['id'],'baseline']['margin']))
        out.append(dict(cut=k,difference=float(np.mean(ds)),families=fams,
                        irrelevant_absolute_margin_drift=float(np.mean(irrelevant_drift))))
    return out

def select_prediction(summary):
    # Fixed priority: greatest absolute matched difference, shallower cut breaks ties.
    ordered=sorted(summary,key=lambda x:(-abs(x['difference']),x['cut']))
    for s in ordered:
        sign=1 if s['difference']>0 else -1
        if (abs(s['difference'])>=.5 and
            all(v['difference']*sign>0 for v in s['families'].values()) and
            abs(s['difference'])>=2*s['irrelevant_absolute_margin_drift']):
            return dict(cut=s['cut'],sign=sign,development_difference=s['difference'],
                        prediction='Same sign in each evaluation family; absolute mean >= 0.5 nat')
    return None

def toy_calibration():
    # Known dependencies, not trained models. Distinct read schedules share every terminal answer.
    def run(schedule,cut,restore=(),bypass=False):
        answer=0; relay=1 if bypass else 0
        for b in range(28):
            allowed=b<cut or any(lo<=b<hi for lo,hi in restore)
            if b in schedule and allowed: answer=1
        return max(answer,relay)
    early=[run([3],k) for k in CUTS]; late=[run([20],k) for k in CUTS]
    repeated=[run([3,20],k) for k in CUTS]
    assert early==[0,1,1,1,1] and late==[0,0,0,1,1]
    assert repeated==early  # Earlier transfer can conceal subsequent rereading.
    assert run([20],0,[(14,21)])==1 and run([20],0,[(7,14)])==0
    assert run([20],0,bypass=True)==1  # An unblocked relay invalidates the isolation argument.
    return dict(kind='synthetic_known_programs_no_LLM_calls',cuts=CUTS,early=early,late=late,
        repeated=repeated,endpoint_answers_identical=True,
        limitations=['A preserved answer does not exclude repeated retrieval.',
                     'An unblocked relay can conceal source dependence.',
                     'These tests do not model off-distribution damage caused by attention removal.'])
