"""Inference-free definitions. A carrier is a token position, not a semantic label."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
LENGTH, BLOCKS = 128, 28
CUTS = [0, 7, 14, 21]
MODES = ['closed', 'frozen', 'live', 'neighbor', 'irrelevant']
MODELS = {'qwen7b': 'research/stronger_model_development_2026-10-15/models/qwen7b',
          'llama3b': 'research/llama_gemma_cross_family_2026-10-15/model'}
FAMILIES = [('g01', 'development', 'Alice', 'Owen', 'greeted', 'greet'),
            ('g02', 'development', 'Hannah', 'Felix', 'helped', 'help'),
            ('g03', 'evaluation', 'Nora', 'Caleb', 'followed', 'follow'),
            ('g04', 'evaluation', 'Iris', 'Leon', 'thanked', 'thank'),
            ('g05', 'evaluation', 'Elena', 'Marcus', 'visited', 'visit'),
            ('g06', 'evaluation', 'Zoe', 'Adam', 'praised', 'praise')]
SCIENTIFIC_MODES = ['closed', 'frozen', 'live', 'irrelevant']

def jobs():
    return [(mode,k) for k in CUTS for mode in SCIENTIFIC_MODES]+[('neighbor',14)]

def sensitivity(rr):
    assert len(rr)==4 and {(r['role'],r['query']) for r in rr}=={(0,0),(0,1),(1,0),(1,1)}
    vals={(r['role'],r['query']):r['class_logits']['Yes']-r['class_logits']['No'] for r in rr}
    d=sum(((-1)**(role+q))*s for (role,q),s in vals.items())/4
    paired=[]
    for q in [0,1]: paired.append(((-1)**q)*(vals[0,q]-vals[1,q]))
    for role in [0,1]: paired.append(((-1)**role)*(vals[role,0]-vals[role,1]))
    return dict(D=float(d),ordinal_pairs=paired,all_ordinal_pairs_positive=all(v>0 for v in paired),
                native_correct=sum(r['correct'] for r in rr),n=4)

def summarize(rows,raw):
    out=[]
    for fam in sorted({r['family'] for r in rows}):
        for voice in ['active','passive']:
            base=[r for r in raw if r['kind']=='scientific' and r['family']==fam and r['voice']==voice and r['mode']=='baseline']
            b=sensitivity(base)
            for mode,k in jobs():
                rr=[r for r in raw if r['kind']=='scientific' and r['family']==fam and r['voice']==voice and r['mode']==mode and r['cut']==k]
                s=sensitivity(rr)
                out.append(dict(family=fam,voice=voice,mode=mode,cut=k,baseline=b,intervention=s,
                                ratio=(s['D']/b['D']) if abs(b['D'])>=.5 else None))
    return out

def select_prediction(summary):
    lookup={(s['family'],s['voice'],s['mode'],s['cut']):s for s in summary}
    cells=sorted({(s['family'],s['voice']) for s in summary})
    for signature in ['cached','refresh']:
        eligible=[]
        for k in [7,14,21]:
            effects=[];ok=True
            for fam,voice in cells:
                g={m:lookup[fam,voice,m,k] for m in SCIENTIFIC_MODES}
                b=g['closed']['baseline']
                if b['D']<.5 or not b['all_ordinal_pairs_positive'] or any(s['ratio'] is None for s in g.values()):
                    ok=False;continue
                r={m:s['ratio'] for m,s in g.items()}
                target,rival=('frozen','closed') if signature=='cached' else ('live','frozen')
                effect=r[target]-r[rival];effects.append(effect)
                ok=ok and abs(r['irrelevant']-1)<=.25 and r[target]>=.5 and r[rival]<=.75 and effect>=.25
            if ok and len(effects)==len(cells):eligible.append(dict(signature=signature,cut=k,min_effect=min(effects)))
        if eligible:return sorted(eligible,key=lambda p:(-p['min_effect'],p['cut']))[0]
    # A distinct alternative: the question-side workspace already retains the
    # response after complete sentence disconnection, with a preceding loss.
    for lo,hi in [(0,7),(7,14),(14,21)]:
        effects=[];ok=True
        for fam,voice in cells:
            a=lookup[fam,voice,'closed',lo];b=lookup[fam,voice,'closed',hi]
            na=lookup[fam,voice,'irrelevant',lo];nb=lookup[fam,voice,'irrelevant',hi]
            native=b['baseline']
            ratios=[s['ratio'] for s in [a,b,na,nb]]
            if native['D']<.5 or not native['all_ordinal_pairs_positive'] or any(r is None for r in ratios):ok=False;continue
            ok=ok and a['ratio']<=.25 and b['ratio']>=.75 and abs(na['ratio']-1)<=.25 and abs(nb['ratio']-1)<=.25
            effects.append(b['ratio']-a['ratio'])
        if ok and len(effects)==len(cells):return dict(signature='suffix',lower_cut=lo,cut=hi,min_effect=min(effects))
    return None

def prediction_cells(summary,prediction):
    if prediction is None:return []
    out=[];k=prediction['cut'];sig=prediction['signature']
    for fam,voice in sorted({(s['family'],s['voice']) for s in summary}):
        if sig=='suffix':
            lo=prediction['lower_cut'];chosen={ (s['mode'],s['cut']):s for s in summary if s['family']==fam and s['voice']==voice }
            a,b,na,nb=[chosen[m,j] for m,j in [('closed',lo),('closed',k),('irrelevant',lo),('irrelevant',k)]]
            base=b['baseline'];valid=base['D']>=.5 and base['all_ordinal_pairs_positive'] and all(s['ratio'] is not None for s in [a,b,na,nb])
            passed=bool(valid and a['ratio']<=.25 and b['ratio']>=.75 and abs(na['ratio']-1)<=.25 and abs(nb['ratio']-1)<=.25)
            out.append(dict(family=fam,voice=voice,passed=passed,baseline=base,
                ratios=dict(closed_lower=a['ratio'],closed_upper=b['ratio'],irrelevant_lower=na['ratio'],irrelevant_upper=nb['ratio']),
                effect=b['ratio']-a['ratio'] if valid else None))
            continue
        g={s['mode']:s for s in summary if s['family']==fam and s['voice']==voice and s['cut']==k}
        b=g['closed']['baseline'];r={m:g[m]['ratio'] for m in SCIENTIFIC_MODES}
        target,rival=('frozen','closed') if sig=='cached' else ('live','frozen')
        usable=b['D']>=.5 and b['all_ordinal_pairs_positive'] and all(v is not None for v in r.values())
        effect=r[target]-r[rival] if usable else None
        passed=bool(usable and abs(r['irrelevant']-1)<=.25 and r[target]>=.5 and r[rival]<=.75 and effect>=.25)
        out.append(dict(family=fam,voice=voice,passed=passed,baseline=b,ratios=r,effect=effect))
    return out

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda: f.read(2**20), b''): h.update(c)
    return h.hexdigest()

def save(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    os.replace(t, p)

def mask(row, mode, cut, block):
    a = np.tri(LENGTH, dtype=bool)
    if block < cut or mode == 'baseline': return a
    s, p = row['source_start'], row['period']
    if mode == 'closed':
        a[p:, s:p] = False
        a[p+1:, p] = False
    elif mode == 'frozen':
        # Block refresh of the carrier itself, and all later bypasses.
        a[p:, s:p] = False
    elif mode == 'live':
        # The carrier can continue to reread the entire source.
        a[p+1:, s:p] = False
    elif mode == 'neighbor':
        n = p-1
        a[n:, s:n] = False
        a[n:, n+1:p+1] = False
    elif mode == 'irrelevant':
        # Translate the exact closed-edge deletion pattern onto neutral keys.
        lo, hi = row['neutral_span']
        assert hi-lo == p-s+1
        a[p:, lo:hi-1] = False
        a[p+1:, hi-1] = False
    else: raise ValueError(mode)
    assert a.any(axis=1).all()
    assert not a[np.triu_indices(LENGTH, 1)].any()
    return a

def lse(x):
    x = np.asarray(x, dtype=np.float64); m = float(np.max(x))
    return float(m + np.log(np.exp(x-m).sum()))

def score(logits, row, decode):
    ids = row['aliases']; best = int(np.argmax(logits))
    cls = {k: lse(logits[v]) for k,v in ids.items()}
    other = 'No' if row['expected'] == 'Yes' else 'Yes'
    margin = cls[row['expected']] - cls[other]
    actual = next((k for k,v in ids.items() if best in v), None)
    top = np.argsort(logits)[-20:][::-1]
    return dict(expected=row['expected'], actual=actual, argmax_id=best,
                argmax_token=decode([best]), gold_margin=margin,
                correct=bool(actual == row['expected'] and margin > 0),
                class_logits=cls, alias_logits={k:[float(logits[i]) for i in v] for k,v in ids.items()},
                top20=[dict(id=int(i), token=decode([int(i)]), logit=float(logits[i])) for i in top])

def calibration():
    # Explicit known dependency schedules, not a fit to LLM outcomes.
    outputs={}
    for program in ['cached','refresh','suffix_cache','late_direct']:
        outputs[program]={}
        for k in CUTS:
            v={}
            for mode in ['closed','frozen','live','irrelevant']:
                source=1; carrier=0; suffix=0
                for b in range(BLOCKS):
                    active=b>=k
                    period_access=not(active and mode in ['closed','frozen'])
                    suffix_direct=not(active and mode in ['closed','frozen','live'])
                    suffix_carrier=not(active and mode=='closed')
                    if program=='cached' and b==3 and period_access: carrier=source
                    if program=='refresh' and b==18 and period_access: carrier=source
                    if program in ['cached','refresh'] and b==23 and suffix_carrier: suffix=carrier
                    if program=='suffix_cache' and b==3 and suffix_direct: suffix=source
                    if program=='late_direct' and b==23 and suffix_direct: suffix=source
                v[mode]=suffix
            outputs[program][k]=v
    assert outputs['cached'][7]==dict(closed=0,frozen=1,live=1,irrelevant=1)
    assert outputs['refresh'][7]==dict(closed=0,frozen=0,live=1,irrelevant=1)
    assert outputs['suffix_cache'][7]==dict(closed=1,frozen=1,live=1,irrelevant=1)
    assert outputs['late_direct'][7]==dict(closed=0,frozen=0,live=0,irrelevant=1)
    return dict(kind='known_schedules_not_empirical',outputs=outputs,
        limits=['A cached syntax key and a cached semantic variable can have the same signature.',
                'Preserved suffix states may already contain source information at the cut.',
                'A null carrier effect does not exclude distributed or other-position carriers.'])
