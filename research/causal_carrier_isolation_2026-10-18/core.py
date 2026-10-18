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
FAMILIES = [('d1', 'development', 'Julia', 'Mark', 'greeted', 'greet'),
            ('e1', 'evaluation', 'Rachel', 'Victor', 'praised', 'praise'),
            ('e2', 'evaluation', 'Diana', 'Simon', 'helped', 'help'),
            ('e3', 'evaluation', 'Laura', 'Peter', 'thanked', 'thank')]

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

def summarize(rows, raw):
    lookup = {(r['id'],r['mode'],r['cut']):r for r in raw if r['kind']=='scientific'}
    output = []
    for family in sorted({r['family'] for r in rows}):
        rr = [r for r in rows if r['family']==family]
        for k in CUTS:
            groups = {m:[lookup[r['id'],m,k] for r in rr] for m in MODES}
            baseline = [lookup[r['id'],'baseline',28] for r in rr]
            margins = {m:float(np.mean([r['gold_margin'] for r in v])) for m,v in groups.items()}
            drift = float(np.mean([abs(a['gold_margin']-b['gold_margin']) for a,b in zip(groups['irrelevant'],baseline)]))
            output.append(dict(family=family,cut=k,n=len(rr),
                correct={m:sum(r['correct'] for r in v) for m,v in groups.items()},
                mean_margin=margins,neutral_absolute_drift=drift,
                frozen_rescue=margins['frozen']-margins['closed'],
                refresh_benefit=margins['live']-margins['frozen'],
                boundary_advantage=margins['frozen']-margins['neighbor']))
    return output

def qualifies(s, mode):
    # Two mutually distinguishable prospective signatures; no label of semantics.
    effect = s['frozen_rescue'] if mode=='cached' else s['refresh_benefit']
    target, rival = ('frozen','closed') if mode=='cached' else ('live','frozen')
    return (s['cut']>0 and s['correct'][target]==8 and s['correct'][rival]<=4
            and s['correct']['irrelevant']==8 and effect>=1.
            and effect>=2*s['neutral_absolute_drift'])

def select_prediction(summary):
    # Cached-carrier signature has explicit priority; otherwise test live refresh.
    for label in ['cached','refresh']:
        eligible=[s for s in summary if qualifies(s,label)]
        if eligible:
            key='frozen_rescue' if label=='cached' else 'refresh_benefit'
            chosen=sorted(eligible,key=lambda s:(-s[key],s['cut']))[0]
            return dict(signature=label,cut=chosen['cut'],development_effect=chosen[key],
                        rule='Each fresh family must satisfy the same numerical and all-answer criteria')
    return None

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
