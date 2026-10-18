"""Synthetic integration and information-path checks; no target model imports."""
import json
from pathlib import Path
import tempfile
import numpy as np
from core import BLOCKS,LENGTH,CONDITIONS,mask,save,score,matched_summary,select_prediction,toy_calibration

def check(rows):
    r=rows[0]; s=r['suffix_start']; lo,hi=r['source_span']
    base=np.tri(LENGTH,dtype=bool)
    zero=next(c for c in CONDITIONS if c['name']=='cut_0')
    assert hi==s
    for c in CONDITIONS:
        for b in range(BLOCKS):
            m=mask(r,c,b)
            assert np.array_equal(m[:s],base[:s]) and m.any(axis=1).all()
            assert np.all(~m|base)
    for c in CONDITIONS:
        if c['name'] in ['cut_28','restore_all']:
            assert all(np.array_equal(mask(r,c,b),base) for b in range(BLOCKS))
    # Boolean structural dependence: no unblocked event-to-suffix relay at cut zero.
    dependency=np.zeros(LENGTH,dtype=bool); dependency[lo:hi]=True
    for b in range(BLOCKS):
        m=mask(r,zero,b)
        dependency|=(m & dependency[None,:]).any(axis=1)
        assert not dependency[s:].any()
    # A final-position-only cut would leak through earlier suffix tokens.
    dependency=np.zeros(LENGTH,dtype=bool); dependency[lo:hi]=True
    wrong=base.copy(); wrong[r['prompt_end'],lo:hi]=False
    for b in range(2): dependency|=(wrong & dependency[None,:]).any(axis=1)
    assert dependency[r['prompt_end']]
    selected=[x for x in rows if x['model']=='qwen7b' and x['split']=='development']
    synthetic=[]
    for row in selected:
        for name in ['baseline']+[c['name'] for c in CONDITIONS]:
            margin=10.
            if name.startswith('cut_'):
                k=int(name.split('_')[1])
                margin-=({0:10,7:1,14:2,21:.2,28:0}[k] if row['query']=='agent' else 0)
            synthetic.append(dict(id=row['id'],condition=name,margin=margin,correct=margin>0))
    summary=matched_summary(selected,synthetic); prediction=select_prediction(summary)
    assert prediction['cut']==14 and prediction['sign']==1
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'raw.json'; save(p,synthetic)
        assert json.loads(p.read_text())==synthetic
        all_ids=[i for ids in r['aliases'].values() for i in ids]
        logits=np.full(max(all_ids)+2,-20.,dtype=np.float32)
        logits[r['aliases'][r['expected']][0]]=20.
        out=score(logits,r,lambda ids:str(ids))
        assert out['correct'] and out['expected']==r['expected']
        np.save(Path(td)/'logits.npy',logits)
        assert np.array_equal(np.load(Path(td)/'logits.npy'),logits)
        save(Path(td)/'scored.json',out)
        assert json.loads((Path(td)/'scored.json').read_text())==out
    toy=toy_calibration()
    return dict(status='PASS_SYNTHETIC_ONLY',target_model_calls=0,
        structural_no_relay=True,final_token_only_leak_detected=True,
        mask_controls=True,scoring_persistence_roundtrip=True,prediction_selection=True,
        known_programs=toy,tokenized_rows_checked=len(rows))
