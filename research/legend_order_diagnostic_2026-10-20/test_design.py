"""CPU-only discriminability checks; synthetic outputs are never empirical data."""
from core import *
def check(rows):
    assert len(rows)==32
    for name in PREDICTIONS:
        synthetic=[{**r,'actual':predicted(r,name),'native_correct':predicted(r,name)==r['expected']} for r in rows]
        assert classify(synthetic)['candidate']==name
    for symbol in ['A','B']:
        synthetic=[]
        for r in rows:
            actual='entailed' if symbol==('A' if r['mapping']==0 else 'B') else 'not_entailed'
            synthetic.append({**r,'actual':actual,'native_correct':actual==r['expected']})
        assert classify(synthetic)['candidate'] is None
    assert all(sum(predicted(r,a)==predicted(r,b) for r in rows)==16 for i,a in enumerate(PREDICTIONS) for b in PREDICTIONS[i+1:])
    assert sum(r['expected']=='entailed' for r in rows)==16
    return dict(real_model_forwards=0,three_predictions_distinct=True,pairwise_prediction_agreement=16,constant_symbol_null_rejected=True,trial_balance_verified=True)
