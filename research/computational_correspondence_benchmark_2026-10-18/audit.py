"""Independent oracle checks of frozen saved outcomes, not an alternate experiment."""
import hashlib,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
z=np.load(P/'results/raw_arrays.npz')
checks=[]
for seed in [11,23,37,41,53,67,79,97]:
    prefix=f's{seed}__'
    ref=z[prefix+'cached__base__natural_dev']
    for kind in ['recompute','threshold','context']:
        a=z[prefix+kind+'__base__natural_dev']
        checks.append((f'{seed}/{kind}/exact_observational_clone',np.array_equal(ref,a)))
    for split in ['dev','held']:
        x=z['cases_'+split]
        for kind in ['cached','recompute','threshold','context']:
            effects=z[prefix+kind+'__base__effects_'+split]
            expect=np.zeros_like(effects)
            expect[:,:,0]=1.
            for i,row in enumerate(x):
                route=kind
                if route=='context': route='cached' if abs(row[6])>.75 else 'recompute'
                expect[i,0,2]=1.
                if route=='recompute': expect[i,1:3,2]=1.
                else:
                    expect[i,1:3,1]=1.
                    expect[i,1:3,3]=.25 if route=='cached' else 0.
                    expect[i,1:3,4]=.75 if route=='cached' else 1.
            error=float(np.max(np.abs(effects-expect)))
            checks.append((f'{seed}/{kind}/{split}/all_case_stage_channel_oracle',error<1e-9))
    a=z[prefix+'cached__nonlinear__effects_held']
    checks.append((f'{seed}/nonlinear/predicted_overshoot',np.max(abs(a[:,1:3,4]-1.125))<1e-9))
summary={'checks':len(checks),'passed':sum(bool(v) for _,v in checks),'failures':[k for k,v in checks if not v],
         'raw_sha256':hashlib.sha256((P/'results/raw_arrays.npz').read_bytes()).hexdigest()}
(P/'results/INDEPENDENT_AUDIT.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
assert all(v for _,v in checks)
