"""Post-run descriptive analysis of saved data only; no new neural forward passes."""
import json,hashlib,platform,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;D=P/'results'
z=np.load(D/'raw_outputs.npz');rows=json.loads((D/'cases_held.json').read_text());pairs=json.loads((D/'PAIR_RESULTS.json').read_text());summary=json.loads((D/'SUMMARY.json').read_text())
patient=np.array([a['Q']<0 for a in rows]);details=[];colors=['#237aab','#e07b32','#785ba1']
for seed in [11,23,37]:
    e=z[f's{seed}__cached__held__nonlinear__effects'];logits=z[f's{seed}__cached__held__nonlinear__logits']
    delta=e[:,:,1:5,2]-e[:,:,1:5,0]
    disagree=logits[:,:,1:5,0]*logits[:,:,1:5,2]<0
    details.append({'seed':seed,'four_interior_doses_only':True,'individual_case_stage_dose_entries':delta.size,
        'rmse':float(np.sqrt(np.mean(delta**2))),'stage_rmse':np.sqrt(np.mean(delta**2,axis=(0,2))).tolist(),
        'abs_error_gt_point05':int(np.sum(abs(delta)>.05)),'abs_error_gt_point1':int(np.sum(abs(delta)>.1)),
        'answer_identity_disagreements':int(disagree.sum()),'stage_answer_disagreement_fraction':disagree.mean(axis=(0,2)).tolist()})
(D/'NONLINEAR_DETAIL.json').write_text(json.dumps(details,indent=2)+'\n')
fig,axs=plt.subplots(2,2,figsize=(12.5,8.6),constrained_layout=True)
for seed,c in zip([11,23,37],colors):
    e=z[f's{seed}__cached__held__identity__effects']
    axs[0,0].plot([1,2,3],e[patient,:,2,0].mean(0),'o-',color=c,label=f'seed {seed}')
    a=next(r for r in details if r['seed']==seed)
    axs[0,1].plot([1,2,3],a['stage_rmse'],'o-',color=c,label=f'blind, seed {seed}')
    axs[1,0].plot([0,.125,.375,.625,.875,1],e[patient,2,:,0].mean(0),'o-',color=c,label=f'seed {seed}')
axs[0,0].axhline(0,color='#555',linestyle='--',label='unused matched cache')
axs[0,0].set(title='A  Developmental responses across trained seeds',xlabel='Encoder stage',ylabel='Mean native response at dose .375',xticks=[1,2,3]);axs[0,0].legend(frameon=False,fontsize=9)
axs[0,1].axhline(0,color='#333',linestyle='--',label='exact transport / linear control')
axs[0,1].set(title='B  Coordinate mismatch concentrates early',xlabel='Encoder stage',ylabel='Blind vs native response RMSE',xticks=[1,2,3]);axs[0,1].legend(frameon=False,fontsize=8)
axs[1,0].axhline(0,color='#555',linestyle='--',label='bypass cache')
axs[1,0].set(title='C  Final-stage dose responses are close',xlabel='Intervention dose',ylabel='Mean normalized response');axs[1,0].legend(frameon=False,fontsize=9)
pred=z['prediction_s23_t37_cached_causal'];truth=z['s37__cached__held__identity__effects'][patient,:,1:5,0]
for k,c in enumerate(colors):axs[1,1].scatter(truth[:,k].ravel(),pred[:,k].ravel(),s=14,alpha=.5,color=c,label=f'stage {k+1}')
lo=min(truth.min(),pred.min());hi=max(truth.max(),pred.max());axs[1,1].plot([lo,hi],[lo,hi],color='#333',lw=1)
axs[1,1].set(title='D  One cross-seed prediction fails the frozen gate',xlabel='Observed target37 response',ylabel='Prediction from seed23')
axs[1,1].text(.04,.96,'23→37 RMSE .147: FAIL\n11→37 RMSE .054: PASS\nFrozen gate: RMSE ≤ .10',transform=axs[1,1].transAxes,va='top',fontsize=9)
axs[1,1].legend(frameon=False,fontsize=8,loc='lower right')
for ax in axs.flat:ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Trained networks: route classification passes; response transfer is incomplete\nThree paired seeds • supplied grammar features • no pretrained LLM claim',fontsize=14)
fig.savefig(D/'learned_transport.svg');fig.savefig(D/'learned_transport.png',dpi=170);plt.close(fig)
freeze=json.loads((P/'FREEZE_RECEIPT.json').read_text());unchanged={n:hashlib.sha256((P/n).read_bytes()).hexdigest()==v for n,v in freeze['sha256'].items()}
assert all(unchanged.values())
(D/'FROZEN_SOURCE_AUDIT.json').write_text(json.dumps({'commit':freeze['commit'],'all_unchanged':True,'files':unchanged},indent=2)+'\n')
import torch
(P/'ENVIRONMENT.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'torch':torch.__version__,'matplotlib':matplotlib.__version__,'device':'CPU, float64, 2 Torch threads','training_steps_per_route':1800},indent=2)+'\n')
entries=[]
for f in sorted(P.rglob('*')):
 if f.is_file() and '__pycache__' not in str(f) and f.name not in ['ARTIFACT_MANIFEST.json','GITHUB_PRESERVATION.json']:
  b=f.read_bytes();entries.append({'path':str(f.relative_to(P)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'git_blob_sha1':hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()})
(P/'ARTIFACT_MANIFEST.json').write_text(json.dumps({'files':entries},indent=2)+'\n')
print(json.dumps({'nonlinear_interior':details,'frozen_unchanged':all(unchanged.values())},indent=2))
