"""Post-run descriptive reporting only; never changes frozen code or reruns outcomes."""
from pathlib import Path
import json,hashlib,platform,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent; R=P/'results'
z=np.load(R/'raw_arrays.npz'); rows=json.loads((R/'pair_results.json').read_text()); dsa=json.loads((R/'dsa_results.json').read_text())
summary=json.loads((R/'SUMMARY.json').read_text())
# Explicit numerical diagnostic; no repaired classification or extra experiment.
from benchmark import cka
h=z['s11__cached__base__natural_dev']; cost=np.array([1-cka(h[:,i],z['s11__cached__'+v+'__natural_dev'][:,j]) for v in ['rotation','width','affine'] for i in range(5) for j in range(5)])
diag={'CKA_nonfinite_distances':sum(not np.isfinite(r['distance']) for r in rows if r['method']=='cka'),
 'example_s11_cross_stage_cost_min':float(cost.min()),'cause':'Floating-point CKA slightly exceeds 1; frozen sqrt(mean(1-CKA)) returns NaN. No corrected rerun.',
 'impact':'CKA distance calibration invalid; finite alignment paths and prediction errors remain descriptive only.'}
(R/'NUMERICAL_LIMITATION.json').write_text(json.dumps(diag,indent=2)+'\n')
# All values below derive from saved arrays, with no new interventions.
fig,axs=plt.subplots(2,2,figsize=(12.5,8.3),constrained_layout=True)
colors={'cached':'#1976a3','recompute':'#d47928','threshold':'#8b5eaa'}
x=np.arange(5); labels=['input','cache','question','answer','readout']
ax=axs[0,0]
for kind,style in [('cached','-'),('recompute','--'),('threshold',':')]:
 a=z[f's11__{kind}__base__natural_dev'][0]
 ax.plot(x,a[:,2],style,color=colors[kind],lw=2.5,label=kind)
ax.set(title='A  Normal paths are exactly identical',ylabel='Stored role coordinate',xticks=x,xticklabels=labels,ylim=(-.12,1.18));ax.legend(frameon=False)
ax=axs[0,1]
for kind in colors:
 a=z[f's11__{kind}__base__effects_dev'][:,:,1].mean(0)
 ax.plot(x,a,'o-',color=colors[kind],lw=2,label=kind)
ax.set(title='B  Full role swaps miss smooth vs threshold',ylabel='Normalized output change',xticks=x,xticklabels=labels,ylim=(-.12,1.18))
ax=axs[1,0]
for kind in colors:
 a=z[f's11__{kind}__base__effects_dev'][:,1]
 values=[0,a[:,3].mean(),a[:,4].mean(),a[:,1].mean()]
 ax.plot([0,.25,.75,1],values,'o-',color=colors[kind],lw=2,label=kind)
ax.set(title='C  Graded swaps distinguish all three routes',xlabel='Role intervention dose at cache stage',ylabel='Normalized output change');ax.legend(frameon=False)
ax=axs[1,1]
q=z['cases_held'][:,3]<0
for name,label,color in [('cached__base','Affine-domain reference','#1976a3'),('cached__nonlinear','Nonlinear coordinates','#b23c51')]:
 a=z[f's11__{name}__effects_held'][q,:,4].mean(0)
 ax.plot(x,a,'o-',label=label,color=color,lw=2)
a=z['s11__context__base__effects_held'];c=z['cases_held'][:,6]
for gate,style,label in [(abs(c)<.75,'--','Context route: recompute'),(abs(c)>.75,':','Context route: cached')]:
 ax.plot(x,a[q&gate,:,4].mean(0),style,color='#587744',lw=2,label=label)
ax.set(title='D  Two declared limits on transfer',ylabel='Held-out dose .75 response',xticks=x,xticklabels=labels,ylim=(-.12,1.23));ax.legend(frameon=False,fontsize=8,loc='upper right')
for ax in axs.flat:
 ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False);ax.tick_params(axis='x',labelsize=9)
fig.suptitle('Identical paths, different causal use\nKnown programs — not pretrained LLM findings',fontsize=15)
fig.savefig(R/'identification_ladder.svg');fig.savefig(R/'identification_ladder.png',dpi=170);plt.close(fig)
# Store strict JSON companion, preserving NaN-bearing originals verbatim.
def strict(v):
 if isinstance(v,float) and not np.isfinite(v):return None
 if isinstance(v,dict):return {k:strict(a) for k,a in v.items()}
 if isinstance(v,list):return [strict(a) for a in v]
 return v
(R/'pair_results.strict.json').write_text(json.dumps(strict(rows),separators=(',',':'),allow_nan=False)+'\n')
(R/'SUMMARY.strict.json').write_text(json.dumps(strict(summary),indent=2,allow_nan=False)+'\n')
# Portable environment record; exact dependency source is vendored for the DSA path.
import scipy,torch
(P/'ENVIRONMENT.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__,'matplotlib':matplotlib.__version__,'inference':'None; deterministic toy programs on CPU'},indent=2)+'\n')
entries=[]
for f in sorted(P.rglob('*')):
 if f.is_file() and '__pycache__' not in str(f) and f.name not in ['ARTIFACT_MANIFEST.json','GITHUB_PRESERVATION.json']:
  b=f.read_bytes();entries.append({'path':str(f.relative_to(P)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'git_blob_sha1':hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()})
(P/'ARTIFACT_MANIFEST.json').write_text(json.dumps({'files':entries},indent=2)+'\n')
print(json.dumps(diag))
