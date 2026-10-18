"""Saved-data report and figure only; no model execution or fitting."""
from pathlib import Path
import json,hashlib,sys,platform
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;D=P/'results'
z=np.load(D/'raw_outputs.npz');meta=json.loads((D/'MODEL_METADATA.json').read_text());inputs=json.loads((D/'INPUTS.json').read_text());baseline=json.loads((D/'CONGRUITY_CODE_BASELINE.json').read_text());pairs=json.loads((D/'PROFILE_COMPARISON.json').read_text())
true=np.array([len(set(s))==4 for s in inputs['held']]);colors={'sorting':'#2578a2','counting':'#d57c29'}
fig,ax=plt.subplots(2,2,figsize=(12.5,8.1),constrained_layout=True)
for k,kind in enumerate(['sorting','counting']):
 y=z[f'{kind}__held__summary_clamp_margins']
 ax[0,0].plot([0,1],[1,y.mean()],'o-',color=colors[kind],lw=2,label=kind)
 e=z[f'{kind}__held__s101__effects'][true]
 for j,style in [(5,'-'),(2,':')]:
  x=np.arange(e.shape[1])/(e.shape[1]-1)
  ax[0,1].plot(x,e[:,:,j].mean(0),style,color=colors[kind],lw=2,label=f'{kind}, dose {1 if j==5 else .375}')
 cut=meta[kind]['summary_cut']/(e.shape[1]-1)
 ax[0,1].axvline(cut,color=colors[kind],alpha=.25)
ax[0,0].axhline(0,color='#888',lw=.8)
ax[0,0].set(title='A  Frozen causal predictions both pass',xticks=[0,1],xticklabels=['untreated','replace tokens after summary'],ylabel='Mean output margin on true permutations',ylim=(-1.2,1.2));ax[0,0].legend(frameon=False)
ax[0,1].set(title='B  Token-source availability across operations',xlabel='Normalized embedding → final readout depth',ylabel='Mean effect on true permutations');ax[0,1].legend(frameon=False,fontsize=8)
ax[1,0].axis('off');table=[]
for a,b in [('sorting','sorting'),('sorting','counting'),('counting','sorting'),('counting','counting')]:
 r=[x for x in baseline if x['split']=='held' and x['source']==a and x['target']==b]
 table.append([f'{a} → {b}',f'{sum(x["reject_same"] for x in r)}/4','0/4' if a==b else '4/4'])
t=ax[1,0].table(cellText=table,colLabels=['Ordered comparison','Reject same','Expected'],cellLoc='left',colLoc='left',loc='center',colWidths=[.53,.24,.23]);t.auto_set_font_size(False);t.set_fontsize(10);t.scale(1,2)
ax[1,0].set_title('C  Released practical classifier: every held decision')
ax[1,0].text(.02,.08,'Restricted sampler • four correlated repetitions\nThis is not the published trained-model benchmark.',transform=ax[1,0].transAxes,fontsize=9)
values=np.zeros((2,2))
for i,a in enumerate(['sorting','counting']):
 for j,b in enumerate(['sorting','counting']):values[i,j]=next(p['development_distance'] for p in pairs if p['source']==a and p['target']==b)
im=ax[1,1].imshow(values,cmap='Blues',vmin=0,vmax=.12)
for i in range(2):
 for j in range(2):ax[1,1].text(j,i,f'{values[i,j]:.4f}',ha='center',va='center',color='black')
ax[1,1].set(title='D  Profile distances: architecture confounded',xticks=[0,1],yticks=[0,1],xticklabels=['sorting','counting'],yticklabels=['sorting','counting'],xlabel='Target algorithm',ylabel='Source algorithm')
fig.colorbar(im,ax=ax[1,1],fraction=.045,pad=.03)
for a in [ax[0,0],ax[0,1]]:a.grid(alpha=.18);a.spines[['top','right']].set_visible(False)
fig.suptitle('Compiled reference Transformers: same task, different causal dependencies\nN=4 • known circuit projection • restricted sampler\nDepth/width confounded: classification is not architecture-independent evidence',fontsize=12)
fig.savefig(D/'reference_comparison.svg');fig.savefig(D/'reference_comparison.png',dpi=170);plt.close(fig)
freeze=json.loads((P/'FREEZE_RECEIPT.json').read_text());check={k:hashlib.sha256((P/k).read_bytes()).hexdigest()==v for k,v in freeze['sha256'].items()};assert all(check.values())
(D/'FROZEN_SOURCE_AUDIT.json').write_text(json.dumps({'commit':freeze['commit'],'all_unchanged':True,'files':check},indent=2)+'\n')
import torch,sklearn
sys.path.insert(0,str(P.parent/'transformer_comparison_feasibility_2026-10-18/runtime'))
import jax,haiku,chex
(P/'ENVIRONMENT.json').write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'torch':torch.__version__,'sklearn':sklearn.__version__,'matplotlib':matplotlib.__version__,'jax':jax.__version__,'haiku':haiku.__version__,'chex':chex.__version__,'tracr_commit':'9ce2b8c82b6ba10e62e86cf6f390e7536d4fd2cd','torch_threads':2,'precision':'float64 CPU'},indent=2)+'\n')
entries=[]
for f in sorted(P.rglob('*')):
 if f.is_file() and '__pycache__' not in str(f) and f.name not in ['ARTIFACT_MANIFEST.json','GITHUB_PRESERVATION.json']:
  b=f.read_bytes();entries.append({'path':str(f.relative_to(P)),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'git_blob_sha1':hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()})
(P/'ARTIFACT_MANIFEST.json').write_text(json.dumps({'files':entries},indent=2)+'\n');print('Figure and frozen-source audit saved')
