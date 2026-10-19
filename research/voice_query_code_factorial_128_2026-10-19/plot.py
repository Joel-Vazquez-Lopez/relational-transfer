"""Descriptive full-factorial figure only when empirical RESULTS exists."""
import json
from core import HERE
if not (HERE/'RESULTS.json').exists():print('No fresh empirical outputs: no scientific figure generated');raise SystemExit(0)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=json.loads((HERE/'RESULTS.json').read_text());profiles=r['profiles'];labels=[(q,l,m) for q in ['active','passive'] for l in ['YN','AB'] for m in [0,1]];fig,axes=plt.subplots(1,3,figsize=(17,5),layout='constrained');limit=max(4,max(abs(c['gap']) for p in profiles for c in p['contrasts']))
for ax,comparison in zip(axes,['same_position0','same_position1','matched_period']):
 data=np.array([[next(c['gap'] for c in p['contrasts'] if (c['question_voice'],c['lexicon'],c['mapping'],c['comparison'])==(q,l,m,comparison)) for q,l,m in labels] for p in profiles]);im=ax.imshow(data,cmap='coolwarm',vmin=-limit,vmax=limit,aspect='auto');ax.set(yticks=range(6),yticklabels=[p['family'] for p in profiles],xticks=range(8),xticklabels=[q+'\n'+l+' m'+str(m) for q,l,m in labels],title=comparison);ax.axhline(2.5,color='black',ls='--')
 for y in range(6):
  for x in range(8):ax.text(x,y,f'{data[y,x]:.1f}',ha='center',va='center',fontsize=8)
fig.colorbar(im,ax=axes,label='Passive−active positive-target score gap (nats)');fig.suptitle('Every factorial contrast; dashed line separates development/evaluation. Descriptive only.')
for ext in ['png','svg','pdf']:fig.savefig(HERE/('FACTORIAL_GAPS.'+ext),dpi=180)
