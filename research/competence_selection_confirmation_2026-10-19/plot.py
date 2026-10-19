"""Descriptive scientific figure; no endpoint selection or additional inference."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import HERE
r=json.loads((HERE/'RESULTS.json').read_text());models=['qwen7b','llama3b'];colors=['#2766ac','#c35b24'];fig,axes=plt.subplots(2,2,figsize=(13,10),layout='constrained')
a=axes[0,0]
for i,m in enumerate(models):
 for j,v in enumerate(['active','passive']):
  u=next(t for t in r['totals'] if (t['model'],t['voice'])==(m,v));x=i*3+j
  a.bar(x-.16,u['native_complete'],.30,color=colors[i],alpha=.9);a.bar(x+.16,u['calibrated_complete'],.30,color=colors[i],alpha=.30,hatch='//');a.text(x-.16,u['native_complete']+.3,str(u['native_complete']),ha='center',fontsize=9);a.text(x+.16,u['calibrated_complete']+.3,str(u['calibrated_complete']),ha='center',fontsize=9)
a.set(xticks=[0,1,3,4],xticklabels=['Qwen\nactive','Qwen\npassive','Llama\nactive','Llama\npassive'],ylim=(0,27),ylabel='Complete four-answer units /24',title='A. Which constructions pass?')
a.bar([],[],color='gray',label='Native next token');a.bar([],[],color='gray',alpha=.3,hatch='//',label='Old fixed threshold');a.legend(fontsize=8)
a=axes[0,1]
for u in r['units']:
 if u['model']!='qwen7b':continue
 y=int(u['family'][1:])+(0.14 if u['voice']=='active' else -.14);lo,hi=u['interval'];c='#2766ac' if u['voice']=='active' else '#ba7132'
 a.plot([lo,hi],[y,y],color=c if lo<hi else '#bb2233',lw=2,alpha=.8);a.plot([lo,hi],[y,y],'.',color=c)
a.plot([],[],color='#2766ac',label='Active');a.plot([],[],color='#ba7132',label='Passive')
a.axvline(0,color='black',lw=1,label='Zero alias-score boundary');a.axvline(-19.507811598961112,color='#418443',ls='--',label='Frozen external threshold');a.set(yticks=range(1,25,2),yticklabels=[f'c{x:02d}' for x in range(1,25,2)],xlabel='Yes–No score (nats)',title='B. Qwen separating intervals (all families)',ylabel='Lexical family');a.invert_yaxis();a.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,-.13),ncol=2)
a=axes[1,0];gates=['all','native_complete','calibrated_complete']
for i,m in enumerate(models):
 vals=[]
 for g in gates:
  p=next(p for p in r['profiles'] if (p['mode'],p['gate'],p['policy'])==('frozen',g,'own_selected'));x=p['models'][m];val=x['mean_retention'];vals.append(np.nan if val is None else val)
  if val is not None:a.text(gates.index(g)+(.04 if i else -.04),val+.025,f"n={x['n_ratio']}",fontsize=8,color=colors[i],ha='left' if i else 'right')
 a.plot(range(3),vals,'o-',color=colors[i],label=m)
a.axhline(1,color='gray',lw=.7,ls=':');a.set(xticks=range(3),xticklabels=['All','Native complete','Fixed-threshold\ncomplete'],ylabel='Mean D(mask) / D(native)',title='C. Fixed period-isolated cut14 assay');a.margins(x=.15,y=.15);a.legend(fontsize=8)
a=axes[1,1]
for i,m in enumerate(models):
 for v,mark in [('active','o'),('passive','^')]:
  us=[u for u in r['units'] if u['model']==m and u['voice']==v];a.scatter([u['D_native'] for u in us],[u['interventions']['frozen']['D'] for u in us],marker=mark,color=colors[i],alpha=.65,s=30,label=m+' '+v)
lims=a.get_xlim();a.plot(lims,lims,color='gray',ls=':',lw=.8);a.set(xlabel='Native role × question contrast D (nats)',ylabel='Intervened contrast D (nats)',title='D. All units retained in raw-score comparison');a.legend(fontsize=8)
fig.suptitle('Fresh-panel competence and causal comparisons\n24 lexical families · fixed prompts, thresholds and cut · AI-only annotations',fontsize=14)
for suffix in ['png','svg','pdf']:fig.savefig(HERE/f'CONFIRMATION.{suffix}',dpi=180)
