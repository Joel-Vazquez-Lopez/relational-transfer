"""Descriptive figures of declared results only; no model calls or new tests."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
ROLES=['Sender','Sent person','Destination person'];FORMS=['active','passive','goal_fronted']
COLORS=['#2864a0','#d88725','#47966a']
def save(fig,name):
 for ext in ['png','svg','pdf']:fig.savefig(HERE/(name+'.'+ext),dpi=180,bbox_inches='tight')
 plt.close(fig)
def main():
 fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
 for j,(folder,title) in enumerate([(HERE.parent/'role_position_source_access_2026-10-20','Llama 3B: exposed discovery'),(HERE,'Qwen 7B: prospective counterpart')]):
  summary=json.loads((folder/'NATIVE_SUMMARY.json').read_text());gate=json.loads((folder/'NATIVE_GATE.json').read_text())
  for k,form in enumerate(FORMS):
   vals=[sum(r['correct'] for r in summary['by_family_form_role'] if r['form']==form and r['role_index']==ri)/sum(r['total'] for r in summary['by_family_form_role'] if r['form']==form and r['role_index']==ri) for ri in range(3)]
   axes[0,j].bar(np.arange(3)+(k-1)*.23,vals,width=.22,label=form.replace('_',' '),color=COLORS[k])
  effects=gate.get('effects')
  if effects is None:
   rows=json.loads((folder/'EXPLORATORY_INTERACTIONS.json').read_text())['rows']
   effects=[dict(role_index=r['role_index'],form=r['form'],interaction=r['effect']) for r in rows]
  for k,form in enumerate(FORMS):
   for ri in range(3):
    v=[r['interaction'] for r in effects if r['form']==form and r['role_index']==ri];x=ri+(k-1)*.23+np.linspace(-.055,.055,len(v))
    axes[1,j].scatter(x,v,s=22,color=COLORS[k],alpha=.75)
  axes[0,j].set(title=title,ylim=(0,1.07),xticks=range(3),xticklabels=ROLES,ylabel='Exact full-answer accuracy')
  axes[1,j].axhline(0,c='black',lw=.7);axes[1,j].axhline(1,c='gray',ls=':',lw=.8)
  axes[1,j].set(xticks=range(3),xticklabels=ROLES,ylabel='Native multi-query role-response R (logits)')
  for ax in axes[:,j]:ax.spines[['top','right']].set_visible(False)
 axes[0,0].legend(frameon=False,fontsize=9)
 fig.suptitle('Answer accuracy and relational response measure different things\nTwo name/question sets, four rotations, one verb; dependent observations',fontsize=14)
 save(fig,'NATIVE_RESPONSE')
 if not (HERE/'RESULTS.json').exists():return
 results=json.loads((HERE/'RESULTS.json').read_text());profiles=results['profiles']
 fig,axes=plt.subplots(1,3,figsize=(13,4.3),layout='constrained',sharey=True)
 for ri,ax in enumerate(axes):
  for k,form in enumerate(FORMS):
   rows=[r for r in profiles if r['role_index']==ri and r['form']==form and r['eligible']]
   for unit in sorted({r['unit'] for r in rows}):
    rr=sorted([r for r in rows if r['unit']==unit],key=lambda r:r['cut']);ax.plot([r['cut'] for r in rr],[r['authority'] for r in rr],color=COLORS[k],alpha=.2,lw=.7)
   means=[np.mean([r['authority'] for r in rows if r['cut']==c]) for c in [7,14,21]]
   ax.plot([7,14,21],means,'o-',c=COLORS[k],lw=2,label=form.replace('_',' '))
  ax.axhline(1,c='gray',ls=':',lw=.7);ax.axhline(-1,c='gray',ls=':',lw=.7);ax.set(title=ROLES[ri],xticks=[7,14,21],xlabel='Outgoing block');ax.spines[['top','right']].set_visible(False)
 axes[0].set_ylabel('Source/suffix sensitivity of role contrast S_R\nUnclipped; eligible edges only');axes[2].legend(frameon=False,fontsize=8)
 fig.suptitle('Declared causal profiles: individual edges and construction means\n'+results['decision'],fontsize=13)
 save(fig,'CAUSAL_PROFILES')
if __name__=='__main__':main()
