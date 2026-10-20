"""An explanatory saved-output case; no new model calls or hypothesis test."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
def main():
 inputs={r['id']:r for r in json.loads((HERE/'INPUTS.json').read_text())};native={r['id']:r for r in map(json.loads,(HERE/'NATIVE.jsonl').read_text().splitlines())}
 unit=next(u for u in json.loads((HERE/'UNITS.json').read_text()) if u['id']=='p01_r0_active_role2');query=unit['queries'][2];aa,bb=native[query['a']],native[query['b']]
 fig,axes=plt.subplots(1,2,figsize=(11,5.4),layout='constrained',sharey=True)
 for ax,r in zip(axes,[aa,bb]):
  row=inputs[r['id']];names=row['names'];scores=r['name_logits'];top=max(scores.values());values=[scores[n]-top for n in names]
  colors=['#8a9aa9' if n not in [r['actual'],r['gold']] else '#b55053' if n==r['actual'] else '#2e916e' for n in names]
  ax.bar(names,values,color=colors,width=.65);ax.axhline(0,c='black',lw=.6)
  winner=names.index(r['actual']);ax.scatter([winner],[0],c='#b55053',s=85,zorder=4)
  ax.get_xticklabels()[winner].set_color('#b55053');ax.get_xticklabels()[winner].set_fontweight('bold')
  for x,v in enumerate(values):ax.text(x,v-.5,f'{v:.2f}',ha='center',va='top',fontsize=10)
  ax.set_title(row['source'].replace('. ','.\n')+'\n\nCorrect answer: '+r['gold']+'; actual answer: '+r['actual'],fontsize=11,pad=18)
  ax.set_ylim(min(values)-4,2);ax.spines[['top','right']].set_visible(False)
 axes[0].set_ylabel('Name-token logit relative to the winning name\nEach prompt shifted to its own maximum; no change to pairwise odds')
 fig.suptitle('The scores track the destination; the answer selects the sent person\nQuestion: “Who was someone sent to?”',fontsize=15)
 fig.supxlabel('Red: emitted wrong name. Green: correct destination. All other answers remain in the raw record.',fontsize=10)
 for ext in ['png','svg','pdf']:fig.savefig(HERE/('WRONG_ANSWER_TRACKED_DESTINATION.'+ext),dpi=180,bbox_inches='tight')
 plt.close(fig)
 evidence=dict(selection='First chronological p01 active destination-swap edge; explanatory after outcomes, not a new confirmatory endpoint.',unit=unit,source_a=inputs[aa['id']]['source'],source_b=inputs[bb['id']]['source'],native_a=aa,native_b=bb,relative_new_old_a=aa['name_logits'][bb['gold']]-aa['name_logits'][aa['gold']],relative_new_old_b=bb['name_logits'][bb['gold']]-bb['name_logits'][aa['gold']])
 (HERE/'ILLUSTRATIVE_CASE.json').write_text(json.dumps(evidence,indent=2)+'\n')
if __name__=='__main__':main()
