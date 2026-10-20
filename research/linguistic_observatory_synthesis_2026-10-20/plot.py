"""Static scientific summary of complete saved results only; no inference or fitting."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent

def main():
    p=ROOT/'research/three_role_llama_causal_depth_map_2026-10-20'
    r=json.loads((p/'RESULTS.json').read_text());joint=json.loads((p/'STRUCTURED_READOUT.json').read_text())
    native=[json.loads((ROOT/f'research/binding_assignment_readout_2026-10-20/{m}_RESULTS.json').read_text()) for m in ['qwen7b','llama3b']]
    depths=[7,14,21];colors={'role':'#1261A0','mention':'#D97919','recipient':'#666666','syntax':'#218C74','neighbor':'#A94772'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    fig,axes=plt.subplots(2,2,figsize=(13,9.6));ax=axes[0,0];x=np.arange(2);width=.32
    v=[n['native_complete'] for n in native];w=[n['structured_complete'] for n in native]
    bars=ax.bar(x-width/2,v,width,label='Native: all three answers correct',color=colors['role']);bars2=ax.bar(x+width/2,w,width,label='External one-to-one readout',color=colors['syntax'])
    ax.bar_label(bars,labels=[f'{i}/36' for i in v],padding=4);ax.bar_label(bars2,labels=[f'{i}/36' for i in w],padding=4)
    ax.set_xticks(x,['Qwen 7B','Llama 3B']);ax.set_ylim(0,42);ax.set_ylabel('Complete source cases');ax.set_title('A  Native expression and joint-score recovery',loc='left',fontweight='bold');ax.legend(fontsize=8,loc='lower left')
    ax=axes[0,1]
    for field,label,color in [('role_table','Donor role prediction',colors['role']),('mention_order','Donor mention-order prediction',colors['mention']),('preserves_recipient_answer','Recipient answer retained',colors['recipient'])]:
        vals=[next(s[field] for s in r['summary'] if s['block']==b and s['condition']=='main') for b in depths]
        ax.plot(depths,vals,'o--',label=label,color=color);[ax.annotate(str(v),(b,v),xytext=(4,5),textcoords='offset points',fontsize=8,color=color) for b,v in zip(depths,vals)]
    ax.set_xticks(depths);ax.set_ylim(-1,40);ax.set_ylabel('Matching individual answers /36');ax.set_xlabel('Outgoing block replaced');ax.set_title('B  Main intervention: all individual answers',loc='left',fontweight='bold');ax.legend(fontsize=8)
    ax=axes[1,0]
    series=[('main','role_table','Main: coherent donor roles',colors['role']),('main','mention_order','Main: coherent mention order',colors['mention']),('same_role_voice','role_table','Same-role syntax: preserved roles',colors['syntax']),('neighbor','role_table','Neighbor: coherent donor roles',colors['neighbor'])]
    for condition,h,label,color in series:
        vals=[sum(u['complete_account'][h] for u in r['units'] if u['block']==b and u['condition']==condition) for b in depths]
        ax.plot(depths,vals,'o--',label=label,color=color)
    ax.set_xticks(depths);ax.set_yticks(range(7));ax.set_ylim(-.25,6.6);ax.set_ylabel('Complete bidirectional six-answer units /6');ax.set_xlabel('Outgoing block replaced');ax.set_title('C  Coherent units and controls',loc='left',fontweight='bold');ax.legend(fontsize=7.8)
    ax=axes[1,1]
    for condition,label,color,offset in [('main','Main',colors['role'],-.35),('same_role_voice','Same-role syntax',colors['syntax'],0),('neighbor','Neighbor',colors['neighbor'],.35)]:
        means=[]
        for b in depths:
            rows=[a for a in joint['ledger'] if a['block']==b and a['condition']==condition]
            vals=[a['role_minus_mention_assignment_score'] for a in rows];means.append(float(np.mean(vals)))
            for a,v in zip(rows,vals):ax.scatter(b+offset,v,s=19,alpha=.38,color=color,marker='o' if a['unit'].startswith('t01') else '^',edgecolors='none')
        ax.plot(np.array(depths)+offset,means,'o--',color=color,label=label,linewidth=1.8)
    ax.axhline(0,color='#333333',linewidth=.8);ax.set_xticks(depths);ax.set_xlabel('Outgoing block replaced');ax.set_ylabel('Role − mention assignment log-score');ax.set_title('D  Joint-score contrasts: every three-question direction',loc='left',fontweight='bold');ax.legend(fontsize=8)
    fig.suptitle('Linguistic assignment: availability, expression and causal transfer',fontsize=16,fontweight='bold',y=.985)
    fig.text(.5,.025,'Two AI-authored name families, one verb, three source forms. Development data only.\nNative readout analysis is exploratory; causal sites and gates were frozen. Lines join three measured depths. No uncertainty intervals or significance claim.',ha='center',fontsize=9,color='#444444')
    fig.tight_layout(rect=[0,.065,1,.955])
    for ext in ['png','pdf','svg']:fig.savefig(HERE/f'FUNCTIONAL_ATLAS.{ext}',dpi=180)
    print('Saved FUNCTIONAL_ATLAS.png/pdf/svg')
if __name__=='__main__':main()
