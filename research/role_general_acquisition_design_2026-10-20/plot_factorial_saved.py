"""Secondary descriptive figures from frozen saved outputs. No model calls/fits."""
from pathlib import Path
import csv,json,os
import numpy as np
HERE=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'plot_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    arms=['neither','source','target','both','source_double','target_double']
    labels=['Neither','Source only','Target only','Both','Source ×2','Target ×2']
    paths={p.parent.name:p.parent for p in (HERE/'factorial_runs/development').rglob('RESULT.json')}
    assert len(paths)==6
    records=[];depth=[];final={}
    for arm in arms:
        run=paths[arm];r=json.loads((run/'RESULT.json').read_text());final[arm]=r['metrics'][-1]
        for metric in r['metrics']:
            step=metric['step']
            with np.load(run/f'traces_{step:05}.npz',allow_pickle=False) as z:
                q=z['query_states']
                for role,pairs,pos in [('source',[(0,1),(4,5)],1),('target',[(2,3),(6,7)],2)]:
                    name=float(np.mean([np.linalg.norm(q[0,i,pos]-q[0,j,pos]) for i,j in pairs]))
                    for layer in range(3):
                        absolute=float(np.mean([np.linalg.norm(q[layer,i,0]-q[layer,j,0]) for i,j in pairs]))
                        relative=float(np.mean([np.linalg.norm(q[layer,i,0]-q[layer,j,0])/(.5*(np.linalg.norm(q[layer,i,0])+np.linalg.norm(q[layer,j,0]))) for i,j in pairs]))
                        depth.append(dict(arm=arm,update=step,role=role,stage=layer,name_token_embedding_delta=name,cls_delta=absolute,cls_relative_delta=relative))
    with (HERE/'DESCRIPTIVE_DEPTH.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(depth[0]));w.writeheader();w.writerows(depth)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(15.5,5.2),gridspec_kw={'width_ratios':[1.4,1,1]})
    x=np.arange(6);width=.37
    for offset,role,color in [(-width/2,'source','#1677b8'),(width/2,'target','#db7923')]:
        y=[100*final[a]['held'][role]['paired_rate'] for a in arms]
        bars=axes[0].bar(x+offset,y,width,label=role.capitalize(),color=color)
        axes[0].bar_label(bars,labels=[f'{v:.1f}' for v in y],fontsize=8,padding=2)
    axes[0].set_xticks(x,labels,rotation=30,ha='right');axes[0].set_ylim(0,105);axes[0].set_ylabel('Both counterfactual answers correct (%)');axes[0].set_title('Held-out behavior');axes[0].legend(frameon=False)
    colors=['#737b84','#1677b8','#db7923','#329367','#735bad','#bb4f78']
    for ax,role in zip(axes[1:],['source','target']):
        for arm,label,color in zip(arms,labels,colors):
            rows=[r for r in depth if r['arm']==arm and r['role']==role and r['update']==3000]
            ax.plot([r['stage'] for r in rows],[r['cls_relative_delta'] for r in rows],marker='o',color=color,label=label,linewidth=1.8)
        ax.set_xticks([0,1,2],['Input','Block 1','Block 2']);ax.set_ylim(0,1.10);ax.set_title(role.capitalize()+' query response');ax.set_ylabel('Relative CLS contrast')
    axes[2].legend(frameon=False,fontsize=8)
    fig.suptitle('Role-demand transfer: one development initialization',fontsize=15,fontweight='bold',y=.98)
    fig.text(.02,.015,'Confirmation stopped: both-role arm missed the 95% required-comparison gate. Depth panels: eight preselected cases; descriptive, no causal layer attribution.',fontsize=9)
    fig.tight_layout(rect=[0,.06,1,.94])
    for ext in ['png','svg','pdf']:fig.savefig(HERE/('ROLE_DEMAND_DEVELOPMENT.'+ext),dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),sharey=True)
    for ax,role in zip(axes,['source','target']):
        for arm,label,color in zip(arms,labels,colors):
            r=json.loads((paths[arm]/'RESULT.json').read_text());steps=[m['step'] for m in r['metrics']];values=[m['held'][role]['paired_rate'] for m in r['metrics']]
            ax.plot(steps,values,marker='o',label=label,color=color)
        ax.set_title(role.capitalize()+' paired correctness');ax.set_xlabel('Training updates');ax.set_ylim(-.02,1.02)
    axes[0].set_ylabel('Fraction of complete counterfactual pairs');axes[1].legend(frameon=False,fontsize=8)
    fig.suptitle('Saved learning curves — development only');fig.tight_layout()
    fig.savefig(HERE/'DEVELOPMENT_LEARNING_CURVES.png',dpi=180);plt.close(fig)
    print('Saved descriptive depth table and figures; no new model inference.')
if __name__=='__main__':main()
