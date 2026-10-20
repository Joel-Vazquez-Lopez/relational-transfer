"""Static scientific summary from saved audit results only."""
import os
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/matched-demand-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import *

def main():
    calibrated=json.loads((HERE/'CALIBRATION_RIVALS.json').read_text())['models']
    lesions=json.loads((HERE/'ROW_NECESSITY.json').read_text())['old_panels']
    fig,axes=plt.subplots(1,3,figsize=(16.6,5.8),gridspec_kw={'width_ratios':[1.2,1.2,1]})
    fig.subplots_adjust(left=.05,right=.98,top=.79,bottom=.27,wspace=.36)
    colors=['#6750a4','#00897b'];x=np.arange(4);w=.34
    labels=['Native / raw','Transferred\nposition prior','Source-column\ncentering','Assignment\nmatching']
    for i,(model,label) in enumerate([('qwen7b','Qwen 7B'),('llama3b','Llama 3B')]):
        d=calibrated[model]
        values=[d['native_complete'],d['methods']['other_family_surface_prior']['complete_sources'],d['methods']['within_source_three_question_column_centering']['complete_sources'],d['assignment_correct']]
        bars=axes[0].bar(x+(i-.5)*w,values,w,color=colors[i],label=label)
        axes[0].bar_label(bars,padding=3,fontsize=9)
    axes[0].set_xticks(x,labels,fontsize=9);axes[0].set_ylim(0,40);axes[0].set_yticks([0,12,24,36])
    axes[0].set_ylabel('Complete three-question sources / 36',fontsize=10)
    axes[0].set_title('A  Readout changes measured success',loc='left',fontsize=12,pad=17)
    axes[0].legend(frameon=False,fontsize=9,loc='upper left')
    values=np.array([[36]+[z['complete_correct'] for z in lesions[m]['by_removed_role']] for m in ['qwen7b','llama3b']])
    axes[1].imshow(values,cmap='Blues',vmin=0,vmax=36,aspect='auto')
    for i in range(2):
        for j in range(4):axes[1].text(j,i,f'{values[i,j]}/36',ha='center',va='center',color='white' if values[i,j]>24 else '#123',fontsize=12,fontweight='bold')
    axes[1].set_xticks(range(4),['All three\nquestions','No\nsender','No sent\nperson','No\ndestination'],fontsize=9)
    axes[1].set_yticks(range(2),['Qwen','Llama'],fontsize=10)
    axes[1].set_title('B  A missing query can be filled in',loc='left',fontsize=12,pad=17)
    axes[1].set_xlabel('Complete assignments after removing one score row',fontsize=9,labelpad=14)
    numerator=[15,14,2,44];denominator=[15,15,15,45]
    bars=axes[2].bar(range(4),np.array(numerator)/denominator,color=['#cf6b35']*3+['#708b96'],width=.68)
    axes[2].bar_label(bars,labels=[f'{n}/{d}' for n,d in zip(numerator,denominator)],padding=4,fontsize=10)
    axes[2].set_ylim(0,1.16);axes[2].set_yticks([0,.5,1],['0%','50%','100%'])
    axes[2].set_xticks(range(4),['Sender','Sent\nperson','Recipient','Mention\norder'],fontsize=9)
    axes[2].set_title('C  New pilot fails native validation',loc='left',fontsize=12,pad=17)
    axes[2].set_ylabel('Correct complete generated answers',fontsize=10)
    for a in [axes[0],axes[2]]:
        a.spines[['top','right']].set_visible(False);a.set_axisbelow(True);a.grid(axis='y',alpha=.15)
    fig.suptitle('Audit what the measurement can establish before tracing a mechanism',fontsize=17,x=.05,ha='left',y=.97)
    fig.text(.05,.89,'Saved Qwen/Llama scores: calibration competitors and question-removal control. New Llama screen: 0 causal interventions.',fontsize=10.5,color='#444')
    fig.text(.05,.13,'A–B: exploratory analyses of the same exposed 36 sources per checkpoint; corrections are external decisions, not new generated answers.',fontsize=10,color='#444')
    fig.text(.05,.085,'C: 75/90 observed answers, including one recovered response; fixed-order incomplete panel. The frozen competence gate is impossible to pass.',fontsize=10,color='#444')
    fig.text(.05,.04,'Next assay: four candidate people, three queried roles; change one target while the other two answers stay fixed. Design only.',fontsize=10,color='#444')
    for suffix in ['png','pdf','svg']:fig.savefig(HERE/f'REFORMULATION_AUDIT.{suffix}',dpi=170,facecolor='white')
    plt.close(fig)

if __name__=='__main__':main()
