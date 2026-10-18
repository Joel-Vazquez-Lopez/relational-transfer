"""Result interpretation and complete-condition figures; no inference or gate edits."""
import json
from pathlib import Path
import numpy as np
from core import save,sha

HERE=Path(__file__).resolve().parent

def main():
    raw=json.loads((HERE/'qwen7b/RAW.json').read_text())['rows']
    audit=json.loads((HERE/'INDEPENDENT_OUTPUT_AUDIT.json').read_text())
    dev=[r for r in raw if r['split']=='development' and r['correct'] is not None]
    names=['baseline','cut_0','cut_7','cut_14','cut_21','cut_28','irrelevant_7','irrelevant_14','irrelevant_21','restore_all','restore_7_14','restore_14_21']
    table=[]
    for c in names:
        rs=[r for r in dev if r['condition']==c]
        masses=[sum(np.exp(x) for x in r['full_name_log_probability'].values()) for r in rs]
        assert all(0<=x<=1+1e-8 for x in masses)
        table.append(dict(condition=c,correct=sum(r['correct'] for r in rs),total=len(rs),
            terminated_wrong=sum(r['terminated'] and not r['correct'] for r in rs),
            truncated=sum(not r['terminated'] for r in rs),
            mean_two_name_prefix_probability=float(np.mean(masses))))
    # Post-outcome diagnostic, not a new primary metric or changed selection rule.
    # Exact algebra separates the predeclared difference into control and cut terms.
    decomposition=[]
    for k in [7,14,21]:
        means={}
        for condition in ['baseline',f'irrelevant_{k}',f'cut_{k}']:
            means[condition]={q:float(np.mean([r['margin'] for r in dev if r['condition']==condition and r['query']==q])) for q in ['agent','first']}
        b=means['baseline']['agent']-means['baseline']['first']
        i=means[f'irrelevant_{k}']['agent']-means[f'irrelevant_{k}']['first']
        c=means[f'cut_{k}']['agent']-means[f'cut_{k}']['first']
        decomposition.append(dict(cut=k,baseline_question_difference=b,irrelevant_question_difference=i,
            event_cut_question_difference=c,primary_difference=i-c))
    save(HERE/'BEHAVIORAL_OUTCOMES.json',dict(development_conditions=table,
        exploratory_margin_decomposition=decomposition,
        decomposition_is_post_outcome_not_confirmatory=True,
        primary_rule_unchanged=True,confirmed_cross_model_sequence=False))
    failed=[r for r in raw if r['condition']=='baseline' and r['split']=='evaluation' and not r['correct']]
    assert len(failed)==1 and failed[0]['id']=='e2_1_first'
    r=failed[0]
    max_recon=max(x.get('reconstruction_relative_error',0) for x in raw)
    max_child=max([c['original_prompt_logit_error'] for x in raw for c in x['child_calls']]+[0])
    header=['## Verified rerun: interpretation and decision','',
        '**Scoring repair verified; development source-access effects observed; fresh evaluation and cross-model confirmation stopped at competence.**',
        '',f"The rerun used **{audit['new_attempted_forwards']}/2000 actual forwards**, including every candidate and generation continuation. The earlier failed attempt used 11 separately preserved forwards, so the source-access programme total is {audit['cumulative_source_access_attempts']}. There were 107 primary records: 16 baselines, three numerical validations and 88 development intervention/control conditions. No evaluation intervention or Llama forward ran.",
        '',f"An independent offline audit recomputed all name-path probabilities and verified every greedy token and EOS from saved logits for **{audit['audited_scientific_records']} scored records**. Maximum probability-summation discrepancy was {audit['max_sequence_probability_error']:.3g}. No additional model calls were used. All frozen files, the exact original stimuli and the original failed attempt remain unchanged.",
        '', '### Actual complete-answer outcomes', '',
        '| Development condition | Correct / total | Terminated wrong | Truncated at four tokens |',
        '|---|---:|---:|---:|']
    for x in table:header.append(f"| {x['condition']} | {x['correct']}/{x['total']} | {x['terminated_wrong']} | {x['truncated']} |")
    header+=['',
        '**What this establishes:** in these two exposed active-sentence families, closing all suffix access to the event at the tested nontrivial cuts disrupted the declared short-name response. The matched irrelevant cuts preserved 24/24 responses; cut-28 shams and full restoration preserved 8/8 each. Zero-access suffix states were exactly invariant across name-swapped orientations. This supports dependence on the source under the specified intervention.',
        '',
        '**What this does not establish:** an onset of role representation, absence of internally stored information, a reusable semantic mechanism, or a different processing sequence between questions/models. Attention removal changes normalization and can produce unfamiliar states. Both demands lost the short-answer endpoint at every tested nontrivial cut. At block 21, seven responses were truncated, so they cannot be described as seven observed completed wrong names. No longer continuation was generated to resolve them.',
        '', '### Frozen prediction and fresh failure', '',
        'The unchanged development rule selected block 14, with Agent-minus-first matched source cost +3.369145 nats and positive differences in both development families. Its raw-development hash was saved before evaluation and independently verified. All eight fresh passive baselines were then evaluated, yielding **7/8**: e1 passed 4/4 and e2 passed 3/4. The required complete-panel gate therefore failed.',
        '',
        '> Sentence: Sarah was visited by James.\n>\n> Question: Who was mentioned first in the sentence?\n>\n> Expected: Sarah. Generated through EOS: James.',
        '',f"The gold-versus-rival full-name margin was {r['margin']:.6f} nats. This is a complete wrong answer under the declared task, distinct from the original partial-token scoring defect. Both complete development families and the Helen/Martin passive family passed; we did not select those successful cases to bypass the failed panel. The prospective **intervention prediction remains untested**, rather than having been falsified by an evaluation intervention. Llama was conditional on Qwen's evaluation success and was not run.",
        '', '### Margin interpretation: exploratory audit, not a revised success rule', '',
        'Raw margin loss alone cannot identify a question-specific route. If both queries have the same retained fraction r after a cut but different baseline margins b_A and b_F, then their cost difference is (1-r)(b_A-b_F), which can be nonzero with identical source dependence. This is a competing explanation, not an inferred mechanism.',
        '',
        'At block 14 the combined probability of either candidate name prefix averaged only **0.0000266** after the cut. Thus a sizeable margin between those two names did not mean the model actually answered with either name. Greedy completions are reported separately above. The exact decomposition below shows how control confidence and residual candidate preference contribute to the selected statistic:',
        '', '| Cut | Baseline Agent−first margin | Irrelevant-cut Agent−first | Event-cut Agent−first | Primary difference |',
        '|---|---:|---:|---:|---:|']
    for x in decomposition:header.append(f"| {x['cut']} | {x['baseline_question_difference']:.6f} | {x['irrelevant_question_difference']:.6f} | {x['event_cut_question_difference']:.6f} | {x['primary_difference']:.6f} |")
    header+=['',
        'This decomposition was added after outcomes to constrain interpretation. It did not alter the selected cut, prediction, endpoint, thresholds or stop decision. The evidence supports a narrow intervention-dependent access observation; the hoped-for generalizing cross-model developmental account is not demonstrated.',
        '', '### Numerical checks and closure', '',
        f'Native/wrapped/dense/repeat logit discrepancies and repeat-state discrepancy were all zero. All development cut-28/full-restoration logit errors and zero-access orientation discrepancies were zero. Maximum last-token block reconstruction relative error across captured primary passes was {max_recon:.9f} (limit .002). Maximum original-prompt logit discrepancy on continuation passes was {max_child:.9f} (limit .05). These are numerical controls, not proof that attention removal preserves a natural computation.',
        '',
        'The scorer is verified. **NO-GO for proceeding to evaluation interventions or a cross-model claim under this protocol.** No prompt repair, replacement family, new cut, longer generation or additional model run follows. Scripts, every raw scored record, every forward logit payload, controls and local tensor hashes are preserved. The figure below reports all fixed development conditions; every bar contains four correlated answers from two families.',
        '', '![Development complete-answer outcomes](ANSWER_OUTCOMES.png)', '']
    p=HERE/'REPORT.md';text=p.read_text();first,rest=text.split('\n',1)
    marker='## Verified rerun: interpretation and decision'
    if marker not in text:p.write_text(first+'\n\n'+'\n'.join(header)+rest)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
    for ax,q in zip(axes,['agent','first']):
        correct=[];wrong=[];truncated=[]
        for condition in names:
            rs=[r for r in dev if r['condition']==condition and r['query']==q]
            correct.append(sum(r['correct'] for r in rs));wrong.append(sum(r['terminated'] and not r['correct'] for r in rs));truncated.append(sum(not r['terminated'] for r in rs))
        xs=np.arange(len(names));ax.bar(xs,correct,color='#238b45',label='Correct, EOS observed')
        ax.bar(xs,wrong,bottom=correct,color='#cb4848',label='Terminated wrong')
        ax.bar(xs,truncated,bottom=np.array(correct)+wrong,color='#db9d27',label='Truncated at four tokens')
        ax.set_ylim(0,4.6);ax.set_yticks([0,1,2,3,4]);ax.set_ylabel('Answers out of 4')
        ax.set_title(('Agent' if q=='agent' else 'First-mentioned-person')+' question — development only')
    axes[0].legend(loc='upper left',bbox_to_anchor=(0,1.33),ncol=3,fontsize=9)
    axes[-1].set_xticks(range(len(names)),names,rotation=40,ha='right')
    fig.suptitle('Source cuts disrupt both tasks; irrelevant cuts and full restoration preserve answers',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(HERE/'ANSWER_OUTCOMES.png',dpi=160);fig.savefig(HERE/'ANSWER_OUTCOMES.svg');plt.close(fig)
    manifest={}
    for f in sorted(HERE.rglob('*')):
        if f.is_file() and f.name not in ['RESULT_MANIFEST.json','GITHUB_PRESERVATION.json'] and '__pycache__' not in f.parts:
            manifest[str(f.relative_to(HERE))]=dict(bytes=f.stat().st_size,sha256=sha(f))
    save(HERE/'RESULT_MANIFEST.json',dict(files=manifest,total_bytes=sum(x['bytes'] for x in manifest.values())))
    print('Final report and complete-answer figure saved; manifest bytes',sum(x['bytes'] for x in manifest.values()))

if __name__=='__main__':main()
