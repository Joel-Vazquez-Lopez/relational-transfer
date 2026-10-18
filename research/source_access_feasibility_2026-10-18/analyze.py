"""Complete ledger/report export; reads saved outcomes, never runs a model."""
import csv
import json
from pathlib import Path
import numpy as np
from core import save,sha

HERE=Path(__file__).resolve().parent

def main():
    all_rows=[]; decisions={}; controls={}
    for m in ['qwen7b','llama3b']:
        p=HERE/m
        if (p/'RAW.json').exists():
            raw=json.loads((p/'RAW.json').read_text())['rows']; all_rows+=raw
        decisions[m]=json.loads((p/'DECISION.json').read_text()) if (p/'DECISION.json').exists() else dict(status='NOT_RUN',reason=json.loads((HERE/'LLAMA_NOT_RUN.json').read_text()) if (HERE/'LLAMA_NOT_RUN.json').exists() else None)
        if (p/'ERROR.json').exists(): decisions[m]['error']=json.loads((p/'ERROR.json').read_text())
        controls[m]={f.name:json.loads(f.read_text()) for f in p.glob('*CONTROLS.json')} if p.exists() else {}
    fields=['model','call','split','family','id','query','orientation','voice','condition','expected','argmax_token','actual','correct','margin','reconstruction_relative_error','prefix_relative_error','before_cut_max_absolute_error']
    for name,rows in [('ANSWER_LEDGER.csv',all_rows),('FAILURES.csv',[r for r in all_rows if not r['correct']])]:
        with (HERE/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    budget=json.loads((HERE/'BUDGET.json').read_text()) if (HERE/'BUDGET.json').exists() else []
    summary=dict(decisions=decisions,controls=controls,calls=len(budget),max_calls=400,
        counts={m:dict(calls=sum(r['model']==m for r in budget),
            baseline_correct=sum(r['correct'] for r in all_rows if r['model']==m and r['condition']=='baseline'),
            baseline_total=sum(r['model']==m and r['condition']=='baseline' for r in all_rows)) for m in decisions},
        errors=[r for r in all_rows if not r['correct']],no_more_inference=True)
    save(HERE/'SUMMARY.json',summary)
    lines=['# Source-access feasibility — results','',f'Closed bounded execution: **{len(budget)}/400 attempted forwards**. No paid API, training or model download.','',
        '## Decisions','', '| Model | Outcome | Baseline answers | Calls |','|---|---|---:|---:|']
    for m,d in decisions.items():
        c=summary['counts'][m]; lines.append(f"| {m} | {d['status']} | {c['baseline_correct']}/{c['baseline_total']} | {c['calls']} |")
    lines+=['','## What this tests','',
        'The intervention closes later access from every question/assistant suffix token to all event-dependent context tokens, while preserving earlier computations. It measures continued-access dependence under that mask. It does not identify when semantic information first exists, establish a reusable role representation, or distinguish a strong syntax-aware copier from semantic composition.','',
        '## Complete baseline outcomes','', '| Model | Family | Correct / total | Entire family correct? |','|---|---|---:|---|']
    for m in decisions:
        for f in ['d1','d2','e1','e2']:
            rs=[r for r in all_rows if r['model']==m and r['family']==f and r['condition']=='baseline']
            if rs: lines.append(f"| {m} | {f} | {sum(r['correct'] for r in rs)}/{len(rs)} | {len(rs)==4 and all(r['correct'] for r in rs)} |")
    for m in decisions:
        p=HERE/m
        for phase in ['DEVELOPMENT','EVALUATION']:
            f=p/f'{phase}_SUMMARY.json'
            if f.exists():
                lines+=['',f'## {m}: {phase.lower()} matched effects','',
                    '| Cut (zero-based block) | Agent minus first-person source cost, nats | Irrelevant absolute drift, nats |','|---|---:|---:|']
                for x in json.loads(f.read_text()): lines.append(f"| {x['cut']} | {x['difference']:.6f} | {x['irrelevant_absolute_margin_drift']:.6f} |")
        f=p/'VALIDATION.json'
        if f.exists(): lines+=['',f'### {m}: validation','', '```json',json.dumps(json.loads(f.read_text()),indent=2),'```']
    lines+=['','## Failures and controls','',
        'ANSWER_LEDGER.csv includes every scored call, including validations and controls. FAILURES.csv includes every incorrect next-token answer, including intentionally source-isolated calls; those are distinct from baseline competence failures. SUMMARY.json retains all decision fields and full control findings. RAW.json includes top-20 outputs and alias logits; full-vocabulary logits and native captures are local.','',
        '## Limitations and prior work','',
        'Two constructed lexical families per phase are not a population sample. Questions differ in wording/length; passive evaluation changes construction and names together. Answer identities are matched on active development but differ between demands in passive evaluation. Quantized local models and one runtime limit generalization. Attention removal changes normalization and may cause off-distribution damage. Matched irrelevant cuts match edge count but not absolute position or attention mass. A preserved answer does not exclude later rereading; a failed answer does not prove absence of an internal representation. No clean cross-model replication is claimed after a failed gate.','',
        'Attention knockout and staged retrieval have close precedents in [Geva et al.](https://aclanthology.org/2023.emnlp-main.751/), [Mixing Mechanisms, appendix F](https://arxiv.org/html/2510.06182v2), and [Lookbacks, appendix K](https://arxiv.org/html/2505.14685v3). [Dai et al.](https://aclanthology.org/2024.emnlp-main.967/) already demonstrates binding-related activation edits. This feasibility run makes no novelty claim.','',
        '## Preservation and closure','',
        'LOCK.json freezes source, stimuli, installed native model code and local weight/tokenizer/config hashes. FREEZE_RECEIPT.json identifies the pre-inference commit. BUDGET.json journals attempts before dispatch. Tensor/logit files remain local with checksums in RESULT_MANIFEST.json; uploading text artifacts does not imply uploading every tensor. Earlier experiments and their gates were not modified. No adaptive follow-up follows this report.']
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    # Plot only collected source profiles, retaining every answer and all declared cuts.
    if any((HERE/m/'DEVELOPMENT_SUMMARY.json').exists() for m in decisions):
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(11,4),squeeze=False)
        for ax,(m,d) in zip(axes[0],decisions.items()):
            for phase,style in [('development','-'),('evaluation','--')]:
                ids={r['id'] for r in all_rows if r['model']==m and r['split']==phase and r['condition']=='baseline'}
                if not (HERE/m/(phase.upper()+'_SUMMARY.json')).exists(): continue
                for q,color in [('agent','tab:blue'),('first','tab:orange')]:
                    values=[]
                    for k in [0,7,14,21,28]:
                        costs=[]
                        for rid in sorted(ids):
                            rs=[r for r in all_rows if r['model']==m and r['id']==rid and r['query']==q]
                            ref=next((r for r in rs if r['condition']=='baseline'),None)
                            cut=next((r for r in rs if r['condition']==f'cut_{k}'),None)
                            if ref and cut: costs.append(ref['margin']-cut['margin'])
                        values.append(float(np.mean(costs)) if costs else np.nan)
                    ax.plot([0,7,14,21,28],values,style+'o',color=color,label=f'{phase}: {q}')
            ax.axhline(0,color='gray',lw=.7); ax.set_title(f"{m}: {d['status']}",fontsize=9)
            ax.set_xlabel('Cut starts at block (0-based)'); ax.set_ylabel('Baseline minus cut margin (nats)')
            if ax.lines and len(ax.lines)>1: ax.legend(fontsize=8)
        fig.suptitle('Dependence on continued source access — all recorded cases')
        fig.tight_layout();fig.savefig(HERE/'SOURCE_ACCESS.png',dpi=160);fig.savefig(HERE/'SOURCE_ACCESS.svg');plt.close(fig)
    manifest={}
    for p in sorted(HERE.rglob('*')):
        if p.is_file() and p.name not in ['RESULT_MANIFEST.json','GITHUB_PRESERVATION.json'] and '__pycache__' not in p.parts:
            manifest[str(p.relative_to(HERE))]=dict(bytes=p.stat().st_size,sha256=sha(p))
    save(HERE/'RESULT_MANIFEST.json',dict(files=manifest,total_bytes=sum(x['bytes'] for x in manifest.values())))
    print('REPORT_SAVED',len(budget),'attempted forwards',flush=True)

if __name__=='__main__': main()
