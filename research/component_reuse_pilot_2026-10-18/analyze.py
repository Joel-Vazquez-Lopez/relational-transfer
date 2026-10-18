"""Report every frozen outcome; component geometry is descriptive only."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def load(name):
    return json.loads((HERE/name).read_text())


def save(name,obj):
    (HERE/name).write_text(json.dumps(obj,indent=2)+'\n')


def ledger(name,rows):
    if not rows:
        (HERE/name).write_text('No conditions executed. See RESULT.json.\n')
        return
    flat = [{k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows]
    fields = list(dict.fromkeys(k for r in flat for k in r))
    with (HERE/name).open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)


def main():
    result,baseline,gate,tech = map(load,['RESULT.json','BASELINES.json','COMPETENCE.json','TECHNICAL.json'])
    lock = load('PROTOCOL.json')
    rows = baseline['rows']
    ledger('ANSWER_LEDGER.csv',[{k:v for k,v in r.items() if k not in ['ids','prompt']} for r in rows])
    ledger('PATCH_LEDGER.csv',load('PATCHES.json'))
    ledger('UNIT_LEDGER.csv',gate['units'])
    errors = [r for r in rows if not r['pass']]
    save('BASELINE_FAILURES.json',errors)
    group = []
    for fam in lock['families']:
        r = [r for r in rows if r['family']==fam['id']]
        group.append({'family':fam['id'],'split':fam['split'],'correct':sum(v['pass'] for v in r),
                      'total':len(r),'all_correct':all(v['pass'] for v in r)})
    save('FAMILY_SUMMARY.json',group)
    strata = []
    for axis in ['mapping','question_key','condition','expected']:
        for value in sorted({r[axis] for r in rows}):
            subset = [r for r in rows if r[axis]==value]
            strata.append({'axis':axis,'value':value,'correct':sum(r['pass'] for r in subset),
                           'total':len(subset)})
    save('BASELINE_STRATA.json',strata)
    z = np.load(HERE/'COMPONENT_STATES.npz')
    sites = list(z['sites'])
    parts = list(z['parts'])
    component_rows = []
    for fam in lock['families']:
        if fam['split']!='development':
            continue
        a = z[f"{fam['id']}__active_a"].astype(np.float64)
        for contrast,condition in [('same_order_role_flip','passive_b'),('same_role_voice_control','cleft_a'),('same_target_role_cleft','cleft_b')]:
            delta = z[f"{fam['id']}__{condition}"].astype(np.float64)-a
            for l in range(a.shape[0]):
                for s,site in enumerate(sites):
                    inc,attn,mlp,out = delta[l,:,s,:]
                    na,nm = np.linalg.norm(attn),np.linalg.norm(mlp)
                    component_rows.append({'family':fam['id'],'contrast':contrast,'layer_zero_indexed':l,'site':site,
                         'incoming_norm':float(np.linalg.norm(inc)),'attention_norm':float(na),
                         'mlp_norm':float(nm),'outgoing_norm':float(np.linalg.norm(out)),
                         'attention_mlp_cosine':float(attn@mlp/(na*nm)) if na*nm else None,
                         'write_cancellation':float(1-np.linalg.norm(attn+mlp)/(na+nm)) if na+nm else None})
    ledger('COMPONENT_LEDGER.csv',component_rows)
    fig,axs = plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for j,fam in enumerate(f for f in lock['families'] if f['split']=='development'):
        r = [r for r in component_rows if r['family']==fam['id'] and r['contrast']=='same_order_role_flip' and r['site']=='period']
        for part,color in [('incoming','#626e7c'),('attention','#2679a7'),('mlp','#d3772a'),('outgoing','#2b8b64')]:
            axs[0,j].plot([v['layer_zero_indexed']+1 for v in r],[v[part+'_norm'] for v in r],label=part,color=color)
        axs[0,j].set(title=f"{fam['id']}: {fam['a']} / {fam['b']} — sentence period",xlabel='Block (1-based)',ylabel='Role-flip contrast norm')
        axs[0,j].legend(fontsize=8)
        display_sites = ['person_a','person_b','verb','period']
        heat = np.array([[next(v['write_cancellation'] for v in component_rows if v['family']==fam['id'] and v['contrast']=='same_order_role_flip' and v['layer_zero_indexed']==l and v['site']==s) for s in display_sites] for l in range(28)])
        im = axs[1,j].imshow(heat,aspect='auto',origin='lower',vmin=0,vmax=1,cmap='magma',extent=(-.5,3.5,.5,28.5))
        axs[1,j].set(xticks=range(4),xticklabels=['Person A','Person B','Verb','Period'],ylabel='Block (1-based)',title='Attention/MLP contrast cancellation')
        fig.colorbar(im,ax=axs[1,j],label='1 − ‖Δattention + ΔMLP‖ / (‖Δattention‖ + ‖ΔMLP‖)')
    fig.suptitle('Separating the layer writes\nDescriptive pre-question map; geometry alone does not establish a reusable role code',fontsize=13)
    fig.savefig(HERE/'COMPONENT_MAP.png',dpi=170)
    fig.savefig(HERE/'COMPONENT_MAP.pdf')
    plt.close(fig)
    summaries = []
    for fam in lock['families']:
        if fam['split']=='development':
            r = [r for r in component_rows if r['family']==fam['id'] and r['contrast']=='same_order_role_flip']
            summaries.append({'family':fam['id'],'mean_write_cancellation':float(np.mean([x['write_cancellation'] for x in r])),
                               'max_write_cancellation':max(x['write_cancellation'] for x in r)})
    save('DESCRIPTIVE_SUMMARY.json',{'families':summaries,'ledger_rows':len(component_rows),
                                  'interpretation':'Captures and algebraic summaries only. No semantic/cross-model mechanism follows.'})
    norm = lambda section,key:max((r[key] for r in tech[section]),default=0)
    stat = {'reconstruction_max':norm('reconstruction','max_relative_error'),
            'prefix_invariance_max':norm('prefix_invariance','max_relative_error'),
            'identity_max_nats':norm('identity','margin_discrepancy') if tech['identity'] else None}
    save('TECHNICAL_SUMMARY.json',stat)
    identity_text = f"{stat['identity_max_nats']:.8g} nat" if stat['identity_max_nats'] is not None else 'untested'
    scientific = 'NARROW' if result['decision']=='BOUNDED_PASS' else 'NO-GO'
    save('SCIENTIFIC_DECISION.json',{'decision':scientific,'frozen_outcome':result['decision'],
                                    'scope':'This pilot only; no universal impossibility or population claim.',
                                    'reason': ('Competence gate failed; prefix-invariance technical gate independently failed; no causal interventions were run.'
                                               if not result['causal_gate_pass'] and not result['technical_pass'] else
                                               'Strong syntax-aware rival survives even a bounded pass; a gate stop or failure supplies no positive causal evidence.')})
    md = ['# Component reuse pilot — 18 October 2026','',
          f"**{scientific} for this pilot. Frozen outcome: {result['decision']}.** {gate['baseline_correct']}/192 baseline answers pass both generation and score checks; {result['patch_count']} component patches executed.",
          '', '## What was tested', '',
          'One existing local Qwen2.5-7B-Instruct model (4-bit MLX); four constructed lexical families, six active/passive/cleft sentences, Agent/Patient/first-person/last-person questions, and reversed X/Y answer codes. The question and output mapping follow the measured sentence. This corrects the earlier person-only prompt conflict. These are fresh prompts, not a natural-corpus sample or a preregistration deposited with an independent registry.',
          '', 'The pre-output protocol and full runner hashes were saved in RUN_STARTED.json before loading the model. PROTOCOL.json declares the complete gates, twelve candidate locations, controls, selection rule and 400-patch maximum. The strongest syntax-aware compound-key rival is explicitly allowed to survive even a successful pilot.',
          '', '## Competence: individual answers, units and families', '',
          '| Family | Split | Correct answers | Entire family correct? |','|---|---|---:|---|']
    md += [f"| {f['family']} | {f['split']} | {f['correct']}/{f['total']} | {f['all_correct']} |" for f in group]
    md += ['',f"Entire-family failures: {sum(not f['all_correct'] for f in group)}/4. These constructed families are the document-level units; individual answers are not independent documents.",
           '', '| Family | Orientation | Correct prerequisite answers | Eligible? |','|---|---|---:|---|']
    md += [f"| {u['family']} | {u['orientation']} | {u['correct']}/24 | {u['eligible']} |" for u in gate['units']]
    md += ['', '| Answer-code mapping | Correct answers |','|---|---:|']
    md += [f"| {r['value']} | {r['correct']}/{r['total']} |" for r in strata if r['axis']=='mapping']
    md += ['', 'All 192 prompts, token positions, raw generations, X/Y log probabilities and failures appear in BASELINES.json. ANSWER_LEDGER.csv, BASELINE_FAILURES.json and UNIT_LEDGER.csv provide separate views. No error was repaired by changing the prompt or substituting a family.',
           '', '## Technical checks', '',
           f"Maximum relative reconstruction error: {stat['reconstruction_max']:.8g} (limit 0.002). Maximum pre-question discrepancy across query/code prompts: {stat['prefix_invariance_max']:.8g} (strict limit 0.001). Maximum measured identity-patch margin discrepancy: {identity_text} (limit 0.05 nat; {len(tech['identity'])} measurements). Empty identity checks are untested, not a successful causal control.",
           '', 'The prefix-invariance check independently fails its frozen tolerance. The common sentence prefix should be invariant in the causal architecture; this discrepancy cannot be interpreted as the model using a future question. Numerical execution sensitivity or a measurement issue is possible, but its source was not established here. No new diagnostic model runs or relaxed tolerances were used to rescue the study. The component map therefore has an additional technical limitation beyond the competence failure.',
           '', '## Causal decision', '', f"Decision: **{result['decision']}**. Selected location: `{result['selected']}`. Development candidates scored: {result['candidate_count']}. Held-out unit/mapping results: {len(result['evaluations'])}.",
           '', 'DEVELOPMENT_DECISION.json reports every candidate, including unsuccessful ones. PATCHES.json preserves every measured intervention and unconstrained greedy next token. RESULT.json separates held-out unit/mapping outcomes and all frozen criteria. Controls not executed after a gate stop remain untested; their absence is not a null effect.',
           '', '## Layer instrument', '',
           'The declared descriptive instrument captures incoming residual, attention output, MLP output and outgoing residual at every one of 28 blocks and six token anchors in both development families, all six sentences. The compressed float32 archive preserves native captures without an additional float16 save. Its checksum is in RESULT.json. Capture as float32 does not undo quantization or native arithmetic error.',
           '', 'COMPONENT_LEDGER.csv reports every development family × three declared contrasts × block × six sites (1,008 rows). The map shows norms and attention/MLP cancellation for the same-name-order role flip. A high cancellation value means opposing component contrasts can obscure a net write; it does not identify a role representation or demonstrate transfer. Entity-anchored positions can occur at different absolute locations, and passive/cleft changes alter several syntactic and lexical features. No geometry threshold is used to invent an additional success.',
           '', '![Component map](COMPONENT_MAP.png)', '', '## Interpretation and prior research', '',
           'A gated failure rejects continuation of this particular single-write design, not the existence of distributed relational computation. It leaves open distributed multi-token mechanisms and different constructions, but this pilot does not test or validate them. It also does not establish shared development across models: only Qwen 7B is used.',
           '', '[Dai, Heinzerling and Inui (2024)](https://aclanthology.org/2024.emnlp-main.967/) already localize low-rank binding information and redirect entity–attribute retrieval by editing it. Our intended advance would require a selective pre-question dependency that survives both role questions, surface-order controls, answer recoding and untouched lexical cases. Generic activation redirection or a similar-looking curve is insufficient for novelty. [Mixing Mechanisms](https://arxiv.org/abs/2510.06182) further motivates testing multiple retrieval explanations rather than assuming one semantic mechanism. A strong syntax-aware compound-key account remains compatible with even the intended success pattern.',
           '', '## Scope and preservation', '',
           'This is a small constructed pilot with competence-based exclusions, no population estimate, no cross-model replication and no unique-mechanism conclusion. All exclusions are exposed. No training, large download or paid API call was used. No adaptive follow-up is part of this run. Scripts, complete raw text outputs, controls, protocols and figures are preserved on the existing research branch and linked to Issue #33. The full component archive is saved locally with its SHA256 manifest; model weights are not uploaded.',
           '', 'Reproduce using the existing MLX environment: `python run.py` in a clean copy without RUN_STARTED.json. The run refuses to overwrite a prior start. `python analyze.py` regenerates descriptive tables and the figure from completed outputs. Reproduction is not authorization for an additional run here.', '']
    (HERE/'REPORT.md').write_text('\n'.join(md))
    artifacts = []
    for p in sorted(HERE.iterdir()):
        if p.is_file() and p.name not in ['ARTIFACT_MANIFEST.json','GITHUB_PRESERVATION.json']:
            b = p.read_bytes()
            artifacts.append({'path':p.name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),
                              'git_blob_sha':hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),
                              'storage':'local_archive' if p.suffix=='.npz' else 'research_branch'})
    model = HERE.parent/'stronger_model_development_2026-10-15/models/qwen7b'
    source = []
    for p in sorted(model.iterdir()):
        if p.is_file():
            digest = hashlib.sha256()
            with p.open('rb') as f:
                for chunk in iter(lambda:f.read(4*1024*1024),b''):
                    digest.update(chunk)
            source.append({'path':str(p),'bytes':p.stat().st_size,'sha256':digest.hexdigest()})
    save('ARTIFACT_MANIFEST.json',{'artifacts':artifacts,'local_model_sources':source,
                                 'protocol_source':'Constructed English templates in PROTOCOL.json; no corpus annotation claims.'})
    print(result['decision'],gate['baseline_correct'],'/192',stat,flush=True)


if __name__=='__main__':
    main()
