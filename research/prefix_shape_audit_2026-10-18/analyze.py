"""Analyze all fixed audit conditions; no further inference."""
import csv
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent


def save(name,x):(HERE/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def maximum(rows,key='relative_l2'):return max((r[key] for r in rows),default=None)


def main():
    raw={k:json.loads((HERE/f'RAW_{k}.json').read_text()) for k in ['qwen7b','llama3b']}
    summary={}
    fig,axs=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for col,(key,d) in enumerate(raw.items()):
        families=[]
        projection_rows=[]
        for f in d['families']:
            rows=d['measurements'][f['id']]
            groups={c:[r for r in rows if r['comparison']==c] for c in sorted({r['comparison'] for r in rows})}
            maxima={c:maximum(v) for c,v in groups.items()}
            good=(maxima['versus_prefix_only']>0.001 and maxima['same_shape_content']<=1e-6 and maxima['repeat']<=1e-6)
            if key=='qwen7b':good=good and maxima['real_padded']<=1e-6
            families.append({'id':f['id'],'prefix_length':f['prefix_length'],'max_relative_discrepancies':maxima,
                             'bounded_shape_pattern':bool(good),
                             'first_nonzero_block':min((r['layer_zero_indexed'] for r in groups['versus_prefix_only'] if r['absolute_l2']>0),default=None)})
            for p in f['projection_metrics']:projection_rows.append({'family':f['id'],**p})
            # Plot all declared lengths, maximum across sites and components.
            values=[]
            for n in f['lengths']:
                r=[r for r in groups['versus_prefix_only'] if r['label']==f'n{n}_s0']
                values.append(maximum(r))
            axs[0,col].plot(f['lengths'],values,'o-',label=f['id'])
            boundary_rows=[r for r in groups['adjacent_lengths'] if r['label']=='n64_to_n65']
            if boundary_rows:
                vals=[maximum([r for r in boundary_rows if r['layer_zero_indexed']==l]) for l in range(28)]
                axs[1,col].plot(range(1,29),vals,label=f['id'])
        stages={s:maximum([r for r in projection_rows if r['stage']==s]) for s in sorted({r['stage'] for r in projection_rows})}
        summary[key]={'families':families,'projection_max_relative':stages,'forwards':d['model_forwards'],
                       'unique_archived_tensors':len({r['archive_key'] for r in d['conditions']}),
                       'conditions':len(d['conditions']),'states_sha256':d['states_sha256']}
        axs[0,col].axvline(64,color='#888',linestyle=':',label='64 tokens')
        axs[0,col].set(title=key,xlabel='Total sequence length',ylabel='Maximum relative prefix-state discrepancy')
        axs[0,col].legend()
        axs[1,col].set(xlabel='Block (1-based)',ylabel='Maximum relative prefix-state discrepancy',title='Identical prefix: 64 → 65 total tokens')
        axs[1,col].legend()
        # Explicit all-row CSV retains every fixed condition/site/component.
        all_rows=[r for rows in d['measurements'].values() for r in rows]
        for name,rows in [(f'MEASUREMENTS_{key}.csv',all_rows),(f'PROJECTIONS_{key}.csv',projection_rows)]:
            with (HERE/name).open('w',newline='') as out:
                w=csv.DictWriter(out,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fig.suptitle('Identical causal prefixes, different execution shapes\nNo measured token can attend to the changed suffix',fontsize=13)
    fig.savefig(HERE/'PREFIX_SHAPE.png',dpi=170)
    fig.savefig(HERE/'PREFIX_SHAPE.pdf')
    plt.close(fig)
    q=summary['qwen7b']
    shape=all(f['bounded_shape_pattern'] for f in q['families'])
    summary['decision']={'bounded_Qwen_shape_explanation':shape,
                         'semantic_interventions_run':0,'previous_failed_studies_reopened':False,
                         'claim':'Local execution-shape diagnostic, not a novel linguistic mechanism or universal backend theorem.'}
    classes=[]
    for key,d in raw.items():
        for f in d['families']:
            buckets={}
            for c in d['conditions']:
                if c['family']==f['id'] and c['kind']=='sweep' and c['suffix']==0:
                    buckets.setdefault(c['tensor_sha256'],[]).append(c['length'])
            classes.append({'model':key,'family':f['id'],'exact_tensor_equivalence_classes':list(buckets.values())})
    save('SHAPE_CLASSES.json',classes)
    save('SUMMARY.json',summary)
    # Exposed-data scale comparison: a descriptive audit, not a new criterion.
    old=np.load(HERE.parent/'component_reuse_pilot_2026-10-18/COMPONENT_STATES.npz')
    new=np.load(HERE/'STATES_qwen7b.npz')
    scale=[]
    for fam in raw['qwen7b']['families']:
        fid=fam['id']
        cond={c['label']:c for c in raw['qwen7b']['conditions'] if c['family']==fid}
        drift=new[cond['n67_s0']['archive_key']].astype(np.float64)-new[cond['n63_s0']['archive_key']].astype(np.float64)
        relation=old[f'{fid}__passive_b'].astype(np.float64)-old[f'{fid}__active_a'].astype(np.float64)
        ratios=np.linalg.norm(drift,axis=-1)/np.maximum(np.linalg.norm(relation,axis=-1),1e-30)
        scale.append({'family':fid,'median_null_to_role_contrast_ratio':float(np.median(ratios)),
                      'max_null_to_role_contrast_ratio':float(np.max(ratios)),
                      'period_final_outgoing_ratio':float(ratios[-1,3,3]),
                      'first_person_full_contrast_exactly_reproduced':bool(np.array_equal(drift[:,:,0],relation[:,:,0])),
                      'first_person_max_unexplained_coordinate':float(np.max(np.abs(drift[:,:,0]-relation[:,:,0]))),
                      'first_person_contrast_total_l2':float(np.linalg.norm(relation[:,:,0]))})
    save('EXPLORATORY_SCALE_COMPARISON.json',{'status':'post hoc exposed-data scale check; not a success criterion','rows':scale})
    md=['# Prefix shape audit — 18 October 2026','',
        '**Question:** Was the pre-question invariance failure caused by future question content, run-to-run noise, or execution shape?',
        '', f"**Bounded Qwen shape explanation: {shape}.** This is a new diagnostic audit on exposed prompts; every earlier scientific gate and failure remains unchanged. Zero semantic interventions were run.",
        '', '## Complete results','',
        '| Model | Family | Across lengths | Different suffix, same length | Exact repeat | Real questions, native shape | Real questions, padded shape |','|---|---|---:|---:|---:|---:|---:|']
    for key,s in summary.items():
        if key=='decision':continue
        for f in s['families']:
            m=f['max_relative_discrepancies']
            show=lambda v:'untested' if v is None else f'{v:.7g}'
            md.append(f"| {key} | {f['id']} | {show(m['versus_prefix_only'])} | {show(m['same_shape_content'])} | {show(m['repeat'])} | {show(m.get('real_native'))} | {show(m.get('real_padded'))} |")
    md+=['','Values are maximum relative L2 discrepancies across all 28 blocks, four components and six sites. Full absolute norms, coordinate errors, cosines and every row are retained. Prefix tokens and positions are identical; suffixes are numerical controls rather than grammatical tasks. Artificial suffix continuations were never scored as linguistic answers.',
         '', 'Both models and both exposed families produced exactly two bit-identical prefix-tensor classes across the tested lengths: lengths up to and including 64, and lengths 65 and above. SHAPE_CLASSES.json records every length. The lower figure panels show the observed 64-to-65 discontinuity; this is a descriptive boundary localization after the fixed sweep, not an extra experiment or a changed success criterion.',
         '', '![Shape audit](PREFIX_SHAPE.png)','', '## Isolating a first operation','',
         '| Model | Embedding | Input RMSNorm | Native Q projection | Q applied row by row | Same quantized Q weights, float32 matmul |','|---|---:|---:|---:|---:|---:|']
    for key in raw:
        p=summary[key]['projection_max_relative']
        md.append('| '+key+' | '+' | '.join(f"{p[k]:.7g}" for k in ['embedding','input_rmsnorm','q_native','q_rowwise','q_dequantized_float32'])+' |')
    md+=['', 'These isolated operations use the same prefix vectors and model weights. A shape-dependent native Q result, together with invariant embeddings/RMSNorm and invariant rowwise Q, localizes at least one source to projection execution. It does not by itself establish the exact Metal kernel, explain every downstream discrepancy or rule out additional attention effects. Float32 here means a temporary dequantized Q matrix from the existing 4-bit model, not original full-precision weights.',
         '', '## What this changes','',
         'Fixed-shape extraction is a justified measurement control when its equal-shape null passes. Padding is appended after the whole prompt, where it cannot causally affect measured prefix tokens in exact arithmetic. Equal-shape agreement is a control on this backend and fixture, not a proof of hardware-independent numerical equivalence or of valid language answering on padded prompts. The reusable capture helper standardizes prefix extraction to a declared total length and reports provenance; it does not repair model competence or prove that activation transport works.',
         '', 'The earlier 161/192 code-answer result remains a competence failure. A technical cause for another failed prerequisite does not supply missing held-out semantic evidence. No old tolerance was raised and no semantic experiment was rerun. The two exposed families and local device/runtime cannot support a population statement or general claims about all LLMs.',
         '', 'A separately marked post hoc scale check compares identical-prefix drift (63 versus 67 tokens) with the already exposed active/passive role contrast. EXPLORATORY_SCALE_COMPARISON.json reports all family summaries. Small numerical differences must not be called the cause of large linguistic effects without a measured scale comparison.',
         '', '**Correction to interpretation of the earlier component map:** at person A in the same-order active/passive contrast, both sentences have an identical causal prefix through that name. In both exposed families, the complete 28-block × four-component role-contrast tensor at that site is reproduced exactly by the shape-only null (zero unexplained coordinate difference). This channel is therefore entirely numerical in these saved fixtures; it cannot encode the later syntactic role assignment. This is a post hoc exact-data check, not a prospective population result. At the final sentence-period outgoing state, the null contrast is only 1.60% and 1.13% of the role-contrast norm in d1/d2 respectively, so the finding does not erase the full sentence-level contrast or explain all previous failures.',
         '', 'The tagged backend source supplies a dispatch explanation consistent with the observed boundary; see INFERENCE_KERNEL_SOURCE.md. The first Q projection already changes despite identical embeddings and input-normalized vectors. This isolates a numerical source without treating a source-code calculation as a direct GPU kernel trace.',
         '', '## Prior work and novelty','',
         '[Thinking Machines Lab (2025)](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) already explains how shape-dependent reductions can produce different inference results despite run-to-run deterministic kernels. The present experiment is a local diagnostic application, not discovery of that general phenomenon. [Zhang and Nanda (2023/2024)](https://arxiv.org/abs/2309.16042) already show that patching methodology and metrics change interpretability conclusions.',
         '', '[Ding et al. (2021)](https://arxiv.org/abs/2108.01661) and [ReSi (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2eef1f75516b0cfd3e944345e5f88c08-Abstract-Conference.html) already argue for grounded tests of representation similarity. [DSA](https://arxiv.org/abs/2306.10168) and [InputDSA](https://openreview.net/forum?id=2FrAvRz1E7) already compare dynamical structure and separate input effects. [Causality ≠ Invariance (2026)](https://arxiv.org/abs/2602.22424) reports that causal task vectors need not be format invariant. [Dai et al. (2024)](https://aclanthology.org/2024.emnlp-main.967/) already shows binding-related causal edits. A new similarity name or another editable vector would not establish novelty.',
         '', '## Reproduction and preservation','',
         'Use the existing local MLX environment. Run each model in its own process with `python run.py qwen7b` and `python run.py llama3b`; each refuses a second start. Then run `python analyze.py`. No downloads or paid APIs. Exact duplicate tensor arrays are deduplicated by SHA256; every condition records its archive key and original input IDs. Full local tensor archives and source hashes are in the manifest. Scripts, complete raw metric outputs, figures and report are preserved on the existing research branch and linked to Issue #33. Verbose RAW JSON and CSV ledgers are committed as lossless .gz files, with decompressed originals retained locally and both checksums listed.','']
    (HERE/'REPORT.md').write_text('\n'.join(md))
    # Complete verbose raw outputs/ledgers are committed as lossless gzip files.
    for p in sorted(HERE.iterdir()):
        if p.is_file() and ((p.name.startswith('RAW_') and p.suffix=='.json') or p.suffix=='.csv'):
            out=p.with_suffix(p.suffix+'.gz')
            with out.open('wb') as f:
                f.write(gzip.compress(p.read_bytes(),compresslevel=9,mtime=0))
            assert gzip.decompress(out.read_bytes())==p.read_bytes()
    manifest=[]
    for p in sorted(HERE.iterdir()):
        if p.is_file() and p.name not in ['MANIFEST.json','GITHUB_PRESERVATION.json']:
            b=p.read_bytes();manifest.append({'path':p.name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),
                                            'git_blob_sha':hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),
                                            'storage':('local_archive' if p.suffix=='.npz' else
                                                       'local_expanded_and_lossless_gzip_on_branch' if (p.name.startswith('RAW_') and p.suffix=='.json') or p.suffix=='.csv' else 'research_branch')})
    save('MANIFEST.json',{'artifacts':manifest,'source_capture_script_sha256':hashlib.sha256((HERE.parent/'component_reuse_pilot_2026-10-18/run.py').read_bytes()).hexdigest(),
                          'source_protocol_sha256':hashlib.sha256((HERE.parent/'component_reuse_pilot_2026-10-18/PROTOCOL.json').read_bytes()).hexdigest()})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
