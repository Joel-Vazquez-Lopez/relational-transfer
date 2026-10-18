"""Write a technical-note draft from the completed frozen benchmark.

This is reporting only. No inference, metric fitting, threshold change or new
success criterion. Secondary category means summarize the saved pair ledger.
"""
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    summary=json.loads((HERE/'SUMMARY.json').read_text())
    pairs=list(csv.DictReader((HERE/'PAIRWISE.csv').open()))
    nulls=list(csv.DictReader((HERE/'NULLS.csv').open()))
    cats=['role_active','role_passive','voice_agent_a','voice_agent_b','unrelated','artifact']
    average=lambda v:sum(v)/len(v) if v else None
    matrix={}
    for a in cats:
        matrix[a]={b:average([float(r['gram_lag8']) for r in pairs if r['model_a']!=r['model_b'] and r['mode']=='fixed' and r['category_a']==a and r['category_b']==b and r['gram_lag8']]) for b in cats}
    (HERE/'CROSS_MODEL_CATEGORY_MEANS.json').write_text(json.dumps({'status':'Descriptive summary of the prespecified full comparison ledger; no new endpoint or fitted metric.','fixed_lag8':matrix},indent=2)+'\n')
    table=['| Model | Role/role | Role/unrelated | Difference | Leave-one-family-out difference | Largest angle correction |','|---|---:|---:|---:|---:|---:|']
    for m,d in summary['models'].items():
        primary=next(r for r in d['primary'] if r['mode']=='fixed' and r['lag']==8 and r['leave_out'] is None)
        loo=[r['specificity_over_unrelated'] for r in d['primary'] if r['mode']=='fixed' and r['lag']==8 and r['leave_out']]
        change=max(r['maximum_angle_change'] for r in d['case_angle_changes'])
        table.append(f"| {m} | {primary['role_within']:.4f} | {primary['role_to_unrelated']:.4f} | +{primary['specificity_over_unrelated']:.4f} | {min(loo):.4f}–{max(loo):.4f} | {change:.6f} |")
    finding=['# What this benchmark actually found','',
             '**Decision: NARROW.** A controlled geometric regularity survives execution-shape correction, but neither a shared semantic algorithm nor a novel mechanism has been identified.',
             '', '## 1. The artifact does not explain the primary sentence-period result','',
             'All 16 family/model content-null checks passed exactly. All 176 declared forward passes completed, with no unavailable 64/65 null pairs. Qwen changed by at most 0.005417 in the active/passive role-response cosine; Llama changed by exactly zero. No family exceeded the frozen 0.05 materiality threshold, and none of the eight primary native/fixed gap signs reversed. This rejects the benchmark-specific prediction that correcting this numerical artifact materially alters the primary period-level result.',
             '', '## 2. There is a modest, stable within-model pattern','',
             'Using only Gram entries for layer steps at least eight apart:', '',*table,
             '', 'Each leave-one-family-out gap is positive. These are sensitivity checks on eight author-constructed families, not confidence intervals or independent replications. The comparison is descriptive: the unrelated paths change names and usually verbs, whereas role reversal preserves the lexical inventory and swaps its order. A lexical-permutation explanation remains untested. The result should be called controlled role-reversal **geometry**, not evidence that the model represents an abstract role variable.',
             '', '## 3. An apparent family difference at the first person disappears','',
             'At the first person token, active_a and passive_b have identical causal token prefixes. Native captures produced a nonzero whole-layer contrast in 8/8 Qwen families and 0/8 Llama families. Fixed-shape capture produced zero in all 16 family/model cases. Qwen native prompts have 63 versus 65 tokens, while the Llama prompts are already above that boundary. This demonstrates a specific way tokenization/prompt length and a shared runtime can masquerade as a model-family difference at a causally ineligible site. It does not explain the valid sentence-period contrast.',
             '', '## 4. Long-range null-path coherence is weak','',
             'Numerical-null within-model Gram correlations fall from 0.319 to 0.0036 for Qwen and 0.745 to 0.0538 for Llama when the minimum lag changes from 1 to 8. Nearby-step structure supplies much of the apparent numerical-path coherence. This is evidence against a strong long-range shared numerical trajectory in this fixture. Absolute null endpoint norms are at most 2.38% and 2.18% of the active role-contrast endpoint norm for Qwen/Llama respectively.',
             '', '## 5. The cross-model matrix does not establish operation matching','',
             'Fixed-shape long-range role/role similarity across models is about 0.167 (active) and 0.191 (passive). However, a Qwen unrelated contrast compared with a Llama active-role contrast is about 0.148. Much of the matrix favors Llama role trajectories irrespective of the Qwen operation. The complete six-by-six means are saved separately. Matched-family advantages are small. High individual entries cannot be treated as successful matching of labeled computations.',
             '', '## Research decision','',
             'The next discriminating linguistic control would preserve roles while matching lexical reordering, rather than use an unrelated sentence as the only rival. This is a proposed control, not an experiment run here. No learned centering or new similarity measure was introduced after outcomes. The numerical artifact line is now bounded; the residual within-model geometry is a lead, not yet a paper-level semantic discovery. A technical-note draft makes the actual claim concrete for review.','']
    (HERE/'FINDINGS.md').write_text('\n'.join(finding))
    article=['# Numerical nulls and linguistic contrast trajectories in two quantized language models','',
             '**Working technical-note draft — not peer reviewed; novelty and publication readiness unestablished.**','',
             '## Abstract','',
             'Layerwise contrasts are appealing tools for comparing how language models transform an input. Their geometric descriptors can also encode generic depth schedules and numerical execution effects. We study this distinction in existing 4-bit MLX versions of Qwen2.5-7B-Instruct and Llama-3.2-3B-Instruct. After a development audit isolated an execution-shape discontinuity, we froze an eight-family active/passive benchmark with native and standardized shapes, exact causal-prefix nulls, unrelated contrasts, and both short- and long-range step-Gram comparisons. All 176 forward passes and all content-null checks completed. Shape correction removed a causally ineligible first-person contrast in all eight Qwen families, while sentence-period role-response angles changed by at most 0.00542; Llama period angles were identical. Long-range within-model role-reversal similarity exceeded role-to-unrelated similarity by 0.096 and 0.152 respectively, with positive leave-one-family-out sensitivity gaps. Numerical-null coherence largely disappeared at long lags. These results distinguish a localized runtime artifact from a modest linguistic-contrast regularity. They do not establish abstract role encoding or a shared cross-model algorithm, because lexical reordering remains a sufficient untested rival.',
             '', '## 1. Research question','',
             'An observed contrast path can combine several sources: altered linguistic input, changed output demands, a common layer schedule and finite-precision execution. A useful comparison should distinguish these sources before interpreting geometric agreement as a shared computation. Our question is deliberately narrower than discovering a universal internal circuit: does controlling a known shape artifact change the apparent specificity of role-reversal trajectories?',
             '', 'An exact causal-prefix null provides a particularly clear check. For a causal decoder in exact arithmetic, a token state cannot depend on suffix tokens it cannot attend to. Two inputs with the same token prefix therefore have equal states at prefix positions. A nonzero contrast there can diagnose numerical or measurement effects without needing a semantic annotation or an answer-accuracy argument. This is an elementary property of the computation graph, not a newly proposed theorem.',
             '', '## 2. Relationship to prior work','',
             '[Thinking Machines Lab (2025)](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) describes shape-dependent reductions in inference. [Khatri (2026)](https://arxiv.org/abs/2609.31796) directly studies numerical robustness of activation probes and local representation changes. Our study instead examines labeled contrast paths at fixed prefix sites; it makes no priority claim for numerical sensitivity.',
             '', '[Ding et al. (2021)](https://arxiv.org/abs/2108.01661) and [ReSi](https://arxiv.org/abs/2408.00531) motivate grounding similarity in function and nuisance controls. [DSA](https://arxiv.org/abs/2306.10168) addresses dynamical comparison. Our feedforward layer index is not assumed to be recurrent time or a stationary transition operator. [Dai et al. (2024)](https://aclanthology.org/2024.emnlp-main.967/) already demonstrates causal binding edits; geometric agreement here is a weaker claim than such causal evidence.',
             '', '## 3. Design and measurements','',
             'Eight lexical families were declared before inference, each with active/passive renderings of both possible Agent assignments. The exact prompts and token IDs are retained. Native-length and right-padded 128-token runs share their measured prefixes. Padding is used only to measure earlier states, never to score a language answer. For each active sentence, an identical-prefix 64/65-token pair supplies a numerical contrast and a same-shape alternative suffix supplies a content control.',
             '', 'We save embeddings and all 28 outgoing residual states at four token anchors. For a contrast delta_l, its block change is v_l = delta_l − delta_(l−1). We compare the full cosine Gram of these changes, their norms and endpoint magnitudes. Minimum Gram lags of 1 and 8 are fixed in advance. Zero-vector directions remain undefined. No coordinate alignment, PCA, smoothing or fitted similarity metric is used.',
             '', 'The primary period-site specificity gap subtracts role-to-unrelated Gram similarity from role-to-role similarity. A cyclic unrelated control changes lexical content; it is deliberately broad and is not a matched word-order control. All within- and cross-model pairings are retained. Eight leave-one-family-out summaries assess sensitivity without treating pair counts as independent samples. A predefined 0.05 angle change and primary-gap sign reversals assess materiality of shape correction.',
             '', '## 4. Results','',*table,
             '', 'All standardized-shape content controls are exactly invariant. Qwen native active/passive lengths straddle 64/65 tokens. The first-person site, whose causal prefix is identical in the compared sentences, has a nonzero native contrast in all eight Qwen families and a zero contrast after correction. Llama shows zero at this site in both modes because its native prompts are already above the observed boundary. This removes one apparent architecture-dependent difference without altering the linguistic input.',
             '', 'The primary sentence-period result is stable: no family reaches the angle-change threshold and no primary gap changes sign. Positive long-range specificity gaps remain in both models and every leave-one-family-out summary. Numerical-null long-range Gram coherence is weak (0.0036 Qwen; 0.0538 Llama), despite stronger short-range values. A broad claim that numerical errors explain the primary period-level pattern is therefore unsupported by this benchmark.',
             '', 'Cross-model category means are weaker evidence. The active role-to-role mean is approximately 0.167, but Qwen unrelated-to-Llama-role similarity is approximately 0.148. This column preference limits an interpretation based on matching operations. Small matched-family advantages and eight-sample CKA are reported descriptively, without a shared-algorithm conclusion.',
             '', '## 5. Interpretation','',
             'Normalized shape and amplitude answer different questions. Multiplying every nonzero step by a common positive factor leaves its cosine Gram unchanged; even a very small error can therefore yield a visually structured descriptor. This scale invariance is elementary. It motivates recording null scales, rather than treating normalization as evidence of computational relevance.',
             '', 'Our results support a bounded separation. An early channel was entirely numerical, while a later linguistic-contrast regularity survived correction and long-range controls. The surviving regularity may reflect lexical permutation, syntactic routing, role information or a mixture. The present unrelated control does not distinguish them. A matched role-preserving reordering control is required before advancing the semantic interpretation.',
             '', '## 6. Limitations and reproducibility','',
             'The eight families are author-constructed, not a natural corpus or an independent population sample. No generated answer or causal intervention was evaluated. Both models use the same backend and device; this is not independent runtime replication. They also share a 28-block depth. Neither the 0.05 materiality threshold nor the observed constants is a universal standard. The numerical source is consistent with a backend dispatch transition, but no GPU dispatch trace was collected.',
             '', 'The frozen protocol, pre-run script hashes, all prompts, every descriptor and comparison, full local state archives, figures and source manifests are retained. Raw verbose text outputs are losslessly compressed for the research branch. No training, downloads, paid APIs or adaptive follow-up occurred. The development audit and this prospectively fixed benchmark remain separate.',
             '', '## 7. Conclusion','',
             'Causal-prefix nulls can expose a qualitative measurement error while leaving a later contrast-level result stable. In this small benchmark, role-reversal paths retain modest within-model geometric specificity after execution correction, but cross-model comparison and semantic interpretation remain unresolved. The appropriate claim is calibrated geometric evidence, not a novel shared semantic mechanism.','']
    (HERE/'TECHNICAL_NOTE_DRAFT.md').write_text('\n'.join(article))
    manifest=[]
    for p in sorted(HERE.iterdir()):
        if p.is_file() and p.name not in ['MANIFEST.json','GITHUB_PRESERVATION.json']:
            b=p.read_bytes();storage='local_tensor_archive' if p.name.startswith('STATES_') else 'research_branch'
            if p.suffix=='.csv' or (p.name.startswith('RAW_') and p.suffix=='.json'):storage='local_expanded_and_gzip_on_branch'
            manifest.append({'path':p.name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),
                             'git_blob_sha':hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),'storage':storage})
    source=HERE.parent/'prefix_shape_audit_2026-10-18/SOURCE_MANIFEST.json'
    (HERE/'MANIFEST.json').write_text(json.dumps({'artifacts':manifest,'source_runtime_model_manifest':str(source.relative_to(HERE.parent)),
                                               'source_manifest_sha256':hashlib.sha256(source.read_bytes()).hexdigest()},indent=2)+'\n')
    print('Wrote findings, full cross-model category means, technical-note draft and refreshed artifact manifest.')


if __name__=='__main__':main()
