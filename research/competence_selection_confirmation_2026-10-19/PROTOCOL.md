# Prospective confirmation: competence screens select constructions

19 October 2026. One bounded study, fixed before fresh model outputs. This follows the explicitly post hoc `readout_selection_synthesis_2026-10-19`; it does not retrofit that synthesis or repair earlier failed gates. No inference beyond this panel is authorized by this protocol.

## Question and competing explanations

Does a native all-four-correct screen preferentially remove active constructions in Qwen, although an externally fixed readout separates their answers? Does changing only eligibility change a fixed cross-checkpoint causal comparison on fresh families?

1. **Transferred readout boundary / selection account:** Qwen has more calibration-rescued complete active than passive units. Native screening changes the construction mixture. At least some removed units have positive role×question contrast and completely separated Yes/No scores. Changing the gate may change the reported intervention comparison, without changing any forward computation.
2. **Relational discrimination deficit:** removed units have overlapping Yes/No score ranges or nonpositive role/question contrasts; a scalar threshold cannot repair them. The old threshold may also fail to transfer even when another threshold could separate the four answers. These are separate outcomes.
3. **Stable causal comparison:** eligibility changes but the fixed intervention comparison remains approximately stable. Filtering is then a selection effect without the previously illustrated magnitude of profile distortion. A constant answer bias alone does not imply a changed causal comparison.

These hypotheses are not exhaustive, mutually exclusive global mechanisms, or proof of semantic binding. A syntax-aware parser/copier or answer-preparation computation can produce the same binary contrasts. Both checkpoints, the exact prompt and the quantization are fixed; no family-wide model claim is possible.

## Panel and annotations

24 predefined fresh name-pair/verb lexical families, each with R2 × voice2 × Q2 = 8 prompts. Each voice-specific **unit consists of four role×question answers**. A family contains two units. The whole-family eight-answer screen is also reported separately. All labels come from deterministic AI-authored templates, with an explicit lexical/annotation manifest. No human/corpus validation is claimed. Complete prompts and sentences are checked against earlier local INPUTS manifests before freezing. Names/verbs are not selected by model answers.

Active: `Agent past Patient.` Passive: `Patient was past by Agent.` Question: `Did A base B?` or its role reversal. Gold Yes iff R=Q. This preserves same-answer/different-relation and same-relation/different-answer cells, but R remains a deterministic function of answer and Q: this panel cannot uniquely identify an internal semantic variable. Passive participles are regular and transitive; unusual plausibility, polysemy and missing complement readings remain limitations.

Use the previous exact background, Yes/No system instruction, native tokenizer/chat template, batch1/128 shape with padding after the original prompt end. Score at the original assistant prompt end. Native correctness means vocabulary-wide next-token argmax is a declared Yes/No alias and matches gold; this is not a collected full generated response. Also report restricted zero-threshold accuracy, off-label argmax, and unchanged external classifiers. Thresholds are copied unchanged from the old eight-example exposed calibration: Qwen −19.507811598961112; Llama −2.4140621280200847. No threshold fitting on this panel.

## Primary endpoint and inference

For family i and voice v, G_native and G_cal are all-four-correct indicators. Define d_i = (G_cal,active − G_native,active) − (G_cal,passive − G_native,passive). **The sole confirmatory test** is the directional Qwen family-level sign test: among nonzero d_i, test whether positive signs exceed fair-binomial expectation. Exact one-sided p = sum_{k=k+}^n C(n,k)/2^n; alpha .05, no retries or alternative primary tests. Report every d_i, ties and effect magnitude. This is a conditional sign-symmetry analysis of an authored finite panel, not a randomized population sample; shared templates and lexical dependencies limit generalization. No layer, answer, or checkpoint is an independent replicate for this test. Llama, native voice differences and whole-family screens are descriptive checks. Negative/zero findings are retained.

Success of this test supports a narrow replicated construction-dependent readout/screen effect, not a novel linguistic mechanism, a general calibration method, or a paper guarantee.

## Fixed causal comparison (secondary; no significance claim)

For every prompt, collect native dense baseline plus **source-isolated period (`frozen`), closed-source, and equal-edge-count irrelevant-span masks**, all at the old fixed cut14 (zero-indexed blocks14..27). No competence exclusion occurs during collection. The original wrapper changes only the native attention mask. The period cannot reread sentence tokens after the cut in frozen; suffix can attend to it. Closed also blocks suffix access to the period. These interventions renormalize attention, can induce alternate routing, and do not isolate a unique semantic carrier.

s = logsumexp(Yes aliases) − logsumexp(No aliases). D = (s00−s01−s10+s11)/4. Retention = D_mask/D_native when |D_native|≥.5. Flag negative native D separately; retain raw D and undefined ratios. No winsorization or selected layer. The prespecified illustration is frozen14; closed14 and irrelevant14 are full retained controls, not alternative primary endpoints. Report every family×voice D, ratio, question-pair sign, baseline/gold labels and new changes.

For all/native-complete/calibrated-complete policies, report model means, Llama−Qwen gap, raw D means, selected units, whole-family gates, and common-selected matched subsets. Delta_gap = native-gated gap − full-panel gap. A gap shift ≥.10 in the previously observed positive direction is a prespecified **descriptive replication criterion**, not a p value. Undefined means/ratios cannot count as success. Require the direction also in raw per-unit gap changes on common selected support before saying it is robust to matched support; otherwise explicitly label mixture dependence. Different gates estimate different conditional samples, not changes to the model itself. A negligible shift or different direction challenges the earlier illustration. No magnitude ratio if the original gap is near zero.

Decompose each baseline unit into mean offset, R term, Q term and R×Q term. Report the open separating interval (max No score, min Yes score): empty/overlapping, contains frozen threshold, or separated but threshold misses. A nonempty interval demonstrates four-score separability, not native competence or semantic sufficiency. Off-label argmax remains separately visible. This decomposition is diagnostic, not an extra confirmatory test.

## Numerical gates, budget and preservation

Per checkpoint: 192 prompts ×4 conditions =768 forwards; first-row native unwrapped/wrapped/repeat validations3; one sham per voice2; first-family cut0 closed disconnection controls8. **Exactly781 planned, maximum781 per model /1562 total**, including every attempted forward. No automatic retries or resume, no outcome-driven additions. Both checkpoints finish the complete panel irrespective of answer correctness. Stop on technical/numerical failure and preserve it.

Wrapper/dense/repeat full-logit tolerance .05; repeated and question-invariant period traces 1e−6; untouched prefixes and before-cut traces exact to 1e−6; sham full logits .05; disconnected role alternatives exact suffix hashes plus logits .05. Record every control before decision. Zero model forwards in preparation/tests. Existing weights/runtime hashes must match the prior lock.

Every attempt is journaled before dispatch, each full output vector saved losslessly before scoring. Native traces contain embeddings + all28 residual outputs at period and answer. Intervention traces preserve anchors0,7,14,21,28 plus exact full-trace hashes; full before-cut checks run before releasing arrays. No attention/MLP traces or KV donation in this study. Keep all unexpected outputs and failures. Reserve ≥1GiB disk; publish scripts, frozen manifests, numerical checks, raw journals, ledgers and every NPZ payload in reconstructable shards. Freeze on the existing research branch before output, use leased append-only publication and Issue35. Stop after this bounded study and report regardless of significance.
