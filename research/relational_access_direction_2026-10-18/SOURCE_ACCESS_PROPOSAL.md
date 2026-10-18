# Causal dependence on continued context access

18 October 2026. Preferred next design direction after the historical overlap audit. Proposal only: no new model inference, frozen protocol, novel-mechanism claim, or authorization for substantial execution.

## One question

How does the answer's dependence on continued access to the event context change during question processing, and does that profile differ systematically across queries, linguistic realizations and models?

The initial measurement is deliberately narrow. It does not reconstruct a complete developmental graph, isolate a semantic variable, or identify when a role representation first exists.

## Why this differs from our completed interventions

The role-path pilot replaced the final-prompt residual state and measured answer redirection. The transport study replaced one pre-question period state at three layers and failed its selectivity gate. The component-reuse intervention stage was not executed. The repaired Atlas captured attention/MLP/residual writes descriptively and stopped on competence.

Those outcomes do not establish the dependence of a suffix computation on continued access to *all* contextual source positions. Searching the local reports/protocols for attention knockout, source/access cuts and KV patching found no completed experiment that executes the intervention specified here. This is a bounded local-history finding, not a claim about every experiment in other chats or branches.

## Concrete intervention

Use a fixed prompt comprising an invariant instruction prefix, event context, and question/assistant suffix. Start with direct natural-name answers, using a tokenizer-valid scorer; avoid adding binary verification or arbitrary answer codes to this study.

For a predeclared block boundary k, leave all earlier computation unchanged. At every attention operation from k onward, prohibit every question/assistant-suffix position from attending to every context-dependent prefix position. Retain causal self-attention within the suffix and attention to the invariant instruction prefix. Do not delete the context tokens or recompute the early suffix without them.

This preserves information that reached the suffix before k while closing subsequent direct access to context. The cut must include context-dependent punctuation, delimiters and other possible prefix relays, not just person names or the sentence period. The retained instruction prefix must precede the context and contain no condition-specific information. Cutting only the final query token would leave relay paths through earlier question tokens and is not the proposed intervention.

For a next-token endpoint there are no generated-token relays. A later full-name scoring implementation must preserve the same cut for teacher-forced continuation tokens and disclose its conditioning; candidate ranking and free generation are distinct outcomes.

Record the complete answer distribution and correct-versus-rival margin for each cut and the unmodified baseline. Keep nonmonotonic responses. A threshold-defined single 'commitment layer' is not the primary object.

## Competing patterns and limits

| Pattern | Compatible account | What it does not establish |
|---|---|---|
| Early cuts preserve accurate answers across counterbalanced names | Remaining suffix computation can answer without later direct context access under this intervention | A compact semantic code, or its construction time |
| Early cuts impair answers; later cuts preserve them | Continued source access matters over an earlier range of computation | That the information was absent earlier; masking may disrupt routing |
| Impairment persists until late cuts | Later context access or sensitivity to the mask intervention | A unique late retrieval mechanism |
| Several separated sensitive intervals or nonmonotonic effects | Multiple dependencies, compensation, or intervention interactions | A mandatory serial list of stages |
| Role and first-mentioned queries have different profiles | A query-dependent difference on the tested prompts | An isolated semantic-versus-lexical distinction; query texts differ |

Incorrect baseline cases stay in the ledger but cannot demonstrate preservation of successful task performance. Report both complete-family competence and individual answers. Do not weaken previous studies' gates or reuse their unsuccessful outcomes as new confirmation.

## Minimal contrasts

First establish implementability and competence in Qwen on a small, separately frozen development panel. One whole-context access intervention, a few fixed boundaries, and two direct-name demands are enough for feasibility; no head search, fitted projection, arbitrary warping or model sweep.

Agent versus first-mentioned-person questions provide a semantic retrieval demand and a surface retrieval demand. Include both role orientations to balance answer identity. This comparison alone cannot eliminate syntax-aware copying or wording effects. Active/passive and new lexical families should be a later frozen prediction test, not additional dimensions adaptively searched for a favorable contrast. Llama is a second-model test only if its declared behavioral and numerical prerequisites pass. A two-model result does not establish family dependence.

The key cross-model target is a reproducible functional correspondence: for example, a dependency on continued source access that shifts in timing but retains a query-specific pattern. A meaningful alternative is that the models differ in that dependency. Fit any mapping using development cases only and test fresh intervention outcomes. Normalized-depth and query-insensitive profiles are baselines. Similar-looking curves alone are not a successful alignment.

## Controls required before interpretation

1. Known-program calibration: carry an answer forward after an early read versus retrieve it only later, keeping terminal outputs identical. Verify that the proposed intervention distinguishes those declared dependencies. Include repeated retrieval/overwrite and bypass-path examples; do not infer a universal stage from a step curve.
2. Exact identity/unmodified-mask and wrapper checks under fixed numerical shapes. Preserve the successful causal-prefix invariance and native reconstruction controls from the repaired Atlas.
3. Endpoint controls: cut before any suffix access versus after all measured computation. The latter is a technical null. Performance above chance under the former requires an explicit leak/prior audit; small balanced sets do not provide a population chance estimate.
4. All suffix positions must receive the cut. Trace the allowed information paths explicitly in the implementation. Masking only the final position or only person tokens is insufficient.
5. A preregistered matched irrelevant-source intervention and a literal retrieval demand to check whether the damage is general. Matching edge count/position is imperfect and must be documented.
6. A planned restoration condition that reopens source access at a fixed interval, alongside the fully restored technical baseline. Rescue constrains some generic-damage explanations but does not prove semantic specificity. Restoration can fail because the cut already damaged a query representation.
7. Full baseline, cut and control outputs, token indices, mask bytes/hashes, attempted calls and failures. Model instrumentation and budget must be reviewed and frozen before target outcomes.

Attention removal changes normalization and may create unfamiliar activation distributions. No control makes that issue disappear. If removal causes comparable collapse in all demands or controls fail, close the feasibility study rather than relabeling collapse as source dependence. A matched donor-cache intervention is a possible separately designed alternative, not an adaptive rescue included in this proposal.

## Prior work and possible contribution

Attention-edge interventions and staged retrieval are established methods/results:

- Geva et al. (EMNLP 2023), *Dissecting Recall of Factual Associations in Auto-Regressive Language Models*, uses attention-edge interventions to study a sequence of information transfer in factual recall. https://aclanthology.org/2023.emnlp-main.751/
- Prakash et al., *Language Models Use Lookbacks to Track Beliefs*, identifies binding, answer and visibility lookbacks in belief tracking. https://arxiv.org/abs/2505.14685
- Gur-Arieh, Geva and Geiger, *Mixing Mechanisms*, demonstrates positional, lexical and reflexive mechanisms across multiple models/tasks. https://arxiv.org/abs/2510.06182
- Dai et al. (EMNLP 2024), *Representational Analysis of Binding in Language Models*, already establishes binding-related low-rank representations and activation edits. https://aclanthology.org/2024.emnlp-main.967/

The abstracts and primary project descriptions establish close overlap. Their full methods must be compared against the eventual precise claim before execution as a novelty-driven study. Neither attention knockout, continued-context dependence nor 'models retrieve information' is available as an invention claim.

A possible contribution would be a new, replicated linguistic condition that predicts when continued source access matters, with a controlled cross-model difference or correspondence and selective causal outcomes on fresh cases. No such result is established here. A source-access curve by itself is a diagnostic, not a paper.

## Alternatives worth retaining

**Cue-triggered syntactic revision:** compare temporarily ambiguous and early-disambiguated sentences, then test effects of the disambiguating cue across tokens/components. This follows linguistic development over text as well as depth. Garden-path mechanisms have close prior work, annotation/competence are demanding, and generic surprisal is a serious rival. No novelty audit or execution is completed for this alternative.

**Measurement-identification benchmark:** test whether proposed cross-model descriptors distinguish known systems with identical answers but different dependencies under coordinate changes and changes in computational depth. This was already proposed in our earlier research decision and partly illustrated by copying counterexamples; it is not a newly conceived direction. It could be a prerequisite for a comparison method, provided it tests genuinely missing cases and existing methods.

## Immediate deliverable and authorization boundary

The present deliverable is this proposal and its historical/prior-work boundary. No model calls or attention interventions were run. Use existing local models only if a separately bounded implementation/feasibility protocol is approved; no paid API, training, model download or large inference run is proposed as an automatic next step.
