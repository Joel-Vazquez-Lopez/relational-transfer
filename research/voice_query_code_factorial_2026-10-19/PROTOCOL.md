# Frozen factorial diagnostic: sentence voice × question voice × answer encoding

19 October2026. One bounded local Qwen2.5-7B-Instruct4bit study; no new models, APIs, training or outcome-driven extensions. Starts from result65719a7 and preserves every older failed gate.

## Identification target

The last fresh panel showed a large Qwen active/passive difference in affirmative scores and a competence-filtering effect. This study tests whether that difference follows (1) source construction across tested controls, (2) a specific source–question same-voice penalty, or (3) literal Yes/No labeling. These are disjoint operational predictions, not exhaustive theories or uniquely identified semantic mechanisms. General copying, position, prompt interpretation and output preparation can remain competitors even if one pattern qualifies. A model need not have clean processing stages.

## Inputs

Six fresh AI-authored/annotated asymmetric predicate/name families, j01..j03 development and j04..j06 evaluation. No human annotation. Explicit entailment task: a converse not licensed by the sentence is not entailed, rather than asserted false. SOURCE_MANIFEST lists lexical uncertainty; no reciprocal 'accompany' item.

Full native factorial per family: role orientation2 × question direction2 × sentence voice2 × question voice2 × symbol lexicon2 (Yes/No or A/B) × mapping2 × prefix-position2 =128 prompts;768 total. All share the exact same entailment system instruction. Response legend follows the question; reversing it preserves the entire earlier token prefix. Both lexicons have normal and reversed mappings. No old threshold is transferred or fitted; target-aligned symbol scores and full-vocabulary argmax are separate outcomes.

Position prefixes are empty or `Record: `. Static tokenizer checks require active+Record and passive+empty to have exactly matched sentence-period, question-end and assistant-end positions, for each fixed content/question/legend. Both prefix levels appear in both voices. This controls the global measured positions, not every word's absolute position or all effects of the prefix itself.

Each family also has8 original-prompt references (old Yes/No instruction, original background, active question wording, no legend/prefix),48 total. These test transfer of the old phenomenon separately from explicit task reframing. They are not interchangeable with the new-task Yes/No cells.

## Scores and fixed competing predictions

s = logsumexp(tokens for entailed symbol) − logsumexp(tokens for nonentailed symbol), with bare and leading-space single-token aliases validated. Native correctness uses vocabulary-wide argmax mapped through the declared legend, without calibration or an additional alias-margin requirement. Report off-format outputs, mapping failures and every wrong answer. Four-answer units cross role×question direction; mu+ is mean s on the two entailed cases, mu− on the two nonentailed cases, D=(mu+−mu−)/2. Answer and question still determine binary role orientation; this is not a semantic-identification experiment.

G = mu+(passive sentence) − mu+(active sentence), at fixed question voice, lexicon, mapping and prefix. Also compute matched-position G using unprefixed passive versus prefixed active. Every factorial contrast is retained, including interactions and changes in negative-case scores. No selected layer or test statistic.

Predefine three **strict per-family patterns**. All require old-reference G≥4nats and at least90% requested-symbol compliance on the full128 new-task cells:

1. **construction_consistent:** every G≥4 across both question voices, both lexicons, both mappings, both same-prefix levels and matched-position comparisons.
2. **voice_match_penalty:** G≥4 with active questions and G≤−4 with passive questions, across all lexicons/mappings and position comparisons. This is a specific same-voice penalty predicted from the old active-question direction, not all possible copying/matching algorithms.
3. **literal_label_linked:** Yes/No G≥4 under normal mapping and G≤−4 under reversed mapping, with |A/B G|≤1 under either mapping, across question voices and position comparisons. This targets a literal-label-linked interaction. A constant label prior alone cancels in G; it is not being falsely tested as an explanation of all construction differences.

Four nats is a materially large score contrast relative to the old mean difference~19.56 and the .05 technical tolerance; one nat defines the declared near-null band. These thresholds are operational, not an information-theoretic estimate or posterior probability. Patterns are mutually exclusive and can all fail. For each phase report every family's classifications. Development selects a candidate only if it qualifies in at least2/3 families; freeze candidate, counts and raw-file hash before any evaluation output. The same candidate must qualify in at least2/3 evaluation families for a **small-panel prospective pattern replication**. No p values: six authored families/shared templates do not support a broad population or paper claim. If no candidate or a failed transfer, say so; do not promote a held-only pattern retrospectively.

If the original reference effect persists but disappears already under explicit Yes/No task framing, report framing sensitivity; this is not evidence uniquely locating output symbols. If A/B/reversed legends are poorly followed, report format/mapping competence failure, not loss of relational representation. No native competence exclusion changes collection or group means.

## Fixed causal panel

For every family, the canonical new-task A/B normal mapping, empty prefix, both question voices has16 prompts. Collect frozen/closed/irrelevant source masks at cut14 on all96 canonical prompts (288 intervention forwards), irrespective of all native outcomes. Masks/wrapper are inherited; only supplied attention edges change, with full prefix and pre-cut numerical checks. Frozen prevents period refresh and suffix sentence-token bypass, while leaving suffix access to the period. Closed additionally blocks suffix period access. Irrelevant translates the exact closed edge count to the earlier background; it is not matched for attention mass or token positions and is not a true sham.

For each family×sentence-voice×question-voice four-answer unit, retain raw D and retention D_mask/D_native only when D_native≥.5; retain ordinal comparisons and native compliance. A separate routing-pattern diagnostic requires positive ordinal comparisons in all4 canonical units and all irrelevant-mask retentions within.25 of1. Define B_q = retention_frozen(active sentence,q) − retention_frozen(passive sentence,q). `source_voice_dependence` requires both B≥.25; `query_match_dependence` requires B_active-question≥.25 and B_passive-question≤−.25. Development/held rule is separately2/3 families, exactly as above. Failure of the neutral-drift gate remains a real control outcome; it prevents calling a clean route pattern but never discards values. Closed-mask rescue values are descriptive and do not establish role content or revive the earlier failed period-message hypothesis.

## Instrumentation and budget

All new calls use batch1, fixed192 tokens with dummy x appended after the original assistant end, scored at that original end. Residual traces save embeddings plus28 block outputs at sentence period, question end and assistant end for native calls. Interventions save depths0,7,14,21,28 plus full-trace/prefix hashes. All full output vectors are lossless. Attention/MLP decomposition is deferred: the current inference does not yet justify a particular component-level mechanism.

**1119 maximum/planned forwards:**816 native scientific prompts +288 fixed masks +15 validation calls. Validations: exposed old fixture at128/192(2); fresh unwrapped/wrapped-native/repeat(3); two sham masks at28(2); first-family canonical active-question closed cut0 role disconnections(8). Count attempts before dispatch, save payload before score, no retries or automatic model resume.

Before fresh outcomes, exposed128/192 full-logit drift must be≤.05; the128 output must match the saved old fixture≤.05. All scientific calls192, so this also flags unsupported cross-shape joins. Wrapper/native/dense/repeat/sham logits≤.05; repeated/sham states and all unchanged pre-cut states≤1e−6. Future question/legend changes must leave the sentence-period traces invariant≤1e−6. Legend changes must leave full question prefixes bit-identical. Cut0 disconnected opposite roles require exact suffix hashes and logits≤.05. Any failed technical gate stops and preserves the error; never raise tolerances.

## Storage and decision

Maintain≥1GiB disk during inference. Save every per-call NPZ losslessly before scoring. After independent analysis, consolidate into uncompressed tar bundles of at most~32MiB (NPZ members are already compressed), verify every member byte/hash, persist bundle index, then remove expanded staging copies. This changes packaging, not data: every original NPZ remains locally and remotely recoverable. Analyze supports both expanded and bundled forms. Never delete earlier raw experiments to make a positive result possible.

Static/CPU synthetic rehearsals use no real model forward and must recover the three distinct patterns and reject a null. Freeze scripts/inputs/protocol on the existing research branch and verify exact Git blobs before all new outcomes. Collect every fixed evaluation cell even with a null development candidate; no post-result site, prompt, hypothesis, scoring or family changes. Report whole-family, four-answer-unit and individual-answer outcomes separately, every control and unexpected answer. Append results with a leased commit and Issue35. Conclude only replicated operational pattern / failed operational prediction / technical stop, and stop after this panel. No new mechanism or novelty follows automatically.
