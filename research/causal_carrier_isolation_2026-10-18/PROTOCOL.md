# Carrier isolation: frozen versus continuously refreshed sentence information

18 October 2026. New bounded pilot authorized by Joel's request to evaluate and execute the best local next step. Earlier failed experiments remain closed. No previous held-out stimuli are reused. AI-only annotation; no training, model download, paid API or external annotation. This is a new operational question, not an attempt to reverse an old failed gate.

## Why this question

The previous language-model study cut access to the full event, while the reference studies used known mappings and eventually exposed an architecture confound. Neither established how a pretrained language model carries sentence information into a later question. Leaving a carrier readable is insufficient to diagnose cached information if that carrier can keep rereading its sources. This pilot explicitly separates those routes. Attention-edge interventions themselves are established methodology.

## Controlled language and endpoint

Four fresh name/verb families, fixed in core.py: d1 development; e1/e2/e3 evaluation. Per family: Agent identity (2) × active/passive voice (2) × question orientation (2), eight prompts. Sentence meaning and answer label are crossed: reversing the later question changes Yes/No while preserving the entire sentence prefix; changing roles and the question together preserves the answer label. Voice preserves roles while changing word order. The lexical inventory is fixed within a family. Voice and word order are not independently isolated.

Prompts ask whether A VERB B, not for arbitrary answer codes or mention-order judgments. Those auxiliary tasks caused earlier failures and are not needed for this primary question. This prospective redesign does not convert earlier failures into passes. Local tokenizers and exact inputs are frozen, with whole-vocabulary argmax and deduplicated single-token Yes/No aliases. A baseline passes only if argmax belongs to the gold class AND gold-versus-rival alias logsumexp is positive. These are next-token classification results; no generated explanation or EOS completion is claimed.

One model at a time, Qwen 7B then Llama 3B independently. Each model must pass all eight development baselines and numerical gates. No replacement families. Development interventions at all fixed sites are then measured. A prespecified development signature is required before any evaluation output for that model. Evaluate all 24 fresh baselines first and stop before evaluation interventions if any fails. Never select competent evaluation subgroups.

## Intervention definitions

All forwards have batch 1 and total length 128; dummy x tokens follow the original assistant end. Measure logits at that original end. No KV cache. Native attention/MLP/norm/residual operations are retained; dense causal masks are supplied uniformly. Blocks zero-indexed; cut k changes blocks k..27. Fixed cuts 0,7,14,21. Cut28 is a sham.

Let S be original sentence tokens before its final period P, and Q all positions after P, including formatting and every later question token. No later formatting token is allowed to evade a source cut.

- **closed:** at/after cutoff, P cannot read S; Q cannot read S or P. P keeps its self edge. Any surviving answer can use pre-cut information already in Q, neutral prefix and question content.
- **frozen:** P and Q cannot read S; Q can read P. P cannot refresh its source information after the cutoff. Its state can still evolve, so frozen means source-isolated, not numerically fixed.
- **live:** Q cannot read S, but can read P; P continues reading S. This admits repeated retrieval through P.
- **neighbor:** protect the immediately preceding lexical token instead of P, close its upstream reads, and block all other sentence sources for it and later positions. This probes alternative carriers, not an equal-attention-mass intervention. At cut0 that name token already contains role-correlated identity, so it is not a no-information null.
- **irrelevant:** translate exactly the closed deleted-edge pattern onto the preceding neutral background, retaining native sentence access. Counts and query indices match closed, but absolute position and attention mass do not.

Closed differs from frozen only by Q→P edges. Frozen differs from live only by P→S edges. Residual changes induced downstream are allowed to differ; this is a pathway intervention, not a claim about a unique internal variable. Preserved Q states may already contain source information at cutoff. Frozen rescue therefore demonstrates incremental usefulness of a source-isolated carrier alongside that information, not that P alone is sufficient.

## Technical and control gates

Native/unwrapped, wrapped-native, dense-baseline and exact-repeat comparisons on first development prompt: max logit error ≤.05; exact-repeat selected states ≤1e-6. Same-sentence period states must be exactly equal under the two later questions at fixed shape. Same-voice opposite-role prompts must have matching source spans/lengths and identical surviving tokens.

For every intervention, tokens strictly before the earliest altered query must match baseline exactly; all stages before cutoff must match baseline at period and prompt end exactly. Cut0 closed/frozen must have identical period and full-suffix states and logits across opposite role assignments for fixed family/voice/question (logit tolerance .05, state max 1e-6). Such a disconnected model cannot know which role assignment was presented; balanced accuracy alone is not a null test. Live must leave all period states exactly native. Each first question per voice receives a closed28 sham whose logits/states must match baseline within .05/1e-6. Every irrelevant-cut answer must remain correct for a clean interpretation. Report any violation and stop before evaluation if development controls fail. No tolerance changes.

## Prospective selection and evaluation

Compute gold-margin effects on all eight prompts per family. At k∈{7,14,21}:

**Cached-carrier signature:** frozen8/8 correct, closed≤4/8, irrelevant8/8; mean(frozen margin−closed margin) ≥1 nat and ≥2×mean absolute irrelevant-minus-baseline margin drift.

**Refresh signature:** live8/8 correct, frozen≤4/8, irrelevant8/8; mean(live margin−frozen margin) meets the same 1 nat and twice-drift criteria.

Choose a cached signature first if any qualifies; otherwise choose refresh. Within a signature choose largest mean effect, breaking ties toward shallower cut. Save model-specific signature/cut and raw-data hashes before fresh evaluation baselines. This is development-based selection, not an independent preregistration registry. Neither model's site may change after evaluation. Llama may have a different development cut: a shared sequence does not require identical layer numbers. If no candidate qualifies, stop that model; all development failures remain evidence.

The prospective prediction succeeds only if **each of all three fresh families** satisfies the identical selected signature at the unchanged cut, all 24 baselines pass, and all evaluation controls pass. Also report separate active/passive, both question orientations, every answer, every nonselected cut and neighbor effects. Do not count layers/answers as independent documents. Four constructed families per model do not support a population law or family-wide generalization. No p-value is required for this pilot.

## Rival predictions and limits

An early cached carrier can show frozen rescue over closed. A late live relay can show live rescue without frozen rescue. Early transfer into question states can make even closed succeed. A later direct-read mechanism can fail all carrier modes while irrelevant passes. Executable known schedules verify these distinguishable signatures before model outputs. Generic damage, competing pathways and attention renormalization can cause other or nonmonotone results; those are failures of the proposed simple account, not permission to reinterpret them as passes.

Strong syntax-aware/compound-key copying, compressed surface information and semantic role summaries can all produce cached signatures. This experiment **does not distinguish those representations**. It tests source-isolated versus continuously refreshed use, not whether the summary has abstract semantics. No method-superiority claim is made. Cross-model causal ordering would require independently passing prospective signatures; visual resemblance is insufficient.

## Bounds and stopping

Per model:32 baselines +32×5 modes×4 cuts=640 interventions +3 validations +8 shams =683 attempted forwards maximum. Total1366, hard cap1400. Development-only maximum8+160+3+2=173. No generation continuations. All attempted forwards journal before execution, including exceptions. One attempt per model, no automatic resume/retry, new prompts, new sites, additional models or adaptive follow-up. A technical exception closes that model. Report and stop after this study.

Persist complete original-end full-vocabulary logits and measured residual arrays locally, all prompts/IDs/offsets, per-answer class logits and top20 remotely, controls, failures, source/model/runtime hashes and archive manifests. Large local tensors may be represented by SHA256/size rather than uploaded; disclose that explicitly. Freeze code/protocol/inputs remotely on the existing branch before any model inference. Append only current-study files, preserving all other work.

## Decision labels

NARROW if a pretrained model passes its prospective operational signature; this validates a limited language-processing dependency. NO-GO if none passes or controls fail. A two-model pass supports only this shared operational ordering on the tested fixtures. This pilot cannot earn GO for a novel semantic mechanism or a paper merely by passing.
