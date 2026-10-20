# Phi-3 cross-family development diagnostic — 15 October 2026

## Outcome

**Construction-dependent role-answering failures occur in Phi-3 as well as Qwen, but the detailed patterns differ. Most notably, the same alternative question wording helps Qwen 7B on object clefts and harms Phi-3.** This is an exploratory crossover, not evidence for a universal internal coordination mechanism.

One selected untouched checkpoint, `mlx-community/Phi-3-mini-4k-instruct-4bit`, revision `5b3819ed6317784fb20eddeae9bed984f778d0d0`. The community and [Microsoft model cards](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) identify the MIT license; the [selected MLX model](https://huggingface.co/mlx-community/Phi-3-mini-4k-instruct-4bit) is converted from that model. Weight bytes and SHA-256 match the pinned Hub object. Installed native MLX Phi-3 support was used with remote-code execution disabled.

## Matched candidate-free results

Each count is events with **both actor and patient correct**, out of the SAME 12 previously examined constructed events. The six participant pairs each contribute both event directions. All conditions use the same semantic questions, names-only instruction, `Answer:` prefill and greedy 12-token maximum; native chat templates and tokenization differ.

### Lexical questions

| Construction | Qwen 0.5B | Qwen 1.5B | Qwen 7B | Phi-3 3.8B |
|---|---:|---:|---:|---:|
| Active | 3/12 | 5/12 | 12/12 | 12/12 |
| Passive | 0/12 | 1/12 | 10/12 | 12/12 |
| Subject cleft | 4/12 | 6/12 | 11/12 | 12/12 |
| Object cleft | 0/12 | 0/12 | 3/12 | 9/12 |

### Alternative / explicit questions

| Construction | Qwen 0.5B | Qwen 1.5B | Qwen 7B | Phi-3 3.8B |
|---|---:|---:|---:|---:|
| Active | 0/12 | 12/12 | 12/12 | 12/12 |
| Passive | 1/12 | 1/12 | 8/12 | 12/12 |
| Subject cleft | 0/12 | 3/12 | 12/12 | 12/12 |
| Object cleft | 0/12 | 0/12 | 8/12 | 1/12 |

### Phi individual roles and contrast directions

| Construction | Wording | Actor correct | Patient correct | Both correct | Reversed contrasts |
|---|---|---:|---:|---:|---:|
| Active | lexical | 12/12 | 12/12 | 12/12 | 0/12 |
| Active | explicit | 12/12 | 12/12 | 12/12 | 0/12 |
| Passive | lexical | 12/12 | 12/12 | 12/12 | 0/12 |
| Passive | explicit | 12/12 | 12/12 | 12/12 | 0/12 |
| Subject cleft | lexical | 12/12 | 12/12 | 12/12 | 0/12 |
| Subject cleft | explicit | 12/12 | 12/12 | 12/12 | 0/12 |
| Object cleft | lexical | 11/12 | 9/12 | 9/12 | 1/12 |
| Object cleft | explicit | 1/12 | 6/12 | 1/12 | 9/12 |

All 192 Phi candidate-free responses are exactly one candidate name, with no length-limit failures. Generic output-format noncompliance cannot account for this panel. Perfect observed control performance is not proof of perfect population performance.

## Paired changes and uncertainty

- Phi object-cleft lexical-to-alternative wording: **0 newly correct and 8 newly incorrect events**, a **−66.7 percentage-point** change; descriptive 95% bootstrap interval **[−91.7, −33.3] pp**. Losses occur in five of six participant-pair groups. Active, passive and subject-cleft joint outcomes are unchanged.
- Qwen 7B on those same events: **7 gains and 2 losses**, a +41.7 pp change. This is a crossover in observed wording effects, not a single universally better prompt.
- Phi object-cleft-minus-active joint gap: lexical **−25 pp [−41.7, −8.3]**; alternative **−91.7 pp [−100, −75]**.

Intervals use 10,000 paired bootstrap samples of SIX name-pair clusters, retaining both directions and all within-event observations. These descriptive intervals are conditional on the reused templates and participant/verb pool. They do not estimate variation over model training seeds, natural-language populations, or unseen constructions. Degenerate intervals at 0%/100% do not establish a zero error rate.

The secondary family difference in wording effects was added AFTER inspection of the completed primary panel. Its exploratory status is recorded in ANALYSIS.json. No prompts, exclusions, checkpoints or primary metrics changed. It must not be presented as prespecified confirmation.

## What kind of failure?

Phi object-cleft reversals increase from **1/12 lexical to 9/12 alternative**. With alternative wording, BOTH role answers are wrong in 6/12 events. This differs from Qwen 7B, whose object-cleft actor answers remain correct under both stronger wordings and whose residual wrong names mostly occur on patient questions. A universal actor-preference explanation is therefore inadequate.

Phi also has **two events per wording** with correctly directed question contrasts but incorrect joint answers, all format-compliant. Thus relative question discrimination and correct final selection can diverge in a second checkpoint family. However, Phi’s large alternative-wording collapse primarily involves actual contrast reversals; it does not simply reproduce Qwen 7B’s residual pattern. A contrast is a behavioral continuation-probability measurement, not a direct observation of internal role representations.

## Candidate controls

On the stronger candidate-list panel, Phi choices flip under order reversal in **4/192 paired comparisons**, all object clefts (4/48). Qwen 7B flips in 15/192, including 11/48 object-cleft comparisons. The older Phi interface has **89/576 contrast-choice flips** under candidate reversal (88/576 generated-answer changes), demonstrating substantial interface dependence. The controls are repeated observations of the same events, not new independent samples.

Candidate provision also matters: Phi object-cleft joint candidate-score correctness is **22/24 event-order cases lexical** and **12/24 alternative**. Candidate-free joint correctness is 9/12 and 1/12 respectively. The two candidate orders do not double the number of independent events. Candidate presence and order are distinct factors; candidate-free failures cannot be explained by candidate-list position alone.

## Simpler explanations

Fixed global first/last-mention, actor-only, patient-only and event-subject rules do not reproduce Phi’s combination of perfect control constructions and wording-dependent object-cleft errors. Nevertheless, noisy or construction/question-dependent shortcuts remain viable. This is behavioral evidence that the effect of wording depends on construction and checkpoint; it does not identify multiple interacting internal mechanisms.

Actor, event grammatical subject and last mention coincide in these object-cleft templates; patient and first mention also coincide. We cannot isolate these cues. Moreover, a fully reversed two-role rule necessarily agrees with opposite-name errors in a two-name task. Its agreement with errors alone is NOT discriminating evidence: it also predicts failures on many actually correct responses. All heuristic fits, including agreements on correct responses, are retained in ANALYSIS.json.

## Original 36-event panel

Below are candidate-score joint correctness and reversal counts for the original AB ordering. All four legacy questions were evaluated, paired as in the archived analysis. BA ordering is an additional within-Phi control.

| Construction | Legacy question pair | Joint candidate-score correctness | Reversals |
|---|---|---:|---:|
| Active | performed / was verb-ed | 33/36 | 0/36 |
| Active | acted on / was acted on | 19/36 | 1/36 |
| Passive | performed / was verb-ed | 14/36 | 0/36 |
| Passive | acted on / was acted on | 2/36 | 0/36 |
| Subject cleft | performed / was verb-ed | 24/36 | 0/36 |
| Subject cleft | acted on / was acted on | 22/36 | 0/36 |
| Object cleft | performed / was verb-ed | 11/36 | 16/36 |
| Object cleft | acted on / was acted on | 13/36 | 10/36 |

Every Phi generation in this run contains one non-EOS name token; its full answer equals its first-token answer in all 1,728 cases. We still preserve the methodological distinction: Qwen legacy/list records were limited to one token, whereas Phi was allowed up to 12. The candidate-free generation comparison has matched limits. Complete role/question/order cells and matched Qwen comparisons are in ANALYSIS.json.

## Scoring, resources and preservation

Phi names encode as one bare-name token or a leading-space token followed by that name token. We score both complete paths, sum their probabilities, and subtract patient log probability from actor log probability. Every step is normalized; no length normalization is applied. Qwen’s frozen score pooled its two single-token variants. Cross-family comparisons use correctness and contrast SIGNS, not raw likelihood magnitudes. Only these predefined bare/space continuation paths are measured.

Cached and uncached smoke logits match exactly; complete path scores match an independent teacher-forced implementation exactly. A missing MLX convenience function was replaced with its mathematically equivalent stable log-softmax expression BEFORE experimental inference; the initial source lock and repair record are preserved. The final inference source and protocol hashes were fixed before predictions.

Completed: **1,728 Phi prompts** (1,152 legacy including both orders; 192 candidate-free; 384 stronger candidate-list). Input prompt tokens: **91,200**; generated non-EOS tokens: **1,728**; auxiliary candidate-scoring prefill tokens: **92,928**. The extra scoring forward pass is measurement overhead, not a longer reasoning trace. Runtime: **982.7 s**, about 16.4 minutes; peak MLX-tracked allocation: **2.48 GB**, not total OS memory. External inference cost: **€0**. No training, paid services, additional checkpoints or GitHub pushes.

All expected unique record keys and completed-output hashes validate. Evaluated Qwen artifacts and their frozen source hashes remain unchanged; all 37 closed diagnostic files remain unchanged. The unreviewed 48-sentence set was not evaluated; sealed data was not accessed. Existing checkpoints were not deleted.

## Research decision for October 23

**This increases confidence in a cross-family construction-and-question-dependent evaluation problem. It decreases confidence in a single universal actor-bias or question-mapping mechanism.** It strengthens the motivation for a focused paper, not an ACL novelty claim: the data are small, reused, constructed, and not independently reviewed natural-language evidence. Different sizes, training procedures, tokenizers and native templates prevent causal attribution to architecture or parameter count.

Next priority: independently validate new sentences, role labels AND the meaning of the question formulations, then freeze a cue-disentanglement panel. Make actor, event subject, mention order and cleft focus yield different predictions, with matched controls. Test the crossover as a prediction on fresh material before mechanism claims or further training. Do not start another broad model sweep from this result.

## Reproduction files

PROTOCOL.json records the fixed checkpoint, data, conditions and scoring. SOURCE_LOCK.json identifies the final inference code; SOURCE_LOCK_BEFORE_SMOKE_API_FIX.json and SMOKE_REPAIRS.json document the pre-inference repair. META.json records native chat template, candidate token paths, stop tokens, model-file hashes, runtime versions and hardware. phi3.jsonl stores prompts, generated IDs, responses and complete candidate-path scores. COMPLETE.json identifies the raw result hash. ANALYSIS.json contains cells, paired transitions, uncertainties and heuristics.

The runner uses the existing sibling stronger_model_development_2026-10-15/.venv runtime and read-only helper functions from that experiment. To reproduce, retain that sibling layout and pinned dependencies; run run.py --smoke, run.py, then analyze_phi.py in a separate reproduction copy to preserve these completed artifacts. Models and transfer fragments are excluded by .gitignore.
