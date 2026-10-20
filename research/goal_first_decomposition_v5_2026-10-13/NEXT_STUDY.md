# Next decisive study: from clause routing to procedure construction

The corrected v4 and v5 pilots identify a narrow bottleneck: the local operation choice transfers when the relevant clause is isolated, and a supervised selector can learn that isolation in a fixed grammar. A simple hand-written boundary rule obtains the same oracle-view score. This is an **engineering feasibility result**, not evidence that a model invents a procedure. It is separate from the course project's existing actor/patient experiments and must not replace their results.

## Hypothesis worth testing

Training a model to derive a *goal-conditioned set of required operations* improves final-answer transfer when the **combination or dependency order** of familiar operations is new. The claim should be about correct procedure construction and use, not about printing a plausible plan.

## Frozen comparison to design before training

Use one common data bank, base model, training-token budget, and output budget. All arms must see the same answer labels. The arms differ only in additional supervision:

1. **Answer-only:** question and final answer.
2. **Forward trace:** question, event/role facts, intermediate result, final answer.
3. **Goal-derived plan:** question, requested result, required conditions, ordered operations, intermediate result, final answer.
4. **Matched-plan placebo:** the same number of extra target tokens with plausible but irrelevant operation annotations, to check whether length or extra supervision explains a gain.
5. **Oracle decomposition diagnostic:** supply the correct plan at inference to all arms. This is a diagnostic ceiling, not a deployable method.

The next benchmark should contain at least **four primitive operations** spanning role lookup and a different dependency type (for example filtering or comparison). Each primitive appears in training in multiple positions. Withhold several **ordered combinations and dependency graphs**, not just unfamiliar names or verbs. Split names, verbs, and story instances across train and test, and vary grammatical form independently. Keep some familiar operation combinations in unseen wording to tell procedure failure from language failure. A rule-based solver should verify every generated answer and check that simple position/name shortcuts fail.

Report these separately: (a) valid generated operations and dependencies, (b) correct intermediate values, (c) exact final answer, (d) sensitivity to sentence form, and (e) behavior when one generated operation is replaced by a controlled alternative. A printed plan is **not** evidence that the model used it; the intervention must cause a predicted answer change on matched items. Evaluate the same held-out items for all arms, with multiple training starts, and retain every start.

The synthetic stage should be followed by a small human-verified natural-language set with unambiguous event roles and dependencies. If the effect disappears there, report the boundary rather than calling the synthetic gain a general LLM repair. The existing provisional AI labels are unsuitable as final gold until reviewed independently.

## Decision rule

Freeze a primary measure and threshold before the next training run. A persuasive result would require a gain in **final answers**, across multiple held-out operation combinations and starts, beyond forward trace and the matched-plan placebo; intact-question inference should work without an oracle mask. Plan validity and intervention behavior should support the interpretation. If only the oracle condition wins, the bottleneck is still decomposition. If plans are valid but answers fail, the bottleneck is execution or routing.

Backward chaining, question decomposition, and plan-tuning already exist. The potentially new contribution would be a controlled finding about **learned construction and causal use of a procedure under withheld dependency compositions**, followed by a natural-language boundary test. That novelty claim requires a broader literature review before publication.
