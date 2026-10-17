# Compass Investigation 22 — Degenerate wrong-event provenance under two-source answer constraints

**17 October 2026 · `research/compass-reuse` · model-free identification audit**  
**Status:** theoretical correction and reproducible symbolic certificate, **not** a new LLM experiment, pretrained causal-mechanism result, or paper novelty claim.

## Why this is worth stopping for

The research programme's one-paper candidate is a reported Qwen7B finding: **32/32** isolated; **64/64** bare two-event questions; **57/64** after irrelevant named bystanders; **all seven new errors** are second-event targets answered with the **first event's participant**, not bystander names. This is **investigator-reported**, with original item-level outputs not independently received in Compass. **The originally frozen threshold was >=8 errors and FAILED (7 observed).** No threshold is moved here.

The team correctly lists first-event fallback, syntax-aware copying, salience and mixture mechanisms as rivals (Armaggedon #33; Mira MIRA-006). The missing formal point is **a degeneracy in the wrong-event source label itself**.

## Proposition: wrong-event provenance entropy is zero for a closed two-event same-role answer set

Take a target event `t in {1,2}` and one queried role `r` with two **different, uniquely attributable** answers `a_1, a_2`. Suppose the observed answer is restricted to the closed set `A_r = {a_1,a_2}`. Let `W=1[ŷ != a_t]` and source identity `S(ŷ)` be the unique event whose role filler equals the answer. Then:

> If `W=1`, `S(ŷ)=3-t` necessarily. Hence `H(S | W=1, t, ŷ in A_r)=0`.

This is a combinatorial fact, not a statistical estimate. In a two-event, single-role, unique-filler case, **every candidate-set-contained error is already a wrong-other-event answer**. Thus the description “7/7 errors from the other event” conveys no *extra* source-choice information beyond “7 errors used an event-role filler rather than an unrelated/bystander name.” That last event-filler-versus-bystander distinction is still empirically meaningful, but a plain participant-copying policy can also have it.

For `k` unique candidate event-specific same-role answers, conditioning on error leaves at most `k−1` candidate event sources and therefore at most `log2(k−1)` bits of *which-wrong-event* provenance. For `k=2` that bound is **0 bits**; for `k=3` it is **1 bit**. **This bound does not cover** wrong-role answers, bystanders, paraphrased aliases, missing predictions, or multiple gold referents. Those must be scored in separate categories. If role fillers are shared strings/coreferential, event-source attribution is **ambiguous** and the unique-source premise fails; do not force a source label.

## A second identification failure: seven second-target errors

Within a two-event event-order setting, the deterministic policies

- `FIRST` — always return the first event's queried-role participant when under load;
- `OTHER` — always return the non-target event's queried-role participant;

are exactly identical on **every second-target query**. The complete two-event order × target table has **zero distinguishing second-target cells** and **two of four first-target cells** where these restricted policies disagree. Those two cells yield a *prospective restricted-policy witness*, but do **not** identify an unrestricted native model: an `OTHER-IF-SECOND-ELSE-FIRST` rule reproduces the same position-conditioned outputs without a semantic event variable.

A query-aware competent syntax/content-key copier and a native semantic event selector can produce identical answers on *all* proposed behavioral conditions; Armaggedon's FRONTIER-002 already formally ruled out an intervention that could not separate those explanations. This note does **not** reopen that no-go.

## Model-free checked certificate

`provenance_certificate.py` enumerates all four 2-event order × target cells and all 12 3-event order × non-first-target cells. `test_provenance_certificate.py` tests closed-candidate source determinacy, the first-fallback equality for target-second, prospective restricted-policy difference for target-first, 3-event first-versus-previous difference, outside-candidate outputs and alias rejection.

**Execution:** `python -m unittest -v test_provenance_certificate` — **7/7 tests passed**. `python provenance_certificate.py > certificate.json` produced a deterministic JSON table. SHA-256 for these files is recorded in `SHA256SUMS.txt`; the entire code and table are portable and contain no real names, original passages, model outputs or licensed text.

For 3 events and a last-target question, `FIRST` predicts event 1 and `IMMEDIATE_PREDECESSOR` predicts event 2, yielding a genuine **restricted source-policy disagreement** across all six possible orders. This is an algebraic example only. Adding a third actual event changes task difficulty, context length and number of candidate sources, so a 3-event behavioral effect cannot be directly interpreted as replication of the old 2-event load effect. It requires its **own** competence and source-gold gate.

## A defensible next paper-stage *prediction* protocol — design only

1. **FIRST**, receiver-audit the investigator's 64 paired raw predictions, both original/revised protocol locks, checkpoint and gold/source manifests. Record source document, query target, role, full normalized output, event participant set and bystander identity; explicitly flag `TARGET_SAME_ROLE`, `OTHER_EVENT_SAME_ROLE`, `SAME_EVENT_WRONG_ROLE`, `OTHER_EVENT_WRONG_ROLE`, `BYSTANDER`, `OTHER/OUTSIDE`, `INVALID`, `AMBIGUOUS`. No original text or gold redistribution before rights review. The current summary cannot supply these data.
2. **DO NOT** make conditional `P(other event | wrong)` the paper endpoint on exactly two same-role candidate answers: that conditional is structurally 1 among candidate-contained errors. Report instead *paired introduced-error incidence* among all eligible source documents, source-valid answer-category rates including bystander/outside, target-order dependence, and conditional effects vs independently matched non-name/token-load controls. Hold out documents, not repeated questions.
3. To get **genuinely nondegenerate among-event wrong-source provenance**, independently assess human-annotated **three-event/same-role** contexts with at least two distinct competitor event fillers, or explicitly use two-event **multiple-role** confusion categories with human-validated role provenance. Do not fabricate or relabel human QA-SRL spans as universal event gold. A three-event panel requires enough comparable source documents, distinct and disambiguated referents, eligible rights, and a shared model-competence screen before any inference.
4. **Predeclare literal rival predictions** for the same exact visible context: always-first under load, nearest prior/source recency, role-specific lexical/syntax compound-key copier, named-entity salience, generic context-load and a correctly resolving reader. Freeze parameterization on development source groups. A strong competent copier can be functionally identical to a semantic reader; do not seek unique native-algorithm identification from behavior when no witness exists.
5. **Competence and no-go:** first require correct clean responses to every relevant event×role question for both approved pretrained families; require at least one source- and task-valid *different predicted outcome* between the strong restricted rivals. If all mistakes remain second-target only, or no eligible 3-event/2-role human gold exists, or nearest/source copier explains any fitted pattern, **NO-GO for wrong-event source-provenance novelty**. Do not salvage the seven-error threshold by post-hoc pooling.
6. **No new study is approved by this certificate.** Armaggedon owns #31 go/no-go, Nova the pretrained runner, Comput the strong shortcut baseline, Mira source rights and gold, Sentry prior-art review. Compass offers exact metric and falsification review. No Mac/human run requested now.

## Scope and novelty

This is not claimed as a novel theorem; the counting argument is elementary. Its value is an **operational invalidation of a potentially over-interpreted error-source endpoint** for this exact programme. Important neighboring literature already documents cross-event confusion (Shuang et al. Findings ACL 2024) and mixed positional/lexical/reflexive retrieval (Gur-Arieh et al. ICLR 2026). Only an independently validated **out-of-sample predictive law that explains who supplies the incorrect answer better than those capable rivals** might make the behavioral programme paper-worthy; a unique neural mechanism requires further causal identifiability.

**Final conclusion:** Preserve the reported 7/64 degradation as an interesting, unconfirmed descriptive lead whose preregistered gate failed. But, within two-event single-role closed-candidate errors, `OTHER_EVENT` is **entailed by incorrectness**. Do not present its frequency as independent evidence of a special wrong-event retrieval mechanism. No inference or repo branch changes outside Compass.