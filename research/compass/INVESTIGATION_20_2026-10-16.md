# Compass Investigation 20 — Counterfactual target acquisition versus false response changes

**16 October 2026 | `research/compass-reuse` | Read-only PyTorch checkpoint audit; synthetic, post hoc, development only.**

## Question
Does the instruction-versus-operand response asymmetry from Investigations 17–19 reflect *correct relational updating*? Or are apparently correct counterfactual outputs contaminated by models already guessing the future gold label before any intervention?

The true algorithmic question—native Transformer relational reuse vs question-time reconstruction—**is not identified** by this audit. Models have poor three-hop whole-world generalization. Here we calibrate a proposed behavioral proxy **before** extending it to pretrained models.

## Completely preserved data and method
Reload the same three **unmodified** Investigation 18 mixed-layout causal Transformers (seeds 19, 29, 41). Reconstruct exactly 188 equal-new-gold pairs per seed/layout: 95 unique A-bijective source worlds, two source/instruction layouts. For each base example, replace final B instruction by A, or edit the actually consulted B relation-table cell to induce the **same** new correct answer. New gold differs from base gold by selection. Separately, for 91 pairs from 66 worlds, perform the **identical old→new token substitution** at an off-path B table cell, which must leave gold unchanged (but occurs at a *different absolute table position*).

Full record count: **1,128** main original/instruction/operand triples; **546** off-path/relevant/base probability triples. Every original argmax was matched exactly to the saved Investigation 19 raw output before computing new probabilities. All logits and targets are sourced exclusively from the **frozen checkpoint forward pass**, not a fitted auxiliary classifier or a new training intervention.

Define **genuine target acquisition** for an edit as `(base_prediction != new_gold) AND (edited_prediction == new_gold)`. This removes the fortuitous case where the unedited model already predicted the target. A stricter baseline-correct variant also requires `base_prediction == base_gold`. For change *precision* report the fraction of all changed answers that land on the new gold. Report both behavioral change and correctness—neither alone is a native-mechanism identification criterion.

## Confound identified in Investigation 19 headline
Across the **1,128** dependent model×layout×stimulus records, **240 base predictions** already equaled the counterfactual new gold (and therefore were wrong under the original gold). Thus even if the edited prediction is equal to new gold, it may have made **no correct semantic update**.

| Diagnostic | Instruction edit | Operand edit |
|:--|--:|--:|
| Changed prediction | 690/1,128 | 302/1,128 |
| Correct edited prediction (previous metric) | 270/1,128 | 457/1,128 |
| **Actual change to correct target** | **189/1,128** | **231/1,128** |
| Changed to wrong target | **501/1,128** | **71/1,128** |
| Fortuitous correct target with unchanged output | 81/1,128 | 226/1,128 |
| Among 445 originally correct answers, new target correct | 152/445 | 201/445 |

The raw success difference of −187 cases **shrinks to −42 cases** when requiring an actual target acquisition. Conditional source-world bootstrap for instruction-minus-operand genuine acquisition: **−3.72 percentage points, [−6.49, −0.80] pp**. The *incorrect response-change* gap is **+38.12 pp, [33.87, 42.24] pp**. A bootstrap resamples the 95 source worlds with all correlated seed/layout observations kept together; it is **conditional/exploratory**, not evidence of external population-level significance.

Raw changed-answer precision is **189/690 = 27.4%** for instruction edits versus **231/302 = 76.5%** for operand edits, but comparing those conditional subsets does not causal-identify an internal mechanism. Some models are initially wrong; target acquisition alone is not guaranteed original-to-counterfactual correctness.

## Full-probability follow-up (new frozen-checkpoint forward pass)
For each input, compute the softmax probability and one-vs-rest log-odds of the **same new correct answer** under base, instruction-edited and operand-edited runs. This avoids top-1 threshold artifacts:

| Metric, pooled descriptive | Instruction edit | Operand edit |
|:--|--:|--:|
| Mean increase in probability of new gold vs base | **+2.24 pp** | **+19.15 pp** |
| Fraction of cases increasing `P(new gold)` | 53.7% | 79.5% |

Instruction-minus-operand mean `P(new gold)` difference **−16.91 pp**, conditional world-cluster bootstrap **[−21.19, −12.45] pp**. For instruction-minus-base `P(new gold)` improvement, conditional interval **[−2.46, +7.29] pp** includes zero; for operand-minus-base it is **[+15.00, +23.64] pp**. This resolves an argmax ambiguity but **does not identify why** each input is weighted.

Mean instruction-minus-operand log-odds of the new target: **−2.125 nats** (conditional bootstrap **[−2.718, −1.527]**). The effect is negative in every seed/layout condition, although strengths differ.

## Off-path token-substitution null on a subset
Within the 91 eligible pairs, replace an irrelevant B source-cell value with exactly the **same old→new symbol change** as the relevant B-cell edit. The oracle answer remains original gold for the irrelevant edit.

| Target-probability metric | Mean difference from base |
|:--|--:|
| Gold-relevant B-cell edit, `P(new gold)` | **+23.78 pp** |
| Gold-irrelevant B-cell edit, `P(new gold)` | **+13.75 pp** |
| Difference relevant − irrelevant | **+10.04 pp**, conditional interval **[+1.03, +18.61] pp** |

The irrelevant edit also reduces `P(original gold)` by **12.66 pp** on average, despite not changing the correct answer. Its argmax changes in 104/546 cases. **Important:** source-cell **position is not matched**, so this is an off-path token-substitution null, not a pure attention/position matched causal control. The eligible subset was selected post hoc; its P(new gold) shifts are not a population property.

## Scientific decision / limitations
- The strongest supported result is **large non-target response reactivity to final instruction edits**, combined with weaker but more target-aligned operand responses. This is observed in poorly compositional models and consistent with surface/shortcut policies.
- A large part of Investigation 19's apparently large operand correctness advantage came from **unchanged fortuitously correct outputs**; always report *actual target acquisition*, *off-target flips*, *base-correct subset*, and *target probability*, not just new-gold accuracy or any-output-change.
- Models saw only one/two-step programs in training; complete held-out three-step world competence remains poor. This cannot establish natural binding, any executed cache, causal circuit, or new ACL-level theory.
- 95 synthetic source worlds × 3 seeds × 2 layouts produce **non-independent comparisons**. Intervals are exploratory, selection-dependent, and not independent confirmation.
- No authentic-language source-rights or human annotation gate was crossed. Mira's EWT/Wikipedia work and Sentry's original QA-SRL paired-question eligibility review remain independent; neither is silently relabelled as actor/patient evidence.
- No paid inference, checkpoint updates, sealed data, other investigators' branch changes or main merge.

**Next valuable gate:** preregister a correctness-based metamorphic intervention score with calibrated *target-acquisition and off-path response precision*, then evaluate on a strong **whole-world competent** model and human-validated authentic multi-event data. Don't continue scaling this weak synthetic task solely to reproduce sensitivity curves.

## Reproducibility
The standalone full-replay archive `COMPASS_INVESTIGATION_20_COUNTERFACTUAL_PRECISION_2026-10-16.zip` includes both new probability replays, all original frozen source/weights, original response rows, strict audit/tests, source-world cluster intervals, raw per-row probability records and a SHA-256 manifest. All results were generated offline with CPU PyTorch. GitHub stores only the scientific report and lightweight checks, not binary weights or copies of source corpora.

**ZIP verified SHA-256:** `60797f315860e70b45334dbe3ded9384f3837ef9315fc8b1ae1410de556b723a`; 26 ZIP entries, 25 file SHA-256 hashes verified. Archive is in the Compass conversation, not in GitHub.
