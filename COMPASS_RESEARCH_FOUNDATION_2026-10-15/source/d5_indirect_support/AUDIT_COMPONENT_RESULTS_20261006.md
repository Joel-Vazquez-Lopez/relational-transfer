# Role-transfer audit and component controls

Date: 2026-10-06. This is an exploratory follow-up to
`RESULTS_20261006.md`. The audit used all 42 existing checkpoints. The
component retraining deliberately selected world `new2`, two seeds, and the
two contrasting graph conditions after their original outcomes were known.
It is a diagnostic study, not a fresh confirmation or a population estimate.

## What the audit established

The new evaluation crossed role direction (2), event (2), answer position
(4), distractor identity (8), and repeat (8): 1,024 four-scene questions per
model and evaluation mode. One mode retained the original scene generator;
the controlled mode used the same distractor in all four candidates. The
same evaluation questions were used for every checkpoint. Four-choice chance
is 25%; the restricted correct-versus-reversed-role choice has 50% chance.

All five models whose original held-pair accuracy was below 80% remained
weak under the controlled evaluation:

| World / condition / seed | Original held accuracy | Controlled four-choice accuracy | Controlled event accuracy | Controlled role-only choice |
| --- | ---: | ---: | ---: | ---: |
| new0 / shared2 / 10401 | 49.1% | 49.9% | 100.0% | 49.9% |
| new0 / shared2 / 10402 | 52.3% | 53.6% | 98.1% | 54.4% |
| new2 / shared2 / 10402 | 49.8% | 46.2% | 93.7% | 49.9% |
| new2 / shared2 / 10403 | 48.0% | 49.1% | 99.2% | 49.3% |
| new3 / shared1 / 10403 | 53.2% | 53.9% | 100.0% | 53.9% |

The errors mostly exchange actor and patient while preserving the event.
The 50% regime is therefore a role-direction failure, **not** four-choice
chance. Crossing answer positions and controlling distractors did not remove
it. This makes the phenomenon less likely to be a test-layout artifact,
although it does not identify its learning cause.

The original training set had 1,280 balanced configurations but only 80
distinct ordered-pair/event meanings. The fourth coded-message token reused
the embedding for entity 0; the component control below checks whether this
alias caused the selected failures. The audit's query and candidate-vector
alignment statistics are observational, so they cannot locate a causal
mechanism by themselves. A separate linear-probe attempt did not decode roles
reliably even on its probe-training examples and is inconclusive.

## Matched retraining controls

All variants used the same world, conditions, seeds, training episodes, 6,000
updates, four-scene objective, and 1,024 controlled held-pair questions.
`original` replays the joint learned query and scene encoders in the exact
diagnostic runner. `padding` gives the fourth message token a dedicated
embedding; all original parameters start identically. `fixed query` uses
fixed source/target/event slots for the query and learns the original scene
encoder. `fixed candidate` learns the original query encoder and encodes each
candidate scene in fixed source/target/event slots. No variant receives an
answer indicator. Both fixed-representation variants change the problem's
representation and optimization, so they are localization controls rather
than a clean causal repair.

| Seed | Condition | Original held | Dedicated padding held | Fixed query held | Fixed candidate held |
| --- | --- | ---: | ---: | ---: | ---: |
| 10402 | shared1 | 99.9% | 99.9% | 98.6% | 100.0% |
| 10402 | shared2 | 45.6% | 43.8% | 96.9% | 100.0% |
| 10403 | shared1 | 99.9% | 99.8% | 99.5% | 100.0% |
| 10403 | shared2 | 49.1% | 45.3% | 94.9% | 100.0% |

Every retrained model scored at least 99.92% on familiar examples.
The original replay reproduces the easy/weak contrast. The dedicated
padding token does not rescue either selected weak run. In contrast, learning
either side against a fixed, role-explicit counterpart gives at least 94.9%
held accuracy here. This focuses the next question on how the two learned
representations acquire a compatible actor/patient geometry. It does **not**
show that the candidate encoder alone is defective, that a particular graph
property causes the failure, or that a single internal variable controls it.

## What this means for the project

The strongest current result is an evaluation finding: near-perfect familiar
performance can mask a robust role-direction failure on an unseen pair.
The originally proposed one-versus-two-common-neighbor explanation did not
generalize across graph layouts. The early internal predictor failed its
independent confirmation, and increasing focal identity-embedding distance
did not repair the bad models (see `RESULTS_20261006.md`). The new component
study supplies a more precise **working hypothesis**, not a confirmed
mechanism: the failure emerges when both sides of the role-matching problem
must be learned jointly from indirect evidence.

The next decisive study should specify a small intervention on the *joint*
model before looking at new outcomes, leave its evaluation and base task
unchanged, and compare it with an equal-compute control on fresh graph
layouts and seeds. A candidate is an explicit role-swap consistency penalty:
pair each training item with its actor/patient-reversed counterpart, then
require the whole four-candidate score pattern to transform with that swap.
The training set already contains both role directions, so merely adding
reversed examples would not test this idea. An equal-compute control should
use the same paired items and updates but omit the consistency penalty.
Predefine a held-role improvement threshold,
familiar-performance floor, and a failure criterion across independent
layouts. This would test whether explicit role alignment improves transfer;
it would not by itself prove why the original graph effect arose.

There is already a separate, exploratory 36-event template pilot with
Qwen2.5-1.5B-Instruct. With the generic actor question it answered both
active and passive items correctly for all 36 events. With a verb-specific
question, it answered 24/36 active and 36/36 passive items correctly; all 12
active errors involved the verb *followed*. These question wordings were
explored on the same small item set, so the contrast is a prompt-sensitivity
lead, not a confirmed effect or evidence about authentic sentences. The
underlying files are `../llm_role_generation_qwen15_generic_result.json`
and `../llm_role_generation_qwen15_verb_specific_result.json`. A proper
linguistic study should use fresh templates and include an authentic-sentence
check. Its outcomes must remain separate from the small-Transformer results.

The controlled-task results are sufficient to justify a focused 15-credit
investigation, while a publishable mechanism or broad claim still needs
fresh confirmation and a thorough related-work review.

## Reproduction files

- `AUDIT_PROTOCOL_20261006.md`, `audit_20261006.py`, and
  `results/audit_20261006.json`: crossed evaluation of 42 checkpoints.
- `COMPONENT_PROTOCOL_20261006.md`, `COMPONENT_ADDENDUM_20261006.md`, and
  `COMPONENT_REPLAY_20261006.md`: diagnostic decisions and timing.
- `component_diagnostic_20261006.py` and
  `results/component_20261006/<variant>/<condition>/seed_<seed>/metrics.json`:
  16 complete runs with checkpoint hashes and full metric records.
- `PROBE_PROTOCOL_20261006.md`, `probe_20261006.py`, and
  `results/probe_20261006.json`: inconclusive linear-probe attempt.
