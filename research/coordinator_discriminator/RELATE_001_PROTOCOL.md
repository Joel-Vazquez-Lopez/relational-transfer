# RELATE-001 — Same queried verb, crossed contextual binding (development protocol)

**Status:** Runnable deterministic stimulus/signature calibration; **no trained-model run** and **not a preregistered confirmation**. Brain-O's E021–030 checkpoints and original training generator remain external to this branch.

## Identifiability question

Brain-O's successful three-layer Transformer changed output under a matched donor query-verb-token hidden-state patch. Could the patch carry **only the identity of the queried verb**, an **event-slot address**, or an **already selected donor answer**? Ordinary output correctness, unrelated/zero patches, and different-verb donors cannot fully separate these alternatives.

For each of 24 unique four-name permutations, two sentence voices, two queried predicates, and both requested roles, keep the *literal query verb and role unchanged* while constructing four structurally valid donor contexts:

- **Predicate swap:** same actor/patient entities at each event slot, swap the two predicates. Query stays fixed; donor answer and its event slot both change.
- **Participant swap:** same predicate-to-slot binding, reverse actor/patient in only the queried event. Donor answer changes; its event slot does not.
- **Event-order swap:** move both *whole* events, preserving their relations and answer; the queried event slot changes.
- **Voice flip:** express exactly the same relations with active vs passive voice; neither correct answer nor event slot changes.

Each of the three idealized intervention hypotheses makes a distinct pattern of *answer changes relative to the recipient*:

| Hypothesis | Predicate swap | Participant swap | Event-order swap | Voice flip |
|---|---|---|---|---|
| Lexical queried-verb identity only | No | No | No | No |
| Event-slot address transplant | Yes | No | Yes | No |
| Donor-answer-content transplant | Yes | Yes | No | No |

The 'event-slot address' counterfactual is explicitly an **absolute slot pointer** assumption: other useful representations may encode *predicate-to-event bindings without using absolute slots*. These signatures do not exhaust possible circuit mechanisms.

## Control checks and limits

The generator produces **768 paired question conditions in 24 independent name-family permutations** (192 per transformation). Predicate/participant/order variants match a lexical-plus-punctuation token **multiset** but do **not** promise identical GPT/Qwen tokenizer IDs, local hidden-state distributions, phrase order or attention paths. Voice-flip is a semantic-invariance control, not a token-bag match. The role- and verb-query text is identical, as required.

**This program does not patch a neural network.** The "hypotheses" are transparent software functions predicting idealized patch outcomes from known event slots and gold participants; labels and event slots are **privileged calibration metadata and must not be included in LLM input**.

The current two-event grammar and unique predicate/name assumptions make local textual parsing easy; Comput's positional, question × voice and punctuation heuristics, and Mira V4/Nova-005 v0.2 graph challenges remain mandatory adversarial comparators. Effective independent sample size is **24 families**, not 768 rows.

## Before experimental inference

1. Use *identical frozen checkpoint(s)* and original Brain-O generator; verify model and data SHA256, layer indexing, output head position, donor/receptor activation tensor shape and teacher-forcing schedule.
2. Predeclare patch location (after layer 1 query-verb token), unpatched recipient output, natural donor output, and the idealized hypotheses above; include same-context sham patch, unrelated contextual donor, source-token-only and answer-position patches.
3. Stratify by active/passive voice, event slot, role, verb and family; examine distributions, not just argmax answer flips. Compare recipient/alternate name log-odds.
4. Generate unseen family and variable 3+ event controls and preserve some as **future untouched** held-out stimuli; the 24 permutations here are development-only.
5. Apply strict format parsing, store **all raw predictions**, model hashes and family IDs; solicit blind semantic-review for authentic language transfer. Avoid overinterpreting off-manifold patching.
6. Have Comput and Mira independently attack any apparent discrimination with a competent token-position or construction-aware solution.

## Reproduce

From repository root:

~~~bash
python3 -m unittest discover -s research/coordinator_discriminator -p 'test_*.py' -v
python3 research/coordinator_discriminator/context_binding_challenge.py --out /tmp/relate001_development.jsonl
~~~

The original local development version passed **5/5 tests** on 15 October 2026. The GitHub code/test pair was transcribed from it and needs CI/reviewer execution on the pinned head to establish repository-level test success.

**Success of these synthetic oracle controls is not a discovery of a Transformer circuit.** The first decisive evidence would be a competent pretrained or trained Transformer whose matched contextual patch results discriminate rival hypotheses after independent adversarial verification.
