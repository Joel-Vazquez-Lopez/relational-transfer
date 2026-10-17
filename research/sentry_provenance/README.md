# SENTRY-002 — Before observing errors, require **source-policy disagreement**

**17 October 2026 · Protocol/method audit · design/fictional-fixture evidence only.**

## Finding

Armaggedon's Qwen7B named-bystander development result is **owner-reported** as 64/64 correct on bare two-event passages, 57/64 correct with irrelevant names; every one of seven errors was on the *second target event* and copied a participant from the first (other) event. The original preregistered `>=8` target-second-error replication threshold **failed**. Raw model generations, tokenization, original protocol and full source annotations are **not independently held by Sentry**. Nothing here claims to verify those seven records.

**Elementary nonidentification (new explicit paper-claim ceiling):** in a two-event passage with target at position two,

```
target = event_2
first event = event_1
previous event = event_1
nearest alternative event = event_1
only other event = event_1
```

Consequently *any observed string that is unambiguously attributable to event one* is consistent with all four restricted source-choice explanations. Seven repetitions, more names or more model seeds **do not separate these explanations** if the design retains this source identity alias. The event-competition story remains an interesting hypothesis, **not a unique account**. This is consistent with [Compass's independent #33 critique](https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/33#issuecomment-6240545756) and [Mira's independently developed counterfactual design](https://github.com/Joel-Vazquez-Lopez/relational-transfer/blob/research/mira-qasrl-independent-audit-2026-10-16/research/mira_qasrl/MIRA_006_IRRELEVANT_NAME_RIVAL_PREDICTIONS.md), but makes the *candidate-answer-source identifiability* gate executable and auditable.

## Novel useful contribution: a prospective **source-policy witness certificate**

The accompanying `provenance_witness.py` accepts **ONLY metadata**: opaque example ID, an ordered list of two or more candidate event IDs with input-only token-position centers and predeclared source-text lexical overlap scores, the target event, and *optionally* an independently verified observed output-source event. **No article text, human original answer strings, gold labels, test text or model output belongs in this file or GitHub.**

It calculates five explicitly restricted prediction policies:

1. `first_event` — first source event regardless of question (can correctly select target when target is first).
2. `previous_event` — most recent event preceding target (abstains when none).
3. `nearest_other` — source event whose token center is closest to target, excluding target; tie => abstain.
4. `lexically_closest_other` — highest source-text lexical overlap with target among other events; tie => abstain. Overlap must be computed without LLM results, annotation gold or hidden test labels. This is a simple *restricted diagnostic*, not Comput's strongest syntax-aware copier.
5. `only_other` — the unique non-target event in a two-event passage (undefined for N>2).

For every rival pair, report the *number of cases where both predict* and the *number/IDs where their predicted source differs*. A single genuine disagreement establishes an eligible diagnostic contrast, **not a neural mechanism**. If no disagreement, `NO WITNESS` is the correct pre-inference decision for that pair. A policy can abstain; undefined is NOT counted as disagreement. Only independently disambiguated `provenance_status="verified"` cases may be used to count matches with observed participant source. Raw model answer strings are **not automatically sufficient to know occurrence/event provenance** if two events share or corefer to the same answer surface.

### Tested structural examples (entirely FICTIONAL metadata)

| Frozen toy case | First | Previous | Nearest-other | Lexical-other | Implication |
|---|---|---|---|---|---|
| Two events E0→E1, ask E1 | E0 | E0 | E0 | E0 | **0/4 rival separation** |
| Three events E0→E1→E2, ask E1, E2 is closer and more lexically similar | E0 | E0 | E2 | E2 | Distinguishes first/previous from near/overlap |
| Four events E0→E1→ET→E3, ask ET, E3 closest and E1 lexically closest | E0 | E1 | E3 | E1 | Three different wrong-source predictions; existence proof |

This demonstrates *design* feasibility under invented metadata, **not availability of rights-cleared natural instances**. Three events suffice to create disagreements for some restricted rivals; four provide distinct answer sources for three policies **in a single trial**. Across separate three-event cases, distinct rivals may also be distinguished. Strong query-aware compound-key syntax/copiers can mimic any of these finite patterns and remain undefeated; the tool cannot prove a latent event-record mechanism.

## Local reproducibility (stdlib only, no GPU)

```bash
cd research/sentry_provenance
python3 -m unittest -v test_provenance_witness
python3 provenance_witness.py /authorized/METADATA_ONLY.json
```

Example fictional input:

```json
[
  {
    "case": "FICTIONAL-POSITION-ONLY-01",
    "target": "E1",
    "events": [
      {"id": "E0", "order": 0, "center": 3, "overlap": 0.1},
      {"id": "E1", "order": 1, "center": 14, "overlap": 0.0},
      {"id": "E2", "order": 2, "center": 16, "overlap": 0.9}
    ]
  }
]
```

**Actually executed local gate:** `13/13` stdlib unittest fixtures passed + both scripts bytecode-compiled. The exact Git blob hashes of the locally tested files are `04b46fb0f18b7e920339029dabd70570b06c5a09` (program) and `8e84b6fa8e6f3287815206540db5b93add8d5464` (tests); **both matching bytes were independently fetched after commit**. No original Qwen raw replay or human source selection has been done.

## What is **not** addressed / experimental stop rules

- Does **NOT** compare strong learned joint syntax/lexical/position/compositional-copying models. Comput owns these baselines; consult ICLR 2026 *Mixing Mechanisms*.
- Does **NOT** validate edited natural-language semantics or token-level lexical matching. [Mira-006](https://github.com/Joel-Vazquez-Lopez/relational-transfer/blob/research/mira-qasrl-independent-audit-2026-10-16/research/mira_qasrl/MIRA_006_IRRELEVANT_NAME_RIVAL_PREDICTIONS.md) plus Mira's existing span/referent validator own candidate integrity and rights.
- Does **NOT** claim the simple nearest policy is plausible for all event contexts; token-center choice and overlap measure must be frozen independently of outcomes.
- Does **NOT** estimate population performance or statistical significance; the seven observed failures are development-exposed, their original threshold failed and source-document independence is unverified.
- Does **NOT** prove any neural representation, causal route, unique semantic binding, independently replicated pretrained effect or ACL novelty.
- For a genuine future confirmation study: require original source-gold event identities and distinct role referents; blind expert rights/semantic review, source-disjoint clusters, target-first/target-last balance, strong policy disagreement against a capable copier, both competent pretrained model families, proper paired outcome accounting and **explicit #31 execution approval**. If no eligible data or no capable-rival disagreement, STOP.
- **Natural paper question, if evidence exists later:** Which source event supplies an error under irrelevant entity load, and can its identity be predicted *out of sample beyond joint lexical/syntactic/positional retrieval*? Do not confuse this predictive error-source task with identifying the model's internal event-selection algorithm.

## Coordination and non-collision

**Sentry** owns the audit of what *source-provenance observations logically distinguish*. **Mira** owns natural counterfactual admissibility/semantic ambiguity, **Comput** owns a strong learned/copier rival, **Compass** owns selective causal identifiability, **Nova** would be the sole approved pretrained runner, **Armaggedon** alone handles experiment gating and paper prioritization. Existing QA-SRL P0 remains separate. This small artifact is a **design-only draft**, does not override existing no-go decisions, and was written during user-authorized analysis without new inference, checkpoint loading, corpus processing, expensive calls or test access.
