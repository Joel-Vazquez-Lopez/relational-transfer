# COMPUT-014 — New Astra raw-response falsification audit

Date: 2026-10-15. Status: **independent retrospective developmental reanalysis**, not new LLM inference or prospective confirmation. Input: `codex/astra-research-2026-10-15:research/astra_snapshot_2026-10-15/completed_adapter_diagnostic/`. All 21 `full_outputs/*.json` files and `FROZEN_ITEMS.json` were inspected read-only, with frozen protocol and origin documented in Astra's `ARTIFACT_SHA256.json`. New code and *derived* summaries are on Comput's branch; source outputs are not copied here.

## Method

The 21 units comprise **2,016 model–sentence comparisons** (three worlds, 96 sentences/model, 3 base checkpoints plus 18 adapted). Each unit records actor direct QA, patient direct QA, joint actor/patient extraction and archived first-token actor. Original sentences and reference role mappings are from `FROZEN_ITEMS.json`.

I independently reparsed **6,048** generated `clean_output` strings with a separate fixed-regex implementation, compared all re-parsed values and validity flags to the archived parser and found **zero mismatches**. Per-world and pooled machine-readable counts, plus each source Git blob SHA, are saved under `research/comput/comput014_evidence/`. The Python replay `comput014_replay.py` checks the archived SHA-256 manifest and independently parses raw outputs; original 21 files were read via the GitHub connection rather than materialized in the Python runtime, so that specific Python entrypoint has not been run locally on the original source directory. A longer offline Python version with a full synthetic end-to-end fixture passed 10 local tests and is in the conversation's reproducibility ZIP.

## Finding A — extraction effectively reduces to one binary orientation at the output level

| Outcome | Count |
|---|---:|
| Extraction attempts | **2,016** |
| Invalid extraction format | **47** |
| Valid extractions | **1,969** |
| Valid extractions containing both names exactly once | **1,968** |
| Both fields correct | **1,362** |
| Both fields reversed | **606** |
| Valid duplicate-field exception | **1** |

**1,968/1,969 = 99.95%** of syntactically valid extractions are permutations of the two participants. Their output behavior is observationally compatible with choosing a single actor/patient orientation and deterministically copying the opposite name into the other field. This does **not** prove that Qwen internally computes only an orientation bit.

Counterexample preserved: `fresh3_830211_unordered_pair`, event `fresh3_0_9_0_praised_active` emits `Actor: Crystal; Patient: Crystal` (valid *syntax*, invalid complementary participant semantics). Do not say 'always complementary'.

## Finding B — direct QA often collapses even when format is valid

| Direct two-question outcome | Count |
|---|---:|
| Same valid name for both questions | **1,349** |
| Different valid single-word names | **434** |
| At least one malformed answer | **233** |
| Both role answers correct | **323** |

Among **1,783** comparisons with two well-formatted direct responses, **1,349 (75.7%)** answer both opposite-role questions with the same name. This means the gap cannot be explained by output-format errors alone.

**Patient-fronted cleft diagnostic (672 comparisons):** 0 invalid direct-answer pairs; 546 same-name direct pairs; 15 jointly correct direct pairs. The direct patient question selects the actor in **571/672** outputs. Joint extraction produces 580 jointly correct, 81 completely role-reversed and 11 invalid. This clearly implicates answer-choice/query conditioning at the behavioral level, without localizing an internal circuit.

## Finding C — source-string-only closed-grammar router is a strong structural alternative

An explicit shallow parser of active, passive and patient-fronted cleft forms, with only the two verbs `praised`/`criticized`, returns the reference actor/patient for **288/288** frozen unique stimuli: 96/96 in each construction. I executed this **directly against the actual archived stimulus strings**, with actor/patient labels used **only to score**, not supplied to the parser. This is a handcrafted template router, *not* a parser that generalizes to natural language. It demonstrates that the current constructed benchmark cannot distinguish this cheap syntax-conditioned rule from abstract event-role computing purely by accuracy.

## Arm-level counts

| Arm | N | Valid complementary extraction | Both extraction fields correct | Direct same-name pairs | Both direct correct | Direct pair invalid |
|---|---:|---:|---:|---:|---:|---:|
| Untouched | 288 | 281 | 177 | 141 | 18 | 118 |
| Ordinary fine-tune | 576 | 576 | 432 | 439 | 91 | 12 |
| Ordered-role auxiliary | 576 | 557 | 379 | 394 | 124 | 18 |
| Participant-only auxiliary | 576 | 554 | 374 | 375 | 90 | 85 |
| Total | **2,016** | **1,968** | **1,362** | **1,349** | **323** | **233** |

No train-arm mechanism is identified: the same underlying worlds, sentences, synthetic templates and different evaluation interfaces are repeatedly used. Do not treat 2,016 as independent natural-language examples. Direct QA takes two generated calls versus extraction's one, without compute matching. The contrast between roles and output format remains observational.

## What the simpler explanation can and cannot claim

- **Supported:** Output-level complementary extraction usually requires just two participant identities plus an orientation decision; it does not empirically demonstrate two independent role-identity decisions.
- **Supported:** A large direct-answer deficit survives when all outputs satisfy the requested one-name format (cleft condition).
- **Not established:** That the frozen Transformer internally uses a position/grammar router, that it lacks an event-role representation, or that an extraction instruction retrieves a known latent structure.
- **Not established:** End-to-end transfer to unseen verbs, discourse structures, human-reviewed natural language or another model family.
- **Competing-mechanism caution:** Compass's recent Investigation 11 demonstrates that powerful probes can learn relational computation from raw inputs; externally decodable roles alone do not prove when a model computed them.

## Next smallest discriminating test

First benchmark text-only frozen position, construction and clause-boundary policies on **human-reviewed**, participant/verb/construction-held-out language where first-mention order, grammatical role and event role vary independently. If a neural adapter or readout improvement survives those controls, use matched full-compute comparisons and targeted causal interventions to distinguish role use from surface routing. Preserve all invalid responses, role reversals and collapsed-output cases.

## Reproduce

`python3 research/comput/comput014_replay.py /path/to/completed_adapter_diagnostic`

Local extended implementation and 10 passing unit tests were packed as `COMPUT_014_COMPLEMENTARITY_FALSIFICATION.zip` and delivered to the user. GitHub's condensed replay implementation has not been executed against the original source in a local Python runtime; the original source was analyzed through connector-side JavaScript with actual records and separate independent raw-string parsing.
