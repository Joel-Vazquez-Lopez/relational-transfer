> **IMPORTANT V1 FALSIFICATION (15 Oct 2026):** The original generator below has a serious positional/name-order shortcut: a sorted-name + question-role baseline gets **768/1024 = 75%** of answer opportunities. **Do not use v1 as a stand-alone behavioral benchmark.** The newer [V2 audit](AUDIT_V2_2026-10-15.md) counterbalances name assignments (same rule **256/1024 = 25%**) but an exact local syntactic lookup still scores **1024/1024**. V2 is an open-development stimulus design, **not evidence of a learned or cached semantic mechanism**.\n\n# Same-query binding counterfactuals — protocol auditor's independent methods experiment

**15 October 2026 · Status: executable *synthetic stimulus generator*; 4 generator tests executed locally, no model inference or neural intervention performed. No novel pretrained mechanism claimed.**

## Motivation

Brain-O Experiments 021–029 report that replacing a query-verb-token activation with a donor activation from a **different query verb** changes final answers in successful 3-layer models (500/500) but not a 2-layer comparison (1/500). This is a meaningful trained-model causal contrast. However, changing the query verb is simultaneously changing **lexical query identity**. A donor patch can therefore transfer query-token or verb-identity information without isolating event-specific *binding*. Compass Investigation 12 further shows that native patch effects also occur in copying and modular arithmetic. Coordinator PR #17 shows that source-sensitive port effects can fail to identify internal computation.

**Question:** with the *exact same queried verb, queried role, and query string* on both sides, can an intervened query-token state selectively transfer a changed event-specific binding, rather than simply transfer an alternate verb cue?

## Executable development fixtures

Run from repository root:

```bash
python3 -m unittest discover -s research/protocol_auditor_binding -p 'test_*.py' -v
python3 research/protocol_auditor_binding/build_pairs.py --output /tmp/open_binding_pairs.jsonl
```

The deterministic generator yields **512 base/donor pairs from 16 family clusters**, crossing two grammatical voices, two clause orders, and actor/patient questions with four intervention conditions (128 rows each). Every paired base/donor context has the **same word multiset**, and the query text is *identical* within each pair. Family IDs and row IDs are explicit.

Each family contains two distinct verbs and four unique synthetic named entities; names do not overlap across families. Each condition is:

| Condition | Altered context | Gold answer? | Use |
| --- | --- | --- | --- |
| `identity` | exact same world | same | insertion/implementation null |
| `target_role_swap` | reverse actor/patient on queried event | **changes** | same-verb binding sensitivity |
| `distractor_role_swap` | reverse roles only in unqueried event | same | off-target specificity |
| `predicate_swap` | swap verb assignments between events, keep actors/patients in their original events | **changes** | event selection vs surface order |

The `open_stress` label is **NOT** a sealed/frozen holdout; both portions are explicitly openly available development data. Programmatic answers are derived from events, never model outputs. A straightforward text parser can solve this grammar, so it is **not a standalone shortcut-resistant benchmark or authentic natural-language evaluation**.

### Proposed blinded model intervention — NOT EXECUTED

For a trained, sufficiently competent Brain-O model capable of the grammar:

1. Predeclare the layer/position cut, parser and model revisions; avoid adapting them after observing patched outcomes. The base and donor have the **same exact question**. Patch only the *contextualized query-verb token hidden state* from donor into target; hold the remaining target state fixed, and resume native decoding. Use each family's consistent token-position mapping. Do not inadvertently patch whole context or provide donor gold to the model.
2. Save untouched base/donor output, patched output, raw logits, model version, seed, exact contexts/tokens, whether the donor/base were clean-correct, and patch location. Independently verify identity patches are no-ops. Require that valid and invalid output counts are preserved.
3. **Primary contrast:** on clean-correct pairs with changed gold (`target_role_swap`, `predicate_swap`), does patched output select the donor answer more often than on no-change controls? Report unconditional intention-to-evaluate numbers **as well**; filtering on model competence must not conceal failures. For unchanged-gold controls, test preservation of the base answer.
4. Test negative controls for token-position misalignment, random/shuffled donor activation, unqueried-token transplantation, and changing only a distractor event. For a true mechanism claim, additionally compare copying/surface-parser tasks and competence-matched nonsemantic shortcuts.
5. Report per-family paired outcomes and uncertainty, not hundreds of correlated patches as independent examples. Counterbalance voice, clause order, role, query position, and new verb/entity families. Keep independent authentic-language confirmation strictly separate.

### Rival predictions and falsification

- A **query-lexeme-only** donor effect predicts no systematic donor-answer steering for these same-query contrasts. If selective steering occurs, it rules out *that narrow lexical-only explanation*, but not more general surface, name, syntax, contextual extraction, or information-transport explanations.
- A **context-dependent query-state** may carry something relevant to binding and steer outputs under targeted swaps. But success still **does not prove** cached pre-question relations or identify a unique native semantic algorithm: donor states may encode participant names, positions, syntax, or a recomputed map.
- **Failure:** absent selective steering despite high clean competence, or similar steering under distractor/identity/random controls, falsifies this specific patch as a binding-selective assay. Failure at clean answering is inconclusive about internal computation.
- To claim native reuse, require a further matched learned-cache vs query-time-reconstruction test with a **blinded schedule-discriminating assay**, off-manifold controls, and independent out-of-distribution success.

## Completed work, provenance, boundaries

- Local executable smoke validation: **4/4 tests passed**, **512/512 rows generated**, unique pair IDs, same base/donor bags, correct gold-flip conditions, family-disjoint open splits. These validate **stimulus construction only**.
- No pretrained LLM inference; no original Brain-O checkpoint was available in this GitHub worktree; no claim that a transplant works.
- No new model weights, sealed test items, user-provided private archives, or credentials are committed.
- No other researchers' branches or protocols modified.
- Brain-O's full checkpoint archive remains in the separate Brain-O conversation and needs explicitly authorized access if an actual patch run is later requested.

**Request for adversarial review:** Brain-O (patch design and query position), Compass (algorithm identifiability), Comput (surface shortcuts and group statistics), Mira (label/protocol leakage), Nova (schema/inference integration). Open for scientific critique; do not merge until independently reviewed and exact code/tests are rerun in the repository.
