# MIRA V5 — Controlled relational competition dataset (16 October 2026)

**Status:** REPRODUCIBLE **OPEN-DEVELOPMENT** synthetic evaluation; **not** a sealed benchmark, authentic-language evaluation, pretrained-model result, or mechanistic discovery.

## Authorized reproducible artifact

The complete 9.3 MB ZIP (Python generator, distinct independent parser, per-family scorer, metadata-safe model inputs, privileged gold, generated graphs, 128-item smoke sample, empirical shortcut audits, scientific report, SHA manifests) is uploaded into the research lead's **ChatGPT persistent Library**:

- **Library path:** `/MIRA_V5_CONTROLLED_COMPETITION_2026-10-16.zip`
- **Exact archive SHA-256:** `01c1b682e6f3810feb00c67023121f7714cb583796e6527c8f8e174cee0580e9`
- **Access:** use the ChatGPT Files Library in your own research conversation. Search the **exact filename**, materialize and verify the SHA; do **not** guess a `sandbox:/mnt/data` path from this readme. Ask the lead if your account/conversation cannot access it. All data are synthetic, unsealed and permissioned for independent dataset review. No model weights or restricted corpora are included.
- This GitHub branch registers the provenance/reading contract only; **generator and full raw data are in the verified Library ZIP**, not committed into the repository. Do not treat the branch alone as the runnable package.

## Design and measured audits

| Motif | Generated questions | Families | Single-constraint ablation |
|---|---:|---:|---:|
| 2-hop double-predicate competition | 1,152 | 12 | 50% |
| 3-hop triple-predicate competition | 4,608 | 12 | 50% |
| 4-hop four-predicate competition | 18,432 | 12 | 50% |
| Conjunctive AND branch with competing target predicates | 2,304 | 12 | 50% when either conjunct or target predicate is ignored |
| **Total** | **26,496** | **48** |  |

- Two open splits: 17,664 development / 8,832 open stress questions, **32/16 disjoint participant-name families**. Nothing is sealed.
- Independently implemented visible-text parser reproduces **26,496/26,496** gold labels. Code is independently implemented within Mira's own workstream, **not externally reviewed**.
- **96,768** single-stage world-bit mutations change gold while preserving the passage word multiset and question; **89,856** question-predicate mutations change gold on fixed passages; **26,496** unqueried-event mutations preserve gold while maintaining word bag/style/voice; **4,608** conjunct checks show each source filter alone admits 2 candidates.
- Two Python processes with different `PYTHONHASHSEED` values produced byte-identical generated data (hashes recorded).
- Independent checker and **five regression tests** pass, including a full 26,496/26,496 oracle scoring test and all 48,384 unique world-bit pair transitions.
- Prespecified text-derived question-time parser still scores **100%** (intended positive reference): passing this synthetic task **does not** demonstrate a native relational algorithm, pre-question caching, or absence of more sophisticated shortcuts.

## Avoid metadata leakage

Use `model_prompts.jsonl` and send **only the `prompt` string** to the evaluated model. Keep `id` for joining outside the prompt. Never feed `gold_private_for_scoring.jsonl`, `privileged_generator_graphs.jsonl`, world bits, motif, split or original answer labels into model context. If a request uses a 128-item smoke sample, label it an **open format/competence preflight**.

## Reproduce from the ZIP

```bash
python3 build_v5.py
python3 independent_validator.py
python3 make_smoke.py
python3 adversarial_shortcuts.py
python3 -m unittest test_release.py -v

# After actual model inference has produced JSONL {"id":"...","response":"raw text"}
python3 score_predictions.py model_predictions.jsonl --out scoring_results.json
```

The scorer checks duplicate/unknown IDs, missing outputs, single-name validity, strict format, correctness, paired counterfactual transitions and descriptive **family-cluster** uncertainty. Do not use row-level p-values.

## Specific independent review requests

- **Comput:** try label-free stage/anchor/position/name/structure shortcuts, regenerate the data, verify the 48 disjoint families and counterfactual labels, and document any contradiction or generator leak. Start with the **negative** interpretation.
- **Nova:** verify ingestion and raw-output scoring in a *small approved preflight*, retain model revision / tokenizer / interface / quantization / seed and decoding. Nova's earlier format-gate failure must be resolved before interpretation of any next run.
- **Brain-O + Compass:** test same-query/different-binding donor mechanisms only after clean task competence, structurally valid intervention controls, ability-matched parsers and reconstruction rivals.
- **Coordinator:** this is **development**, not independent confirmation; avoid adapting to sealed natural language and preserve failed controls.
- **Mira:** human-reviewed authentic semantics remain the next external-validity bottleneck.

## Required outcome

**V5 is ready for adversarial dataset review and reproducible synthetic model development, not for paper-level confirmatory claims.** A separate human-annotated, document-held-out natural-language test and successful mechanism controls remain mandatory for any strong ACL claim.
