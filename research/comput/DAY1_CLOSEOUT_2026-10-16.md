# COMPUT — Day 1 scientific closeout and verified handoffs

**Closeout:** 16 October 2026, after the project's first research day (15–16 Oct). **Branch:** `research/comput-falsification`. No new pretrained inference, paid compute, checkpoint unpickling, neural training, sealed evaluation, or PR merge occurred in this closeout.

## 1. Confirmed Comput contributions

- **R011**: original Qwen2.5-1.5B archival `llm_role_internal_v2/RESULT.json` (source SHA256 `653a33a0a56f34a84c5c39649dda7bdb100990462daea0390cfcb2fcb8d7ea27`): 384 dependent archived responses, 240 correct, simple question×voice positional rule matches 368/384. This is archived-output behavioral reconstruction, not native circuit identification.
- **R012/R013**: historical predicate-sensitive margin variation; leave-verb-out failure of rich verb-specific fitting. Reference actor/patient-indexed margin is diagnostic, NOT a fully label-free raw-text answering algorithm.
- **R014**: independent raw-string reparse of Astra's 21 archived units; 2,016 dependent sentence-model comparisons / 6,048 responses; 0 archived parser disagreements. 1,968/1,969 valid joint extractions return the two distinct names exactly once; direct QA often repeats one name. A formatting-only explanation is insufficient, and joint extraction does not establish precomputed semantic bindings.
- **R015**: source-feature/generator-aware V2 benchmark rule scores 768/1024 (all 512 base, 256 donor) without parsing the actor/patient relation; **0/256** on gold-changing donor cases. The scientific coordinator independently ran the actual, byte-identical upstream generator and Comput predictor (provenance in [issue #7](https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/7#issuecomment-6203465177)); result independently reproduced against authoritative source. PR #20 remains a development protocol, not a shortcut-resistant semantic evaluation. The upstream generator additionally varies the base world with requested question role; it must not support same-world cross-role claims.

## 2. Current team evidence reviewed

- **Brain-O E037/E038 corrected** a passive participant-token indexing error. The crossed 2×2 synthetic activation intervention yields 200/200 and 197/200 donor restorations (reported); E039–E042 shows a trained factorized copier also reproduces the E038 output signature, so **semantic algorithm uniqueness is explicitly falsified** although the original 3-layer task-effective neural pathway is not. E041 unconstrained 3L run is incomplete; do not cite it as failure.
- **Compass I13/I14**: native whole-world outcomes remain weak; externally imposed structural constraints largely account for a sizable apparent coherence improvement. **I15** additionally reports weak multi-step nonbijective OOD composition and a strong omission heuristic; details remain in source and were not independently replayed by Comput.
- **Brain-X BX-002**: paired 4,000-world neural edge-information sensitivity without beneficial accuracy over local 65.3% or simple 3-vote 69.875%; reported source and tests are committed, independent Comput replay still pending.
- **Mira V4/V5**: generated development material, not authentic language or sealed confirmation. V4 has 2,592 rows in 60 families with shortcut and metadata-ID issues. V5 has 26,496 rows in 48 name-disjoint families and a text-only parser at 100%; no pretrained model results. Treat as adversarial dataset stimulus only.
- **Nova CAL-01**: original 1.5B/7B Qwen outputs are now available privately for independent audit; original reported 7B 0/8 two-hop on one synthetic world is not generalization. A helper-copy/scope rival remains untested by Comput.
- **Research goal**: authentic human-authored, licensed, independently doubly adjudicated semantic roles; **at least two distinct untouched pretrained model families**, not Qwen 1.5B versus Qwen 7B counted as two. Both streams parallel, per [issue #23](https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/23). No native pretrained relational mechanism / ACL-level novelty currently validated.

## 3. Day 1 recipient-verified source access (NEW)

Files were listed from the research lead's authorized private ChatGPT File Library, materialized in an isolated session directory, SHA-256 hashed from **actual bytes**, and verified by ZIP CRC:

| Handoff | Exact filename | Recipient SHA-256 | Expected match | ZIP CRC | Content |
|---|---|---|---|---|---|
| H1 Mira V4 | `MIRA_V3_FALSIFICATION_AND_MULTISTRUCTURE_V4_2026-10-15.zip` | `50451ea957fb5e81a03ea7b637e34fd976906e9968186683f5118cd4c66b58d7` | YES | PASS | 19 ZIP members, unsealed generator, 2592-row data and scorer |
| H3 Nova CAL-01 | `NOVA_CAL01_H3_REVIEW_HANDOFF_2026-10-16.zip` | `d241a1f18d55b6539fbfd69454658d40953fb107f3a149e9fa90ccfbc20c7852` | YES | PASS | 11 members, 48 raw Qwen responses, isolated model-only inputs, scorer-only gold, verifier |

**Accurate state:** `RECEIVED + SHA256 VERIFIED + ZIP CRC PASS`. **Not yet:** verified all internal manifest member hashes, independent per-row replay of V4/Nova, executed verifiers, model inference, scientific conclusions from their raw content, or redistribution authorization beyond private reviewed research. Original files remain unmodified; never send `id`, `split`, `gold`, or `generator_edges` to an evaluated model.

## 4. Day 2 explicit handoff / priority

1. **H1:** Inspect V4 manifest and run source-locked integrity/shortcut audit, including predicate-competition and branched-edge ablation, family-level statistics and leakage. Compare against Mira's reports without copying privileged source labels into predictors.
2. **H3:** Execute the H3 cold verifier offline; independently rescore all 48 outputs for strict compliance, helper-copy, two-hop/gold confusion and repeated-world dependence.
3. **H6:** Run committed Brain-X BX-002 deterministic tests and inspect available numerical summaries; full training regeneration requires separate explicit scope/compute review.
4. **Branch maintenance:** execute full Comput test suite, preserve results and reconcile PR #13 current scope/review gates. The previously stale README has been corrected with prominent R007 oracle-leakage disclosure.
5. **External validity:** freeze label-safe shortcut and paired metrics for Mira's genuinely double-annotated, licensed, human-authored document-family-disjoint pilot. Prepare an untouched two-family LLM behavioral test; do not run until validated and approved.

**Day 1 completed; neither EUREKA nor confirmatory acceptance is claimed.** No new inference or urgent human-run request.
