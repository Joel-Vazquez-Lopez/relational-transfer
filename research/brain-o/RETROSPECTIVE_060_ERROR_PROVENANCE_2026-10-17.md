# Brain-O R060 — Correct counterfactual updating and wrong-event error provenance (archive-only)

**17 October 2026 · READ-ONLY RETROSPECTIVE, NOT a new experiment.** Existing frozen E058 results were re-audited in response to the team's single-paper wrong-event-substitution priority and Compass I20 target-acquisition correction. No model inference, retraining, fresh sample generation, downloaded corpus, paid API, hidden test, branch merge, or pretrained-mechanism claim.

## Exact source / reproducibility
- Frozen parent archive: `BRAIN_O_E057_059_NATIVE_BRIDGE_2026-10-16.zip`, SHA256 `9912ea0c0372ea6b4df0d67d4f7a1b77628576d225bfdef82c44baee0b97ab56`.
- Verified archive CRC, original SHA256 manifest **31/31** member hashes.
- Replayed original deterministic E040/E058 NumPy generator **only to recover the 300 existing source worlds/seed**; all gold participant IDs, base/donor event indices and six unique participant names per world matched **3,600 frozen CSV prediction records** (2 seeds × 3 trained bridge arms × 600 role queries).
- Independent small archive `BRAIN_O_R060_ARCHIVED_ERROR_PROVENANCE_2026-10-17.zip`, SHA256 `ad280cfd7301aa2494d5ce5da200d31bf5b8f922251ff35bdc0a6a7ef80dfed9`, contains source audit, six passing adversarial tests, full derived 3,600 synthetic prediction+provenance CSV, JSON/paired world-cluster intervals, detailed scientific report, logs and member checksums. Original weights remain *only* in parent archive, not GitHub. User must explicitly share ZIP to other sessions for independent receipt; do not assume that uploading here is a cross-chat handoff.

## Metrics — avoid previously ambiguous 'new gold correct' label
Define **acquisition** = original predicted answer != new gold AND patched prediction == new gold; **loss** = original predicted answer == new gold AND patched prediction != new gold. Net correct updating = acquisition - loss, by 300 paired two-role worlds per model, not independent question rows.

| Frozen E058 arm / seed | Any changed /600 | Genuine new-target acquisitions /600 | Lost pre-existing correct target /600 | Changed to wrong /600 | Correct *after* patch /600 | Net newly correct |
|---|---:|---:|---:|---:|---:|---:|
| Contextual query-state bridge 1040 | 190 | **63** | 51 | 127 | 180 | **+12** |
| Self-role sham 1040 | 3 | 1 | 0 | 2 | 194 | +1 |
| Raw lexical sham 1040 | 2 | 0 | 1 | 2 | 195 | -1 |
| Contextual query-state bridge 1041 | 254 | **103** | 60 | 151 | 219 | **+43** |
| Self-role sham 1041 | 2 | 0 | 1 | 2 | 208 | -1 |
| Raw lexical sham 1041 | 0 | 0 | 0 | 0 | 191 | 0 |

**Crucial:** the originally cited **180/600 and 219/600** correct donor-target predictions included **117 and 116 outputs that were already correct before patching**. The genuine correct acquisitions are just **63 and 103**. Of the contextual arm's changed outputs, **127/190 and 151/254 remain wrong**. Restricting to base-originally-correct → donor-gold-correct gives **35/600 and 53/600**.

## NEW source-role taxonomy from original frozen world labels
All 1,200 contextual-patched answers were names appearing in their respective six-name synthetic source. Among *incorrect changed* outputs:

| Wrong changed source | Seed 1040 /127 | Seed 1041 /151 |
|---|---:|---:|
| Old target's **requested-role** participant | 50 | 58 |
| Third event's **requested-role** participant | 56 | 79 |
| Any **wrong-role** participant (including new target) | 21 | 14 |
| **Wrong event, correct requested role** (sum above) | **106/127 (83.5%)** | **137/151 (90.7%)** |

Across *all* contextual patched outputs, correct **synthetic role** is **565/600 (94.2%)** and **581/600 (96.8%)** but correct **donor target event+role** only **180/600 (30.0%)** and **219/600 (36.5%)**. A simple **role-conditional copier with an unreliable per-role event selector** can produce this exact qualitative syndrome: source-name outputs, preserved requested role, wrong event, no shared latent event bottleneck. This is **not evidence of pretrained LLM role or event representation**.

### No positive evidence of a common selector across the two role questions
For each of 300 parent worlds, check whether the two patched role answers refer to *the same original source event* (even if wrong). Observed:
- Seed 1040 **84/300 = 28.0%**, vs **100.94/300** expected from independent cross-world permutation of the two observed role-specific source-category marginals.
- Seed 1041 **97/300 = 32.3%**, vs **100.57/300** expected from the same null.
- Exploratory world-permutation 95% null ranges both **85–117/300**. This is post hoc and **cannot establish independent internal circuits** (sentence-specific correlations lost under permutation). However there is **no excess cross-role event coherence** to support a shared event-selector claim. Both roles correct on **24/300** and **38/300** after patch.

### World-clustered descriptive intervals, conditioned on two trained seeds
Resampling 300 parent worlds with both role queries (20,000 frozen deterministic bootstrap draws):
- Contextual **net correct** change: **+2.00 pp [−1.33,+5.50]** seed1040; **+7.17 pp [+3.17,+11.17]** seed1041.
- Contextual minus self **acquisition**: **+10.33 pp [7.83,13.00]** seed1040; **+17.17 pp [14.17,20.33]** seed1041.
- Contextual minus self **net correct**: **+1.83 pp [−1.67,+5.33]**; **+7.33 pp [+3.33,+11.33]**.
Intervals do not capture variation across models, authentic documents or dataset families and are not preregistered tests.

## Scientific decision and relation to current coordinator gate
- **SUPPORTED in synthetic development:** Contextual answer pathway patch has more **genuine target acquisitions** than equal-parameter shams; most changed outputs are still wrong, usually right-role/wrong-event.
- **FALSIFIED overclaim:** Changed answers and donor-gold-after-patch counts do **not** show reliable correct updating or a native shared abstract event selector. Outputs are consistent with **independent role-conditioned source lookup/copying**, in line with prior E039 and coordinator's shared-selector nonidentification proof.
- **NO-GO for mechanism/novelty conclusion:** No competent complete-world event×role answering, no independent replicate, no real-language model, no validation of the team's Qwen7B seven bystander-induced cases. Prior-art collision with entity binding, mixed retrieval and representation/deployment literature remains.
- **Value to NORTH STAR (#33):** provides an exact **error-provenance audit schema** to apply *when authorized* to real Qwen7B saved item-level outputs: genuine target acquisitions/losses, other-event correct-role vs wrong-role vs bystander/out-of-source errors, both-role common-event coherence vs conditional/marginal copying rivals. Do not silently infer any source identity from a mere changed answer; provenance requires gold role/event labels and source evidence.
- **#31 experiment ledger:** user explicitly asked Brain-O to proceed; this is *analysis of existing frozen data* only, not permission for a new model experiment. Future empirical work needs a separate nonduplicate witness, coordinator go/no-go and source/rights review. No collision with Nova authentic QA-SRL, Astra field-order or Codex frontier gates.

**Decision:** R060 archive-only corrected reanalysis complete. **NO EUREKA.** All earlier contradictory/negative counts preserved.
