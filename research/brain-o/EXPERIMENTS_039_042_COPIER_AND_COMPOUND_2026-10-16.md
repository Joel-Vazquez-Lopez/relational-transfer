# Brain-O E039–E042 — Copier counterexample and compound-key generalization controls
**16 October 2026 · Branch `research/brain-o-ideation` · Status: executed synthetic development, no pretrained inference.**

## Scientific result: E038 is not mechanism-identifying
Brain-O E037–038 showed approximately composable event-address and participant-name causal effects in two learned unrestricted 3-layer synthetic Transformers. On **400 new disjoint 2-event worlds**, our *existing trained E016 factorized copier* (seeds 1016/1017) produced **400/400 correct** native base/A/B/AB answers and **400/400 correct** A-only, B-only and composed A+B interventions **for each seed**. A exchanges predicate-to-event assignments; B swaps within-event actor/patient names; all four targets have different gold labels and all queries remain identical. Explicit learned verb-key matching + learned voice/field pointer suffices for this output-level causal-factorization signature.

**Very important:** E016's source/field gates are *architecturally exposed*, unlike the unrestricted 3L model; these E039 interventions are not the same neural activation patches and the systems are not parameter-/compute-matched. Therefore E039 **does not falsify the learned token-to-token route** localized in E031–038. It **does** refute any claim that the crossed factorial by itself is unique evidence for a novel semantic relational algorithm. A generic key→record→payload copier remains a live, competent explanation.

## New development task: E040
Built a three-event grammar with colliding keys: `(verb1,tag1), (verb1,tag2), (verb2,tag1)`, random presentation order, independently randomized active/passive layout and six distinct participants, query targets one of two output fields for a `(verb,tag)` conjunction. Names: 24; verbs: 8; context tags: 6. Simple single-key lookup cannot uniquely locate every event.

**Audit:** 500 generated worlds × 6 role/query combinations = **3,000 QA**. First-record shortcut **1,000/3,000 (33.3%)**; verb-only first-match **2,000/3,000 (66.7%)**; tag-only **2,000/3,000 (66.7%)**; exact compound-key grammar-aware lookup **3,000/3,000**. Generator produced **1,000 within-world gold-changing edits** preserving exact question and complete token multiset. These examples are synthetic and still solvable by a competent record copier.

An explicitly structured, *learned* 17,188-parameter compound-key copier (seeds 1040/1041, 250 updates) achieved **1,200/1,200 answers and 200/200 full six-question worlds per seed** on independent generated test worlds. It is not an unrestricted Transformer or linguistic semantic generalizer.

## E042 — One compound-key pairing completely withheld from training
Disallow one particular verb×tag token combination in *every training record*, while both individual component tokens occur elsewhere. Evaluate 200 fresh worlds per seed containing the withheld pairing (6 questions/world); 400 questions/seed query the withheld pairing. **Structured copier seeds 1042/1043: 400/400 withheld-pair queries, 1,200/1,200 total and 200/200 complete worlds each** after 250 updates. This establishes limited held-out *key combination* generalization for the architecturally biased copier, **not unseen vocabulary, natural language, or compositionality in unrestricted Transformers**.

## E041 incomplete (do not cite as model failure)
A larger unrestricted three-layer causal Transformer with local three-way event auxiliary was started at a planned 500-update setting. Execution stopped during training (~200 steps observed) without a completed evaluation/checkpoint. No valid full-budget result is available. Do NOT compare its partial training-batch values to E040/E042.

## Cross-team context
- Compass I12: similar causal handoffs appear in copying and modular arithmetic; I13/14: question-level accuracy and externally imposed whole-world coherence do not prove systematic relational computation.
- Brain-X BX002: intervention sensitivity need not provide better accuracy than a simpler local baseline; an orthogonal version of the *information-used versus information-helpful* problem.
- Comput R014/15, Mira and Auditor: lexical/format/position/generator-index shortcuts remain serious. Avoid counting questions from one world as independent evidence.
- Coordinator issue #23: a publishable bridge **requires** independently adjudicated human-authored authentic language and at least two untouched pretrained model families, in parallel with synthetic calibrations.

## A preregistrable synthetic→language falsification
On authentic, human-adjudicated event groups, compare:
1. single-predicate cues vs *repeated predicate with two valid event referents*, matched in length, roles, syntax and name order;
2. repeated predicate with a *second independently informative event cue* vs an irrelevant cue (matched length/token budget);
3. direct verb- and context-specific agent/patient requests vs joint extraction in both field orders;
4. full event/world-level paired correctness, not only individual query accuracy.

**Prediction worth killing:** if natural models use a single-key retrieval shortcut, adding a repeated predicate should selectively reduce context-specific role retrieval even when one-hop role identification remains correct. If they can combine independent event cues, disambiguation should rescue relative to a matched irrelevant cue. A grammar-aware lexical copier can also pass; this tests behavior, not a unique neural algorithm. Hold out documents/event families and evaluate multiple untouched model families. Do not execute models until #23's source/label/format/competence gates pass.

## Exact artifact / replay
Conversation attachment: `BRAIN_O_E039_042_COPIER_AND_COMPOUND_KEYS_2026-10-16.zip`, SHA256 `dc63b76b2a73c174949d88c231f580d67c95c6860780ffba8bac8be6b58ff868`, 16 files (source, two pre-existing learned copier weights, two new withheld-pair weights, JSON metrics, per-file hashes, E041 incomplete script, research note, replay utility). ZIP CRC verified. After clean extraction: `python verify_saved.py` returns `REPLAY PASS` for E039, E040 and E042 without retraining. Raw binary artifact is **not in GitHub**; do not assume other researcher conversations can access it.

**Decision:** **FALSIFICATION of *semantic uniqueness* inferred from the E038 signature**, but **NOT** falsification of its task-effective causal route. Next gate: *competence-matched, ordinary-architecture learned copier control*, then authentic multi-event/LLM evidence. Negative results and architecture confounds preserved. No sealed sets, no unapproved paid inference, no other branches altered.
