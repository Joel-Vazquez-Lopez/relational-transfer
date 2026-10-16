# Mira — Pre-specified authentic-language behavioral evaluation plan (FEASIBILITY DRAFT, not executed)

**Created 16 October 2026. Scientific ownership:** Mira = source suitability/human-role audit; Nova = raw-output infrastructure; Astra = field-order/interface intervention execution; Comput = adversarial baselines; coordinator = approval/freeze. No duplicate Astra inference requested.

## Scientific question
Can an untouched pretrained LM use **event-specific context** to select predicate-appropriate participants when the same predicate label occurs more than once? Does informative context improve **correct semantic role assignment**, as opposed to forcing two different names or causing generic longer-prompt effects?

### Rival predictions
- **Contextual event selection:** event-resolving cues improve correct full-event actor/undergoer answers selectively; mere irrelevant context does not. Identical verb labels need not imply identical events.
- **First/last-position copier:** chooses nearby/frequent role name, particularly when repeated predicate appears; position changes dominate.
- **Format/complement constraint:** joint extraction discourages repeating one name, potentially increasing joint exact correctness while also increasing wrong role orientation.
- **Underspecified query:** verb-only question may legitimately have multiple answers even for competent model; a disambiguating question is not evidence of internal processing by itself.
- **Native-vs-reconstructed route:** behavior alone CANNOT distinguish pre-question binding from capable question-time sentence parsing. Mechanistic claims require separate interventions with competent learned copier controls.

## Current development intake and limits
- The linked suitability pass comprises **24 source-pinned unscored EWT challenge groups**: 8 priority, 8 conditional, 8 excluded from the main behavioral assay. Shortlist comprises **16 document-disjoint groups / 24 unfilled predicate annotation slots**.
- Prioritization by an investigator observing text creates selection bias. It is NOT a representative, untouched, independently blinded benchmark. EWT current source rights unresolved.
- Priority includes **negative cases with unexpressed agents**; do not score these as though they require a person/name answer.
- Shortlist is **not model-ready** until source rights/privacy, two truly independent human annotations, human adjudication of role applicability/scope, and a format/competence calibration pass.

## Intended factorial, after clearance and human adjudication
**Test A: event scope (primary behavior).** In the subset with truly distinct predicate occurrences, compare (i) original source sentence alone, (ii) original permitted contiguous context containing a human-verified disambiguating cue. Hold target predicate/event/question and its human-labeled answer fixed. If creating matched irrelevant-length context, mark it explicitly as a **derived controlled edit**, not an unchanged naturally authored corpus. If a source document has no naturally informative cue, classify as `NO_VALID_SCOPE_CONTRAST` rather than fabricating a positive.
**Test B: interface (Astra-owned).** At fixed event context, compare two independent direct verb-specific role questions with single joint extraction in field orders `Agent;Undergoer` vs `Undergoer;Agent`. Because two direct calls vs one joint call are not compute-identical, report call count/token budget, length, decoding, and prompt flags separately. Maintain comparable candidate-free one-name format for direct outputs. Do not assume every predicate has both roles.
**Test C: baseline controls.** Prespecify: first and last named participant, lexical repeat-match, local grammatical subject/object with passive correction, last-mentioned event, name-complement policy, simple full-source syntax/semantic copier where licenses permit; evaluate on same events. If context is uninformative or gold ambiguous, score abstention/scope rather than force binary gold.

## Rules for model and scoring readiness
1. **Before any inference:** settle individual excerpt usage rights; obtain two locked human sheets and adjudicator acceptance; ensure source-family/document split integrity and separately seal an *unexamined* confirmation set; validate exact model-input template contains no source IDs, splits, UD tags, or gold.
2. **Choose at least two distinct pretrained model families**, not merely two Qwen sizes; record exact model, tokenizer and quantization revisions, prompt template, engine, decoding seed and hardware. No model choice or inference is authorized by this document.
3. **Small open calibration first:** one-hop role recognition and exact response-format compliance scored separately, including empty/ambiguous role-handling; models failing prespecified validity/competence thresholds cannot support higher-order claims. Thresholds will be locked by coordinator BEFORE seeing predictions, not retrospectively chosen.
4. **Primary unit of analysis:** independent source document/event family, not each paraphrase, question, prediction or model seed. A 16-group feasibility sample is **descriptive only**; no row-level significance tests or ACL novelty claims.
5. **Primary feasibility output:** fraction of admissible annotated groups with at least one valid informative-context matched contrast, number with genuinely distinct same-lemma event occurrences, independent human agreement, number excluded for rights/privacy/ambiguity. This gate is met BEFORE LM accuracy interpretation.
6. **If proceeding:** report event-level **joint semantic correctness**, separate correct agent/undergoer, exact format validity, role collapse/reversal, wrong-event choice, unresolved/abstain and licensed/nonlicensed exclusions. Preserve every raw response, input context, event ID and evaluator normalization code. A role-not-applicable case must not be silently included in a two-role binary score.
7. **Paired analyses:** matched informative-vs-irrelevant context and field-order cross within document; bootstrap at document-family level if sufficient independent families; otherwise descriptive transitions only. Negative/failure cases preserved.

## Freeze/stop criterion
This document freezes **hypotheses and planned outcome types for discussion**, not source rights, gold labels, model checkpoint or executable inference. The coordinator must issue a separately timestamped, source-hashed, human-adjudicated exact prompt list and decision thresholds BEFORE any prospectively labeled experiment is called frozen.

**Falsifier:** if event disambiguation improves no more than irrelevant context, or all gains are reproduced by deterministic verb/position/copier/format baselines, reject the stronger contextual-binding interpretation. If no authentic role-disambiguating pairs survive human review, the *dataset* gate fails; do not convert ambiguous items into gold.

No training, models or paid inference in this work.
