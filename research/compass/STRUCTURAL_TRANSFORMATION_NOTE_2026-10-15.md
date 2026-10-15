# Compass Research Note: Transformations Are Not Roles
**Date:** 2026-10-15  
**Status:** Exploratory archived-data reanalysis (no new inference, no verified mechanism).

## Finding
An independent recomputation of existing `strong_free` constructed-prompt predictions distinguishes:
1. **Role discrimination:** Do actor and patient questions produce different candidate names?
2. **Direction sensitivity:** When the same named participants exchange actor/patient roles, does the output name change?
3. **Construction stability:** For the same event, does the answer remain the same through active, passive and cleft paraphrases?
4. **Correctness:** Are both role answers actually right?

For untouched Qwen2.5 1.5B, lexical wording, on this development panel:
- **12/48** event conditions: both role answers correct;
- **14/48** event conditions: two distinct *valid* names returned;
- **42/48** role-specific comparisons: answer name changes when actor/patient directions are reversed within a name pair and construction;
- **36/72** answer comparisons: same name returned when active-voice event is rephrased using passive, subject-cleft, or object-cleft form.

**These different denominators do not establish a dissociation of internal computational faculties.** In particular, a first/last-position heuristic can flip the output when the sentence direction reverses without ever computing the actor/patient relationship.

This audit spans seven model configurations and two wording conditions, but uses repeatedly examined constructed development events, only six name-pair groups, and multiple non-independent transformations. It is not a confirmatory or causal test.

## New scientific opening
A model may be highly sensitive to **entity/position changes** yet fail to keep the *same semantic relationship* invariant under changes in syntactic construction. That motivates a test of whether models transform relational coordinates consistently, independently from their surface positions.

Think of the observed two-role answers as four possible maps: identity (correct), permutation (roles swapped), projection onto actor (same actor twice), projection onto patient (same patient twice). Wording and syntax move the observed behavior among these maps. These are **behavioral mappings**, not discovered internal computational states.

## What could invalidate the exciting interpretation?
- A position-only model may reproduce the full pattern.
- Reused lexical examples and construction templates may drive all consistency statistics.
- No statement about pre-question relation caching versus post-question reconstruction follows from these outputs.
- Gemma's perfect panel already shows that these constructed tasks can be solved; making other models fail is not the scientific aim.

## Next falsifying experiment (small and cheap before model intervention)
Use *new, independently validated* event configurations with 3+ participants and syntactic changes that decouple semantic roles from ordinal mention position. Prespecify rival algorithms: first mention, last mention, subject selection, question-wording-dependent constant role, role-correct event binder, and question-dependent reconstructor. Test invariances and crossed transformations that these algorithms predict differently. Only then spend approved compute on native-path causal analysis, with explicit controls for bypassing the source text.

## Reproducibility
Source: uploaded `COMPASS_RESEARCH_FOUNDATION_2026-10-15.zip`; archive-integrity verification previously passed. Local Compass audit file: `structural_transformation_audit.py` (in the Companion Compass structural-transformation audit package delivered in chat), plus `structural_transformation_metrics.csv` and `collapse_positions.csv`. Historical raw predictions unchanged.

## Reporting status
**Interesting clue; not yet a discovery.** No mechanism, generalization, or novelty claim is made. Competing explanations remain active.
