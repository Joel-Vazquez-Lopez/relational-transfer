# Phase 2 development result — 14 October 2026

**Status:** All 160 planned models completed. This is development on eight **previously examined** graph families, not independent confirmation. The frozen protocol, source manifest, experiment, analysis, runner, blueprint, and preflight hashes match; the ledger has 160 unique starts and 160 successful finishes, with no exclusions or technical reruns. `analyze.py` verified every saved checkpoint/model hash. The separately declared all-45-pair audit completed for all 160 final checkpoints; its source/protocol hashes and saved model hashes match.

## Frozen primary contrast

At 6,000 updates on scene suite 260107, the participant-only high-minus-sham focal binary-role accuracy penalty averaged **−26.83 percentage points**; ordered-role training's penalty averaged **−4.26 points**. The ordered-minus-participant attenuation was **+22.57 points**, with family effects **+17.38, +24.02, +40.14, +23.73, +27.25, +4.30, +21.97, +21.78**. All eight were positive, but the sixth family was weak. The descriptive family-level 95% t interval is **+14.24 to +30.91 points**; the exact two-sided family sign-flip p value is **0.0078125**. These statistics cannot be presented as independent replication because these families and earlier outcomes were already examined.

The second scene suite yielded +22.17 points, also positive in all eight families. No arm/family met the protocol's ceiling flag of at least 97% focal role accuracy in **both** graph conditions, but sham focal accuracy was often near 100%, so ceiling/scale effects still matter.

## Other arms and performance matching

The proposal's role-swap score-consistency arm did **not** show a distinct benefit over the permutation compute control: mean attenuation **+0.33 points**, family interval **−5.31 to +5.97**, exact sign-flip p **0.875**, with mixed directions. It is a global identity-relabeling consistency constraint, not explicit gold-role supervision. Ordered-role supervision outperformed it on the focal interaction; they must not be described as the same intervention.

The prespecified performance match **failed**. Ordered-role models at 6,000 updates reached **95.68%** nonfocal unseen binary-role accuracy on the eight selected pairs. The closest eligible participant-only checkpoint was 1,500 updates at **85.96%**, a **9.73-point** gap, far beyond the frozen two-point tolerance. The standard, role-swap, and permutation arms also missed the tolerance by more than eleven points. Accordingly, the listed checkpoint contrast under the matching procedure is **invalid as a performance-matched inference**. The primary result establishes a measured absolute-scale interaction, but it does **not** isolate reduced structural dependence from a large general transfer improvement.

At the fixed final checkpoint, focal binary-role accuracy averaged **96.09%** for ordered-role and **85.07%** for participant-only. Nonfocal unseen accuracy on the selected pairs averaged **95.68%** versus **85.01%**. Familiar four-choice accuracy averaged **99.94%** for both. Ordered-role focal event accuracy was **94.65%**, versus **96.03%** for participant-only: a small secondary trade-off, not a uniform improvement.

## Full-pair validity audit

Every final checkpoint was evaluated on all 45 entity pairs, with 128 balanced renderings per pair: 20 observed, one focal unseen, and 24 other unseen pairs. All 160 files have the expected counts and model SHA-256. Across sham/high cells, familiar four-choice accuracy remained approximately **99.9%** for every arm. Ordered-role nonfocal binary-role accuracy was **96.13% sham / 95.13% high**, versus participant-only **83.44% / 85.53%**. Thus the broader transfer gain is real in this task and cannot be dismissed as an artifact of the eight selected nonfocal pairs. The mean sham-to-high change across nonfocal pairs was only −1.01 points for ordered-role and +2.09 for participant-only; this supplementary diagnostic suggests the large penalty is concentrated at the focal pair, but it is not a quality-matched test and was not the frozen primary endpoint.

## Scientific decision

**A. Reduced structural sensitivity beyond general improvement:** unresolved. The measured focal interaction is strong and consistent, but all alternative arms remain 9.7–11.7 points below ordered-role on the performance-matching outcome. Do not claim the stronger causal learning-principle interpretation.

**B. Distinct role-swap benefit:** not supported in the frozen final-checkpoint contrast.

**C. Across-family consistency:** ordered-role attenuation is positive in all eight, with substantial heterogeneity (+4.30 to +40.14 points).

**D. Fresh confirmation:** the narrow absolute-scale attenuation is worth considering for a new cohort, especially given the earlier prospective rewire result. But repeating it alone would not settle the key rival explanation. The next *design* step should establish a genuinely comparable nonfocal performance operating point (or another prespecified ability-controlled estimand) on development data, then freeze a fresh confirmation protocol before seeing any new graph outcomes. No fresh confirmation or pretrained-LLM run was launched here.

The rewire changes several local graph properties, not just common-partner count. No unique structural mechanism or pretrained-LLM result follows from this phase.

## Reproduction

Frozen files: `PROTOCOL.json`, `BLUEPRINT.md`, `FULL_PAIR_AUDIT_PROTOCOL.json`. Raw results: `full/`, `RUN_LEDGER.jsonl`, `logs/`. Machine-readable frozen analysis: `SUMMARY.json`. Re-run from this directory with the recorded Python environment: `python analyze.py`; `python full_pair_audit.py` skips completed audit files.
