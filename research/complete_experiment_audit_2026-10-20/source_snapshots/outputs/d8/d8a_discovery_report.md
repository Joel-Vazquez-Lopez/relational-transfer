# D8A — Protected Exploratory Mechanism Discovery

## 1. Executive Summary

D8A finds **construction-dominant exploratory evidence**: endpoint identity is still present at the changed token in unsupported graph roles, but the first CLS summary write—block-0 attention—constructs a large, functionally useful endpoint distinction only for observed graph roles. Later computation amplifies that gap. Event information remains healthy across the same stages.

The limited causal interchange is informative but negative for a reusable operation. Replacing the supported q-prime block-0 CLS attention value with the matched q value reduced the discovery endpoint margin from 5.724114 to -0.274359 and the internal-validation margin from 5.624211 to -0.515820. However, it also damaged event and complementary margins and was driven mainly by target substitutions. The exact P-isomorphic natural-difference transfer did not rescue withheld margins: discovery 0.038066 sham versus -0.022361 transfer; internal validation -0.100243 versus -0.094587.

Accordingly, D8A does **not** justify preregistering this intervention on protected models. A causal mechanism is not confirmed.

## 2. Scientific Status

D8A is exploratory. Only D5 A/B seeds 1701–1703 were used for internal analysis. Seeds 1704–1706 remain protected: their checkpoint files were hashed for inventory but never instantiated, and no protected hidden states, gradients, components, interventions, or new mechanism metrics were inspected. D8B and D8C were not run.

## 3. Discovery/Test Firewall

- Discovery models: A/B × seeds 1701, 1702, 1703 (six models).
- Protected models: A/B × seeds 1704, 1705, 1706 (six models).
- Context universe: 2,880 legal ordered endpoint-substitution cells.
- Discovery cells: 1,884; internal-validation cells: 996.
- Split rule: frozen SHA-256 modulo-3 assignment after B-to-A identity canonicalization; A/B-isomorphic cells share assignments.
- Plan hash: `57ccb73fc44ad3568ec8e2fcdd5cee88ee9bc9285111e327180f1f578ded08d1`.
- Discovery-manifest hash: `4f49640dd253a82a09997c828a3bddcc54192e9287ee423ae42cb4d5b7da8faf`.
- Protected-manifest hash: `e6215cbb15dac21133f101a55440a18b321ddac7cbbd3c24c01473571ce43e02`.
- Candidate hash frozen before validation: `d17e8fe992e15c3be990a9e78036f59cb3fbadb299d0a43d5fdb3edd73c07304`.

## 4. D5 Scientific Foundation

D5 validly showed that moving the absent relational clique from {0,1,2} to {3,4,5} moves final-query endpoint compression and endpoint-redirection failure. It did not locate a computation, establish mediation, or identify a reusable operation. D6 is invalid and D7-R is methodological context only; neither is used as mechanism evidence here.

## 5. Exact Model Computation

The oracle message is `[source identity, target identity, event token, PAD]`. Each one-hot token is multiplied by the learned symbol table, a learned CLS token is prepended, and learned positions are added. Two pre-norm transformer blocks apply LayerNorm→four-head self-attention→residual and LayerNorm→two-layer GELU MLP→residual. Final LayerNorm is applied to the sequence; CLS is extracted and L2-normalized.

Each 3×14 candidate scene passes through a separate two-block pre-norm encoder with a learned input projection, CLS, positions, final LayerNorm, and L2 normalization. Scores are exactly `10 * dot(normalized query, normalized candidate)`.

The atlas records token+position, each norm input, attention update, post-attention residual, MLP input/update, post-MLP residual, final message norm, and final L2 query. The candidate pathway is summarized at the final exact scoring interface by candidate-difference magnitude and alignment.

## 6. Instrumentation Validation

Across all six discovery models, maximum absolute score difference was 0, prediction agreement was 1.0, and exact allclose passed. Models were in eval mode, no optimizer existed, gradients were enabled only on detached local activations for diagnostic derivatives, and every loaded model state hash was unchanged after its analysis.

## 7. Layerwise Endpoint Development

At the changed token, endpoint displacement already differed modestly by graph role: observed 1.146290, withheld 0.771272. Thus endpoint identity was not absent in unsupported inputs. CLS displacement is zero before attention by construction. The first CLS write creates the decisive gap: block-0 attention observed 5.221720, withheld 0.570645. The gap grows after block 0 MLP (8.914145 vs 0.642810), after block 1 (21.066000 vs 0.888988), and at the L2 query (1.197764 vs 0.050754).

Source/target asymmetry is substantial at the first CLS write (observed source 2.128067, target 8.382993; withheld source 0.336587, target 0.850498), but final observed separation converges (source 1.189901, target 1.205798). Cross-context endpoint-delta cosine remains structured rather than random; at the final query it is 0.936987 observed and 0.736497 withheld.

## 8. Event Control

Event CLS displacement is nearly invariant to endpoint graph role: block-0 attention observed 6.449523, withheld 6.282232; final L2 query observed 1.458585, withheld 1.446979. Final event margin changes are likewise large in both groups (−18.898183 vs −18.853748). The endpoint-specific failure therefore does not reflect global collapse of the query pathway.

## 9. Functional Alignment With Candidate Decision

At the exact final interface, observed versus withheld means were:

| quantity | observed | withheld |
|---|---:|---:|
| query endpoint-delta norm | 1.197764 | 0.050754 |
| candidate-difference norm | 1.117997 | 0.259790 |
| query-delta / candidate-difference cosine | -0.830180 | -0.029702 |
| exact natural margin change | -11.394794 | -0.004749 |

The final identity is exact up to floating-point error. Earlier first-order diagnostics usually recover the sign but not the magnitude in observed cells: sign agreement rises from 0.941892 after block-0 attention to 0.997297 after block-1 MLP, while mean absolute error falls from 9.119166 to 4.345021. These are local sensitivity diagnostics, not causal effects. Withheld endpoint functional change is already near zero after block-0 attention (mean finite −0.004749), whereas event sensitivity remains large.

Candidate-side endpoint differences are also compressed (1.117997 observed vs 0.259790 withheld). This independently limits any query-only rescue and is a major unresolved alternative.

## 10. Construction vs Transmission vs Use

**For construction:** unsupported endpoint identity remains at its token but is only weakly written into CLS at the first attention boundary; this graph-role gap is present in every discovery model and event construction is intact. **Against a pure construction-only account:** the initial symbol displacement is already smaller for withheld pairs, and later blocks further amplify/transform the gap.

**For transmission:** the observed/withheld ratio grows across later residual stages, and direction changes occur in MLP/attention updates. **Against transmission-dominant:** the largest qualitative separation is present at the first CLS construction, not after initially comparable CLS summaries.

**For use/readout:** candidate-difference magnitude and query/candidate alignment both collapse in withheld roles, so downstream compatibility matters. **Against use/readout-dominant:** the unsupported CLS displacement itself is already tiny at the first summary write.

**Mixed/distributed evidence:** all four block-0 heads contribute, later blocks amplify the difference, source/target effects differ, and candidate-side compression co-occurs. Best current interpretation: construction-dominant summary formation with distributed downstream amplification and candidate-side co-compression. Strongest alternative: the intervention removes a broad task summary and candidate-encoder compression, rather than a selective endpoint operation, is the proximal behavioral bottleneck.

## 11. Component-Level Analysis

The layer atlas justified inspection of block-0 attention. All four heads were searched and logged; no neurons were searched. For observed substitutions, projected CLS-delta norms by head were 1.975927, 1.895065, 1.379151, and 1.063788. Replacing each q-prime head value with q changed endpoint margin by −2.050340, −1.993390, −0.714830, and −0.866969 respectively. Effects were distributed and strongly target-heavy; no single head constitutes a clean mechanism.

## 12. Exploratory Causal Interchange

All attempted discovery interventions are retained in `exploratory_interventions_discovery.csv`: baseline/sham, recipient natural delta, matched P-isomorphic donor, reversed donor, opposite-role donor, event-delta control, and supported natural-value replacement. No amplitude search was performed; natural scale was exactly 1.

Supported replacement caused a large endpoint-margin reduction but also reduced event margin from 8.378039 to 4.909507 and complementary margin from 9.202958 to 2.428410. The endpoint effect was small for source substitutions (5.751432→4.654774) and large for target substitutions (5.696200→−5.311233). Matched transfer failed and did not outperform reversed or opposite-role controls.

## 13. D8A Internal Validation

Candidate C1 was frozen before validation and evaluated once on the 996 held-out cells without modification. Supported suppression replicated: endpoint margin 5.624211→-0.515820. Its role asymmetry also replicated (source 5.533578→4.424452; target 5.711392→−5.267891), as did collateral loss: event 8.439649→4.757609; complementary 9.167156→2.076035.

Matched withheld transfer remained null: -0.100243 sham versus -0.094587 transfer. Thus the full mixed discovery pattern survived: reproducible suppression, nonselectivity, and no rescue.

## 14. Candidate Mechanism

One candidate was frozen: `C1_block0_cls_attention_natural_endpoint_interchange`. Its hook, donor/recipient rules, operation, role/event/partner matching, natural amplitude, controls, discovery evidence, failure cases, and ambiguity are fully specified in `candidate_mechanisms.json`. It is a plausible suppression locus, not a confirmed mechanism or reusable operation.

## 15. Strongest Alternative Explanation

The network retains endpoint identity in token states but learns a task-specific CLS summary and candidate representation that jointly collapse distinctions absent from relational training. The successful target-heavy replacement may therefore perturb a broad task summary; it does not isolate a selective endpoint computation. Candidate-side compression could prevent query-only rescue even if the donor query direction were meaningful.

## 16. What D8A Does Not Establish

D8A does not confirm mediation, a head/neuron mechanism, a reusable relational primitive, compositional grammar, language generality, or LLM generality. Internal validation remains exploratory and uses discovery models. No protected-model evidence exists.

## 17. Proposed Next Test (Not Executed)

Do **not** preregister C1 on protected models yet. The single most informative next discovery analysis is a matched causal decomposition of endpoint differences through the candidate encoder, especially its first CLS attention write, using the same natural substitutions and graph-role relocation. A future protected test should be frozen only if that analysis yields a selective joint query/candidate intervention with donor specificity, event preservation, and discovery-cell-to-internal-holdout rescue.

## 18. Reproducibility and Integrity

All six discovery checkpoint hashes match before versus after. All per-model state checks passed. No checkpoint was written, no optimizer was created or stepped, and protected checkpoints were not loaded. Source/config hashes are in `source_config_hashes.json`; exact raw rows and reduced tables are in the D8 results directory. The only non-critical operational warning was an undefined cosine at stages where CLS displacement is exactly zero; these entries are retained as NaN and did not change the protocol.

## 19. Final Exploratory Classification

**construction-dominant evidence**

## 20. End-of-D8A Questions

**Q1.** The first modest graph-role difference is already in the changed token embedding. The first decisive, model-consistent CLS divergence is the block-0 attention update.

**Q2.** A mixture led by magnitude and functional alignment: early summary magnitude collapses, candidate-difference magnitude also shrinks, and final query/candidate alignment disappears. Final normalization preserves rather than creates the gap.

**Q3.** Yes. At the final query, within-{0,1,2} displacement is 0.050416 in A and 1.213381 in B; within-{3,4,5} is 1.239825 in A and 0.051091 in B. The trajectory swaps with graph role.

**Q4.** Endpoint information is present in token states but weakly constructed into the unsupported CLS summary, with later distributed amplification: construction-dominant, not absent-identity or pure late readout failure.

**Q5.** Block-0 CLS attention natural endpoint interchange is a plausible causal locus for suppression, but it is not a reusable candidate operation because effects are distributed/nonselective and transfer fails.

**Q6.** Yes, suppression impairs supported endpoint margin, especially target substitutions, but not selectively: event and complementary margins also decline.

**Q7.** No. Matched P-isomorphic transfer does not improve withheld endpoint discrimination over sham and does not beat controls.

**Q8.** The frozen candidate's mixed pattern survives internal holdout without modification: strong target-heavy suppression, collateral damage, and failed rescue.

**Q9.** A broad task-summary perturbation plus candidate-encoder co-compression, rather than a selective reusable endpoint operation.

**Q10. NO.** Selective suppression and donor-specific rescue are missing. The next discovery analysis should causally localize candidate-encoder endpoint compression before any protected test is preregistered.
