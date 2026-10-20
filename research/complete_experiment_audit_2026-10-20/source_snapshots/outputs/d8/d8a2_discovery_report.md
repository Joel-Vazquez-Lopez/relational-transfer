# D8A.2 — Candidate-Side and Query–Candidate Interface Mechanism Discovery

## 1. Executive Summary

D8A.2 is best classified as **BROAD/DISTRIBUTED** exploratory evidence. Candidate endpoint information is locally available and observed/withheld candidate CLS contrasts remain comparable through block 0. The first decisive candidate-side divergence occurs at **block-1 attention**, one block later than the query-side block-0 divergence. Event contrasts remain healthy.

The endpoint failure is joint at the natural interface—query construction, candidate construction, and compatibility all collapse—but the frozen natural transfers do not yield a clean operation. Query-only transfer is null, candidate-only transfer moves margins in the wrong direction, and joint transfer produces opposite source/target effects. The target improvement is reproduced by an unrelated-context donor, so it lacks donor specificity. Candidate suppression is powerful but damages event and complementary margins almost one-for-one.

Protected confirmation is **not justified**.

## 2. Scientific Status and Firewall

D8A.2 is exploratory. Only A/B seeds 1701–1703 were instantiated. Seeds 1704–1706 remain internally untouched. The exact D8A context split was reused (SHA-256 `c841ecc8c0729911a070ddb2975d44ff3210811496a6712abdce53a4b687d1c5`): 1,884 discovery cells and 996 embargoed internal-validation cells. Candidate C2 was frozen with SHA-256 `26ffa0b22751fb877f5bfc1227ad7b0190a5ac55de7e29bc7af2b77bfd0fb845` before validation and evaluated once without modification. D8B/D8C were not run.

## 3. Frozen D8A Findings Motivating D8A.2

D8A remains unchanged: query endpoint identity was locally present, the first decisive query CLS divergence occurred at block-0 attention, C1 suppression was target-heavy/nonselective, and query-only P-isomorphic transfer failed. D8A.2 tests the unresolved candidate-side and compatibility explanations; it does not revise C1.

## 4. Candidate Encoder Architecture

Each 3×14 candidate scene is linearly projected into three object tokens. A learned CLS is prepended and learned positions are added. Two pre-norm blocks each apply LayerNorm→four-head self-attention→residual and LayerNorm→two-layer GELU MLP→residual. Final LayerNorm is applied to CLS, which is then L2-normalized for `10 * dot(q,c)` scoring. D8A.2 traces the input projection, token+position state, every norm/update/residual boundary, final candidate norm, and final L2 representation.

## 5. Instrumentation Validation

For every discovery model, candidate output difference and score difference were exactly 0, prediction agreement was 1.0, and strict allclose passed. No optimizer, training, parameter edit, or checkpoint write occurred.

## 6. Candidate-Side Endpoint Development

Identity-aligned object-token contrast is already substantial and graph-role comparable at input projection (whole-object norm 3.136780 observed, 3.174068 withheld). CLS contrast is also comparable after block-0 attention (1.254607 observed, 1.192502 withheld) and block-0 MLP (1.305200, 1.230058). The sharp split begins at block-1 attention (6.728970 vs 2.307979), grows after the MLP (12.137856 vs 2.887581), and persists after L2 normalization (1.105055 vs 0.273193).

Cross-context CLS-difference cosine changes at the same boundary: block-1 attention 0.801135 observed versus 0.042188 withheld; final output 0.755162 versus 0.023363. Supported candidate differences become reproducible across partners; withheld differences remain context-dependent.

## 7. Candidate-Side Event Control

Event construction is preserved: block-1 attention event contrast is 8.604199 observed and 7.873012 withheld; final L2 event contrast is 1.436142 and 1.347287. Candidate compression is endpoint-selective in natural computation, although the causal suppression itself is not selective.

## 8. First Candidate-Side Divergence

The first decisive, seed-consistent divergence is the **block-1 attention update**, not input projection, block 0, or final normalization. Final normalization preserves an already-large gap.

## 9. Source/Target Asymmetry

Natural candidate construction is much more symmetric than the query pathway. Final observed contrast is 1.108615 source versus 1.101417 target; withheld is 0.301494 source versus 0.239354 target. The intervention interface is not symmetric: joint discovery margin is -3.062333 source and 1.510556 target; validation is -2.583881 and 1.735267.

## 10. Candidate Component Analysis

All four block-1 heads were searched and retained; no neurons were searched. Observed projected head-contrast norms are 2.428150, 2.447419, 2.764656, and 2.235351. Each head contains both value/output and attention-weight terms; the exact symmetric decomposition reconstruction error is below 5.6e-7. Natural head replacement changes observed margins by −1.162116, −1.192948, −1.414439, and −0.982091. The transition is distributed, not a single-head mechanism.

## 11. Exact Query–Candidate Interface Decomposition

| quantity | observed | withheld |
|---|---:|---:|
| query endpoint-delta norm | 1.197764 | 0.050754 |
| candidate endpoint-difference norm | 1.105055 | 0.273193 |
| query / candidate-difference cosine | 0.506774 | -0.019922 |
| exact endpoint margin | 5.701521 | -0.093535 |

No defensible scalar fraction exists: query contrast shrinks about 24-fold, candidate contrast about fourfold, and alignment moves from strongly positive to approximately zero. These terms interact multiplicatively at the exact score interface.

## 12. 2×2 Counterfactual Decomposition

Using only exact P-isomorphic natural donors and unit amplitude, discovery WW/OW/WO/OO margins were 0.097511, 0.042591, -3.447168, and -0.979631. Validation margins were -0.110797, -0.110210, -3.092858, and 0.027697. Query-only does not rescue. Candidate-only increases contrast norm but rotates it into a strongly wrong direction. Joint transfer is not coherent across roles: targets improve, sources worsen.

## 13. Competing Hypotheses

- **H-Q:** not supported as dominant; OW is indistinguishable from WW.
- **H-C:** contradicted; WO is strongly worse despite larger candidate norm.
- **H-JOINT:** contradicted in aggregate; OO is role-opposed rather than generally restorative.
- **H-ALIGN:** alignment is essential descriptively, but the matched natural donor is not sufficient or donor-specific, so alignment-dominant classification is too strong.
- **H-BROAD:** best supported. Natural endpoint computation is distributed, suppression is broad, and simple donor differences do not transfer coherently.

## 14. Exploratory Candidate-Side Causal Interchange

Observed candidate suppression reduces endpoint margin from 5.672241 to 0.098030. It also reduces redirected event margin from 9.505668 to 3.931456, and complementary margin from 9.352151 to 3.777940. It is causal but broad.

Candidate-only transfer changes withheld discovery margin from 0.097511 to -3.447168; validation changes -0.110797 to -3.092858. This is failure, not rescue.

## 15. Exploratory Joint Query/Candidate Interchange

Joint discovery margin is -0.979631; validation is 0.027697. The aggregate masks reproducible role opposition: source margins are negative while target margins are positive. Therefore joint transfer is not a general interface repair.

## 16. Controls and Donor Specificity

Matched joint transfer does not outperform unrelated-context transfer: discovery -0.979631 matched versus -0.920553 unrelated; validation 0.027697 versus -0.019781. Target-only values are similarly indistinguishable. Reversed, wrong-endpoint, opposite-role, event, and norm-matched controls are all retained in raw tables. No donor-specific rescue exists.

## 17. D8A.2 Internal Validation

The frozen pattern replicated without modification: suppression remained strong and nonselective; query-only remained null; candidate-only remained harmful; joint transfer remained source-negative/target-positive; unrelated-context joint transfer matched the nominal donor. The holdout therefore strengthens the negative conclusion.

## 18. Candidate Mechanism

One candidate was frozen: `C2_candidate_block1_cls_attention_natural_interface_interchange`. Its complete hook, donor, recipient, operation, role/event/partner matching, natural amplitude, controls, predictions, discovery failures, and ambiguity are preserved in `candidate_mechanisms.json`. It is a diagnostic localization, not a reusable mechanism.

## 19. Strongest Remaining Alternative

The learned query and candidate spaces may use identity- and context-specific distributed coordinates. P-isomorphic semantic matching does not imply a transferable vector basis within an independently trained model. Broad CLS replacement can therefore disrupt many decisions, while adding a donor difference can increase norm yet rotate the recipient into an incompatible direction.

## 20. Implications for D5/D8

D5's endpoint failure is not explained by query compression alone. Query construction diverges at block 0; candidate construction diverges at block 1; final compatibility collapses. These are coordinated learned effects, but D8A.2 does not identify a simple transferable joint operation.

## 21. What D8A.2 Does Not Establish

It does not confirm a causal mechanism, reusable relational interface, primitive, grammar, language generality, or LLM generality. It does not authorize protected-model inspection.

## 22. Whether Protected Confirmation Is Justified

**NO.** Suppression is nonselective, neither single-side transfer rescues, joint effects are role-opposed, and matched donors do not beat unrelated donors. The single most informative next scientific step is to test whether a deterministic, architecture-native alignment between identity-specific query and candidate coordinate systems can be derived from model weights alone—without fitting on outcomes—before proposing any new intervention.

## 23. Reproducibility and Integrity

Candidate instrumentation reproduced outputs and scores exactly. All discovery checkpoint hashes match before/after; all model-state checks passed. D8A foundations and the reused context split are hash-identical. Protected checkpoints were not loaded. The only operational repair occurred before candidate freeze: a redirected-meaning event foil was added because the initial event column addressed the original meaning; the original column is retained, and no validation cell was opened during the repair.

## 24. Final Exploratory Classification

**BROAD/DISTRIBUTED**

## End Questions

**Q1.** Candidate-side observed/withheld divergence first decisively emerges at block-1 attention.

**Q2.** Yes. Final within-{0,1,2} contrast is 0.275325 in A and 1.163264 in B; within-{3,4,5} is 1.155960 in A and 0.271061 in B.

**Q3.** Yes. Identity-aligned object states and block-0 CLS contrasts are comparable before block-1 compression.

**Q4.** Natural candidate construction is approximately symmetric; causal joint transfer is strongly role-asymmetric and opposite-signed.

**Q5.** Yes in natural computation. Final event contrast remains 1.436142 observed and 1.347287 withheld. The suppression intervention itself is not selective.

**Q6.** Failure is jointly associated with query construction, candidate construction, and alignment. Their interaction does not support a unique scalar fraction.

**Q7.** No. OW does not improve over WW.

**Q8.** No. WO substantially worsens the endpoint margin.

**Q9.** No in aggregate. OO improves targets but harms sources.

**Q10.** No. Matched joint intervention does not consistently outperform query-only, candidate-only, unrelated-context, reversed, mismatched, event, or sham controls.

**Q11.** No. Suppression damages event/complementary margins, and apparent target rescue lacks donor specificity.

**Q12.** The frozen candidate's negative pattern survives internal holdout without modification.

**Q13.** Distributed identity/context-specific coordinate systems make natural semantic donor vectors non-transferable despite graph isomorphism.

**Q14. NO.** The next step is an outcome-independent analysis of whether model weights define a deterministic query-to-candidate coordinate alignment; do not open protected models.
