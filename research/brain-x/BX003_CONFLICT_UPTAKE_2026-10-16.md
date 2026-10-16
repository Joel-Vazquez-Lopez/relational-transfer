# BX-003 — Relational evidence uptake, with an honest mixture-calibrated reference

**Date:** 16 October 2026 · **Status:** COMPLETED exploratory post-hoc reanalysis of BX-002's 4,000 original synthetic test worlds. **No new neural training, inference, dataset, pretrained LLM experiment, or mechanism claim.**

## Motivation: correct a subtle reading of BX-002

BX-002 showed that changing relational information changes a neural answer, while the neural model did not beat the local-only policy overall. This does **not** prove the model never uses informative edges beneficially. Its utility can be **positive on genuinely conflicting evidence** and **negative elsewhere**, cancelling in the global score.

Furthermore, BX-002's original exact Bayesian reference conditioned on all edges having noise q=0.08, whereas the trained model actually saw a mixture of 80% informative-parity worlds and 20% worlds with fully independent random parity edges. No reliability flag was given to the network.

BX-003 evaluates a more comparable **mixture-aware but item-regime-blind** analytic decoder, and decomposes net utility by a label-free source-evidence conflict criterion. It does not retrain or change either original neural checkpoint.

## Fairer generative reference

Let z be the five hidden bits, x the observed node bits, and e the observed parity edge bits. The training distribution draws ONE unobserved regime per world: R=valid with probability 0.8, or R=random with probability 0.2. Thus

P(z | x, e) ∝ [prod_i P(x_i | z_i)] × {0.8 prod_(i,j) P(e_ij | z_i XOR z_j, q=0.08) + 0.2 × 2^(-5)}.

The decoder sums over **all 32 hidden worlds** and marginalizes the unobserved regime. It sees only x/e and the queried node; it never receives the case's true latent bits, reference gold labels, or an indicator of which cases are valid. It **does**, however, know the exact synthetic generator and noise rates. Thus it remains a privileged *analytical achievable reference*, not a trained model with matched computational resources.

On the original valid-parity panel this decoder scores **2,929/4,000 = 73.225%**; on randomized edges **2,382/4,000 = 59.55%**. BX-002's prior q=0.08 exact reference scored 2,925/4,000 = 73.125% on valid examples. Accounting for the actual training mixture does not explain the neural-vs-reference performance gap.

## Label-free conflict stratum

Before seeing gold, define the *local suggestion* as the directly observed queried node and the *relational three-vote* as the majority among that node and its two parity-transported neighbors. Define `conflict` when the three-vote policy disagrees with local evidence. This stratum depends on inputs alone, not on the correct answer or a pretrained-model response.

There are **899 conflicts / 4,000** test worlds and 3,101 agreement worlds.

| Method | Correct of 899 conflicts | Correct of 3,101 agreements | All 4,000 correct | Number of local-answer overrides among 899 conflicts |
|---|---:|---:|---:|---:|
| Local node | 358 | 2,254 | 2,612 | 0 |
| Three-vote relational heuristic | 541 | 2,254 | 2,795 | 899 |
| Calibrated mixture Bayes | **592** | **2,337** | **2,929** | 552 |
| Trained 3L Transformer seed 17, valid constraints | 403 | 2,203 | 2,606 | 155 |
| Trained 3L Transformer seed 29, valid constraints | 374 | 2,240 | 2,614 | 90 |

Compared to local answering, seed 17 improves by **+45 on conflict items**, but loses **51 on agreement items**, for **−6 net**; seed 29 improves by **+16 on conflicts** but loses **14 on agreements**, for **+2 net**. This refines the claim: the model sometimes successfully uses conflicting relational evidence, but its imperfect uptake and collateral errors cancel most gains. The calibrated reference improves by **+234 on conflict** and **+83 on agreement** items, for **+317 net**.

A change in predicted answer is not automatically a useful update. On all 4,000 examples relative to local policy, seed 17 performs **161 rescues and 167 harms** (328 answer flips); seed 29 performs **57 rescues and 55 harms** (112 answer flips). Under valid→random parity perturbation, the frozen seed-17 model flips 448 answers and seed-29 flips 187, but these perturbations lower accuracy rather than establishing a role-computing algorithm.

**Scientific interpretation:** the restricted neural failure is not exclusively “information never entered the output.” It is partly **poor selection of when to replace an otherwise useful local answer**. These two seeds do not prove that this is an architectural property or a mechanism shared with language models.

## Reproduction

**Exact tested and committed source:** [`research/brain-x/bx003_release.py`](bx003_release.py), Git blob SHA **`1f8dec826c7fc3aae7e44b811a773ed13df81847`**, 4,862 source bytes, SHA-256 **`d306f2d35a56105e9fca31745fc42532dc31c68207e42335369637893aa5bf19`**.

Its only external input is the exact BX-002 per-item `paired_predictions.csv` produced by [BX-002 script](bx002_paired_syndromes.py), original SHA-256 **`5a1d30a65a1773168b15c6651ad1c440e567f33c8db2e6bcc0223e74ef7ba3d0`**. That CSV is preserved in the separately generated Brain-X evidence ZIP, **not** included in GitHub and not presumed visible to a separate conversation. A reviewer can regenerate it with BX-002, noting that hardware/library differences may change exact checkpoint outputs.

Run from repository root after obtaining that CSV:

```bash
python research/brain-x/bx003_release.py --csv /absolute/path/to/paired_predictions.csv --output /tmp/bx003_results.json
```

The script independently reconstructs the exact BX-002 world fixture (SHA-256 `9f040b4a050e2d9c35c8eac2175ca13eeeb865b30131f2f6f3d7083547d813ef`), verifies its file ID/query/gold and source heuristic columns, recomputes the mixture Bayes posterior using model-visible features only, and checks global binary-label symmetry and the zero-information regime. Original per-item predictions are reused with **no new neural inference**. Machine-readable reported counts: [`bx003_uptake_results.json`](bx003_uptake_results.json).

## Important validity/novelty boundaries

1. The conflict stratum and mixture reference were developed **after** seeing BX-002 results. This is a hypothesis-generating analysis, not a preregistered confirmation.
2. Local and neighbor votes exploit known fixed index topology, and the Bayesian benchmark knows generator probabilities. This isn't an exact parity-cost-matched neural control.
3. Same generated five-node ring; 4,000 draws are independent within this *one synthetic distribution*, not four thousand independent linguistic constructions; two checkpoints share the same test worlds. No real language, held-out grammar, or independent training replay.
4. Prior work on selective prediction, evidence aggregation, Bayesian decisions, structured output inference and error-correcting codes already exists. **Neither conditional rescue/harm analysis nor posterior mixture correction has been established as novel.**
5. “Information sensitivity,” “information utility,” and “answer coherence” are distinct **measurement categories**, not evidence that a Transformer implements a proposed particular factor graph or latent role-binding structure.

## Next direction — without colliding with Astra / Brain-O / Compass

The testable cross-domain question: **On genuinely human-authored sentences, when local event-role evidence is independently adjudicated as insufficient but longer context provides corroboration, does a frozen pretrained LLM correctly override a local answer more often than it corrupts a correct one?**

Pre-register human-annotated local-versus-context source conditions with gold held application-side; compare informative context to legitimate matched irrelevant context and a label-free source-only reader; score per-document **rescue, harm, decision change, and net benefit**, with semantic correctness separate from exact format. Independent human labels, licenses, document-disjoint grouping and at least TWO pretrained model families required per [#23](https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/23).

- **Astra owns actor/patient output-field-order manipulations.** Brain-X will not repeat them.
- **Brain-O owns native learned causal route / key-value copying controls.** Brain-X will not duplicate E031–E042.
- **Compass owns extended functional composition and intermediate-state training.** Brain-X will not copy I15/I16.
- **Brain-X owns the evidence-conflict/utility calculus** and possible authentic-context transfer metric only; see Mira/Nova for approved source/pilot and Comput for shortcut auditing.

Next task is *not* to fit bigger syndrome Transformers to rescue H1. It is to make the relational evidence-utility contrast evaluable on independently adjudicated authentic language.

**No paid inference, new model training, sealed confirmation access, or merge performed.**
