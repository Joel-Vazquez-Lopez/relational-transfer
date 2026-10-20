# Preferred next investigation: where a learned distinction transfers

20 October 2026. **Design candidate for review, not a frozen protocol or a completed training experiment.** No new model inference, training, downloads or API inference. The accompanying archive check uses saved CSVs and Python's standard library only.

## The decision

Investigate one acquisition question:

> When training requires distinguishing two entities in one argument role, does the learned distinction transfer to the other role, remain role-specific, or require comparisons in both roles?

This uses the strongest causal foothold in the existing project. It narrows the broad claim about computational-development similarity into a test of a relationship between two developmental responses. It makes no commitment to identical hidden directions or identical timing.

The initial microscope is the existing small query/candidate Transformer system. Source and target are task-defined ordered arguments. **Its fixed message slots cannot establish natural-language semantic roles independently of position.** A linguistic extension must vary argument role and surface position independently within the same model; it cannot simply rename the source slot “Agent.”

## What was already done

- D25–D29 changed both source and target comparison requirements and observed strong transfer. Repeating this package would add little.
- D52–D59 corrected a positive-frequency confound, compared foil-only and positive-only conditions, and partitioned collision/noncollision comparisons. They did not isolate the proposed source-demand × target-demand factorial in the indexed records inspected here.
- D51 showed improvement across all three withheld identity pairs, weakening a strictly focal-pair explanation.
- D62/D63 failed to predict later acquisition using merge/release assays. We must not repeat those assays as if the problem were new.
- The older fixed-query/fixed-candidate controls used **role-explicit supplied representations**, not merely frozen learned branches. They do not establish a learned-encoder bootstrapping mechanism.
- B-D33 localized a sufficient adaptation route in another cue/relation task. It does not identify the mechanism of the graph effect.

This is an indexed overlap check, not a proof that no unindexed intermediate run ever tested a similar idea.

## A concrete correction from the saved data

The historical field `redirected` in the inspected D15/D21/D45 scripts means “the edited query selects the designated target.” It does not require that the original query selected its original gold. It is a valid endpoint-accuracy measurement, but a misleading name for actual switching.

The archive-only recheck found:

| D15 source change | Original correct | Target selected after change | Both original and changed answers correct |
|---|---:|---:|---:|
| A seed1701 | 49/128 | 80/128 | **2/128** |
| A seed1702 | 58/128 | 71/128 | **3/128** |
| A seed1703 | 51/128 | 76/128 | **0/128** |
| E seed2701 | 125/128 | 128/128 | **125/128** |
| E seed2702 | 125/128 | 128/128 | **125/128** |
| E seed2703 | 120/128 | 128/128 | **120/128** |

Thus A's reported 55–63% “redirection” mostly consisted of target answers without a correct baseline. E's substantial paired success survives. The A/E difference becomes clearer under the stricter endpoint definition; this is **not a new independent replication**.

Among 144 saved developmental summary rows, 20 have reported mean query change exactly zero and positive target-after accuracy. One tied initial state has target-after accuracy82.8125% with reported query change0. An unchanged query cannot cause a switch when scored against unchanged candidates. The code confirms that this statistic counted target-after predictions only. These aggregate CSVs cannot recover actual paired switching at those training stages because they omit per-scene baseline predictions.

See [the complete recheck](ARCHIVE_RECHECK.json), [per-model counts](PAIRED_ENDPOINTS.csv), and [the executable script](archive_check.py). D15 and D21 share A1701 cases and must not be pooled as independent observations. This correction applies to these archived diagnostics. It does not automatically invalidate original D5 support effects or recent Qwen interventions scored differently.

## Competing operational predictions

| Restricted explanation | Source comparisons only | Target comparisons only | Comparisons in both roles |
|---|---|---|---|
| Global identity protection | Benefits can transfer to both source and target distinctions | Benefits can transfer to both | Usually little uniquely role-specific necessity for the conjunction |
| Role-specific acquisition | Source transfer improves more than target transfer | Target transfer improves more than source transfer | Both improve; crossed training-role × tested-role interaction |
| Joint requirement or dose threshold | Single-role conditions insufficient | Single-role conditions insufficient | Both improve; **these two alternatives remain confounded unless dose controls separate them** |

These are restricted predictive models, not exhaustive accounts of neural computation. A mixture may be the correct outcome. We cannot infer a dedicated role circuit from a behavioral interaction.

## Smallest design worth building

First construct a source-comparison × target-comparison factorial with four conditions: neither, source only, target only, both. All conditions use the same positive meaning schedule, initialization, optimizer, total examples, update count, four-candidate format, rendering seeds and answer-position balancing.

At each off-factor slot, substitute a nonfocal identity comparison rather than deleting the example. This keeps comparison count constant. Preserve counts for every supervised identity/role/event; audit candidates and foils as well as queries. Report remaining geometry, difficulty and gradient differences explicitly.

**Before any training**, enumerate the full finite input design and verify:

1. Source-only examples require the focal source distinction and introduce no focal target-comparison requirement; vice versa for target-only. Inspect every candidate pair, not just positive-versus-foil pairs.
2. No nominally withheld relation occurs as a query, correct candidate or foil. Inspect full episodes; positive-label support alone is insufficient.
3. Every candidate can serve as the target equally often under the evaluated block construction, preventing a candidate-only answer shortcut.
4. Identity and source/target labels are permuted across independent worlds; object slots and answer positions are randomized/balanced.
5. A merger-based necessity claim must be demonstrated on **complete encoded inputs**: exhibit identical merged observations with incompatible labels, or compute the appropriate finite conditional Bayes ceiling. A collision of symbolic meanings alone is not a proof because nuisance geometry could distinguish examples.

If the factorial cannot meet these requirements, stop. Do not substitute a nominally balanced but semantically entangled dataset.

Four arms can test role-selective versus broad transfer. If we want to interpret a both-only improvement, add two dose-matched single-role arms: source-only and target-only with the same total focal comparison exposure as the both arm. **A four-arm both-only result alone cannot establish joint computational necessity.** This makes six arms the minimum for that stronger question. Matching initial cross entropy or gradient norm alone would still not match the entire experienced gradient field.

The number of fresh worlds/seeds, numerical effects required, uncertainty procedure and runtime/storage caps remain to be set in a real pre-output lock. No significance or power is claimed for an unspecified small cohort. A feasibility pilot on one disclosed development world must not be counted as confirmation. Reuse the existing trainer; do not download or fine-tune a7B model to settle this first question.

## What to record

**Primary:** all-case paired correctness for original and counterfactual queries, separately for source and target, at the final fixed update. Average repeated scenes within each model; retain world/seed hierarchy. Report all arms including familiar-task failures. Conditioning on baseline competence is a separately labeled secondary result, because treatment changes competence.

**Secondary developmental observations:** input, both encoder blocks, all relevant token states, query/candidate contrasts and their alignment. Save both pre/post predictions and margins for every case, identity/no-change controls, losses, full model and optimizer/RNG states at fixed training updates. No old-style target-after-only “switch” statistic.

Do not call decodability evidence of causal use. Do not call the algebraic relationship between contrasts/alignment and the score a learning mechanism. A later component intervention must be chosen from development data, validated against identity/norm/position controls, and tested without retuning on fresh cases. It is a separate gate, not something we promise will succeed.

## Linguistic bridge and ambition

If a selective transfer pattern survives the factorial and dose controls, ask whether it tracks **semantic role across surface forms**. Use controlled active/passive and role-preserving reorderings, multiple question demands and unfamiliar names. The same relation should be realized at different token positions within each model. This is where the source/target microscope reconnects to the Linguistic Development Observatory.

Only then compare architectures using held-out transfer/intervention predictions rather than raw path cosine. If the controlled effect does not transfer to language, report that boundary; it cannot be rescued by assuming the process is hidden.

## Prior research and claim boundary

- [Dai et al.,2024](https://aclanthology.org/2024.emnlp-main.967/) already locate a low-rank binding-related subspace in pretrained LMs and manipulate bindings by editing it. Another binding-swap demonstration alone would not establish our contribution. The proposed question concerns acquisition under manipulated training comparisons.
- [Pezeshki et al.,2021, Gradient Starvation](https://papers.nips.cc/paper_files/paper/2021/hash/0987b8b338d6c90bbedd8631bc499221-Abstract.html) already formalize how predictive features can remain unlearned under cross-entropy dynamics. Better transfer after harder comparisons is compatible with this family of explanation.
- [Li et al.,2025, Feature Averaging](https://arxiv.org/abs/2410.10322) analyze feature combination/averaging and finer supervision in a two-layer setting. Identity separation after additional supervision is not sufficient novelty.
- [How Abstract Is Linguistic Generalization in Large Language Models?](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00608/118116/How-Abstract-Is-Linguistic-Generalization-in-Large) already studies generalization between argument contexts, including voice-related contexts. Its publisher page was403 during this check; the available primary search excerpt supports that scope, not a detailed full-text assessment.

This is a focused related-work check, not a completed novelty review. The possible contribution would require a predictive, causally validated account of **where a learned distinction transfers and where it fails**, beyond general identity discrimination, role-slot memorization, dose and generic difficulty. No publishable novelty is established by the proposal or today's archive recheck.

## Next concrete operation

Implement and exhaustively audit the factorial generator. Then freeze its rival predictions, effect criteria and resources. Training begins only if that preflight passes and the substantial inference design has been reviewed, as requested in the standing research instructions. Today's work supplies the candidate and a verified measurement correction; it does not claim a new mechanism.
