# Prospective early-signal test (locked before training)

Date: 2026-10-06. This test follows the exploratory PATH2 result. Its result is
new evidence, but the hypothesis was chosen after viewing the older 18 models.

## Question

Does separation between the two focal entities in a model's query vectors at
step 1000 predict the final held-pair role margin at step 6000? This is a
predictive question, not proof that the separation causes generalization.

## Design

Five newly selected, nonisomorphic, connected 10-node, 4-regular graphs. Each
has one nonedge with one common neighbor and one with two. The four focal
endpoints have matched triangle counts within each graph. The direct focal
edge is always absent. Both conditions have the same 1280 training meanings,
episodes, candidate positions, optimizer, model initialization, and steps up
to an identity relabeling. Each condition uses paired seeds 10401, 10402,
10403. The graphs were selected from deterministic random-regular-graph
generation by the first five qualifying seeds (2, 3, 5, 6, 8), before model
training. No model outcome was used to pick them.

Model: the existing 2-layer Transformer listener with 48-dimensional hidden
vectors. Training: 6000 steps, batch 128, AdamW at 0.0008. The only early
checkpoint used in the primary test is step 1000. No held-out scene is
evaluated before final training. The early signal is 1 minus cosine similarity
between query vectors differing only in focal identity 0 versus 1, averaged
over both directions, other entities, and event types, as defined by
`run.py:query_separation`. It uses unlabeled query inputs, including some
combinations absent from training; it does not use final held-scene answers.

## Outcomes and analysis

The primary outcome is the step-6000 mean correct-minus-reversed-role score
on 1024 fresh held scenes. The score is in [-20, 20] because each candidate
score is 10 times a cosine similarity. Positive is better role discrimination.
The gate is familiar accuracy >= 0.95 for all 30 models. If it fails, report
the models and do not interpret transfer differences as learned transfer.

Primary analysis is the Spearman association between the paired condition
difference in step-1000 query separation and the paired condition difference
in final held margin, across 15 world-seed pairs. Report the raw observations,
Spearman rho, an exact/Monte Carlo within-world permutation p value, and
leave-one-world-out rho. Report the 5 separate world means as well, since
seeds within a graph are not independent graph replications. A positive rho
with consistent held-world behavior would motivate a larger replication; a
near-zero or reversing association rejects this predictor in this setting.
No claim about a general graph law follows from this small sample.

Secondary comparisons (descriptive only): initial query separation, step-1000
training loss, step-1000 entity-embedding cosine distance, and final query
separation. Step-3000 measurements are exploratory. Report familiar accuracy,
held accuracy, and the final margin by condition, but do not select a preferred
condition after seeing outcomes. No prompt-based LLM or natural-language
result follows from this study.

The new graph degree differs from the earlier 3-regular PATH2 study. This is
a deliberate architecture-preserving transport test; it also means a failed
association could be specific to this new data regime.
