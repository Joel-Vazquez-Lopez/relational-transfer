# Legend-order diagnostic: completed32-case pilot

## Main finding: category-linked output, not a last-position policy

**NARROW:** Qwen emitted the code assigned to **not entailed on32/32 fresh cases**. This held for both sentence families, both legend orders and both A/B mappings. When not-entailed appeared first, it chose that first option on16/16 cases. The pure last-option prediction matched16/32 and failed the frozen qualification criterion. Correct relational answering was16/32, with0/8 complete four-answer units; the assay remains unready for causal interventions.

The model's output tracks the association between a legend key and its assigned code. This does **not** establish that it semantically judged every relation false or lacked relational information. A syntax-aware rule such as `return legend["not entailed"]`, ignoring the sentence, produces exactly the same observed choices as a negative-category prior. Those explanations remain indistinguishable here. The pilot resolves the proposed last-option rival at the behavioral level without identifying an internal mechanism.

The numerical compatibility control reproduced its old full output vector exactly (maximum error0). All33 collected payloads passed independent score, argmax, shape, hash and journal checks. Every wrong answer and full output vector is retained. Earlier failed gates and the larger assay's NO-GO decision remain unchanged.


## Fixed decision

Qualifying operational pattern: **negative_category**. Native first-token correctness:16/32; complete four-answer units:0/8. Causal-task competence gate (all32 correct): **False**.

|Prediction|Matches /32|Family l01 /16|Family l02 /16|Entailed-first /16|Nonentailed-first /16|Qualifies|
|---|---:|---:|---:|---:|---:|---|
|negative_category|32|16|16|16|16|True|
|last_option|16|8|8|16|0|False|
|relational_answer|16|8|8|8|8|False|

Qualification was frozen before outcomes:≥28/32 overall,≥14/16 in each family and each legend order. These are strict operational patterns, not fitted explanations or population tests. All32 cases were fresh, AI-authored/reviewed clauses. Only active source/question syntax and A/B codes were used.

## Every condition

|Family|Legend order|Mapping|Correct /4|Complete unit|
|---|---|---:|---:|---|
|l01|entailed_first|0|2|False|
|l01|entailed_first|1|2|False|
|l01|nonentailed_first|0|2|False|
|l01|nonentailed_first|1|2|False|
|l02|entailed_first|0|2|False|
|l02|entailed_first|1|2|False|
|l02|nonentailed_first|0|2|False|
|l02|nonentailed_first|1|2|False|

## First-token choices by legend order

|Legend order|Predicted entailed|Predicted not entailed|Off format|Correct /16|
|---|---:|---:|---:|---:|
|entailed_first|0|16|0|8|
|nonentailed_first|0|16|0|8|

## Interpretation and limits

Changing the legend order changes category recency and the last-listed symbol together. Following the last option supports order-sensitive output steering over a fixed negative-category preference in this panel; it does not uniquely identify literal token copying or a neural circuit. A negative-category result is an output-category preference, not proof the model evaluated the relation and rejected it. Correct relational answering is a behavioral criterion, not mechanistic identification. Mixed outcomes remain unclassified.

This32-case/two-family pilot has no population p value or novelty claim. Four-answer units and individual decisions are kept separate. The all32-correct competence prerequisite is independent of pattern qualification. No causal intervention or extension is authorized by a positive result here. Scores are first-token decisions, not complete generated-response accuracy.

All33 full output vectors, every wrong/off-format token, source manifests, model/runtime hashes, static controls and resource records are preserved. The one exposed numerical control must pass the inherited .05 full-vector tolerance before fresh cases. The audit independently verified every payload hash, shape, score, argmax and journal entry. The original32-case collection used batch1 and128 tokens, scored at the original assistant end; no answer conditioning or cache was carried between requests.

A connection to established prompt/answer bias ([Zhao et al.,2021](https://proceedings.mlr.press/v139/zhao21c.html)) remains plausible. This is not a new binding-ID subspace or intervention result of the kind studied by [Dai et al.,2024](https://aclanthology.org/2024.emnlp-main.967/). No comparison establishes novelty.

**This bounded pilot is complete. No additional inference or adaptive experiment is scheduled.**
