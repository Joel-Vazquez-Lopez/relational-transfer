# COMPUT-022 — Locate existing human directional gold: SemEval-2010 Task 8

**16 October 2026. Status: independent upstream description review and exploratory protocol proposal; NOT a full corpus-byte cold replay or pretrained inference.**

## Critical source result

Official original Task 8 data description (preserved in third-party mirror `nb20593/Semeval-2010-task-8-Relational-extraction`, file `Sem eval 2010 task 8 Description.txt`) explicitly says the accompanied 8,000 training sentence dataset was released under **Creative Commons Attribution 3.0 Unported**. The same description lists **nine** semantic relations: Cause-Effect, Instrument-Agency, Product-Producer, Content-Container, Entity-Origin, Entity-Destination, Component-Whole, Member-Collection, Message-Topic. Labels distinguish `Rel(e1,e2)` and `Rel(e2,e1)`, plus `Other` = **19 possible labels**, with entity markers `<e1>` and `<e2>` in sentence order. e1/e2 order is surface order and must NOT be confused with semantic direction. Original evaluation uses accuracy and macro-F1 over 18 oriented relation classes excluding `Other`.

A binary `reverse` task can be scored *only conditional on correctly recognizing relation type*: predictions require the full 19-way class, otherwise a direction-only score can conceal type mismatch. The original test set is a long-public benchmark; **never call it unseen by pretrained weights**. Reuse the 8,000 released training examples solely as *development* for source-locked questions if source-rights allowed; reserve any genuine confirmation for independently sourced, disjoint human gold.

## Why important / research scope

QA-SRL Gold is original *human-written predicate-conditioned question → accepted answer spans* (7,188 dev questions, 1,000 gold-annotated sentences); does not automatically provide binary actor/patient direction labels. SemEval Task8 gives authentic annotated relation **direction** as well as relation type, directly complementing QA-SRL and Compute-018's synthetic 50% undirected ceiling. It is NOT event actor/patient supervision or a proof of a pretrained LLM relational mechanism; syntax, marked-entity order, lexical memorization and pretraining benchmark exposure can explain performance.

**Not a source-verified corpus result yet:** GitHub `JoelNiklaus/SemEval2010Task8` tree lists `SemEval2010_task8_all_data.zip` (1,964,087 bytes), but this session's shell cannot retrieve external raw bytes. The original 8k count/license are based on upstream dataset DESCRIPTION checked, not a run against the binary. Verify official archive bytes and licenses before any model-facing use.

## Necessary adversarial controls / minimal experiment (do not duplicate Nova)

1. For every released training record, verify exactly two tagged entity spans, one of 19 classes, relation type and orientation, comment isolation, original record identity and licensing provenance. Do NOT use human comment as model prompt.
2. Check relation×orientation counts, lexical e1/e2 order rules, entity-string cooccurrence / memorized facts, mention-distance / sentence position, class priors and reverse-direction balance. Predeclare grouped source-based splits where source document lineage exists; avoid reporting examples as independent just because text differs.
3. Two untouched pretrained **model families** (candidate Qwen2.5-7B and Phi-3-mini) receive only sentence+marked entities, strict 19-label answer. Track exact type×direction classification, confusion within each directed pair, output validity and family-balanced denominator. Baselines: majority class; type-only; choose e1->e2 for all; syntactic pattern (e.g. `X caused Y`); candidate-pair lexical memorization. Per-direction causal counterfactual prompt swaps would be synthetic *development* only and need separate human applicability verification.
4. Integrate with Nova's runner only after explicit model processing permission and exact checkpoint/budget checks. Nova owns model inference; Comput owns baseline and shortcut audit; Mira owns interpretation of QA-SRL / authentic source. Do NOT replace QA-SRL P0 while discussing SemEval as independent complementary positive-directionality diagnostic.

## Scientific stop/go

**Go** for locating *human-labeled directional relation development gold* without requiring new annotation. **Do not go** to model claims until exact original archive is independently downloaded+validated, source use documented, prompts frozen and non-oracle baselines validated. Even perfect 19-way classification on 2010 data would be competence evidence only, not a neural relational circuit.

**Artifact status:** text/doc-only investigation. No new inference, model downloads, spending, copy of restricted source sentences or merge.