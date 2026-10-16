# Mira authentic-language candidate pilot — 16 October 2026

**Status:** selected, source-pinned and internally validated references; **0/25 independently human-annotated semantic-role items**, **0** adjudications, **0** model runs. This is an open-development feasibility pilot, **not** a finished benchmark.

## Scope
- **25 sentences in 25 distinct source documents**, stratified as 5 each from **answers, email, newsgroup, reviews and weblog**. Original text remains at the upstream source; no text excerpts are committed here.
- Source: [UD English Web Treebank](https://github.com/UniversalDependencies/UD_English-EWT), `en_ewt-ud-dev.conllu`, pinned commit `4a4d77f599ea53cc405f85d0cec4b2f14f81d42b`, [exact file](https://github.com/UniversalDependencies/UD_English-EWT/blob/4a4d77f599ea53cc405f85d0cec4b2f14f81d42b/en_ewt-ud-dev.conllu).
- Seed selection rule: each sentence has a `VERB` root with one UD `nsubj` and one UD `obj`, 55–180 source-text characters, plus initial content filtering; one candidate per document. **This syntactic filter is not semantic gold and excludes many complex/passive constructions.** These are candidates to discover annotation difficulty, not a representative benchmark.
- `candidate_manifest.jsonl` contains exact IDs, verb root surface/lemma and token ID, but **no text or actor/patient answers**. Annotation templates contain **neither UD heads nor model outputs**, to avoid bias.
- Files `annotator_A_template.csv`, `annotator_B_template.csv`, `ANNOTATION_PROTOCOL.md`, `SOURCE_AUDIT.md`, and `SELECTION_INTEGRITY.json` are included.

## Rights, ethics and release block
The [upstream README](https://github.com/UniversalDependencies/UD_English-EWT/blob/4a4d77f599ea53cc405f85d0cec4b2f14f81d42b/README.md) licenses *annotations/database rights* under CC BY-SA 4.0, while identifying multiple third-party copyrights in the underlying text. **Do not assume all original sentences are redistributable under the annotation license.** Each document needs a specific source/rights decision before excerpt circulation, benchmark publication, or model testing. Maintain restricted references and attribution. Privacy and sensitive details require review; source IDs alone do not establish suitability.

## Prior authentic-pilot erratum
The earlier 20-row pilot had **19/20 exact text–source-ID matches** when checked against the pinned original. Old `EWTDEV-011` was joined to the wrong sentence: its source text is `...-0005` rather than recorded `...-0007`. **Quarantine the original** and create a separately versioned corrected row only after source/role review; don't silently relabel historical data.

## Humans needed
Two independent human reviewers must label full agent/undergoer spans *if semantically applicable*, predicate/event scope, ambiguity, coreference and non-agent subjects. Humans must annotate without seeing each other's choices or candidate syntax heads. A third adjudicator records disagreements and genuinely unresolved cases. Current files are **blank**; AI checking does not meet the two-human requirement.

## Confirmation boundary and next gate
- All 25 candidate sentences were inspected during selection. **No frozen/held-out claim** can be made using them.
- Reserve future **unexamined** documents, ideally from different document/source groups; predeclare rights, document split, source lineage, gold construction, and group-aware scoring.
- Then, and only after human adjudication/licensing, plan small two-**distinct**-family pretrained LLM format/competence smoke with Nova and shortcut baselines with Comput. Do not expose source IDs, UD annotations, reviewer fields or gold to model prompts.
- See [external validity issue #23](https://github.com/Joel-Vazquez-Lopez/relational-transfer/issues/23).
