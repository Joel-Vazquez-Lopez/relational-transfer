# Mira — Source rights and ethical reuse decision, 16 October 2026

**Decision: EWT original-text rights remain unresolved for ALL 49 selected source documents.** This is a reference-only audit; none has been granted publication, private human-annotation, or third-party inference permission by this document.

## Confirmed facts, with primary sources
- UD English EWT README at pinned source commit `4a4d77f599ea53cc405f85d0cec4b2f14f81d42b` states the **annotations and database** are CC BY-SA 4.0 but the **underlying text** was collected from multiple parties and may belong to Google, Yahoo!, the University of Pennsylvania and/or original authors. The UD treebank webpage's headline CC license does not erase this explicit distinction: https://github.com/UniversalDependencies/UD_English-EWT/blob/4a4d77f599ea53cc405f85d0cec4b2f14f81d42b/README.md
- LDC2012T13 is the source of the EWT raw text; checking an upstream GitHub sentence is not the same as receiving an individual copyright/license grant: https://catalog.ldc.upenn.edu/LDC2012T13
- Most English Wikipedia text is available under CC BY-SA 4.0. Reuse normally requires attribution, link to article/history, license notice and ShareAlike for adaptations; page-specific imported copyrighted excerpts and other rights can be exceptions. The official Wikimedia reuse instructions require checking the actual page and license: https://www.mediawiki.org/wiki/Wikimedia_APIs/Content_reuse and https://foundation.wikimedia.org/wiki/Terms_of_Use
- The Creative Commons BY-SA 4.0 deed confirms reuse/adaptation is permitted **subject to** attribution, ShareAlike, indication of changes and other rights: https://creativecommons.org/licenses/by-sa/4.0/
- U.S. government employee-authored works are generally outside U.S. copyright protection under 17 USC §105, but this does not automatically include contractor/third-party materials, rights in other jurisdictions or personal information: https://www.govinfo.gov/content/pkg/USCODE-2024-title17/html/USCODE-2024-title17-chap1-sec105.htm

## Decision matrix (NOT legal advice)

| Source option | Native human-authored language | Rights predictability | Task fit | Decision |
| --- | --- | --- | --- | --- |
| **EWT dev, 49 referenced documents** | Yes | Mixed third-party rights, unresolved for every document | Good authentic informal genres and UD-pinned predicate anchors; sample heuristically biased | Retain IDs-only open-development feasibility **BLOCK RAW TEXT RELEASE/MODEL INPUT** until individual document-use review |
| **English Wikipedia source revisions** | Yes (except generated/imported portions require scrutiny) | Usually explicitly CC BY-SA 4.0 with page history and attribution; per-revision exceptions remain | Multi-sentence coreference and recurrent predicates possible, but encyclopedic register and memorization risks | **Preferred parallel candidate source** for a redistribution-capable review, AFTER selecting and rights-checking exact revisions |
| **US government authored text, e.g. agency releases** | Yes | Generally federal-employee text not copyrighted in USA; exceptions for contractors/third parties and outside-US rules | Scope, agents and events; formal register, limited everyday language | Secondary candidate; retain provenance and jurisdictional review |
| **UD English GUM** | Yes | CC BY-NC-SA restrictions and underlying-source provenance | Richer genres/coreference | Defer absent explicit compatibility with intended release/use |
| **QA-SRL Bank** | Underlying authentic sentences; role questions crowd-written | Need sentence-source/license check | Good role-span candidate seed | Potential triangulation dataset, not automatic gold agent/patient |

## Rights workflow
The accompanying `source_rights_triage_49.jsonl` has 49 distinct source-document rows, **all statuses `PENDING_DOCUMENT_SPECIFIC_REVIEW`**. For each selected excerpt the rights reviewer must record: original rights holder/license, primary evidence URL and date, allowed internal annotation, allowed on-device processing vs remote provider processing, allowed public text redistribution, attributable contributors, sensitive/person-identifying content, necessary exclusions. Unknown must remain UNKNOWN, not silently become YES.

**Default before review:** don't commit source text to the user's research GitHub, don't submit source text to hosted LLM APIs, don't distribute a text-containing artifact publicly, don't describe the panel as licensed for reuse. Source-pinned IDs and empty annotation forms are permitted without reproducing the underlying paragraphs.

## Pragmatic way forward
Begin with two or three revision-pinned Wikipedia paragraphs, each from a different article, chosen only after checking page copyright/import indicators, attribution, no copyrighted quotes and any biographical privacy concerns. Record article title, page ID, revision ID, text fragment offsets, revision timestamp, source license/attribution, and whether text was modified. **Do not claim unseen/sealed confirmation**; any viewed paragraph is development. Compare the annotation process before replacing EWT entirely.

No specific Wikipedia article has been selected, cleared or annotated in this note. A legal or institutional rights review may still be appropriate for public redistribution or commercial reuse.
