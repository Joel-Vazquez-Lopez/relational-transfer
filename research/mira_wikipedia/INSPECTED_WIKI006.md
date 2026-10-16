# First directly inspected historical revision — WIKI-006 Marie Curie

**Review status:** Source subsection visually inspected in **actual pinned rendered historical revision** at https://en.wikipedia.org/w/index.php?oldid=1379969707&title=Marie_Curie . This is the **only** one of the seven initial candidates for which the historical revision successfully rendered in the browser on 16 Oct 2026.

## Candidate location
- Article `Marie Curie`, page ID `20408`, revision ID `1379969707`.
- Section `Life and career → New elements`, paragraph beginning with the year **1895**.
- Two separate uses of the discovery predicate are visible nearby in the same paragraph: **1895 Wilhelm Röntgen/X-rays** versus **1896 Henri Becquerel/uranium rays**. These distinguish **potentially different discovery events** with temporal and object cues. They are *not* annotated actor/patient slots and are NOT human semantic gold.
- This is a strong **format/annotation-pipeline feasibility example** but a weak paper-level heldout because these discoveries and names are common factual knowledge and likely familiar to pretrained LMs.
- No source paragraph copied to this repository; Wikipedia authorship attribution is through the pinned page and its history. No claim that the relevant paragraph is free from imported quotations has been certified yet.
- HTML-rendered subsection presence was inspected; **raw original wikitext hash, exact paragraph byte offsets, copyright/import exception review and two-human semantic labels remain PENDING**.

## Failure mode to avoid
If a model correctly answers year/object-specific direct questions, it might retrieve encyclopedic world knowledge rather than bind the correct event from the supplied passage. Use no claim of *in-context relational extraction* unless context-conditioned alternatives and competent source-copying and memory baselines are measured. For future genuinely novel event-selection claims, select less culturally ubiquitous and independently human-adjudicated sources after rights checks.

**No model inference run.** Do not put even tentative agent/patient gold into model evaluation as verified labels.
