# Proposed matched adapter readout diagnostic (development only)

**Status:** proposal for inspection, not frozen. The four-arm smoke run is a format and load preflight, not evidence for a behavioral or mechanistic claim.

## Question

When ordinary answer fine-tuning causes a relational regression, does an explicit actor/patient extraction request show the same errors as the archived direct actor question? Compare the untouched Qwen2.5-0.5B-Instruct base with matched ordinary, participant-only, and ordered-role adapters. This is a behavioral interface comparison. It cannot establish what representation the model stores or where the regression occurs.

## Proposed full development panel

- Use the three existing name worlds and both existing seeds. For each world, compare one untouched base and all three adapters at each seed: 3 base evaluations plus 18 trained evaluations, with matched same-world/same-seed comparisons.
- Construct eight held-out unordered name pairs per world from the saved graph manifest, selected deterministically before model inference. Use both actor directions, two new reversible verbs, and active, passive, and patient-fronted cleft renderings: 8 × 2 × 2 × 3 = 96 sentences per world. These are newly authored synthetic development sentences; no historical sealed tests, review labels, or natural frontier sets are read.
- Apply two fixed interfaces to every sentence: (1) the archived direct QA chat prompt with “Who performed the action?” and first-token exact-name scoring; (2) a single explicit actor/patient extraction chat prompt with deterministic generation and strict parsing of `Actor: NAME; Patient: NAME`. Preserve full prompt, raw output, and parsed fields separately. Do not tune prompts against the observed accuracy.
- Total: 96 sentences × 7 model evaluations per world × 3 worlds = 2,016 sentence-model comparisons, or 4,032 interface prompts. Analyze pair-level, world-level, and seed-level matched differences, invalid-format rates, and extraction-versus-QA discordance. Treat repeated sentences and shared graph seeds as dependent; do not claim prompt-level independent replication.
- Primary descriptive contrast: for patient-fronted clefts, compare each trained arm’s direct QA and explicit extraction actor accuracy against the untouched base and against its same-start ordinary arm. Retention on active/passive sentences and full actor/patient extraction accuracy are required companion outcomes. Report every item, including invalid output.

The smoke revealed frequent trailing periods after the patient name. Before a full run, specify whether the semantic parser accepts one optional final period while recording that deviation from the requested format, or treats it as invalid. The raw output and a separate exact-format flag should be retained either way. This is a format decision, not a prompt change or accuracy-driven selection.

## Gates and limits

Before a full run, freeze exact pair IDs, two verbs, sentence strings, prompts, parser, model/checkpoint hashes, generation length, and analysis code. Technical smoke outputs are excluded from the full panel. The panel would still be one model family with synthetic templates and cannot establish an internal causal mechanism. Existing 1.5B whole-state patch scripts use a different model and are not part of this proposed run.
