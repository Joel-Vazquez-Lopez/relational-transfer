# Authentic-sentence transfer check

This evaluates the **existing** Qwen2.5-0.5B-Instruct adapters on selected QA-SRL Gold sentences. It makes no model updates. The comparison uses the same 63 events and 504 prompts for every arm: actor/patient question, two question wordings, and both answer-option orders. There are six matched graph/start units (three worlds × two seeds) for ordinary, ordered-role, and participant-only training, plus one untouched pretrained baseline.

The items come from the [QA-SRL Gold Standard](https://github.com/plroit/qasrl-gs) already copied into this project. The exact 63 items were previously used in a separate Qwen 1.5B pilot, so this is **exploratory**, not a blind confirmation. Pretraining overlap is unknown. The item selection and source annotations need independent human review before making a paper-level claim. The review sheet is [HUMAN_REVIEW.csv](HUMAN_REVIEW.csv); every `include_yes_no` cell is intentionally blank. Exclude or correct items only through a new documented dataset version, not by silently changing this frozen evaluation after results are seen.

## Frozen files

- [PROTOCOL.json](PROTOCOL.json): question, scoring, interpretation, source hashes, and caveats.
- [ITEMS.json](ITEMS.json): exact prompt rows, including original sentence, target verb, both answer spans, labels, wording, and option order.
- [HUMAN_REVIEW.csv](HUMAN_REVIEW.csv): one row per source event for independent label and suitability review.

## Run

The project has a working Python environment at `/Users/joelvazquezlopez/Documents/Codex/2026-08-19/wha/work/pilot-venv/bin/python`. The default `python3` on this host lacks PyTorch.

From `/Users/joelvazquezlopez/Documents/ChatGPT/project null`:

```sh
/Users/joelvazquezlopez/Documents/Codex/2026-08-19/wha/work/pilot-venv/bin/python llm_role_authentic_transfer.py validate
/Users/joelvazquezlopez/Documents/Codex/2026-08-19/wha/work/pilot-venv/bin/python llm_role_authentic_transfer.py batch
```

`batch` resumes by skipping completed runs and then writes `ANALYSIS.json`. Individual runs use `run --world fresh1 --seed 830211 --arm ordinary`, with arms `ordinary`, `ordered_roles`, and `unordered_pair`; the untouched baseline uses `run --arm pretrained`. `--limit 8` makes a nonreportable smoke result. Full results are stored by world/start/arm with each prompt's first-token answer, validity, and correctness. Analysis reports primary actor accuracy, patient accuracy, all-eight-prompt event reliability, wording/order cells, and matched differences against ordinary training. Repeated prompts for the same event are not independent observations.

## Interpretation

A gain here would show a change on this selected natural-sentence *multiple-choice* task. It would not establish that the model learned an internal role representation, that the result applies to all natural language, or that the sentences were absent from pretraining. A null result could reflect the move from the synthetic name-output task to longer natural sentences and A/B answers. Report both directions and invalid outputs, not only overall accuracy.
