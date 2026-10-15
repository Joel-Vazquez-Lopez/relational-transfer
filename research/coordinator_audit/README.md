# Independent frozen adapter rescore (coordinator)

This workstream responds to the missing **independent event-level numerical verification** gap without retraining the model or changing any frozen protocol.

## Input
Obtain the original unsealed directory `research/astra_snapshot_2026-10-15/completed_adapter_diagnostic/` from Astra branch commit `7d4a807d467da637a93369df8542104082d855f3`; preserve `FROZEN_ITEMS.json` and its `full_outputs/` directory, all 21 unit JSON files. The coordinator audit does **not** include these raw files or protected datasets in its own branch.

## Run
```bash
python3 research/coordinator_audit/independent_adapter_rescore.py /path/to/completed_adapter_diagnostic --output independent_audit.json
python3 -m unittest discover -s research/coordinator_audit -p 'test_*.py' -v
```

No dependencies outside Python 3 standard library; no network, inference, or model weights.

## Validation
During this chat, the coordinator **independently rescored all 21 units from GitHub source JSON** using a separate JavaScript implementation, split into three world batches because of connector limits. All 21 files returned 96 records; actor, patient, joint, archived-actor and input-last-person metrics matched the reported primary patient-cleft table when pooled by baseline/adapter arm.

The checked patient-cleft outcomes, in order (untouched/ordinary/ordered/participant), were:
- Archived actor: 81.25% / 62.50% / 77.08% / 89.06%.
- Direct actor: 82.29% / 69.79% / 63.54% / 72.92%.
- Joint extraction both correct: 90.625% / 85.94% / 84.90% / 85.94%.

**Important distinction:** the Python script is an independent, portable implementation written after the JS audit; its unit tests were included for execution but **have not been executed in the coordinator runtime or CI**. Do not claim fully independent code execution or model replication until it has been run with exact pinned inputs and logs independently checked.

## Limits
- Uses the archived `parsed` objects, not a wholly independent semantic parser of `raw_output`.
- Verifies descriptive denominators and scores, **not** exact bootstrap CIs.
- All name-worlds are synthetic, same underlying per-world test families recur across arms, no new pretrained inference or causal interventions.
- Extraction position agreement is an observed behavior, **not** a discovery of an internal pointer circuit.
- The entire original Astra snapshot still requires its own safety/provenance screen before merge.

Next step: independent re-parsing of raw outputs, exact per-pair transitions, partner-review of cluster uncertainty, and an output-field order / verb-specific question factorial with preregistered surface baselines. Keep the design/frozen evidence separate.
