# Role-general acquisition: design and archive-only measurement check

## Executed study outcome

**STOP_DEVELOPMENT_GATE:** six small-model development arms completed; no confirmation models ran. Start with [INTERPRETATION.md](INTERPRETATION.md) for the role-selective development pattern, the failed combined-arm training criterion, all limitations and preservation links. The [machine outcome](FACTORIAL_RESULTS.json), [raw error table](WRONG_ANSWERS.csv) and verified RAW_FACTORIAL archive parts preserve the full record.

Start with [DESIGN.md](DESIGN.md) for the original proposal. The authorized execution is specified in [FACTORIAL_PROTOCOL.md](FACTORIAL_PROTOCOL.md), with the generator checks in [PREFLIGHT.json](PREFLIGHT.json), pre-output hashes in FROZEN_LOCK.json, and collection/decision records produced by the bounded runner. The proposal itself is not an experimental result.

Reproduce today's saved-data calculation from the project root:

```sh
python3 research/role_general_acquisition_design_2026-10-20/archive_check.py
```

Outputs: [ARCHIVE_RECHECK.json](ARCHIVE_RECHECK.json), including exact input hashes and limitations, and [PAIRED_ENDPOINTS.csv](PAIRED_ENDPOINTS.csv).

That archive-only calculation uses no model package, checkpoint, paid inference or training. Its source CSVs and scripts belong to the older adaptive discovery archive. Repeated source rows are not independent replications. The subsequently authorized factorial is a separate small-model training experiment under the frozen execution protocol. The historical measurement clarification is attached to the [complete experiment audit](../complete_experiment_audit_2026-10-20/ADDENDUM_METRIC_DEFINITION.md).
