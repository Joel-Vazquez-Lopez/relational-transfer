# Role-general acquisition: design and archive-only measurement check

Start with [DESIGN.md](DESIGN.md). This is a candidate investigation, not a frozen protocol or a completed new training study.

Reproduce today's saved-data calculation from the project root:

```sh
python3 research/role_general_acquisition_design_2026-10-20/archive_check.py
```

Outputs: [ARCHIVE_RECHECK.json](ARCHIVE_RECHECK.json), including exact input hashes and limitations, and [PAIRED_ENDPOINTS.csv](PAIRED_ENDPOINTS.csv).

No model package, checkpoint, paid inference or training is used. The source CSVs and scripts belong to the older adaptive discovery archive. Repeated source rows are not independent replications. The measurement clarification is also attached to the [complete experiment audit](../complete_experiment_audit_2026-10-20/ADDENDUM_METRIC_DEFINITION.md).
