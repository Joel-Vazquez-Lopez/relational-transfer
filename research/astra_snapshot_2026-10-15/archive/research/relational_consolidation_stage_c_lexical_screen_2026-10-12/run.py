#!/usr/bin/env python3
"""V2 lexical-composition screen using the frozen V1 runner without editing it."""
from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
PRIOR = (HERE / "../relational_consolidation_stage_c_exploratory_2026-10-12").resolve()
sys.path.insert(0, str(PRIOR))
import train as base  # noqa: E402

cfg = json.loads((HERE / "PROTOCOL.json").read_text())
base.ROOT = HERE
original_bundle = base.data_bundle


def restricted_bundle() -> dict:
    bundle = original_bundle()
    s2_file = (base.BENCH / "generated/family_00/S2_TRAIN_DERIVATIONS.jsonl")
    pairs = base.read_jsonl(s2_file)
    assert len(bundle["transformed"]) == 2 * len(pairs)
    allowed = set(cfg["transformed_train_verbs"])
    kept = []
    for i, pair in enumerate(pairs):
        verb = pair["source_event_id"].rsplit("-", 1)[-1]
        if verb in allowed:
            kept.extend(bundle["transformed"][2*i:2*i+2])
    assert len(kept) == 1728
    bundle["transformed"] = kept
    target = cfg["primary_development_verb"]
    bundle["dev"] = [row for row in bundle["dev"] if row[0].rsplit("-", 1)[-1] == target]
    assert len(bundle["dev"]) == 300
    return bundle


base.data_bundle = restricted_bundle

if __name__ == "__main__":
    base.run(cfg["seeds"][0])
