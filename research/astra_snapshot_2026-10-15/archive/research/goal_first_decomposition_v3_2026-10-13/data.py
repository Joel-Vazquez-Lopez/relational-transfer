#!/usr/bin/env python3
"""Fresh split for matched operation-decomposition training; no old test reuse."""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / "goal_first_decomposition_pilot_2026-10-13"
V2 = ROOT.parent / "goal_first_decomposition_v2_2026-10-13"
sys.path.insert(0, str(OLD))
import benchmark as source  # noqa: E402

SPLITS = {"train": (6000, ("FF", "RR"), 951),
          "development": (400, ("FF", "RR"), 952),
          "in_domain_test": (400, ("FF", "RR"), 953),
          "mixed_test": (400, ("FR", "RF"), 954)}


def make() -> tuple[dict, dict]:
    old = json.loads((OLD / "DATA.json").read_text())
    v2 = json.loads((V2 / "DATA.json").read_text())
    seen = {source.signature(row) for bank in (old, v2)
            for rows in bank.values() for row in rows}
    prior_count = len(seen)
    out = {}
    for split, (count, patterns, seed) in SPLITS.items():
        rng = random.Random(seed)
        rows = []
        while len(rows) < count:
            row = source.generate_one(rng, patterns[len(rows) % len(patterns)])
            sig = source.signature(row)
            if sig in seen:
                continue
            assert source.solve(row) == row["answer"]
            seen.add(sig)
            row["id"] = f"v3-{split}-{len(rows):04d}"
            rows.append(row)
        out[split] = rows
    assert len(seen) == prior_count + sum(len(rows) for rows in out.values())
    assert {r["pattern"] for r in out["train"]} == {"FF", "RR"}
    assert {r["pattern"] for r in out["mixed_test"]} == {"FR", "RF"}
    audit = {split: {"n": len(rows),
                     "patterns": dict(Counter(r["pattern"] for r in rows)),
                     "answer_in_question": sum(r["answer"] in r["question"].split() for r in rows),
                     "first_fact_actor_correct": sum(r["answer"] == r["facts"][0][0] for r in rows),
                     "first_fact_patient_correct": sum(r["answer"] == r["facts"][0][1] for r in rows)}
             for split, rows in out.items()}
    return out, {"old_signatures_excluded": prior_count, "splits": audit}


if __name__ == "__main__":
    if (ROOT / "DATA.json").exists():
        raise FileExistsError(ROOT / "DATA.json")
    data, audit = make()
    (ROOT / "DATA.json").write_text(json.dumps(data, indent=2) + "\n")
    (ROOT / "DATA_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
