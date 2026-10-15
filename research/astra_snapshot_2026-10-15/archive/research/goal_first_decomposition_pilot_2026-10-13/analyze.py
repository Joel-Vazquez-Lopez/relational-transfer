#!/usr/bin/env python3
"""Descriptive analysis of both saved feasibility runs; no checkpoint selection."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "DATA.json").read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def correct_ops(output, row, arm):
    if arm == "direct":
        return None
    v1, v2 = row["verbs"]
    d1, d2 = row["operations"]
    try:
        if arm == "forward":
            return output[2] == v1 and output[3] == d1 and output[6] == v2 and output[7] == d2
        return output[2] == v2 and output[3] == d2 and output[5] == v1 and output[6] == d1
    except IndexError:
        return False


def correct_first_operation(output, row, arm):
    if arm == "direct":
        return None
    v1, v2 = row["verbs"]
    d1, d2 = row["operations"]
    try:
        if arm == "forward":
            return output[2] == v1 and output[3] == d1
        return output[2] == v2 and output[3] == d2
    except IndexError:
        return False


out = {"frozen_first": {}, "adaptive_capacity": {}, "paired": {}}
for phase, prefix, protocol in (("frozen_first", "EVAL_", "PROTOCOL.json"),
                                ("adaptive_capacity", "CAP_EVAL_", "CAPACITY_PROTOCOL.json")):
    for arm in ("direct", "forward", "goal_first"):
        data = json.loads((ROOT / f"{prefix}{arm}.json").read_text())
        assert data["protocol_sha256"] == sha(ROOT / protocol)
        out[phase][arm] = {}
        for split, rows in data["records"].items():
            originals = {r["id"]: r for r in DATA[split]}
            assert len(rows) == len(originals)
            assert {r["id"] for r in rows} == set(originals)
            metrics = {"n": len(rows),
                       "answer_correct": sum(r["correct"] for r in rows),
                       "invalid_answer": sum(r["answer"] is None for r in rows),
                       "by_pattern": dict(Counter({p: sum(r["correct"] for r in rows if r["pattern"] == p)
                                                    for p in {r["pattern"] for r in rows}}))}
            if arm != "direct":
                metrics.update({
                    "full_plan_valid": sum(r["plan_valid"] for r in rows),
                    "both_operations_correct": sum(correct_ops(r["output"], originals[r["id"]], arm)
                                                   for r in rows),
                    "first_emitted_operation_correct": sum(correct_first_operation(r["output"], originals[r["id"]], arm)
                                                           for r in rows),
                })
            out[phase][arm][split] = metrics
    for split in ("development", "in_domain_test", "heldout_mixed"):
        a = json.loads((ROOT / f"{prefix}forward.json").read_text())["records"][split]
        b = json.loads((ROOT / f"{prefix}goal_first.json").read_text())["records"][split]
        by = {r["id"]: r for r in b}
        assert len(a) == len(by)
        out["paired"][f"{phase}/{split}"] = {
            "goal_only_correct": sum(not r["correct"] and by[r["id"]]["correct"] for r in a),
            "forward_only_correct": sum(r["correct"] and not by[r["id"]]["correct"] for r in a),
        }
(ROOT / "ANALYSIS.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
