#!/usr/bin/env python3
"""Summarize all paired v2 starts after the frozen evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
cfg = json.loads((ROOT / "PROTOCOL.json").read_text())
protocol_hash = hashlib.sha256((ROOT / "PROTOCOL.json").read_bytes()).hexdigest()
out = {"protocol_sha256": protocol_hash, "seeds": {}, "screen": {}}
for seed in cfg["seeds"]:
    out["seeds"][str(seed)] = {}
    for arm in cfg["arms"]:
        eval_file = json.loads((ROOT / f"EVAL_{seed}_{arm}.json").read_text())
        train_file = json.loads((ROOT / f"TRAIN_{seed}_{arm}.json").read_text())
        assert eval_file["protocol_sha256"] == train_file["protocol_sha256"] == protocol_hash
        assert eval_file["checkpoint_sha256"] == train_file["checkpoint_sha256"]
        assert train_file["updates"] == 470
        out["seeds"][str(seed)][arm] = {}
        for split, rows in eval_file["records"].items():
            assert len(rows) == 400 and len({r["id"] for r in rows}) == 400
            out["seeds"][str(seed)][arm][split] = {
                "n": 400, "full_both": sum(r["both_correct"] for r in rows),
                "masked_both": sum(r["masked_both_correct"] for r in rows),
                "full_step1": sum(r["pred"][0] == r["gold"][0] for r in rows),
                "full_step2": sum(r["pred"][1] == r["gold"][1] for r in rows),
                "full_same_direction": sum(r["pred"][0] == r["pred"][1] for r in rows),
            }
    j = out["seeds"][str(seed)]["joint_replay"]
    d = out["seeds"][str(seed)]["clause_decomposition"]
    out["screen"][str(seed)] = {
        "in_domain_gate": j["in_domain_test"]["full_both"] >= 360 and d["in_domain_test"]["full_both"] >= 360,
        "mixed_gain": d["mixed_test"]["full_both"] - j["mixed_test"]["full_both"],
        "positive_seed": d["mixed_test"]["full_both"] - j["mixed_test"]["full_both"] >= 40
                         and d["in_domain_test"]["full_both"] >= j["in_domain_test"]["full_both"] - 10,
    }
out["screen"]["overall_pass"] = (all(out["screen"][str(s)]["in_domain_gate"] for s in cfg["seeds"])
                                   and sum(out["screen"][str(s)]["positive_seed"] for s in cfg["seeds"]) >= 4
                                   and sum(out["screen"][str(s)]["mixed_gain"] for s in cfg["seeds"]) > 0)
(ROOT / "ANALYSIS.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out["screen"], indent=2))
