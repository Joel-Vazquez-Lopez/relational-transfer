#!/usr/bin/env python3
"""Analyze every cell of the frozen question-role diagnostic."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
cfg = json.loads((ROOT / "PROTOCOL.json").read_text())
items = json.loads((ROOT / "ITEMS.json").read_text())
ids = {item["id"] for item in items}
assert len(ids) == len(items) == 40


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(value):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (value or "").casefold())).strip()


def summarize(start, arm, condition):
    path = ROOT / f"EVAL_{start}_{arm}_{condition}.json"
    data = json.loads(path.read_text())
    assert data["protocol_sha256"] == sha(ROOT / "PROTOCOL.json")
    assert data["checkpoint_sha256"] == cfg["checkpoint_sha256"][f"{start}:{arm}"]
    records = data["records"]
    by = {(r["id"], r["role"], r["order"]): r for r in records}
    assert len(records) == len(by) == 160
    assert {r["id"] for r in records} == ids
    exact = sum(bool(r["correct"]) for r in records)
    joint = sum(all(by[(id_, role, order)]["correct"]
                    for role in ("actor", "patient") for order in (0, 1)) for id_ in ids)
    result = {"exact_answers": exact, "four_correct": joint,
              "parsed": sum(bool(r["parsed"]) for r in records),
              "by_role_order": {f"{role}/order{order}": sum(by[(id_, role, order)]["correct"] for id_ in ids)
                                for role in ("actor", "patient") for order in (0, 1)},
              "position": {"first": 0, "second": 0, "other": 0}}
    for r in records:
        answer = norm(r["answer"])
        if answer == norm(r["candidate_pair"][0]):
            result["position"]["first"] += 1
        elif answer == norm(r["candidate_pair"][1]):
            result["position"]["second"] += 1
        else:
            result["position"]["other"] += 1
    if arm == "role_first":
        routed = sum(bool(r["routed_correct"]) for r in records)
        result.update({"routed_exact": routed,
                       "routed_four_correct": sum(all(by[(id_, role, order)]["routed_correct"]
                                                  for role in ("actor", "patient") for order in (0, 1))
                                                  for id_ in ids),
                       "actor_field_correct": sum(bool(r["actor_correct"]) for r in records),
                       "patient_field_correct": sum(bool(r["patient_correct"]) for r in records)})
        same_answer = stable_fields = 0
        for id_ in ids:
            for order in (0, 1):
                a, p = by[(id_, "actor", order)], by[(id_, "patient", order)]
                same_answer += norm(a["answer"]) == norm(p["answer"])
                stable_fields += norm(a["actor"]) == norm(p["actor"]) and norm(a["patient"]) == norm(p["patient"])
        result["same_answer_actor_patient_question_pairs"] = same_answer
        result["stable_fields_across_question_pairs"] = stable_fields
        pair = Counter()
        for r in records:
            pair[f"route_{int(bool(r['routed_correct']))}_direct_{int(bool(r['correct']))}"] += 1
        result["routed_direct_pair_table"] = dict(pair)
    return result, by


out = {"protocol_sha256": sha(ROOT / "PROTOCOL.json"), "cells": {}, "paired": {}, "screen": {}}
all_by = {}
for start in cfg["starts"]:
    for arm in cfg["arms"]:
        for condition in cfg["conditions"]:
            key = f"{start}/{arm}/{condition}"
            out["cells"][key], all_by[key] = summarize(start, arm, condition)
for start in cfg["starts"]:
    for arm in cfg["arms"]:
        base = all_by[f"{start}/{arm}/ordinary"]
        tagged = all_by[f"{start}/{arm}/tagged"]
        rescue = harm = 0
        for key, a in base.items():
            b = tagged[key]
            rescue += not a["correct"] and b["correct"]
            harm += a["correct"] and not b["correct"]
        out["paired"][f"{start}/{arm}/tagged-vs-ordinary"] = {
            "tag_rescues": rescue, "tag_harms": harm, "net": rescue - harm}
    tagged = out["cells"][f"{start}/role_first/tagged"]
    ordinary = out["cells"][f"{start}/role_first/ordinary"]
    out["screen"][start] = {
        "tag_direct_gain": tagged["exact_answers"] - ordinary["exact_answers"],
        "tag_routed_change": tagged["routed_exact"] - ordinary["routed_exact"],
        "residual_tagged_route_minus_direct": tagged["routed_exact"] - tagged["exact_answers"],
        "pass": tagged["exact_answers"] - ordinary["exact_answers"] >= 16
                and tagged["routed_exact"] - ordinary["routed_exact"] >= -8,
    }
out["positive_development_screen"] = all(out["screen"][s]["pass"] for s in cfg["starts"])
(ROOT / "ANALYSIS.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({"screen": out["screen"], "positive_development_screen": out["positive_development_screen"],
                  "cells": {k: {name: v[name] for name in ("exact_answers", "four_correct")
                                + (("routed_exact", "routed_four_correct", "same_answer_actor_patient_question_pairs")
                                   if "routed_exact" in v else ())}
                            for k, v in out["cells"].items()},
                  "paired": out["paired"]}, indent=2))
