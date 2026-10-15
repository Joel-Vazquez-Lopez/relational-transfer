#!/usr/bin/env python3
"""Report all frozen development and exploratory natural-data cells."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
cfg = json.loads((ROOT / "PROTOCOL.json").read_text())
protocol_hash = hashlib.sha256((ROOT / "PROTOCOL.json").read_bytes()).hexdigest()
splits = json.loads((ROOT / "SPLITS.json").read_text())
gold = json.loads((ROOT.parent / "role_slot_routing_qasrl_gold_2026-10-13" / "ITEMS.json").read_text())


def norm(value):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (value or "").casefold())).strip()


def summarize(start, arm, which, stress, items):
    path = ROOT / f"EVAL_{start}_{arm}_{which}_{stress}.json"
    data = json.loads(path.read_text())
    assert data["protocol_sha256"] == protocol_hash
    assert data["checkpoint_sha256"] == hashlib.sha256((ROOT / f"{start}_{arm}" / "adapter.pt").read_bytes()).hexdigest()
    records = data["records"]
    n = len(items)
    by = {(r["id"], r["role"], r["order"]): r for r in records}
    assert len(records) == len(by) == 4*n
    ids = [item["id"] for item in items]
    assert len(ids) == len(set(ids)) == n
    exact = sum(r["correct"] for r in records)
    primary = sum(all(by[(id_, role, order)]["correct"] for role in ("actor", "patient")
                      for order in (0, 1)) for id_ in ids)
    by_role_order = {f"{role}/order{order}": sum(by[(id_, role, order)]["correct"] for id_ in ids)
                     for role in ("actor", "patient") for order in (0, 1)}
    pos = {"first": 0, "second": 0, "other": 0}
    draft_correct = draft_follow = draft_present = 0
    for r in records:
        answer = norm(r["answer"])
        pair = r["candidate_pair"]
        if answer == norm(pair[0]):
            pos["first"] += 1
        elif answer == norm(pair[1]):
            pos["second"] += 1
        else:
            pos["other"] += 1
        draft = r.get("draft")
        if draft:
            draft_present += 1
            requested = draft[0] if r["role"] == "actor" else draft[1]
            draft_correct += norm(requested) == norm(r["gold"])
            draft_follow += answer == norm(requested)
    result = {"items": n, "prompts": 4*n, "four_correct": primary,
              "exact_answers": exact, "parsed": sum(r["parsed"] for r in records),
              "by_role_order": by_role_order, "answer_candidate_position": pos,
              "draft_present": draft_present, "requested_draft_correct": draft_correct,
              "answer_follows_requested_draft": draft_follow,
              "raw_output_examples": [r["raw"] for r in records[:2]]}
    if arm not in ("answer_only", "candidate_first"):
        result["correct_actor_fields"] = sum(r["actor_correct"] for r in records)
        result["correct_patient_fields"] = sum(r["patient_correct"] for r in records)
        result["routed_exact_answers"] = sum(r["routed_correct"] for r in records)
        result["routed_four_correct"] = sum(all(by[(id_, role, order)]["routed_correct"]
                                                for role in ("actor", "patient") for order in (0, 1))
                                            for id_ in ids)
    return result


out = {"protocol_sha256": protocol_hash, "training_budgets": {}, "datasets": {}, "screen": {}}
for start in cfg["starts"]:
    for arm in cfg["arms"]:
        path = ROOT / f"{start}_{arm}" / "TRAINING.json"
        data = json.loads(path.read_text())
        assert data["protocol_sha256"] == protocol_hash
        assert data["checkpoint_sha256"] == hashlib.sha256((ROOT / f"{start}_{arm}" / "adapter.pt").read_bytes()).hexdigest()
        out["training_budgets"][f"{start}_{arm}"] = {k: data[k] for k in ("updates", "examples", "target_tokens", "elapsed_seconds")}

for which, items in (("development", splits["development"]), ("gold_exploratory", gold)):
    out["datasets"][which] = {}
    for start in cfg["starts"]:
        for arm in cfg["arms"]:
            for stress in (("normal", "swapped") if arm.startswith("verifier_") else ("normal",)):
                key = f"{start}_{arm}_{stress}"
                out["datasets"][which][key] = summarize(start, arm, which, stress, items)

for start in cfg["starts"]:
    d = out["datasets"]["development"]
    noisy = d[f"{start}_verifier_noisy_normal"]
    clean = d[f"{start}_verifier_clean_normal"]
    role = d[f"{start}_role_first_normal"]
    out["screen"][start] = {
        "noisy_minus_clean_four_correct": noisy["four_correct"]-clean["four_correct"],
        "noisy_minus_role_first_four_correct": noisy["four_correct"]-role["four_correct"],
        "noisy_parsed_rate": noisy["parsed"]/noisy["prompts"],
        "pass": noisy["four_correct"]-clean["four_correct"] >= 4 and
                noisy["four_correct"]-role["four_correct"] >= -4 and
                noisy["parsed"] >= .8*noisy["prompts"],
    }
out["positive_development_screen"] = all(out["screen"][s]["pass"] for s in cfg["starts"])
(ROOT / "ANALYSIS.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({"screen": out["screen"], "positive_development_screen": out["positive_development_screen"],
                  "development": {k: {"four_correct": v["four_correct"], "exact_answers": v["exact_answers"],
                                       "routed_exact_answers": v.get("routed_exact_answers")}
                                  for k,v in out["datasets"]["development"].items()},
                  "gold_exploratory": {k: {"four_correct": v["four_correct"], "exact_answers": v["exact_answers"],
                                          "routed_exact_answers": v.get("routed_exact_answers")}
                                       for k,v in out["datasets"]["gold_exploratory"].items()}}, indent=2))
