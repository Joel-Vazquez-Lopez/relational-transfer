#!/usr/bin/env python3
"""Summarize the frozen QA-SRL Gold probe without filtering outputs."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
cfg = json.loads((ROOT / "PROTOCOL.json").read_text())
protocol_hash = hashlib.sha256((ROOT / "PROTOCOL.json").read_bytes()).hexdigest()
items = json.loads((ROOT / "ITEMS.json").read_text())
ids = [item["id"] for item in items]
items_by_id = {item["id"]: item for item in items}
assert len(ids) == len(set(ids)) == 51


def norm(value):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (value or "").casefold())).strip()
out = {"protocol_sha256": protocol_hash, "n_items": 51,
       "n_unique_sentences": len({item["sentence"] for item in items}),
       "arms": {}, "screen": {}}

for start in cfg["starts"]:
    for arm in cfg["arms"]:
        key = f"{start}_{arm}"
        data = json.loads((ROOT / f"EVAL_{key}.json").read_text())
        assert data["protocol_sha256"] == protocol_hash
        assert data["checkpoint_sha256"] == cfg["checkpoint_sha256"][key]
        records = data["records"]
        assert len(records) == 204
        by = {(r["id"], r["role"], r["order"]): r for r in records}
        assert len(by) == 204
        per_role_order = {f"{role}/order{order}": sum(by[(i, role, order)]["correct"] for i in ids)
                          for role in ("actor", "patient") for order in (0, 1)}
        both_per_order = {f"order{order}": sum(by[(i, "actor", order)]["correct"] and
                                               by[(i, "patient", order)]["correct"] for i in ids)
                          for order in (0, 1)}
        primary = sum(all(by[(i, role, order)]["correct"]
                          for role in ("actor", "patient") for order in (0, 1))
                      for i in ids)
        for record in records:
            item = items_by_id[record["id"]]
            pair = [item["actor"], item["patient"]]
            if record["order"]:
                pair.reverse()
            record["answer_candidate_position"] = (1 if norm(record["answer"]) == norm(pair[0])
                                                   else 2 if norm(record["answer"]) == norm(pair[1]) else 0)
            record["routed_correct"] = (norm(record["actor"] if record["role"] == "actor"
                                             else record["patient"]) == norm(record["gold"])) if arm != "candidate_first" else None
        single_word_ids = {item["id"] for item in items if len(item["actor"].split()) == 1 and
                            len(item["patient"].split()) == 1}
        by_partition = {part: {"items": sum(item["partition"] == part for item in items),
                               "four_correct": sum(item["partition"] == part and
                                                   all(by[(item["id"], role, order)]["correct"]
                                                       for role in ("actor", "patient") for order in (0, 1))
                                                   for item in items)}
                        for part in sorted({item["partition"] for item in items})}
        out["arms"][key] = {
            "primary_four_correct": primary,
            "both_roles_by_order": both_per_order,
            "per_role_order": per_role_order,
            "exact_answers": sum(r["correct"] for r in records),
            "parsed": sum(r["parsed"] for r in records),
            "slot_joint_correct": sum(r["slot_joint_correct"] for r in records) if arm != "candidate_first" else None,
            "answer_matches_generated_slot": sum(r["answer_matches_generated_slot"] for r in records) if arm != "candidate_first" else None,
            "answer_candidate_position": {
                "first": sum(r["answer_candidate_position"] == 1 for r in records),
                "second": sum(r["answer_candidate_position"] == 2 for r in records),
                "other": sum(r["answer_candidate_position"] == 0 for r in records),
            },
            "single_word_items": len(single_word_ids),
            "single_word_exact_answers": sum(r["correct"] for r in records if r["id"] in single_word_ids),
            "single_word_primary_four_correct": sum(all(by[(i, role, order)]["correct"]
                                                       for role in ("actor", "patient") for order in (0, 1))
                                                    for i in single_word_ids),
            "routed_exact_answers": sum(r["routed_correct"] for r in records) if arm != "candidate_first" else None,
            "routed_four_correct": sum(all(by[(i, role, order)]["routed_correct"]
                                           for role in ("actor", "patient") for order in (0, 1))
                                       for i in ids) if arm != "candidate_first" else None,
            "by_partition": by_partition,
            "raw_output_examples": [r["raw"] for r in records[:4]],
        }

for start in cfg["starts"]:
    slot = out["arms"][f"{start}_slot_first"]
    gains = {arm: slot["primary_four_correct"] - out["arms"][f"{start}_{arm}"]["primary_four_correct"]
             for arm in ("answer_first", "candidate_first")}
    out["screen"][start] = {"gains": gains,
                             "slot_parsed_rate": slot["parsed"] / 204,
                             "pass": all(gain >= 6 for gain in gains.values()) and slot["parsed"] >= 164}
out["positive_screen"] = all(out["screen"][s]["pass"] for s in cfg["starts"])
(ROOT / "ANALYSIS.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
