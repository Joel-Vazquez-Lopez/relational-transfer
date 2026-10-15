#!/usr/bin/env python3
"""Read-only analysis of the frozen matched role/subject pilot."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "PROTOCOL.json").read_text())
FORMS = CONFIG["test_forms"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(start: str, arm: str) -> list[dict]:
    data = json.loads((ROOT / f"EVAL_{start}_{arm}.json").read_text())
    assert data["protocol_sha256"] == sha(ROOT / "PROTOCOL.json")
    assert data["start"] == start and data["arm"] == arm
    rows = data["records"]
    assert len(rows) == 432 and len({(r["id"], r["form"], r["role"]) for r in rows}) == 432
    return rows


def first_mention(row: dict) -> str:
    a, b = CONFIG["test_pairs"][row["pair_index"]]
    actor, patient = (a, b) if row["direction"] == 0 else (b, a)
    return patient if row["form"] in ("passive", "object_cleft", "relative_actor") else actor


def local_subject(row: dict) -> str:
    a, b = CONFIG["test_pairs"][row["pair_index"]]
    actor, patient = (a, b) if row["direction"] == 0 else (b, a)
    return patient if row["form"] in ("passive", "relative_patient") else actor


def summarize(rows: list[dict]) -> dict:
    by_form = {}
    indexed = {(r["id"], r["form"], r["role"]): r for r in rows}
    ids = sorted({r["id"] for r in rows})
    for form in FORMS:
        cell = [r for r in rows if r["form"] == form]
        by_form[form] = {
            "actor_correct": sum(r["correct"] for r in cell if r["role"] == "actor"),
            "patient_correct": sum(r["correct"] for r in cell if r["role"] == "patient"),
            "joint_events": sum(indexed[(i, form, "actor")]["correct"] and
                                indexed[(i, form, "patient")]["correct"] for i in ids),
            "subject_answers": sum(r["answer"] == local_subject(r) for r in cell),
            "first_mention_answers": sum(r["answer"] == first_mention(r) for r in cell),
            "by_pair_joint": [sum(indexed[(i, form, "actor")]["correct"] and
                                  indexed[(i, form, "patient")]["correct"]
                                  for i in ids if indexed[(i, form, "actor")]["pair_index"] == p)
                              for p in range(6)],
        }
    return {"all_prompt_correct": sum(r["correct"] for r in rows),
            "valid_outputs": sum(r["valid"] for r in rows),
            "all_six_forms_joint_events": sum(all(indexed[(i, f, role)]["correct"]
                                               for f in FORMS for role in ("actor", "patient"))
                                              for i in ids),
            "by_form": by_form}


def main() -> None:
    result = {"status": "analysis_of_frozen_exploratory_pilot",
              "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
              "summaries": {}, "matched_changes": {}}
    for start in CONFIG["starts"]:
        for arm in ("source", *CONFIG["arms"]):
            rows = load(start, arm)
            result["summaries"][f"{start}_{arm}"] = summarize(rows)
        a = result["summaries"][f"{start}_aligned"]["by_form"]
        b = result["summaries"][f"{start}_decoupled"]["by_form"]
        primary_gain = b["relative_patient"]["joint_events"] - a["relative_patient"]["joint_events"]
        relative_actor_loss = a["relative_actor"]["joint_events"] - b["relative_actor"]["joint_events"]
        active_a = a["active"]["actor_correct"] + a["active"]["patient_correct"]
        active_b = b["active"]["actor_correct"] + b["active"]["patient_correct"]
        result["matched_changes"][start] = {
            "relative_patient_joint_gain": primary_gain,
            "relative_actor_joint_loss": relative_actor_loss,
            "active_question_accuracy_pp_change": 100 * (active_b - active_a) / 72,
            "primary_by_pair_joint_deltas": [y-x for x, y in zip(
                a["relative_patient"]["by_pair_joint"], b["relative_patient"]["by_pair_joint"])],
            "passes_positive_screen": primary_gain >= 6 and relative_actor_loss <= 4 and
                                       100 * (active_b - active_a) / 72 >= -5,
        }
    result["positive_screen_both_starts"] = all(x["passes_positive_screen"]
                                                 for x in result["matched_changes"].values())
    (ROOT / "ANALYSIS.json").write_text(json.dumps(result, indent=2) + "\n")
    for name, s in result["summaries"].items():
        print(name, "correct", s["all_prompt_correct"], "/432",
              "all-joint", s["all_six_forms_joint_events"], "/36")
        for f in FORMS:
            x = s["by_form"][f]
            print(" ", f, "actor", x["actor_correct"], "patient", x["patient_correct"],
                  "joint", x["joint_events"], "/36")
    print("Matched changes:", json.dumps(result["matched_changes"], indent=2))
    print("Both starts pass:", result["positive_screen_both_starts"])


if __name__ == "__main__":
    main()
