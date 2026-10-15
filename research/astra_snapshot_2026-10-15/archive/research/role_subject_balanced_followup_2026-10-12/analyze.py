#!/usr/bin/env python3
"""Read-only analysis of the prospective mixed-form follow-up."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "PROTOCOL.json").read_text())
FORMS = CFG["forms"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(start: str, arm: str) -> list[dict]:
    data = json.loads((ROOT / f"EVAL_{start}_{arm}.json").read_text())
    assert data["protocol_sha256"] == sha(ROOT / "PROTOCOL.json")
    assert data["start"] == start and data["arm"] == arm
    rows = data["records"]
    assert len(rows) == 432 and len({(r["id"], r["form"], r["role"]) for r in rows}) == 432
    return rows


def summarize(rows: list[dict]) -> dict:
    idx = {(r["id"], r["form"], r["role"]): r for r in rows}
    ids = sorted({r["id"] for r in rows})
    by_form = {}
    for f in FORMS:
        cells = [r for r in rows if r["form"] == f]
        by_form[f] = {
            "actor_correct": sum(r["correct"] for r in cells if r["role"] == "actor"),
            "patient_correct": sum(r["correct"] for r in cells if r["role"] == "patient"),
            "joint": sum(idx[(i, f, "actor")]["correct"] and idx[(i, f, "patient")]["correct"]
                         for i in ids),
            "joint_by_pair": [sum(idx[(i, f, "actor")]["correct"] and idx[(i, f, "patient")]["correct"]
                                  for i in ids if idx[(i, f, "actor")]["pair_index"] == p)
                              for p in range(6)],
        }
    return {"prompt_correct": sum(r["correct"] for r in rows),
            "valid_outputs": sum(r["valid"] for r in rows),
            "sum_form_joint": sum(x["joint"] for x in by_form.values()),
            "all_forms_joint_events": sum(all(idx[(i, f, role)]["correct"]
                                               for f in FORMS for role in ("actor", "patient"))
                                          for i in ids),
            "by_form": by_form}


def main() -> None:
    out = {"status": "analysis_of_prospective_exploratory_followup",
           "protocol_sha256": sha(ROOT / "PROTOCOL.json"),
           "summaries": {}, "matched_comparisons": {}}
    for start in CFG["starts"]:
        for arm in CFG["evaluation_arms"]:
            out["summaries"][f"{start}_{arm}"] = summarize(load(start, arm))
        summaries = {arm: out["summaries"][f"{start}_{arm}"] for arm in CFG["evaluation_arms"]}
        mixed = summaries["mixed"]
        old = (summaries["aligned"], summaries["decoupled"])
        gain_over_best = mixed["sum_form_joint"] - max(x["sum_form_joint"] for x in old)
        deficits = {f: max(x["by_form"][f]["joint"] for x in old) - mixed["by_form"][f]["joint"]
                    for f in ("active", "passive", "relative_patient")}
        out["matched_comparisons"][start] = {
            "sum_joint_gain_over_best_old_arm": gain_over_best,
            "joint_deficits_vs_best_by_form": deficits,
            "passes_frozen_screen": gain_over_best >= 6 and all(v <= 3 for v in deficits.values()),
        }
    out["both_starts_pass"] = all(v["passes_frozen_screen"]
                                  for v in out["matched_comparisons"].values())
    (ROOT / "ANALYSIS.json").write_text(json.dumps(out, indent=2) + "\n")
    for name, x in out["summaries"].items():
        print(name, "prompts", x["prompt_correct"], "/432", "form-joint",
              x["sum_form_joint"], "/216", "all-forms-joint", x["all_forms_joint_events"], "/36")
        print(" ", " ".join(f"{f}:{x['by_form'][f]['joint']}" for f in FORMS))
    print(json.dumps(out["matched_comparisons"], indent=2))
    print("Both starts pass:", out["both_starts_pass"])


if __name__ == "__main__":
    main()
