#!/usr/bin/env python3
"""Fixed endpoint analysis for paired role/prompt-coupled pilot."""
from __future__ import annotations

import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FORMS = ("active", "passive", "subject_cleft", "object_cleft")
QUESTIONS = ("actor_performed", "actor_did", "patient_verb",
             "actor_carried", "patient_which")
ACTOR = tuple(q for q in QUESTIONS if q.startswith("actor"))


def read(name: str):
    x = json.loads((ROOT / f"EVAL_{name}.json").read_text())
    records = x["records"]
    assert len(records) == 720
    by = collections.defaultdict(dict)
    for r in records:
        key = (r["event"], r["form"])
        assert r["wording"] not in by[key]
        by[key][r["wording"]] = r
    assert len(by) == 144 and all(set(g) == set(QUESTIONS) for g in by.values())
    return by


def summarize(by: dict):
    result = {"valid": sum(r["valid"] for g in by.values() for r in g.values()),
              "total": 720, "forms": {}}
    for form in FORMS:
        groups = [v for (e, f), v in by.items() if f == form]
        assert len(groups) == 36
        result["forms"][form] = {
            "correct_per_wording_of_36": {
                q: sum(g[q]["correct"] for g in groups) for q in QUESTIONS},
            "heldout_both_roles_correct_of_36": sum(
                g["actor_carried"]["correct"] and g["patient_which"]["correct"]
                for g in groups),
            "all_five_correct_of_36": sum(
                all(g[q]["correct"] for q in QUESTIONS) for g in groups),
            "actor_contradictions_of_36": sum(
                len({g[q]["answer"] for q in ACTOR}) > 1 for g in groups),
        }
    result["heldout_both_roles_correct_of_144"] = sum(
        x["heldout_both_roles_correct_of_36"] for x in result["forms"].values())
    result["all_five_correct_of_144"] = sum(
        x["all_five_correct_of_36"] for x in result["forms"].values())
    result["actor_contradictions_of_144"] = sum(
        x["actor_contradictions_of_36"] for x in result["forms"].values())
    return result


def main():
    out = {}
    for start in ("fresh1", "fresh3"):
        groups = {}
        for arm in ("balanced_sft", "role_coupled"):
            name = f"{start}_{arm}"
            by = read(name)
            groups[arm] = by
            out[name] = summarize(by)
        a, b = groups["balanced_sft"], groups["role_coupled"]
        assert a.keys() == b.keys()
        joint = lambda g: g["actor_carried"]["correct"] and g["patient_which"]["correct"]
        paired = collections.Counter((bool(joint(a[k])), bool(joint(b[k]))) for k in a)
        out[f"{start}_paired"] = {
            f"sft_{int(x)}_coupled_{int(y)}": n for (x, y), n in sorted(paired.items())}
    (ROOT / "ANALYSIS.json").write_text(json.dumps(out, indent=2) + "\n")
    for start in ("fresh1", "fresh3"):
        print("\n", start, "paired", out[f"{start}_paired"])
        for arm in ("balanced_sft", "role_coupled"):
            x = out[f"{start}_{arm}"]
            print(arm, "heldout_joint", x["heldout_both_roles_correct_of_144"],
                  "all_five", x["all_five_correct_of_144"],
                  "actor_contradictions", x["actor_contradictions_of_144"],
                  "valid", x["valid"])
            for form, z in x["forms"].items():
                print(form, z["correct_per_wording_of_36"],
                      "heldout_joint", z["heldout_both_roles_correct_of_36"])


if __name__ == "__main__":
    main()
