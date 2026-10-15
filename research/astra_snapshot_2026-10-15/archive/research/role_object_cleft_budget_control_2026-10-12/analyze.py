#!/usr/bin/env python3
"""Analyze fixed object-cleft exposure comparison."""
from __future__ import annotations

import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FORMS = ("active", "passive", "subject_cleft", "object_cleft")
QS = ("actor_performed", "actor_did", "patient_verb",
      "actor_carried", "patient_which")


def load(start: str, arm: str):
    raw = json.loads((ROOT / f"EVAL_{start}_{arm}.json").read_text())
    records = raw["records"]
    assert len(records) == 720
    groups = collections.defaultdict(dict)
    for r in records:
        k = (r["event"], r["form"])
        assert r["wording"] not in groups[k]
        groups[k][r["wording"]] = r
    assert len(groups) == 144 and all(set(v) == set(QS) for v in groups.values())
    return raw, groups


def stats(groups):
    result = {"valid": sum(r["valid"] for g in groups.values() for r in g.values()),
              "forms": {}}
    for form in FORMS:
        items = [v for (e, f), v in groups.items() if f == form]
        assert len(items) == 36
        result["forms"][form] = {
            "wording_correct_of_36": {q: sum(g[q]["correct"] for g in items) for q in QS},
            "heldout_two_role_joint_of_36": sum(
                g["actor_carried"]["correct"] and g["patient_which"]["correct"]
                for g in items),
            "all_five_correct_of_36": sum(all(g[q]["correct"] for q in QS) for g in items),
            "actor_contradictions_of_36": sum(
                len({g[q]["answer"] for q in QS if q.startswith("actor")}) > 1
                for g in items),
        }
    result["broad_heldout_joint_of_144"] = sum(
        v["heldout_two_role_joint_of_36"] for v in result["forms"].values())
    return result


def main():
    out = {}
    for start in ("fresh1", "fresh3"):
        group_by_arm = {}
        for arm in ("old_sft", "old_coupled", "boost_sft", "boost_coupled"):
            _, groups = load(start, arm)
            group_by_arm[arm] = groups
            out[f"{start}_{arm}"] = stats(groups)
        for base, treatment in (("old_sft", "boost_sft"),
                                ("boost_sft", "boost_coupled")):
            a, b = group_by_arm[base], group_by_arm[treatment]
            assert a.keys() == b.keys()
            joint = lambda g: g["actor_carried"]["correct"] and g["patient_which"]["correct"]
            out[f"{start}_{base}_vs_{treatment}"] = {
                f"base_{int(x)}_treat_{int(y)}": n
                for (x, y), n in sorted(collections.Counter(
                    (bool(joint(a[k])), bool(joint(b[k]))) for k in a).items())}
    (ROOT / "ANALYSIS.json").write_text(json.dumps(out, indent=2) + "\n")
    for start in ("fresh1", "fresh3"):
        print("\n", start)
        for arm in ("old_sft", "old_coupled", "boost_sft", "boost_coupled"):
            x = out[f"{start}_{arm}"]
            print(arm, "broad", x["broad_heldout_joint_of_144"],
                  "object", x["forms"]["object_cleft"]["heldout_two_role_joint_of_36"],
                  "active", x["forms"]["active"]["heldout_two_role_joint_of_36"],
                  "passive", x["forms"]["passive"]["heldout_two_role_joint_of_36"],
                  "object_patient", x["forms"]["object_cleft"]["wording_correct_of_36"]["patient_verb"],
                  "valid", x["valid"])
        print("paired exposure", out[f"{start}_old_sft_vs_boost_sft"])
        print("paired coupled", out[f"{start}_boost_sft_vs_boost_coupled"])


if __name__ == "__main__":
    main()
