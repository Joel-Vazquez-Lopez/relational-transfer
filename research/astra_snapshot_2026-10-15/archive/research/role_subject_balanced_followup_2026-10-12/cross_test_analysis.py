#!/usr/bin/env python3
"""Paired interaction audit across the two disjoint constructed test banks."""
from __future__ import annotations

import json
from pathlib import Path

WORK = Path(__file__).resolve().parents[2]
FIRST = WORK / "research/role_subject_decoupling_pilot_2026-10-12"
SECOND = WORK / "research/role_subject_balanced_followup_2026-10-12"
FORMS = ("active", "passive", "subject_cleft", "object_cleft",
         "relative_actor", "relative_patient")


def read(directory: Path, start: str, arm: str) -> dict:
    data = json.loads((directory / f"EVAL_{start}_{arm}.json").read_text())
    return {(r["id"], r["form"], r["role"]): r for r in data["records"]}


def compare(directory: Path, start: str) -> dict:
    a = read(directory, start, "aligned")
    b = read(directory, start, "decoupled")
    assert set(a) == set(b) and len(a) == 432
    ids = sorted({k[0] for k in a})
    results = {}
    for form in FORMS:
        old = {i: a[(i, form, "actor")]["correct"] and a[(i, form, "patient")]["correct"]
               for i in ids}
        new = {i: b[(i, form, "actor")]["correct"] and b[(i, form, "patient")]["correct"]
               for i in ids}
        gained = sum(not old[i] and new[i] for i in ids)
        lost = sum(old[i] and not new[i] for i in ids)
        by_pair = [sum(int(new[i]) - int(old[i]) for i in ids
                       if a[(i, form, "actor")]["pair_index"] == p)
                   for p in range(6)]
        invalid_old = sum(not a[k]["valid"] for k in a if k[1] == form)
        invalid_new = sum(not b[k]["valid"] for k in b if k[1] == form)
        results[form] = {"aligned_joint": sum(old.values()),
                         "decoupled_joint": sum(new.values()),
                         "gained": gained, "lost": lost, "net": gained-lost,
                         "by_pair_net": by_pair,
                         "invalid_first_tokens_aligned": invalid_old,
                         "invalid_first_tokens_decoupled": invalid_new}
    return results


def main() -> None:
    panels = {label: {start: compare(directory, start) for start in ("fresh1", "fresh3")}
              for label, directory in (("test_1", FIRST), ("test_2", SECOND))}
    output = {"status": "post_hoc_descriptive_synthesis_of_two_prospective_exploratory_tests",
              "warning": "The same two source starts recur across tests and form-events share events; panel counts must not be treated as independent training replications.",
              "panels": panels}
    (SECOND / "CROSS_TEST_ANALYSIS.json").write_text(json.dumps(output, indent=2) + "\n")
    for label, panel in panels.items():
        for start, forms in panel.items():
            print(label, start, " ".join(f"{f}:{forms[f]['net']:+d}"
                                         for f in FORMS))
            print("  active/passive interaction",
                  forms["passive"]["net"] - forms["active"]["net"], "/36")
    print("Saved CROSS_TEST_ANALYSIS.json")


if __name__ == "__main__":
    main()
