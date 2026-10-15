#!/usr/bin/env python3
"""Analyze the prospectively fixed role-question wording audit."""
from __future__ import annotations

import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ACTOR = ("actor_performed", "actor_did", "actor_carried")
PATIENT = ("patient_verb", "patient_which")
FORMS = ("active", "passive", "subject_cleft", "object_cleft")


def main() -> None:
    result = {}
    for start in ("fresh1", "fresh3"):
        for arm in ("source", "aligned", "decoupled", "mixed"):
            name = f"{start}_{arm}"
            p = ROOT / f"EVAL_{name}.json"
            data = json.loads(p.read_text())
            records = data["records"]
            assert len(records) == 720
            by = collections.defaultdict(dict)
            for r in records:
                key = (r["event"], r["form"])
                assert r["wording"] not in by[key]
                by[key][r["wording"]] = r
            assert len(by) == 144
            s = {"valid": sum(r["valid"] for r in records), "total": len(records), "forms": {}}
            for form in FORMS:
                groups = [v for (event, f), v in by.items() if f == form]
                assert len(groups) == 36
                counts = {q: sum(g[q]["correct"] for g in groups) for q in ACTOR + PATIENT}
                any_actor_contradiction = sum(len({g[q]["answer"] for q in ACTOR}) > 1 for g in groups)
                all_actor_correct = sum(all(g[q]["correct"] for q in ACTOR) for g in groups)
                all_patient_correct = sum(all(g[q]["correct"] for q in PATIENT) for g in groups)
                all_five_correct = sum(all(g[q]["correct"] for q in ACTOR + PATIENT) for g in groups)
                did_performed = collections.Counter(
                    (g["actor_did"]["correct"], g["actor_performed"]["correct"])
                    for g in groups)
                s["forms"][form] = {
                    "correct_per_wording_of_36": counts,
                    "actor_answer_contradictions_of_36": any_actor_contradiction,
                    "actor_all_three_correct_of_36": all_actor_correct,
                    "patient_both_correct_of_36": all_patient_correct,
                    "all_five_correct_of_36": all_five_correct,
                    "actor_did_vs_performed_paired_correctness": {
                        f"did_{int(d)}_performed_{int(p)}": n
                        for (d, p), n in sorted(did_performed.items())
                    },
                }
            result[name] = s
    (ROOT / "ANALYSIS.json").write_text(json.dumps(result, indent=2) + "\n")
    print("name form did perf carried patient_verb patient_which actor_all3 all5 contradictions")
    for name, data in result.items():
        for form, stats in data["forms"].items():
            c = stats["correct_per_wording_of_36"]
            print(name, form, c["actor_did"], c["actor_performed"], c["actor_carried"],
                  c["patient_verb"], c["patient_which"],
                  stats["actor_all_three_correct_of_36"], stats["all_five_correct_of_36"],
                  stats["actor_answer_contradictions_of_36"])


if __name__ == "__main__":
    main()
