"""Post-hoc diagnostic of option-choice behavior in completed natural eval."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARMS = ("pretrained", "ordinary", "ordered_roles", "unordered_pair")


def files_for(arm: str) -> list[Path]:
    if arm == "pretrained":
        return [ROOT / "fresh1_0_pretrained.json"]
    return sorted(ROOT.glob(f"fresh*_83021?_{arm}.json"))


def main() -> None:
    report = {"status": "post_hoc_diagnostic_after_all_19_runs"}
    for arm in ARMS:
        files = files_for(arm)
        assert len(files) == (1 if arm == "pretrained" else 6)
        stats = {"runs": len(files), "total_prompts": 0, "answer_A": 0,
                 "answer_B": 0, "invalid": 0, "roles": {}}
        by_role = {role: {"both_orders_correct": 0, "pairs": 0,
                          "order0_correct": 0, "order1_correct": 0}
                   for role in ("actor", "patient")}
        for file in files:
            records = json.loads(file.read_text())["records"]
            assert len(records) == 504
            stats["total_prompts"] += len(records)
            stats["answer_A"] += sum(r["answer"] == "A" for r in records)
            stats["answer_B"] += sum(r["answer"] == "B" for r in records)
            stats["invalid"] += sum(not r["valid"] for r in records)
            grouped = defaultdict(list)
            for row in records:
                grouped[(row["id"], row["role"], row["wording"])].append(row)
            assert len(grouped) == 252
            for (_, role, _), group in grouped.items():
                assert {r["order"] for r in group} == {0, 1}
                cell = by_role[role]
                cell["pairs"] += 1
                cell["both_orders_correct"] += all(r["correct"] for r in group)
                cell["order0_correct"] += next(r["correct"] for r in group if r["order"] == 0)
                cell["order1_correct"] += next(r["correct"] for r in group if r["order"] == 1)
        stats["roles"] = by_role
        report[arm] = stats
    (ROOT / "DIAGNOSTICS.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
