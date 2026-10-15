"""Post-hoc error audit; does not alter the frozen exact-match analysis."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
from llm_role_authentic_spans import norm  # noqa: E402


def files(arm: str) -> list[Path]:
    return ([ROOT / "fresh1_0_pretrained.json"] if arm == "pretrained"
            else sorted(ROOT.glob(f"fresh*_83021?_{arm}.json")))


def main() -> None:
    report = {"status": "post_hoc_error_audit_after_19_completed_runs"}
    for arm in ("pretrained", "ordinary", "ordered_roles", "unordered_pair"):
        paths = files(arm)
        assert len(paths) == (1 if arm == "pretrained" else 6)
        counts = Counter()
        for path in paths:
            rows = json.loads(path.read_text())["records"]
            assert len(rows) == 126
            grouped = defaultdict(dict)
            for row in rows:
                grouped[row["id"]][row["role"]] = row
            for group in grouped.values():
                subj, other = group["subject_answer"], group["other_participant_answer"]
                counts["events"] += 1
                counts["same_answer_for_both_roles"] += norm(subj["answer"]) == norm(other["answer"])
                for row, opposite in ((subj, other), (other, subj)):
                    answer, gold = norm(row["answer"]), norm(row["gold_span"])
                    counts["answers"] += 1
                    counts["exact"] += row["correct"]
                    counts["copied_from_sentence"] += row["copied"]
                    counts["opposite_gold_exact"] += answer == norm(opposite["gold_span"])
                    counts["literal_placeholder"] += answer in ("someone", "something")
                    counts["answer_contains_gold"] += gold in answer and answer != gold
                    counts["gold_contains_answer"] += bool(answer) and answer in gold and answer != gold
        report[arm] = dict(counts)
    (ROOT / "DIAGNOSTICS.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
