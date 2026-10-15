"""Audit matched goal-first versus role-first-with-goal-order outcomes."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STARTS = ("fresh1", "fresh3")
ARMS = ("goal_first", "goal_after_roles")
SPLITS = ("fresh",)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(rows):
    groups = {}
    for row in rows:
        key = (row["id"], row["role"], row["order"])
        assert key not in groups
        groups[key] = row
    items = {x["id"] for x in rows}
    for item in items:
        assert {(role, order) for (id_, role, order) in groups if id_ == item} == {
            ("actor", 0), ("actor", 1), ("patient", 0), ("patient", 1)}
    return {
        "items": len(items), "prompts": len(rows),
        "all_four_answer_correct": sum(all(groups[(item, role, order)]["correct"]
                                        for role in ("actor", "patient") for order in (0, 1))
                                       for item in items),
        "exact_answer": sum(row["correct"] for row in rows),
        "parsed": sum(row["parsed"] for row in rows),
        "need_correct": sum(row["need_correct"] for row in rows),
        "actor_correct": sum(row["actor_correct"] for row in rows),
        "patient_correct": sum(row["patient_correct"] for row in rows),
        "both_fields_correct": sum(row["actor_correct"] and row["patient_correct"] for row in rows),
        "routed_correct": sum(row["routed_correct"] for row in rows),
        "answer_matches_field": sum(row["answer_matches_field"] for row in rows),
    }


def main():
    cfg = json.loads((ROOT / "PROTOCOL.json").read_text())
    phash = sha(ROOT / "PROTOCOL.json")
    assert tuple(cfg["starts"]) == STARTS and tuple(cfg["arms"]) == ARMS
    summary = {"protocol_sha256": phash, "matched": {}, "screens": {}}
    for start in STARTS:
        summary["matched"][start] = {}
        for arm in ARMS:
            cp = ROOT / f"{start}_{arm}" / "adapter.pt"
            train = json.loads((cp.parent / "TRAINING.json").read_text())
            assert train["protocol_sha256"] == phash and train["checkpoint_sha256"] == sha(cp)
            assert train["updates"] == cfg["updates_per_checkpoint"] == 160
            record = {"training": {"updates": train["updates"], "target_tokens": train["target_tokens"]}, "splits": {}}
            for split in SPLITS:
                evaluation = json.loads((ROOT / f"EVAL_{start}_{arm}_{split}.json").read_text())
                assert evaluation["protocol_sha256"] == phash
                assert evaluation["checkpoint_sha256"] == sha(cp)
                assert evaluation["start"] == start and evaluation["arm"] == arm and evaluation["split"] == split
                assert len(evaluation["records"]) == 160
                record["splits"][split] = score(evaluation["records"])
            summary["matched"][start][arm] = record
        first = summary["matched"][start]["goal_first"]["splits"]["fresh"]
        after = summary["matched"][start]["goal_after_roles"]["splits"]["fresh"]
        summary["screens"][start] = {
            "all_four_gain": first["all_four_answer_correct"] - after["all_four_answer_correct"],
            "exact_answer_gain": first["exact_answer"] - after["exact_answer"],
            "field_joint_difference": first["both_fields_correct"] - after["both_fields_correct"],
            "pass": (first["all_four_answer_correct"] - after["all_four_answer_correct"] >= 6
                     and first["parsed"] >= 152
                     and first["both_fields_correct"] >= after["both_fields_correct"] - 8),
        }
    summary["screen_pass"] = all(summary["screens"][start]["pass"] for start in STARTS)
    (ROOT / "ANALYSIS.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"screens": summary["screens"], "pass": summary["screen_pass"]}, indent=2))


if __name__ == "__main__":
    main()
