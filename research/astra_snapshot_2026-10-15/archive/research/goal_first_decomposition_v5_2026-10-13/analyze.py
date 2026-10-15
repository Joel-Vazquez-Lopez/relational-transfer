"""Audit frozen v5 selector-plus-planner evaluations."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
V4 = ROOT.parent / "goal_first_decomposition_v4_2026-10-13"
SEEDS = (11, 23, 37, 41, 53)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    protocol = json.loads((ROOT / "PROTOCOL.json").read_text())
    protocol_hash = sha(ROOT / "PROTOCOL.json")
    assert tuple(protocol["seeds"]) == SEEDS
    current_bank = json.loads((ROOT / "DATA.json").read_text())
    prior_bank = json.loads((V4 / "DATA.json").read_text())
    earlier_train_questions = {r["question"] for r in prior_bank["train"]}
    earlier_mixed_questions = {r["question"] for r in prior_bank["mixed_test"]}
    novel_mixed_ids = {r["id"] for r in current_bank["mixed"] if r["question"] not in earlier_mixed_questions}
    summary = {"protocol_sha256": protocol_hash, "seeds": {}, "gate_pass": False,
               "question_overlap": {
                   "mixed_rows_in_earlier_train": sum(r["question"] in earlier_train_questions for r in current_bank["mixed"]),
                   "mixed_rows_in_earlier_mixed_eval": 400 - len(novel_mixed_ids),
                   "mixed_rows_not_in_earlier_mixed_eval": len(novel_mixed_ids),
               }}
    ids = {}
    golds = {}
    for seed in SEEDS:
        evaluation = json.loads((ROOT / f"EVAL_{seed}.json").read_text())
        assert evaluation["seed"] == seed
        assert evaluation["protocol_sha256"] == protocol_hash
        assert evaluation["planner_checkpoint_sha256"] == sha(V4 / f"MODEL_{seed}_clause_decomposition.pt")
        summary["seeds"][str(seed)] = {}
        for split in ("in_domain", "mixed"):
            rows = evaluation["records"][split]
            assert len(rows) == 400
            row_ids = [r["id"] for r in rows]
            row_gold = [r["gold"] for r in rows]
            assert len(set(row_ids)) == 400
            if split in ids:
                assert ids[split] == row_ids and golds[split] == row_gold
            else:
                ids[split], golds[split] = row_ids, row_gold
            assert all(r["pattern"] == "".join(r["gold"]) for r in rows)
            summary["seeds"][str(seed)][split] = {
                "n": len(rows),
                "selector_both_exact": sum(all(r["selector_exact"]) for r in rows),
                "selector_step1_exact": sum(r["selector_exact"][0] for r in rows),
                "selector_step2_exact": sum(r["selector_exact"][1] for r in rows),
                "predicted_mask_both_correct": sum(r["predicted_mask_operations"] == r["gold"] for r in rows),
                "oracle_mask_both_correct": sum(r["oracle_mask_operations"] == r["gold"] for r in rows),
                "full_question_both_correct": sum(r["full_question_operations"] == r["gold"] for r in rows),
            }
            if split == "mixed":
                fresh_rows = [r for r in rows if r["id"] in novel_mixed_ids]
                summary["seeds"][str(seed)][split]["question_new_since_v4_mixed"] = {
                    "n": len(fresh_rows),
                    "selector_both_exact": sum(all(r["selector_exact"]) for r in fresh_rows),
                    "predicted_mask_both_correct": sum(r["predicted_mask_operations"] == r["gold"] for r in fresh_rows),
                    "full_question_both_correct": sum(r["full_question_operations"] == r["gold"] for r in fresh_rows),
                }
    summary["gate_pass"] = sum(
        summary["seeds"][str(seed)]["mixed"]["selector_both_exact"] >= 380
        and summary["seeds"][str(seed)]["mixed"]["predicted_mask_both_correct"] >= 380
        for seed in SEEDS
    ) >= 4
    (ROOT / "ANALYSIS.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"mixed": {s: v["mixed"] for s, v in summary["seeds"].items()},
                      "gate_pass": summary["gate_pass"]}, indent=2))


if __name__ == "__main__":
    main()
