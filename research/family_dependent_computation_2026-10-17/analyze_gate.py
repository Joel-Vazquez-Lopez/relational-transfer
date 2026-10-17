"""Summarize the frozen gate; reads raw JSON and performs no model inference."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOCK = json.loads((HERE / "GATE_LOCK.json").read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    by_model = {}
    for key in LOCK["models"]:
        path = HERE / f"GATE_{key}.json"
        data = json.loads(path.read_text())
        assert data["model"] == key
        assert data["lock_sha256"] == sha(HERE / "GATE_LOCK.json")
        assert data["script_sha256"] == sha(HERE / "run_gate.py")
        assert len(data["families"]) == len(LOCK["families"])
        rows = []
        competent_ids = []
        for saved, planned in zip(data["families"], LOCK["families"]):
            assert saved["family"] == planned
            assert [x["condition"] for x in saved["rows"]] == LOCK["conditions"]
            for row in saved["rows"]:
                condition = row["condition"]
                assert row["sentence"] == LOCK["sentence_templates"][condition].format(**planned)
                assert row["question"] == LOCK["question"].format(**planned)
                assert row["expected"] == (planned["a"] if condition.startswith("a_agent") else planned["b"])
            competent = all(x["greedy_correct"] and x["score_correct"] for x in saved["rows"])
            assert competent == saved["competent"]
            if competent:
                competent_ids.append(planned["id"])
            rows.extend(saved["rows"])
        by_model[key] = {
            "raw_sha256": sha(path), "competent_ids": competent_ids,
            "competent_families": len(competent_ids),
            "greedy_correct_answers": sum(r["greedy_correct"] for r in rows),
            "score_correct_answers": sum(r["score_correct"] for r in rows),
            "joint_correct_answers": sum(r["greedy_correct"] and r["score_correct"] for r in rows),
            "total_answers": len(rows),
            "greedy_score_disagreements": sum(r["greedy_correct"] != r["score_correct"] for r in rows),
            "by_condition": {
                c: {
                    "joint_correct": sum(r["greedy_correct"] and r["score_correct"] for r in rows if r["condition"] == c),
                    "total": sum(r["condition"] == c for r in rows),
                } for c in LOCK["conditions"]
            },
        }
    family_status = []
    for fam in LOCK["families"]:
        competent = {key: fam["id"] in by_model[key]["competent_ids"] for key in LOCK["models"]}
        family_status.append({"id": fam["id"], "split": fam["split"], "verb": fam["verb"],
                              "competent": competent,
                              "common_primary": all(competent[m] for m in LOCK["primary_models"])})
    common_dev = sum(f["common_primary"] and f["split"] == "development" for f in family_status)
    common_eval = sum(f["common_primary"] and f["split"] == "evaluation" for f in family_status)
    summary = {
        "status": "gate_failed_no_causal_work_authorized_by_this_lock",
        "lock_sha256": sha(HERE / "GATE_LOCK.json"),
        "runner_sha256": sha(HERE / "run_gate.py"),
        "scorer_sha256": sha(HERE.parent / "causal_developmental_alignment_2026-10-17/factorial_gate.py"),
        "by_model": by_model,
        "family_status": family_status,
        "common_development_families": common_dev,
        "common_evaluation_families": common_eval,
        "pass": common_dev >= 2 and common_eval >= 2,
    }
    assert not summary["pass"]
    out = HERE / "GATE_SUMMARY.json"
    assert not out.exists(), f"refusing overwrite: {out}"
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"models": by_model, "common_development": common_dev,
                      "common_evaluation": common_eval, "pass": summary["pass"]}, indent=2))


if __name__ == "__main__":
    main()
