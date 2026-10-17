"""Summarize frozen factorial competence results without model inference."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tally(rows, key):
    return {str(value): {"correct": sum(r["greedy_correct"] and r["score_correct"]
                                      for r in rows if r[key] == value),
                         "total": sum(r[key] == value for r in rows)}
            for value in sorted(set(r[key] for r in rows))}


def main():
    out = {"status": "frozen_competence_gate_analysis_no_causal_patches",
           "lock_sha256": sha(HERE / "FACTORIAL_GATE_LOCK.json"), "models": {}}
    for m in ("qwen7b", "llama3b"):
        path = HERE / f"FACTORIAL_GATE_{m}.json"
        data = json.loads(path.read_text())
        assert data["lock_sha256"] == out["lock_sha256"]
        rows = [r for f in data["families"] for r in f["rows"]]
        unique = {r["prompt"]: r for r in rows}
        assert len(rows) == 64 and len(unique) == 48
        failures = []
        for family in data["families"]:
            for r in family["rows"]:
                if not (r["greedy_correct"] and r["score_correct"]):
                    failures.append({
                        "family": family["family"]["id"], "split": family["family"]["split"],
                        "frame": r["frame"], "r": r["r"], "v": r["v"], "role": r["role"],
                        "sentence": r["sentence"], "question": r["question"],
                        "generated": r["generated"], "expected": r["expected"],
                        "score_margin": r["score_margin"]
                    })
        out["models"][m] = {
            "raw_sha256": sha(path), "old_scorer_validation_passed": data["old_scorer_validation_passed"],
            "old_correct": sum(r["greedy_correct"] and r["score_correct"] for r in data["old_validation"]),
            "old_total": len(data["old_validation"]),
            "families": [{"id": f["family"]["id"], "split": f["family"]["split"],
                          "competent": f["competent"], "correct_rows": f["correct_count"],
                          "total_rows": len(f["rows"])} for f in data["families"]],
            "competent_families": sum(f["competent"] for f in data["families"]),
            "correct_rows": sum(r["greedy_correct"] and r["score_correct"] for r in rows),
            "total_rows": len(rows),
            "correct_unique_prompts": sum(r["greedy_correct"] and r["score_correct"] for r in unique.values()),
            "unique_prompts": len(unique),
            "by_frame": tally(rows, "frame"), "by_role": tally(rows, "role"),
            "by_voice": tally(rows, "v"), "by_swap": tally(rows, "r"),
            "greedy_score_disagreements": sum(r["greedy_correct"] != r["score_correct"] for r in rows),
            "failures": failures
        }
    out["cross_model_gate_passed"] = all(
        sum(f["competent"] for f in out["models"][m]["families"] if f["split"] == split) >= 1
        for m in out["models"] for split in ("development", "evaluation")
    )
    (HERE / "FACTORIAL_GATE_SUMMARY.json").write_text(json.dumps(out, indent=2) + "\n")
    print({m: (v["competent_families"], v["correct_unique_prompts"], len(v["failures"]))
           for m, v in out["models"].items()})
    print("cross_model_gate_passed", out["cross_model_gate_passed"])


if __name__ == "__main__":
    main()
