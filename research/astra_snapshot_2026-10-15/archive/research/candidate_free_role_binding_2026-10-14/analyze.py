"""Frozen candidate-free development analysis; report both roles and routing separately."""
from __future__ import annotations

import json
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def score(records: list[dict]) -> dict:
    by_id = {}
    for row in records:
        assert row["role"] in ("actor", "patient")
        assert (row["id"], row["role"]) not in by_id
        by_id[row["id"], row["role"]] = row
    ids = sorted({key[0] for key in by_id})
    assert len(ids) == 100 and len(records) == 200
    assert all((id_, role) in by_id for id_ in ids for role in ("actor", "patient"))
    return {"exact": sum(r["correct"] for r in records),
            "both_answers": sum(by_id[id_, "actor"]["correct"] and by_id[id_, "patient"]["correct"] for id_ in ids),
            "actor_exact": sum(r["correct"] for r in records if r["role"] == "actor"),
            "patient_exact": sum(r["correct"] for r in records if r["role"] == "patient"),
            "both_printed_roles": sum(r["actor_correct"] and r["patient_correct"] for r in records),
            "routed_exact": sum(r["routed_correct"] for r in records),
            "answer_matches_field": sum(r["answer_matches_field"] for r in records),
            "parsed": sum(r["parsed"] for r in records),
            "tag_correct": sum(r["tag_correct"] for r in records)}


def main() -> None:
    spec = importlib.util.spec_from_file_location("candidate_free_pilot", ROOT / "pilot.py")
    assert spec and spec.loader
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    sha, validate = pilot.sha, pilot.validate
    validate()
    cfg = json.loads((ROOT / "PROTOCOL.json").read_text())
    scores = {}
    for start in cfg["starts"]:
        scores[start] = {}
        for arm in cfg["arms"]:
            cp = ROOT / f"{start}_{arm}" / "adapter.pt"
            training = json.loads((cp.parent / "TRAINING.json").read_text())
            ev = json.loads((ROOT / f"EVAL_{start}_{arm}.json").read_text())
            assert training["updates"] == 160
            assert training["checkpoint_sha256"] == ev["checkpoint_sha256"] == sha(cp)
            assert training["protocol_sha256"] == ev["protocol_sha256"] == sha(ROOT / "PROTOCOL.json")
            scores[start][arm] = {"training": {"target_tokens": training["target_tokens"]},
                                   **score(ev["records"])}
        a, b = scores[start]["typed"], scores[start]["neutral"]
        assert a["training"]["target_tokens"] == b["training"]["target_tokens"]
    screen = {start: scores[start]["typed"]["both_answers"] - scores[start]["neutral"]["both_answers"] >= 10
              and scores[start]["typed"]["exact"] - scores[start]["neutral"]["exact"] >= 20
              and scores[start]["typed"]["both_printed_roles"] - scores[start]["neutral"]["both_printed_roles"] >= -10
              for start in cfg["starts"]}
    result = {"status": "exploratory_development_only", "scores": scores,
              "positive_screen_by_start": screen, "positive_screen_all": all(screen.values()),
              "limits": cfg["limits"]}
    (ROOT / "ANALYSIS.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
