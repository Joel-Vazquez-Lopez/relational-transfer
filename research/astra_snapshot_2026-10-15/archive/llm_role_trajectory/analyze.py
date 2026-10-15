"""Verify the trajectory, development choice, and held-out-wording comparison."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT.parent / "llm_role_repair_confirmation"
UNORDERED = ROOT.parent / "llm_role_unordered_control"
WORLDS = ("fresh1", "fresh2", "fresh3")
SEEDS = (830211, 830212)
STEPS = (20, 40, 60, 80, 100, 120)


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counts(rows):
    patient = [r for r in rows if r["form"] == "patient_cleft"]
    familiar = [r for r in rows if r["form"] in ("active", "passive")]
    return {"patient_correct": sum(r["correct"] for r in patient), "patient_total": len(patient),
            "familiar_correct": sum(r["correct"] for r in familiar), "familiar_total": len(familiar),
            "invalid": sum(not r["valid"] for r in rows)}


def main():
    protocol = load(ROOT / "PROTOCOL.json")
    assert protocol["source_sha256"] == sha(ROOT.parent / "llm_role_trajectory.py")
    selection = load(ROOT / "SELECTION.json")
    assert selection["trajectory_protocol_sha256"] == sha(ROOT / "PROTOCOL.json")
    assert {a: v["selected"]["step"] for a, v in selection["choices"].items()} == {
        "ordered_roles": 80, "unordered_pair": 80}
    units = []
    for world in WORLDS:
        for seed in SEEDS:
            training = load(ROOT / f"{world}_{seed}_ordinary" / "TRAINING.json")
            assert training["step120_exact_reference_match"] and training["updates"] == 120
            points = {}
            for step in STEPS:
                result = load(ROOT / f"{world}_{seed}_ordinary" / f"step_{step:03d}" / "result.json")
                assert result["protocol_sha256"] == sha(ROOT / "PROTOCOL.json")
                assert len(result["records"]) == 750
                points[str(step)] = counts(result["records"])
            units.append({"world": world, "seed": seed, "screen": points})
    confirmation = []
    for seed in SEEDS:
        ordinary = load(ROOT / f"confirm_fresh3_{seed}_step_080.json")
        assert ordinary["selection_sha256"] == sha(ROOT / "SELECTION.json")
        assert ordinary["trajectory_protocol_sha256"] == sha(ROOT / "PROTOCOL.json")
        assert len(ordinary["records"]) == 1500
        scores = {"ordinary_80": counts(ordinary["records"])}
        for arm, directory in (("ordered_roles", REFERENCE), ("unordered_pair", UNORDERED)):
            result = load(directory / f"fresh3_{seed}_{arm}" / "result.json")
            selected = [r for r in result["records"]
                        if r["form"] in ("active", "passive", "patient_cleft")
                        and r["wording"] in ("did", "carried_out")]
            for a, b in zip(ordinary["records"], selected, strict=True):
                assert {k: v for k, v in a.items() if k not in ("answer", "correct", "valid")} == \
                       {k: v for k, v in b.items() if k not in ("answer", "correct", "valid")}
            scores[arm] = counts(selected)
        confirmation.append({"seed": seed, "scores": scores})
    result = {"protocol_sha256": sha(ROOT / "PROTOCOL.json"),
              "selection_sha256": sha(ROOT / "SELECTION.json"),
              "selection": selection["choices"], "units": units,
              "confirmation": confirmation}
    (ROOT / "ANALYSIS.json").write_text(json.dumps(result, indent=2) + "\n")
    for item in confirmation:
        print(item["seed"], item["scores"])


if __name__ == "__main__":
    main()
