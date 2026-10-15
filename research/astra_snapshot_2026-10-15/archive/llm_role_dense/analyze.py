"""Verify dense sweep, selection, and reserved-wording comparisons."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT.parent / "llm_role_repair_confirmation"
UNORDERED = ROOT.parent / "llm_role_unordered_control"
WORLDS = ("fresh1", "fresh2", "fresh3")
SEEDS = (830211, 830212)
STEPS = (25, 30, 35)


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
    assert protocol["source_sha256"] == sha(ROOT.parent / "llm_role_dense.py")
    selection = load(ROOT / "SELECTION.json")
    assert selection["dense_protocol_sha256"] == sha(ROOT / "PROTOCOL.json")
    assert selection["choices"]["ordered_roles"]["selected"]["step"] == 30
    assert selection["choices"]["unordered_pair"]["selected"]["step"] == 25
    units = []
    for world in WORLDS:
        for seed in SEEDS:
            training = load(ROOT / f"{world}_{seed}_ordinary" / "TRAINING.json")
            assert training["step40_exact_match"] and training["protocol_sha256"] == sha(ROOT / "PROTOCOL.json")
            points = {}
            for step in STEPS:
                result = load(ROOT / f"{world}_{seed}_ordinary" / f"step_{step:03d}" / "result.json")
                assert result["protocol_sha256"] == sha(ROOT / "PROTOCOL.json")
                assert len(result["records"]) == 750
                points[str(step)] = counts(result["records"])
            units.append({"world": world, "seed": seed, "screen": points})
    confirmation = []
    for seed in SEEDS:
        item = {"seed": seed, "comparisons": {}}
        for arm, step, directory in (("ordered_roles", 30, REFERENCE),
                                     ("unordered_pair", 25, UNORDERED)):
            ordinary = load(ROOT / f"confirm_fresh3_{seed}_step_{step:03d}.json")
            assert ordinary["selection_sha256"] == sha(ROOT / "SELECTION.json")
            assert ordinary["dense_protocol_sha256"] == sha(ROOT / "PROTOCOL.json")
            assert len(ordinary["records"]) == 1500
            aux = load(directory / f"fresh3_{seed}_{arm}" / "result.json")
            selected = [r for r in aux["records"] if r["form"] in ("active", "passive", "patient_cleft")
                        and r["wording"] in ("did", "carried_out")]
            for a, b in zip(ordinary["records"], selected, strict=True):
                assert {k: v for k, v in a.items() if k not in ("answer", "correct", "valid")} == \
                       {k: v for k, v in b.items() if k not in ("answer", "correct", "valid")}
            item["comparisons"][arm] = {"ordinary_step": step,
                                        "ordinary": counts(ordinary["records"]),
                                        "auxiliary": counts(selected)}
        confirmation.append(item)
    result = {"protocol_sha256": sha(ROOT / "PROTOCOL.json"),
              "selection_sha256": sha(ROOT / "SELECTION.json"),
              "selection": selection["choices"], "units": units,
              "confirmation": confirmation}
    (ROOT / "ANALYSIS.json").write_text(json.dumps(result, indent=2) + "\n")
    for item in confirmation:
        print(item["seed"], item["comparisons"])


if __name__ == "__main__":
    main()
