"""Apply the checkpoint rule written in PROTOCOL.json, using fresh1 and fresh2 only."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT.parent / "llm_role_repair_confirmation"
UNORDERED = ROOT.parent / "llm_role_unordered_control"
STEPS = (20, 40, 60, 80, 100, 120)
WORLDS = ("fresh1", "fresh2")
SEEDS = (830211, 830212)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def counts(records):
    patient = [r for r in records if r["form"] == "patient_cleft" and r["wording"] == "performed"]
    familiar = [r for r in records if r["form"] in ("active", "passive") and r["wording"] == "performed"]
    assert len(patient) == 250 and len(familiar) == 500
    return sum(r["correct"] for r in patient), sum(r["correct"] for r in familiar)


def main():
    p = load(ROOT / "PROTOCOL.json")
    assert p["checkpoints"] == list(STEPS)
    screen = {}
    for world in WORLDS:
        for seed in SEEDS:
            key = f"{world}_{seed}"
            screen[key] = {}
            for step in STEPS:
                result = load(ROOT / f"{key}_ordinary" / f"step_{step:03d}" / "result.json")
                assert result["protocol_sha256"] == sha(ROOT / "PROTOCOL.json")
                screen[key][str(step)] = counts(result["records"])
            for arm, directory in (("ordered_roles", REFERENCE), ("unordered_pair", UNORDERED)):
                result = load(directory / f"{key}_{arm}" / "result.json")
                screen[key][arm] = counts(result["records"])
    choices = {}
    for arm in ("ordered_roles", "unordered_pair"):
        target_patient = sum(screen[k][arm][0] for k in screen) / (4 * 250)
        target_familiar = sum(screen[k][arm][1] for k in screen) / (4 * 500)
        candidates = []
        for step in STEPS:
            patient = sum(screen[k][str(step)][0] for k in screen) / (4 * 250)
            familiar = sum(screen[k][str(step)][1] for k in screen) / (4 * 500)
            candidates.append({"step": step, "patient": patient, "familiar": familiar,
                               "eligible": familiar >= target_familiar})
        eligible = [c for c in candidates if c["eligible"]]
        chosen = sorted(eligible, key=lambda c: (-c["patient"], c["step"]))[0] if eligible else None
        choices[arm] = {"auxiliary_dev_patient": target_patient,
                        "auxiliary_dev_familiar": target_familiar,
                        "candidates": candidates, "selected": chosen}
    result = {"trajectory_protocol_sha256": sha(ROOT / "PROTOCOL.json"),
              "development_worlds": WORLDS, "screen": screen, "choices": choices}
    (ROOT / "SELECTION.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(choices, indent=2))


if __name__ == "__main__":
    main()
