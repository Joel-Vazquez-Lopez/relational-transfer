"""Apply the dense protocol's development selection rule."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAJECTORY = ROOT.parent / "llm_role_trajectory"
STEPS = (20, 25, 30, 35, 40, 60, 80, 100, 120)
WORLDS = ("fresh1", "fresh2")
SEEDS = (830211, 830212)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def main():
    p = load(ROOT / "PROTOCOL.json")
    assert p["new_checkpoints"] == [25, 30, 35]
    prior = load(TRAJECTORY / "SELECTION.json")
    assert prior["trajectory_protocol_sha256"] == sha(TRAJECTORY / "PROTOCOL.json")
    choices = {}
    for arm in ("ordered_roles", "unordered_pair"):
        target_familiar = prior["choices"][arm]["auxiliary_dev_familiar"]
        target_patient = prior["choices"][arm]["auxiliary_dev_patient"]
        candidates = {c["step"]: c for c in prior["choices"][arm]["candidates"]}
        for step in (25, 30, 35):
            patient = familiar = 0
            for world in WORLDS:
                for seed in SEEDS:
                    result = load(ROOT / f"{world}_{seed}_ordinary" / f"step_{step:03d}" / "result.json")
                    assert result["protocol_sha256"] == sha(ROOT / "PROTOCOL.json")
                    assert len(result["records"]) == 750
                    patient += result["scores"]["patient_cleft"]["correct"]
                    familiar += result["scores"]["active"]["correct"] + result["scores"]["passive"]["correct"]
            candidates[step] = {"step": step, "patient": patient / 1000,
                                "familiar": familiar / 2000,
                                "eligible": familiar / 2000 >= target_familiar}
        all_candidates = [candidates[step] for step in STEPS]
        eligible = [c for c in all_candidates if c["eligible"]]
        chosen = sorted(eligible, key=lambda c: (-c["patient"], c["step"]))[0] if eligible else None
        choices[arm] = {"auxiliary_dev_patient": target_patient,
                        "auxiliary_dev_familiar": target_familiar,
                        "candidates": all_candidates, "selected": chosen}
    result = {"dense_protocol_sha256": sha(ROOT / "PROTOCOL.json"),
              "prior_selection_sha256": sha(TRAJECTORY / "SELECTION.json"),
              "choices": choices}
    (ROOT / "SELECTION.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(choices, indent=2))


if __name__ == "__main__":
    main()
