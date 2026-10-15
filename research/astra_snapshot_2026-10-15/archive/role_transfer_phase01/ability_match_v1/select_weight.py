"""Select role-loss strength from nonfocal ability only; never read focal scores."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import statistics
from pathlib import Path

import match_experiment as exp

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "phase2_development_v1" / "full"


def prior_ability(family: int, condition: str, seed: int) -> float:
    file = PRIOR / f"family{family}" / condition / f"seed_{seed}" / "participant_only" / "checkpoint_6000.json"
    return json.loads(file.read_text())["evaluation"][str(exp.SCENE)]["nonfocal_unseen_binary_role"]


def cells(weight: float, families: range) -> list[dict]:
    out = []
    for family, condition, seed in itertools.product(families, exp.CONDITIONS, exp.SEEDS):
        folder = exp.dest(family, condition, seed, weight)
        record = json.loads((folder / "metrics.json").read_text())
        assert (record["family"], record["condition"], record["seed"], record["weight"]) == (family, condition, seed, weight)
        assert record["model_sha256"] == exp.digest(folder / "model_step_6000.pt")
        assert record["protocol_sha256"] == exp.digest(HERE / "PROTOCOL.json")
        out.append({"family": family, "condition": condition, "seed": seed,
                    "role_ability": record["ability_only"]["nonfocal_binary_role"],
                    "familiar": record["ability_only"]["familiar_four_choice"],
                    "participant_ability": prior_ability(family, condition, seed)})
    return out


def summarize(weight: float, families: range) -> dict:
    rows = cells(weight, families)
    role = statistics.mean(x["role_ability"] for x in rows)
    participant = statistics.mean(x["participant_ability"] for x in rows)
    return {"weight": weight, "role_nonfocal": role,
            "participant_nonfocal": participant, "signed_gap": role - participant,
            "absolute_gap": abs(role - participant), "minimum_familiar": min(x["familiar"] for x in rows),
            "family_gaps": [statistics.mean(x["role_ability"] - x["participant_ability"]
                                           for x in rows if x["family"] == family)
                            for family in families]}


def calibration() -> None:
    exp.audit()
    p = exp.protocol()
    rows = [summarize(weight, range(2)) for weight in exp.WEIGHTS]
    eligible = [x for x in rows if x["minimum_familiar"] >= p["familiar_minimum"]]
    assert eligible, "no candidate preserved familiar accuracy"
    chosen = min(eligible, key=lambda x: (x["absolute_gap"], -x["weight"]))
    result = {"phase": "calibration", "candidates": rows, "selected_weight": chosen["weight"],
              "calibration_match": chosen["absolute_gap"] <= p["tolerance"],
              "source_sha256": exp.digest(Path(__file__)),
              "protocol_sha256": exp.digest(HERE / "PROTOCOL.json")}
    (HERE / "CALIBRATION.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def full_ability() -> None:
    exp.audit()
    p = exp.protocol()
    choice = json.loads((HERE / "CALIBRATION.json").read_text())
    assert choice["calibration_match"]
    chosen = choice["selected_weight"]
    row = summarize(chosen, range(8))
    row.update({"phase": "full_development_ability_gate",
                "valid_global_match": row["absolute_gap"] <= p["tolerance"]
                                      and row["minimum_familiar"] >= p["familiar_minimum"],
                "source_sha256": exp.digest(Path(__file__)),
                "protocol_sha256": exp.digest(HERE / "PROTOCOL.json")})
    (HERE / "ABILITY_GATE.json").write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("calibration", "full"))
    args = ap.parse_args()
    if args.stage == "calibration":
        calibration()
    else:
        full_ability()
