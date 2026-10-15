"""Post-selection focal evaluation; refuses an unsuccessful ability match."""
from __future__ import annotations

import itertools
import json
import math
import statistics
from pathlib import Path

import scipy.stats
import torch

import match_experiment as exp

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "phase2_development_v1" / "full"


def exact_signflip(values: list[float]) -> float:
    observed = abs(statistics.mean(values))
    count = sum(abs(statistics.mean(s * x for s, x in zip(signs, values))) >= observed - 1e-12
                for signs in itertools.product((-1, 1), repeat=len(values)))
    return count / (2 ** len(values))


def main() -> None:
    exp.audit()
    gate = json.loads((HERE / "ABILITY_GATE.json").read_text())
    if not gate["valid_global_match"]:
        raise RuntimeError("nonfocal ability match failed; focal evaluation remains sealed")
    weight = gate["weight"]
    family_rows = []
    for family in range(8):
        cells = []
        for condition, seed in itertools.product(exp.CONDITIONS, exp.SEEDS):
            folder = exp.dest(family, condition, seed, weight)
            record = json.loads((folder / "metrics.json").read_text())
            checkpoint = folder / "model_step_6000.pt"
            assert record["model_sha256"] == exp.digest(checkpoint)
            exp.prior.configure(family, condition)
            model = exp.prior.auxiliary.RoleListener()
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            model.eval()
            pair = tuple(exp.prior.graph(family, condition)[0]["focal_pair"])
            pair_index = list(itertools.combinations(range(10), 2)).index(pair)
            score = exp.prior.eval_pair(model, *pair, pair_index, exp.SCENE, repeats=4)
            ref = json.loads((PRIOR / f"family{family}" / condition /
                              f"seed_{seed}" / "participant_only" /
                              "checkpoint_6000.json").read_text())["evaluation"][str(exp.SCENE)]["focal"]
            cells.append({"condition": condition, "seed": seed,
                          "ordered_focal_binary_role": score["binary_role_accuracy"],
                          "participant_focal_binary_role": ref["binary_role_accuracy"],
                          "ordered_focal_four_choice": score["four_choice_accuracy"],
                          "ordered_focal_event": score["event_accuracy"]})
        def penalty(key: str) -> float:
            return statistics.mean(next(x[key] for x in cells if x["condition"] == "high" and x["seed"] == seed)
                                   - next(x[key] for x in cells if x["condition"] == "sham" and x["seed"] == seed)
                                   for seed in exp.SEEDS)
        ordered = penalty("ordered_focal_binary_role")
        participant = penalty("participant_focal_binary_role")
        family_rows.append({"family": family, "ordered_penalty": ordered,
                            "participant_penalty": participant, "attenuation": ordered - participant,
                            "ability_gap": gate["family_gaps"][family], "cells": cells})
    effects = [x["attenuation"] for x in family_rows]
    mean = statistics.mean(effects)
    half = scipy.stats.t.ppf(.975, len(effects) - 1) * statistics.stdev(effects) / math.sqrt(len(effects))
    result = {"status": "development_only", "weight": weight,
              "ability_gate": gate, "mean_attenuation": mean,
              "95pct_family_t_interval": [mean - half, mean + half],
              "exact_two_sided_family_signflip_p": exact_signflip(effects),
              "family_rows": family_rows,
              "limitations": ["This weight was calibrated on examined graphs using nonfocal ability only.",
                              "The comparison is development evidence, not independent confirmation.",
                              "An aggregate ability match may mask family-specific gaps."]}
    (HERE / "FOCAL_DEVELOPMENT.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"weight": weight, "mean_attenuation": mean,
                      "95pct_family_t_interval": result["95pct_family_t_interval"],
                      "family_effects": effects, "ability_family_gaps": gate["family_gaps"]}, indent=2))


if __name__ == "__main__":
    main()
