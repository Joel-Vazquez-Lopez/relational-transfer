"""Frozen Phase 2 development analysis; graph family is the independent unit."""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t

import experiment as exp

HERE = Path(__file__).resolve().parent
PRIMARY_SCENE = "260107"


def mean(x):
    return float(np.mean(x))


def signflip(x):
    values = np.asarray(x, dtype=float)
    threshold = abs(mean(values))
    return sum(abs(mean(values * signs)) >= threshold - 1e-12
               for signs in itertools.product((-1, 1), repeat=len(values))) / (2 ** len(values))


def interval(x):
    values = np.asarray(x, dtype=float)
    half = float(t.ppf(.975, len(values) - 1) * np.std(values, ddof=1) / math.sqrt(len(values)))
    return [mean(values) - half, mean(values) + half]


def logodds(accuracy: float, n: int = 512):
    successes = accuracy * n
    return math.log((successes + .5) / (n - successes + .5))


def load_all():
    p = exp.protocol()
    assert p["status"] == "frozen_before_full_training"
    assert p["source_manifest_sha256"] == exp.digest(exp.SOURCE_MANIFEST)
    assert p["script_sha256"] == exp.digest(exp.Path(exp.__file__))
    records = {}
    for i, condition, seed, arm in itertools.product(
            range(8), exp.CONDITIONS, exp.SEEDS, exp.ARMS):
        dest = exp.output_dir(i, condition, seed, arm, False, exp.SWAP_WEIGHT)
        data = json.loads((dest / "metrics.json").read_text())
        assert data["family"] == i and data["condition"] == condition
        assert data["seed"] == seed and data["arm"] == arm
        assert data["steps"] == exp.TRAIN_STEPS
        assert data["source_manifest_sha256"] == p["source_manifest_sha256"]
        assert data["script_sha256"] == p["script_sha256"]
        assert data["protocol_sha256"] == exp.digest(exp.PROTOCOL)
        assert data["model_sha256"] == exp.digest(dest / f"model_step_{exp.TRAIN_STEPS}.pt")
        checkpoints = {}
        for step in exp.CHECKPOINTS:
            cp = json.loads((dest / f"checkpoint_{step}.json").read_text())
            assert cp["step"] == step
            assert cp["model_sha256"] == exp.digest(dest / f"model_step_{step}.pt")
            checkpoints[step] = cp["evaluation"]
        records[(i, condition, seed, arm)] = checkpoints
    assert len(records) == 160
    return records


def cell(records, family, condition, seed, arm, step, scene=PRIMARY_SCENE):
    return records[(family, condition, seed, arm)][step][scene]


def family_penalty(records, family, arm, step, scene=PRIMARY_SCENE, metric="binary_role_accuracy"):
    return mean([
        cell(records, family, "high", seed, arm, step, scene)["focal"][metric]
        - cell(records, family, "sham", seed, arm, step, scene)["focal"][metric]
        for seed in exp.SEEDS])


def contrast(records, left, right, steps, scene=PRIMARY_SCENE):
    effects = [family_penalty(records, i, left, steps[left], scene)
               - family_penalty(records, i, right, steps[right], scene)
               for i in range(8)]
    return {"mean": mean(effects), "95pct_family_t_interval": interval(effects),
            "exact_two_sided_family_signflip_p": signflip(effects),
            "family_effects": effects, "positive_families": sum(v > 0 for v in effects)}


def performance_match(records):
    target = mean([
        cell(records, i, condition, seed, "ordered_roles", 6000)["nonfocal_unseen_binary_role"]
        for i, condition, seed in itertools.product(range(8), exp.CONDITIONS, exp.SEEDS)])
    selected = {"ordered_roles": 6000}
    rows = {"ordered_roles": {"selected_step": 6000, "nonfocal_accuracy": target,
                               "absolute_gap": 0.0, "within_2_points": True}}
    for arm in exp.ARMS:
        if arm == "ordered_roles":
            continue
        candidates = []
        for step in exp.CHECKPOINTS:
            cells = [cell(records, i, condition, seed, arm, step)
                     for i, condition, seed in itertools.product(range(8), exp.CONDITIONS, exp.SEEDS)]
            familiar_min = min(c["familiar_four_choice"] for c in cells)
            accuracy = mean([c["nonfocal_unseen_binary_role"] for c in cells])
            if familiar_min >= .95:
                candidates.append((abs(accuracy - target), step, accuracy, familiar_min))
        if not candidates:
            rows[arm] = {"selected_step": None, "reason": "no checkpoint passed familiar 95%"}
            continue
        gap, step, accuracy, familiar_min = min(candidates)
        selected[arm] = step
        rows[arm] = {"selected_step": step, "nonfocal_accuracy": accuracy,
                     "absolute_gap": gap, "within_2_points": gap <= .02,
                     "minimum_familiar_four_choice": familiar_min}
    return selected, rows


def analyze(records):
    fixed = {arm: 6000 for arm in exp.ARMS}
    matched_steps, matching = performance_match(records)
    family_rows = []
    for i in range(8):
        row = {"family": i, "penalties": {}, "focal_by_cell": {}}
        for arm in exp.ARMS:
            row["penalties"][arm] = family_penalty(records, i, arm, 6000)
            row["focal_by_cell"][arm] = {
                condition: mean([cell(records, i, condition, seed, arm, 6000)["focal"]["binary_role_accuracy"]
                                 for seed in exp.SEEDS]) for condition in exp.CONDITIONS}
        family_rows.append(row)
    primary = contrast(records, "ordered_roles", "participant_only", fixed)
    swap = contrast(records, "role_swap", "permutation_control", fixed)
    swap_vs_ordered = contrast(records, "role_swap", "ordered_roles", fixed)
    second = contrast(records, "ordered_roles", "participant_only", fixed, scene="260108")
    matched = {}
    for left, right in (("ordered_roles", "participant_only"),
                        ("role_swap", "permutation_control"),
                        ("ordered_roles", "standard")):
        if left in matched_steps and right in matched_steps:
            result = contrast(records, left, right, matched_steps)
            result["valid_matching"] = matching[left]["within_2_points"] and matching[right]["within_2_points"]
            matched[f"{left}_minus_{right}"] = result
    cells = [cell(records, i, condition, seed, arm, 6000)
             for i, condition, seed, arm in itertools.product(
                 range(8), exp.CONDITIONS, exp.SEEDS, exp.ARMS)]
    by_arm = {}
    for arm in exp.ARMS:
        xs = [cell(records, i, condition, seed, arm, 6000)
              for i, condition, seed in itertools.product(range(8), exp.CONDITIONS, exp.SEEDS)]
        by_arm[arm] = {
            "mean_focal_binary_role": mean([x["focal"]["binary_role_accuracy"] for x in xs]),
            "mean_nonfocal_unseen_binary_role": mean([x["nonfocal_unseen_binary_role"] for x in xs]),
            "min_familiar_four_choice": min(x["familiar_four_choice"] for x in xs),
            "mean_familiar_four_choice": mean([x["familiar_four_choice"] for x in xs]),
            "mean_focal_event": mean([x["focal"]["event_accuracy"] for x in xs]),
            "mean_focal_four_choice": mean([x["focal"]["four_choice_accuracy"] for x in xs]),
            "ceiling_saturated_families": [i for i in range(8)
                if all(mean([cell(records, i, c, s, arm, 6000)["focal"]["binary_role_accuracy"]
                             for s in exp.SEEDS]) >= .97 for c in exp.CONDITIONS)],
        }
    logodds_penalties = {}
    for arm in exp.ARMS:
        per_family = [mean([
            logodds(cell(records, i, "high", seed, arm, 6000)["focal"]["binary_role_accuracy"])
            - logodds(cell(records, i, "sham", seed, arm, 6000)["focal"]["binary_role_accuracy"])
            for seed in exp.SEEDS]) for i in range(8)]
        logodds_penalties[arm] = {"mean": mean(per_family), "family_effects": per_family}
    return {"status": "development_only", "models": len(records),
            "primary_ordered_minus_participant": primary,
            "role_swap_minus_permutation_control": swap,
            "role_swap_minus_ordered": swap_vs_ordered,
            "second_scene_ordered_minus_participant": second,
            "performance_matching": matching, "performance_matched_interactions": matched,
            "fixed_checkpoint_by_arm": by_arm,
            "logodds_high_minus_sham": logodds_penalties,
            "family_rows": family_rows,
            "minimum_familiar_accuracy_all_models": min(x["familiar_four_choice"] for x in cells),
            "limitations": ["These graph families and their earlier outcomes were already examined.",
                            "The sham/high rewire changes multiple graph properties.",
                            "A flat penalty at ceiling is not evidence of changed structural dependence.",
                            "The role-swap arm enforces identity-relabeling equivariance, not gold roles."]}


def main():
    records = load_all()
    result = analyze(records)
    (HERE / "SUMMARY.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"models": result["models"],
                      "primary": result["primary_ordered_minus_participant"],
                      "role_swap": result["role_swap_minus_permutation_control"],
                      "performance_matching": result["performance_matching"]}, indent=2))


if __name__ == "__main__":
    main()
