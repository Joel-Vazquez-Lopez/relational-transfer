"""Frozen-checkpoint role-state test on the fresh rewiring repair models."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import t

import role_mechanism_v1 as prior_probe
import role_mechanism_v2 as prior_projection
import role_mechanism_v4 as prior_intervention
import rq2_rewire_repair_v1 as repair

HERE = Path(__file__).resolve().parent
ROOT = HERE / "mechanism_rewire_v1"
PROTOCOL = ROOT / "PROTOCOL.json"
SEEDS = repair.SEEDS
CONDITIONS = repair.CONDITIONS
ARMS = repair.ARMS
FAMILIES = repair.FAMILIES


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol() -> dict:
    sources = [HERE / "mechanism_rewire_v1.py", HERE / "role_mechanism_v1.py",
               HERE / "role_mechanism_v2.py", HERE / "role_mechanism_v4.py",
               HERE / "rq2_rewire_repair_v1.py"]
    return {"status": "frozen_before_internal_state_analysis", "date": "2026-10-09",
            "checkpoint_study": "rq2_rewire_repair_v1, all 96 fixed checkpoints, no weight updates",
            "checkpoint_manifest_sha256": digest(repair.MANIFEST),
            "checkpoint_summary_sha256": digest(repair.ROOT / "SUMMARY.json"),
            "question": "On fresh rewiring graphs, does ordered-role training make actor/patient order more decodable and more consequential to the held-out pair decision?",
            "data": {"probe_train": "all 20 observed unordered pairs, both directions, two events, eight distractors: 640 independent scene renderings per model, seed 620107",
                     "probe_test": "all 25 unseen pairs: 800 independent renderings, seed 620108; focal subset 32 renderings",
                     "intervention_test": "same focal unseen pair, 128 balanced episodes from fixed scene suite 260108; source message unchanged",
                     "models": FAMILIES * len(SEEDS) * len(CONDITIONS) * len(ARMS)},
            "probe": "fixed ridge 0.1 linear readout from final scene-tower state to separate actor and patient IDs; train on observed pairs only",
            "intervention": "project correct-versus-reversed candidate-state shift into nine-dimensional actor-minus-patient readout subspace, exchange that component, and renormalize; five random nine-dimensional subspaces scaled to the same per-episode L2 shift",
            "primary_endpoint": "on high-rewiring focal pair, random-control mean minus role-swap correct-versus-reversed score margin; ordered minus unordered, paired by seed then averaged equally over eight families",
            "primary_test": "exact two-sided family-level sign flip; describe as support if mean > 0 and p <= .05",
            "secondary": ["observed-trained probe joint actor/patient accuracy on unseen and focal scene states", "intervention accuracy change", "sham-condition contrast", "role-shift energy"],
            "uncertainty": "report all eight family contrasts and t(7) 95 percent interval; seeds and episodes are nested, not independent replications",
            "limitations": "linear probe and hybrid-state intervention do not prove a unique neuron or complete circuit; movement and identity cues remain available in generated scenes; this is not an LLM result",
            "source_hashes": {str(path.relative_to(HERE.parent)): digest(path) for path in sources}}


def prepare() -> None:
    if PROTOCOL.exists():
        raise FileExistsError(PROTOCOL)
    ROOT.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(protocol(), indent=2) + "\n")
    print(json.dumps({"protocol": str(PROTOCOL), "models": 96,
                      "protocol_sha256": digest(PROTOCOL)}, indent=2))


def audit() -> dict:
    assert json.loads(PROTOCOL.read_text()) == protocol()
    assert (repair.ROOT / "SUMMARY.json").exists()
    repair.audit()
    return {"valid": True, "models": 96, "protocol_sha256": digest(PROTOCOL)}


def one_model(index: int, condition: str, arm: str, seed: int,
              train_suite: tuple, test_suite: tuple, focal_episodes: tuple) -> dict:
    repair.configure()
    repair.gate.configure(repair.world_name(index, condition))
    path = repair.model_path(index, condition, arm, seed) / "model.pt"
    model = repair.auxiliary.RoleListener()
    model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    model.eval()
    family = repair.load()[0]["families"][index]
    observed = {tuple(edge) for edge in family["conditions"][condition]["edges"]}
    focal = tuple(family["focal_pair"])
    assert focal not in observed
    train_scene, train_labels, train_pairs = train_suite
    test_scene, test_labels, test_pairs = test_suite
    seen = np.asarray([tuple(pair) in observed for pair in train_pairs])
    unseen = np.asarray([tuple(pair) not in observed for pair in test_pairs])
    focal_mask = np.asarray([tuple(pair) == focal for pair in test_pairs])
    assert seen.sum() == 640 and unseen.sum() == 800 and focal_mask.sum() == 32
    x_train = prior_probe.scene_states(model, train_scene[seen])["final"]
    x_test = prior_probe.scene_states(model, test_scene[unseen])["final"]
    x_focal = prior_probe.scene_states(model, test_scene[focal_mask])["final"]
    unseen_probe, weight = prior_probe.ridge_roles(
        x_train, train_labels[seen], x_test, test_labels[unseen])
    focal_probe, _ = prior_probe.ridge_roles(
        x_train, train_labels[seen], x_focal, test_labels[focal_mask])
    role_project = prior_projection.project(weight[:, :10] - weight[:, 10:])
    q, correct, reverse = prior_probe.embeddings(model, focal_episodes)
    assert len(q) == 128
    delta = reverse - correct
    role_shift = delta @ role_project
    target_norm = np.linalg.norm(role_shift, axis=1, keepdims=True)
    assert np.all(target_norm > 1e-8)
    ordinary = prior_probe.assess_swap(q, correct, reverse, None)
    role = prior_intervention.evaluate_shift(q, correct, reverse, role_shift)
    random_controls = []
    for random_index in range(5):
        rng = np.random.default_rng(np.random.SeedSequence(
            [910107, index, random_index]))
        basis, _ = np.linalg.qr(rng.normal(size=(48, 9)), mode="reduced")
        direction = delta @ (basis @ basis.T)
        direction_norm = np.linalg.norm(direction, axis=1, keepdims=True)
        assert np.all(direction_norm > 1e-8)
        matched = direction / direction_norm * target_norm
        random_controls.append(prior_intervention.evaluate_shift(
            q, correct, reverse, matched))
    random_margin = float(np.mean([row["mean_margin"] for row in random_controls]))
    random_accuracy = float(np.mean([row["binary_role_accuracy"] for row in random_controls]))
    return {"family": index, "condition": condition, "arm": arm, "seed": seed,
            "model_sha256": digest(path), "focal_pair": list(focal),
            "probe_unseen": unseen_probe, "probe_focal": focal_probe,
            "ordinary": ordinary, "role_swap": role,
            "random_controls": random_controls,
            "role_specific_margin_drop": random_margin - role["mean_margin"],
            "role_specific_accuracy_drop": random_accuracy - role["binary_role_accuracy"],
            "mean_role_shift_norm": float(target_norm.mean())}


def run_family(index: int) -> None:
    assert 0 <= index < FAMILIES
    target = ROOT / f"family{index}.json"
    if target.exists():
        raise FileExistsError(target)
    assert json.loads(PROTOCOL.read_text()) == protocol()
    repair.configure()
    repair.gate.configure(repair.world_name(index, "high"))
    torch.set_num_threads(2)
    train_suite = prior_probe.scene_suite(620107)
    test_suite = prior_probe.scene_suite(620108)
    episodes = prior_probe.eval_rows()
    focal = tuple(repair.load()[0]["families"][index]["focal_pair"])
    records = [one_model(index, condition, arm, seed,
                         train_suite, test_suite, episodes[focal])
               for seed, condition, arm in itertools.product(SEEDS, CONDITIONS, ARMS)]
    target.write_text(json.dumps({"protocol_sha256": digest(PROTOCOL),
                                  "records": records}, indent=2) + "\n")
    print(json.dumps({"family": index, "models": len(records),
                      "high_arm_role_specific_margin_drop": {
                          arm: float(np.mean([r["role_specific_margin_drop"] for r in records
                                              if r["condition"] == "high" and r["arm"] == arm]))
                          for arm in ARMS}}, indent=2), flush=True)


def signflip(values: list[float]) -> float:
    observed = abs(float(np.mean(values)))
    return sum(abs(float(np.mean(np.asarray(values) * signs))) >= observed - 1e-12
               for signs in itertools.product((-1, 1), repeat=len(values))) / (2 ** len(values))


def summarize() -> dict:
    families = []
    for index in range(FAMILIES):
        data = json.loads((ROOT / f"family{index}.json").read_text())
        assert data["protocol_sha256"] == digest(PROTOCOL)
        records = data["records"]
        assert len(records) == len(SEEDS) * len(CONDITIONS) * len(ARMS)
        for record in records:
            path = repair.model_path(index, record["condition"], record["arm"],
                                     record["seed"]) / "model.pt"
            assert digest(path) == record["model_sha256"]
        def mean(condition: str, arm: str, key: str) -> float:
            return float(np.mean([r[key] for r in records
                                  if r["condition"] == condition and r["arm"] == arm]))
        high_ordered = mean("high", "ordered_roles", "role_specific_margin_drop")
        high_unordered = mean("high", "unordered_identities", "role_specific_margin_drop")
        families.append({"family": index, "high_ordered_role_specific_margin_drop": high_ordered,
                         "high_unordered_role_specific_margin_drop": high_unordered,
                         "high_ordered_minus_unordered": high_ordered - high_unordered,
                         "records": records})
    values = [row["high_ordered_minus_unordered"] for row in families]
    average = float(np.mean(values))
    half = float(t.ppf(.975, FAMILIES - 1)
                 * np.std(values, ddof=1) / math.sqrt(FAMILIES))
    return {"complete_models": FAMILIES * len(SEEDS) * len(CONDITIONS) * len(ARMS),
            "primary_mean": average, "primary_p_exact_two_sided": signflip(values),
            "primary_95pct_t_interval": [average - half, average + half],
            "prespecified_support": average > 0 and signflip(values) <= .05,
            "family_rows": families}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "audit", "family", "summarize"))
    parser.add_argument("--family", type=int)
    args = parser.parse_args()
    if args.action == "prepare": prepare()
    elif args.action == "audit": print(json.dumps(audit(), indent=2))
    elif args.action == "family": run_family(args.family)
    else: print(json.dumps(summarize(), indent=2))


if __name__ == "__main__": main()
