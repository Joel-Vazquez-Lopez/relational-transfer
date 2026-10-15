"""Sealed fresh-graph confirmation runner and graph-level analysis."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.stats import t

import generator_gate_v1 as gate
import role_aux_v1 as auxiliary
import fresh_graph_manifest_v1 as selector

ROOT = Path(__file__).resolve().parent / "role_aux_confirmation_v1"
MANIFEST = json.loads((ROOT / "GRAPH_MANIFEST.json").read_text())
PROTOCOL = json.loads((ROOT / "PROTOCOL.json").read_text())
SEEDS = tuple(MANIFEST["training_seeds"])
ARMS = ("balanced_baseline", "unordered_identities", "ordered_roles")
WORLDS = tuple(MANIFEST["worlds"])


def configure_modules() -> None:
    gate.WORLDS = {
        name: {"edges": tuple(tuple(edge) for edge in info["canonical_edges"])}
        for name, info in MANIFEST["worlds"].items()}
    gate.SEEDS = SEEDS
    gate.OUTPUT = ROOT / "baseline"
    auxiliary.ROOT = ROOT / "auxiliary"

    def configure(world: str) -> tuple[int, ...]:
        info = MANIFEST["worlds"][world]
        gate.base.N_ID = 10
        gate.base.VOCAB = gate.VOCAB
        gate.base.BASE_EDGES = gate.WORLDS[world]["edges"]
        gate.base.MAPPINGS = {"canonical": tuple(range(10))}
        return tuple(info["label_permutation"])

    gate.configure = configure


def model_path(world: str, seed: int, arm: str) -> Path:
    if arm == "balanced_baseline":
        return ROOT / "baseline" / "full" / world / "balanced_distractor" / f"seed_{seed}"
    return ROOT / "auxiliary" / "full" / world / arm / f"seed_{seed}"


def audit() -> dict:
    expected = selector.generate()
    assert expected == MANIFEST, "Graph manifest or source hashes changed"
    assert PROTOCOL["status"] == "frozen_before_confirmation_training"
    assert len(WORLDS) == 8 and len(SEEDS) == 3 and len(ARMS) == 3
    prior = [selector.graph(info["edges"]) for info in selector.PREVIOUS]
    fresh = []
    configure_modules()
    for world in WORLDS:
        info = MANIFEST["worlds"][world]
        graph = selector.graph(info["canonical_edges"])
        assert not any(nx.is_isomorphic(graph, older) for older in prior + fresh)
        fresh.append(graph)
        mapping = gate.configure(world)
        assert sorted(mapping) == list(range(10))
        labeled_edges = sorted(sorted([mapping[a], mapping[b]])
                               for a, b in info["canonical_edges"])
        assert labeled_edges == info["labeled_edges"]
        assert len(gate.base.make_pool("canonical")) == 1280
    return {"valid": True, "fresh_graphs": len(fresh), "historical_graphs_excluded": len(prior),
            "starts_per_graph": len(SEEDS), "models": len(WORLDS) * len(SEEDS) * len(ARMS),
            "manifest_sha256": hashlib.sha256((ROOT / "GRAPH_MANIFEST.json").read_bytes()).hexdigest(),
            "protocol_sha256": hashlib.sha256((ROOT / "PROTOCOL.json").read_bytes()).hexdigest()}


def train(world: str, seed: int, arm: str) -> None:
    assert world in WORLDS and seed in SEEDS and arm in ARMS
    audit()
    configure_modules()
    if arm == "balanced_baseline":
        gate.train(world, "balanced_distractor", seed, gate.STEPS, False)
    else:
        auxiliary.train(world, arm, seed)


def signflip_p(values: list[float]) -> float:
    observed = abs(float(np.mean(values)))
    extreme = 0
    for signs in itertools.product((-1, 1), repeat=len(values)):
        permuted = abs(float(np.mean(np.asarray(values) * signs)))
        extreme += permuted >= observed - 1e-12
    return extreme / (2 ** len(values))


def summarize() -> dict:
    records = []
    missing = []
    for world, seed, arm in itertools.product(WORLDS, SEEDS, ARMS):
        directory = model_path(world, seed, arm)
        metrics_path = directory / "metrics.json"
        checkpoint = directory / "model.pt"
        if not metrics_path.exists() or not checkpoint.exists():
            missing.append({"world": world, "seed": seed, "arm": arm})
            continue
        data = json.loads(metrics_path.read_text())
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == data["model_sha256"]
        primary = data["summary"]["260107"]
        second = data["summary"]["260108"]
        records.append({"world": world, "seed": seed, "arm": arm,
                        "seen_four_choice": primary["seen"]["four_choice_accuracy"],
                        "unseen_binary_role": primary["unseen"]["binary_role_accuracy"],
                        "unseen_four_choice": primary["unseen"]["four_choice_accuracy"],
                        "unseen_event": primary["unseen"]["event_accuracy"],
                        "second_unseen_binary_role": second["unseen"]["binary_role_accuracy"],
                        "weak_pairs": sum(not row["observed"] and row["binary_role_accuracy"] < .8
                                          for row in data["evaluation"]["260107"])})
    output = {"complete_models": len(records), "planned_models": 72, "missing": missing}
    if missing:
        return output
    index = {(row["world"], row["seed"], row["arm"]): row for row in records}
    graph_rows = []
    for world in WORLDS:
        values = {}
        for arm in ARMS:
            rows = [index[(world, seed, arm)] for seed in SEEDS]
            values[arm] = {metric: float(np.mean([row[metric] for row in rows]))
                           for metric in ("seen_four_choice", "unseen_binary_role",
                                          "unseen_four_choice", "unseen_event",
                                          "second_unseen_binary_role", "weak_pairs")}
        graph_rows.append({"world": world, "arms": values,
                           "ordered_minus_unordered": values["ordered_roles"]["unseen_binary_role"]
                           - values["unordered_identities"]["unseen_binary_role"],
                           "ordered_minus_baseline": values["ordered_roles"]["unseen_binary_role"]
                           - values["balanced_baseline"]["unseen_binary_role"]})
    differences = [row["ordered_minus_unordered"] for row in graph_rows]
    mean = float(np.mean(differences))
    halfwidth = float(t.ppf(.975, df=7) * np.std(differences, ddof=1) / math.sqrt(8))
    output.update({
        "graph_rows": graph_rows,
        "primary_mean_difference": mean,
        "primary_exact_two_sided_signflip_p": signflip_p(differences),
        "primary_graph_t_interval_95": [mean - halfwidth, mean + halfwidth],
        "mean_seen_four_choice_loss_vs_unordered": float(np.mean([
            row["arms"]["unordered_identities"]["seen_four_choice"]
            - row["arms"]["ordered_roles"]["seen_four_choice"] for row in graph_rows])),
        "mean_unseen_event_loss_vs_unordered": float(np.mean([
            row["arms"]["unordered_identities"]["unseen_event"]
            - row["arms"]["ordered_roles"]["unseen_event"] for row in graph_rows])),
        "arm_means": {arm: {metric: float(np.mean([
            row["arms"][arm][metric] for row in graph_rows]))
            for metric in ("seen_four_choice", "unseen_binary_role", "unseen_four_choice",
                           "unseen_event", "second_unseen_binary_role", "weak_pairs")}
            for arm in ARMS},
    })
    output["success_rule_passed"] = (
        mean >= .05 and output["primary_exact_two_sided_signflip_p"] <= .05
        and output["mean_seen_four_choice_loss_vs_unordered"] <= .01
        and output["mean_unseen_event_loss_vs_unordered"] <= .02)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("audit")
    sub.add_parser("summarize")
    run = sub.add_parser("train")
    run.add_argument("--world", choices=WORLDS, required=True)
    run.add_argument("--seed", type=int, choices=SEEDS, required=True)
    run.add_argument("--arm", choices=ARMS, required=True)
    args = parser.parse_args()
    if args.command == "audit":
        print(json.dumps(audit(), indent=2))
    elif args.command == "summarize":
        print(json.dumps(summarize(), indent=2))
    else:
        train(args.world, args.seed, args.arm)


if __name__ == "__main__":
    main()
