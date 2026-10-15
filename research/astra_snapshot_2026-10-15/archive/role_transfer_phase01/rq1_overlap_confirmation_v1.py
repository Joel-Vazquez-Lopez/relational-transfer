"""Frozen fresh-graph confirmation of shared-partner risk and role supervision."""
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

import fresh_graph_manifest_v1 as old_selector
import generator_gate_v1 as gate
import role_aux_v1 as auxiliary

HERE = Path(__file__).resolve().parent
ROOT = HERE / "rq1_overlap_confirmation_v1"
MANIFEST_PATH = ROOT / "GRAPH_MANIFEST.json"
PROTOCOL_PATH = ROOT / "PROTOCOL.json"
GRAPH_COUNT = 8
SEEDS = (291001, 291002, 291003)
ARMS = ("balanced_baseline", "unordered_identities", "ordered_roles")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prior_graphs() -> list[nx.Graph]:
    previous = [old_selector.graph(row["edges"]) for row in old_selector.PREVIOUS]
    old_manifest = json.loads(old_selector.DESTINATION.read_text())
    previous += [old_selector.graph(row["canonical_edges"])
                 for row in old_manifest["worlds"].values()]
    triangle_manifest = json.loads((HERE / "rq1_structure_v1" / "GRAPH_MANIFEST.json").read_text())
    previous += [old_selector.graph(row["canonical_edges"])
                 for pair in triangle_manifest["pairs"] for row in pair["graphs"].values()]
    assert len(previous) == 32
    assert all(not nx.is_isomorphic(left, right)
               for i, left in enumerate(previous) for right in previous[i + 1:])
    return previous


def overlap_groups(graph: nx.Graph) -> tuple[list[list[int]], list[list[int]], dict[str, int]]:
    counts = {str(count): 0 for count in range(5)}
    low, high = [], []
    for a, b in itertools.combinations(range(10), 2):
        if graph.has_edge(a, b):
            continue
        count = len(set(graph.neighbors(a)) & set(graph.neighbors(b)))
        counts[str(count)] += 1
        if count <= 1:
            low.append([a, b])
        if count >= 3:
            high.append([a, b])
    assert sum(counts.values()) == 25
    return low, high, counts


def generate_manifest() -> dict:
    previous = prior_graphs()
    selected: list[dict] = []
    for generator_seed in range(40000, 200000):
        graph = nx.random_regular_graph(4, 10, seed=generator_seed)
        low, high, _ = overlap_groups(graph)
        if len(low) < 3 or len(high) < 3:
            continue
        if any(nx.is_isomorphic(graph, older) for older in previous):
            continue
        if any(nx.is_isomorphic(graph, row["graph"]) for row in selected):
            continue
        permutation = np.random.default_rng(np.random.SeedSequence(
            [400000, generator_seed])).permutation(10).tolist()
        selected.append({"graph": graph, "generator_seed": generator_seed,
                         "label_permutation": permutation})
        if len(selected) == GRAPH_COUNT:
            break
    assert len(selected) == GRAPH_COUNT
    worlds = {}
    for i, row in enumerate(selected):
        edges = sorted([sorted(edge) for edge in row["graph"].edges()])
        mapping = row["label_permutation"]
        labeled = sorted([sorted([mapping[a], mapping[b]]) for a, b in edges])
        graph = old_selector.graph(labeled)
        low, high, counts = overlap_groups(graph)
        assert len(low) >= 3 and len(high) >= 3
        worlds[f"overlap{i}"] = {"generator_seed": row["generator_seed"],
                                 "canonical_edges": edges,
                                 "label_permutation": mapping,
                                 "labeled_edges": labeled,
                                 "low_overlap_unseen_pairs": low,
                                 "high_overlap_unseen_pairs": high,
                                 "unseen_overlap_counts": counts}
    paths = (HERE / "generator_gate_v1.py", HERE / "role_aux_v1.py",
             HERE / "all_pair_audit.py", HERE.parent / "d5_indirect_support" / "run.py",
             HERE.parent / "d5_indirect_support" / "path2_control.py")
    return {"status": "selected_by_graph_only_rule_before_training",
            "selection": "first eight new nonisomorphic connected 10-node degree-4 graphs from networkx random_regular_graph seeds >=40000, each with at least three unseen pairs at overlap 0-1 and at least three at overlap 3-4, excluding 32 prior graph classes",
            "historical_graph_classes_excluded": len(previous),
            "training_seeds": list(SEEDS), "worlds": worlds,
            "source_hashes": {str(path.relative_to(HERE.parent)): sha256(path) for path in paths}}


def protocol() -> dict:
    return {
        "status": "frozen_before_training", "date": "2026-10-08",
        "questions": [
            "Within a fresh training network, are unseen pairs with 3-4 shared observed partners harder on actor/patient direction than unseen pairs with 0-1 shared partners?",
            "Does ordered-role auxiliary supervision reduce that high-overlap penalty relative to a parameter- and compute-matched unordered-identity auxiliary control?"],
        "design": {"graphs": GRAPH_COUNT, "starts_per_graph": len(SEEDS),
                   "arms": list(ARMS), "models": GRAPH_COUNT * len(SEEDS) * len(ARMS),
                   "entities": 10, "observed_unordered_pairs": 20,
                   "degree_per_entity": 4, "unseen_unordered_pairs": 25,
                   "minimum_low_overlap_pairs_per_graph": 3,
                   "minimum_high_overlap_pairs_per_graph": 3,
                   "low_overlap": "0 or 1 common observed partner",
                   "high_overlap": "3 or 4 common observed partners",
                   "updates": 6000, "batch_size": 128,
                   "generator": "balanced distractors and dedicated padding token",
                   "auxiliary_weight": 0.5,
                   "endpoint": "final checkpoint only, no test-based checkpoint selection"},
        "primary_endpoint": "baseline high-overlap minus low-overlap binary role accuracy, each pair averaged within category, then three starts averaged within graph, then equal-weight mean over eight graphs",
        "primary_test": "exact two-sided sign-flip over eight graph-level penalties, all 256 sign assignments",
        "primary_success_rule": "mean penalty <= -0.10, exact two-sided p <= 0.05, and every baseline model at least 0.95 familiar four-choice accuracy",
        "secondary_endpoint": "(ordered high-minus-low penalty) minus (unordered high-minus-low penalty), averaged over paired starts and graphs",
        "secondary_test": "exact two-sided sign-flip over eight graph-level paired attenuation differences, evaluated as a confirmatory second test only if primary rule passes",
        "secondary_success_rule": "mean attenuation >= +0.10 and exact two-sided p <= 0.05; mean familiar loss <=0.01 and mean unseen-event loss <=0.02 relative to unordered control",
        "multiplicity": "hierarchical gate: test secondary claim only after primary success; report both effect estimates regardless",
        "uncertainty": "report all eight graph effects and graph-level t(7) 95 percent intervals; pairs, starts, and scene renderings are nested, not independent graph replications",
        "fixed_evaluation": "all 20 seen and 25 unseen pairs, two existing 512-rendering scene suites with seeds 260107 and 260108; primary uses 260107",
        "other_secondary_measures": ["unseen four-choice accuracy", "unseen event accuracy",
                                     "seen four-choice accuracy", "weak-pair incidence",
                                     "second-rendering robustness", "accuracy by exact overlap count 0-4"],
        "no_adaptive_replacement": "all eight graphs and three starts remain regardless of learning or transfer; only same-configuration technical reruns are allowed",
        "scope_limit": "A positive within-graph association is predictive, not proof that overlap itself causes the failure; correlated local graph properties and training distractor co-occurrences remain alternatives. The task uses coded messages and generated scenes, not natural language or pretrained LLMs."
    }


def prepare() -> None:
    if MANIFEST_PATH.exists() or PROTOCOL_PATH.exists():
        raise FileExistsError("Frozen study files already exist")
    manifest = generate_manifest()
    ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    PROTOCOL_PATH.write_text(json.dumps(protocol(), indent=2) + "\n")
    print(json.dumps({"manifest": str(MANIFEST_PATH), "protocol": str(PROTOCOL_PATH),
                      "worlds": {name: {"seed": info["generator_seed"],
                                        "low": len(info["low_overlap_unseen_pairs"]),
                                        "high": len(info["high_overlap_unseen_pairs"])}
                                 for name, info in manifest["worlds"].items()}}, indent=2))


def load() -> tuple[dict, dict]:
    return json.loads(MANIFEST_PATH.read_text()), json.loads(PROTOCOL_PATH.read_text())


def configure_modules() -> None:
    manifest, _ = load()
    gate.WORLDS = {world: {"edges": tuple(map(tuple, info["canonical_edges"]))}
                   for world, info in manifest["worlds"].items()}
    gate.SEEDS = SEEDS
    gate.OUTPUT = ROOT / "baseline"
    auxiliary.ROOT = ROOT / "auxiliary"
    def configure(world: str) -> tuple[int, ...]:
        gate.base.N_ID = 10
        gate.base.VOCAB = gate.VOCAB
        gate.base.BASE_EDGES = gate.WORLDS[world]["edges"]
        gate.base.MAPPINGS = {"canonical": tuple(range(10))}
        return tuple(manifest["worlds"][world]["label_permutation"])
    gate.configure = configure


def model_path(world: str, seed: int, arm: str) -> Path:
    if arm == "balanced_baseline":
        return ROOT / "baseline" / "full" / world / "balanced_distractor" / f"seed_{seed}"
    return ROOT / "auxiliary" / "full" / world / arm / f"seed_{seed}"


def audit() -> dict:
    manifest, plan = load()
    assert manifest == generate_manifest(), "Manifest, graph selection or source hashes changed"
    assert plan == protocol(), "Protocol changed"
    assert len(manifest["worlds"]) == GRAPH_COUNT
    configure_modules()
    for world, info in manifest["worlds"].items():
        assert sorted(gate.configure(world)) == list(range(10))
        graph = old_selector.graph(info["labeled_edges"])
        low, high, counts = overlap_groups(graph)
        assert low == info["low_overlap_unseen_pairs"]
        assert high == info["high_overlap_unseen_pairs"]
        assert counts == info["unseen_overlap_counts"]
        assert len(gate.base.make_pool("canonical")) == 1280
    return {"valid": True, "new_graphs": GRAPH_COUNT,
            "historical_graph_classes_excluded": manifest["historical_graph_classes_excluded"],
            "models": GRAPH_COUNT * len(SEEDS) * len(ARMS),
            "manifest_sha256": sha256(MANIFEST_PATH), "protocol_sha256": sha256(PROTOCOL_PATH)}


def train(world: str, seed: int, arm: str) -> None:
    assert world in load()[0]["worlds"] and seed in SEEDS and arm in ARMS
    audit()
    configure_modules()
    if arm == "balanced_baseline":
        gate.train(world, "balanced_distractor", seed, gate.STEPS, False)
    else:
        auxiliary.train(world, arm, seed)


def exact_signflip(values: list[float]) -> float:
    observed = abs(float(np.mean(values)))
    count = sum(abs(float(np.mean(np.asarray(values) * signs))) >= observed - 1e-12
                for signs in itertools.product((-1, 1), repeat=len(values)))
    return count / (2 ** len(values))


def graph_t_interval(values: list[float]) -> list[float]:
    mean = float(np.mean(values))
    half = float(t.ppf(.975, len(values)-1) * np.std(values, ddof=1) / math.sqrt(len(values)))
    return [mean-half, mean+half]


def summarize() -> dict:
    manifest, _ = load()
    graph_rows = []
    missing = []
    for world, info in manifest["worlds"].items():
        categories = {"low": {tuple(pair) for pair in info["low_overlap_unseen_pairs"]},
                      "high": {tuple(pair) for pair in info["high_overlap_unseen_pairs"]}}
        arms = {}
        for arm in ARMS:
            starts = []
            for seed in SEEDS:
                directory = model_path(world, seed, arm)
                metrics_path, checkpoint = directory / "metrics.json", directory / "model.pt"
                if not metrics_path.exists() or not checkpoint.exists():
                    missing.append({"world": world, "seed": seed, "arm": arm})
                    continue
                metrics = json.loads(metrics_path.read_text())
                assert sha256(checkpoint) == metrics["model_sha256"]
                assert metrics["steps"] == 6000 and metrics["seed"] == seed
                assert tuple(metrics["mapping"]) == tuple(info["label_permutation"])
                suites = {}
                for scene_seed in (260107, 260108):
                    evaluation = metrics["evaluation"][str(scene_seed)]
                    assert len(evaluation) == 45
                    by_pair = {tuple(row["pair"]): row for row in evaluation}
                    assert all(not by_pair[pair]["observed"] for pair in categories["low"] | categories["high"])
                    suites[str(scene_seed)] = {category: {
                        metric: float(np.mean([by_pair[pair][metric] for pair in pairs]))
                        for metric in ("binary_role_accuracy", "four_choice_accuracy", "event_accuracy")}
                        for category, pairs in categories.items()}
                starts.append({"seed": seed, "categories": suites,
                               "seen_four_choice": metrics["summary"]["260107"]["seen"]["four_choice_accuracy"],
                               "unseen_event": metrics["summary"]["260107"]["unseen"]["event_accuracy"],
                               "unseen_role_all25": metrics["summary"]["260107"]["unseen"]["binary_role_accuracy"]})
            arms[arm] = starts
        if all(len(starts) == len(SEEDS) for starts in arms.values()):
            penalties = {}
            for arm, starts in arms.items():
                penalties[arm] = float(np.mean([
                    row["categories"]["260107"]["high"]["binary_role_accuracy"]
                    - row["categories"]["260107"]["low"]["binary_role_accuracy"]
                    for row in starts]))
            graph_rows.append({"world": world, "low_pairs": len(categories["low"]),
                               "high_pairs": len(categories["high"]),
                               "arm_penalties": penalties,
                               "ordered_minus_unordered_penalty":
                               penalties["ordered_roles"]-penalties["unordered_identities"],
                               "starts": arms})
    result = {"complete_graphs": len(graph_rows), "planned_graphs": GRAPH_COUNT,
              "complete_models": GRAPH_COUNT*len(SEEDS)*len(ARMS)-len(missing),
              "planned_models": GRAPH_COUNT*len(SEEDS)*len(ARMS),
              "missing_models": missing, "graph_rows": graph_rows}
    if missing:
        return result
    primary = [row["arm_penalties"]["balanced_baseline"] for row in graph_rows]
    secondary = [row["ordered_minus_unordered_penalty"] for row in graph_rows]
    result.update({
        "primary_mean_high_minus_low": float(np.mean(primary)),
        "primary_exact_two_sided_signflip_p": exact_signflip(primary),
        "primary_graph_t_interval_95": graph_t_interval(primary),
        "primary_success": float(np.mean(primary)) <= -.10
            and exact_signflip(primary) <= .05
            and all(start["seen_four_choice"] >= .95
                    for row in graph_rows for start in row["starts"]["balanced_baseline"]),
        "secondary_mean_penalty_attenuation": float(np.mean(secondary)),
        "secondary_exact_two_sided_signflip_p": exact_signflip(secondary),
        "secondary_graph_t_interval_95": graph_t_interval(secondary),
        "mean_familiar_loss_ordered_vs_unordered": float(np.mean([
            np.mean([start["seen_four_choice"] for start in row["starts"]["unordered_identities"]])
            - np.mean([start["seen_four_choice"] for start in row["starts"]["ordered_roles"]])
            for row in graph_rows])),
        "mean_unseen_event_loss_ordered_vs_unordered": float(np.mean([
            np.mean([start["unseen_event"] for start in row["starts"]["unordered_identities"]])
            - np.mean([start["unseen_event"] for start in row["starts"]["ordered_roles"]])
            for row in graph_rows])),
    })
    result["secondary_success_under_hierarchical_gate"] = (
        result["primary_success"]
        and result["secondary_mean_penalty_attenuation"] >= .10
        and result["secondary_exact_two_sided_signflip_p"] <= .05
        and result["mean_familiar_loss_ordered_vs_unordered"] <= .01
        and result["mean_unseen_event_loss_ordered_vs_unordered"] <= .02)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "audit", "summarize"):
        sub.add_parser(command)
    run = sub.add_parser("train")
    run.add_argument("--world", choices=[f"overlap{i}" for i in range(GRAPH_COUNT)], required=True)
    run.add_argument("--seed", type=int, choices=SEEDS, required=True)
    run.add_argument("--arm", choices=ARMS, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "audit":
        print(json.dumps(audit(), indent=2))
    elif args.command == "summarize":
        print(json.dumps(summarize(), indent=2))
    else:
        train(args.world, args.seed, args.arm)


if __name__ == "__main__":
    main()
