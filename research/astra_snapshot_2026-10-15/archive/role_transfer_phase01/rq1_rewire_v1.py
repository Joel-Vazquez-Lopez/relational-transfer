"""Prospective matched edge-rewiring study of focal role transfer."""
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

HERE = Path(__file__).resolve().parent
ROOT = HERE / "rq1_rewire_v1"
MANIFEST = ROOT / "GRAPH_MANIFEST.json"
PROTOCOL = ROOT / "PROTOCOL.json"
SEEDS = (301001, 301002, 301003)
ARMS = ("base", "sham", "high")
FAMILIES = 8


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def edges(graph: nx.Graph) -> list[list[int]]:
    return sorted([sorted(e) for e in graph.edges()])


def graph_from_edges(items: list[list[int]]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(10))
    graph.add_edges_from(items)
    assert nx.is_connected(graph) and len(graph.edges()) == 20
    assert all(degree == 4 for _, degree in graph.degree())
    return graph


def shared(graph: nx.Graph, pair: tuple[int, int]) -> int:
    a, b = pair
    return len(set(graph.neighbors(a)) & set(graph.neighbors(b)))


def signature(base: nx.Graph, variant: nx.Graph, pair: tuple[int, int]):
    a, b = pair
    if not nx.is_connected(variant) or variant.has_edge(a, b):
        return None
    if len(set(base.edges()) ^ set(variant.edges())) != 8:
        return None
    if any(len(set(base.neighbors(x)) - set(variant.neighbors(x))) != 1 for x in pair):
        return None
    triangles = nx.triangles(variant)
    return (sum(triangles.values()) // 3, triangles[a], triangles[b])


def find_variants(base: nx.Graph, pair: tuple[int, int]):
    choices: dict[tuple, dict[str, tuple[int, nx.Graph]]] = {}
    for swap_seed in range(5000):
        candidate = base.copy()
        nx.double_edge_swap(candidate, nswap=2, max_tries=100, seed=swap_seed)
        count = shared(candidate, pair)
        if count not in (1, 3):
            continue
        sig = signature(base, candidate, pair)
        if sig is None:
            continue
        key = "sham" if count == 1 else "high"
        bucket = choices.setdefault(sig, {})
        bucket.setdefault(key, (swap_seed, candidate))
        if len(bucket) == 2:
            return {arm: {"swap_seed": bucket[arm][0], "edges": edges(bucket[arm][1])}
                    for arm in ("sham", "high")}, sig
    return None


def make_manifest() -> dict:
    selected = []
    base_graphs = []
    for generator_seed in range(50014, 70000):
        base = nx.random_regular_graph(4, 10, seed=generator_seed)
        if not nx.is_connected(base):
            continue
        if any(nx.is_isomorphic(base, old) for old in base_graphs):
            continue
        focal = next(((a, b) for a, b in itertools.combinations(range(10), 2)
                      if not base.has_edge(a, b) and shared(base, (a, b)) == 1), None)
        if focal is None:
            continue
        found = find_variants(base, focal)
        if found is None:
            continue
        variants, sig = found
        permutation = np.random.default_rng(np.random.SeedSequence(
            [810000, generator_seed])).permutation(10).tolist()
        def mapped(items):
            return sorted([sorted([permutation[a], permutation[b]]) for a, b in items])
        selected.append({"generator_seed": generator_seed,
                         "focal_pair": sorted([permutation[x] for x in focal]),
                         "canonical_focal_pair": list(focal),
                         "label_permutation": permutation,
                         "matched_triangle_signature": list(sig),
                         "arms": {"base": {"edges": mapped(edges(base))},
                                  **{arm: {"swap_seed": row["swap_seed"],
                                           "edges": mapped(row["edges"])}
                                     for arm, row in variants.items()}}})
        base_graphs.append(base)
        if len(selected) == FAMILIES:
            break
    assert len(selected) == FAMILIES
    sources = [HERE / "rq1_rewire_v1.py", HERE / "generator_gate_v1.py",
               HERE / "all_pair_audit.py", HERE.parent / "d5_indirect_support" / "run.py",
               HERE.parent / "d5_indirect_support" / "path2_control.py"]
    return {"status": "graph_only_selection_before_model_training",
            "selection": "First eight nonisomorphic base graph classes from seeds >=50014 with a focal nonedge at one common partner and a two-swap sham/high pair matching global and focal triangle counts and one changed neighbor per focal endpoint. No model outcomes used.",
            "families": selected, "training_seeds": list(SEEDS),
            "source_hashes": {str(p.relative_to(HERE.parent)): digest(p) for p in sources}}


def make_protocol() -> dict:
    return {"status": "frozen_before_training", "date": "2026-10-09",
            "design": {"families": FAMILIES, "seeds_per_family": len(SEEDS),
                       "conditions": list(ARMS), "models": FAMILIES * len(SEEDS) * len(ARMS),
                       "entities": 10, "degree": 4, "observed_pairs": 20,
                       "focal_pair": "same labeled unseen pair in all three conditions",
                       "base_sham_overlap": 1, "high_overlap": 3,
                       "sham_high_matching": "two effective edge swaps from base; 8-edge symmetric difference; each focal endpoint loses one and gains one neighbor; same global and focal-endpoint triangle counts",
                       "updates": 6000, "generator": "balanced distractors, dedicated padding",
                       "evaluation": "all 45 pairs; fixed 512-rendering suites 260107 primary and 260108 robustness"},
            "primary_endpoint": "high minus sham focal-pair binary role accuracy, averaged over three paired starts per graph family, then equally over eight families; final checkpoint, first scene suite",
            "primary_test": "exact two-sided sign flip over eight family differences, all 256 sign assignments",
            "primary_success": "mean difference <= -0.10 and exact p <= 0.05; every model at least 0.95 familiar four-choice accuracy",
            "secondary": ["high minus base focal-pair role accuracy", "sham minus base focal-pair role accuracy", "second scene suite", "all other unseen pairs grouped by overlap", "familiar four-choice and event accuracy"],
            "uncertainty": "report all eight family effects and t(7) 95 percent interval; seeds and scene items are nested within family",
            "missingness": "no adaptive family or seed replacement; only same-configuration technical reruns",
            "interpretation": "A change after the matched rewiring supports a causal effect of the local graph modification, not unique causality of shared-partner overlap. Other local paths and training co-occurrences may change."}


def prepare() -> None:
    if MANIFEST.exists() or PROTOCOL.exists():
        raise FileExistsError("Frozen files already exist")
    manifest = make_manifest()
    ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    PROTOCOL.write_text(json.dumps(make_protocol(), indent=2) + "\n")
    print(json.dumps({"manifest": str(MANIFEST), "protocol": str(PROTOCOL),
                      "graph_seeds": [row["generator_seed"] for row in manifest["families"]],
                      "models": FAMILIES * len(SEEDS) * len(ARMS)}, indent=2))


def load() -> tuple[dict, dict]:
    return json.loads(MANIFEST.read_text()), json.loads(PROTOCOL.read_text())


def world_name(index: int, arm: str) -> str:
    return f"family{index}_{arm}"


def configure() -> None:
    manifest, _ = load()
    gate.WORLDS = {world_name(i, arm): {"edges": tuple(map(tuple, family["arms"][arm]["edges"]))}
                   for i, family in enumerate(manifest["families"]) for arm in ARMS}
    gate.SEEDS = SEEDS
    gate.OUTPUT = ROOT
    def configure_world(world: str) -> tuple[int, ...]:
        gate.base.N_ID = 10
        gate.base.VOCAB = gate.VOCAB
        gate.base.BASE_EDGES = gate.WORLDS[world]["edges"]
        gate.base.MAPPINGS = {"canonical": tuple(range(10))}
        return tuple(range(10))
    gate.configure = configure_world


def model_path(index: int, arm: str, seed: int) -> Path:
    return ROOT / "full" / world_name(index, arm) / "balanced_distractor" / f"seed_{seed}"


def audit() -> dict:
    manifest, protocol = load()
    assert manifest == make_manifest(), "Graph selection or source code changed"
    assert protocol == make_protocol(), "Protocol changed"
    configure()
    for i, family in enumerate(manifest["families"]):
        pair = tuple(family["focal_pair"])
        graphs = {arm: graph_from_edges(family["arms"][arm]["edges"]) for arm in ARMS}
        assert all(not graph.has_edge(*pair) for graph in graphs.values())
        assert [shared(graphs[arm], pair) for arm in ARMS] == [1, 1, 3]
        saved = family["matched_triangle_signature"]
        for arm in ("sham", "high"):
            observed = signature(graphs["base"], graphs[arm], pair)
            assert observed is not None and observed[0] == saved[0]
            assert sorted(observed[1:]) == sorted(saved[1:])
        for arm in ARMS:
            world = world_name(i, arm)
            gate.configure(world)
            assert len(gate.base.make_pool("canonical")) == 1280
    return {"valid": True, "families": FAMILIES, "models": FAMILIES * len(SEEDS) * len(ARMS),
            "manifest_sha256": digest(MANIFEST), "protocol_sha256": digest(PROTOCOL)}


def train(index: int, arm: str, seed: int) -> None:
    assert 0 <= index < FAMILIES and arm in ARMS and seed in SEEDS
    audit()
    configure()
    gate.train(world_name(index, arm), "balanced_distractor", seed, gate.STEPS, False)


def signflip(values: list[float]) -> float:
    observed = abs(float(np.mean(values)))
    return sum(abs(float(np.mean(np.asarray(values) * signs))) >= observed - 1e-12
               for signs in itertools.product((-1, 1), repeat=len(values))) / (2 ** len(values))


def summarize() -> dict:
    manifest, _ = load()
    rows = []
    for i, family in enumerate(manifest["families"]):
        pair = family["focal_pair"]
        by_arm = {}
        for arm in ARMS:
            starts = []
            for seed in SEEDS:
                path = model_path(i, arm, seed)
                data = json.loads((path / "metrics.json").read_text())
                assert digest(path / "model.pt") == data["model_sha256"]
                assert data["steps"] == gate.STEPS and data["seed"] == seed
                suite = data["evaluation"]["260107"]
                focal = next(row for row in suite if row["pair"] == pair)
                assert not focal["observed"]
                starts.append({"seed": seed,
                               "focal_role": focal["binary_role_accuracy"],
                               "focal_four_choice": focal["four_choice_accuracy"],
                               "second_suite_role": next(row["binary_role_accuracy"] for row in data["evaluation"]["260108"] if row["pair"] == pair),
                               "familiar_four_choice": data["summary"]["260107"]["seen"]["four_choice_accuracy"],
                               "unseen_event": data["summary"]["260107"]["unseen"]["event_accuracy"]})
            by_arm[arm] = {"starts": starts,
                           "mean_focal_role": float(np.mean([row["focal_role"] for row in starts])),
                           "mean_second_suite_role": float(np.mean([row["second_suite_role"] for row in starts]))}
        rows.append({"family": i, "generator_seed": family["generator_seed"],
                     "focal_pair": pair, "arms": by_arm,
                     "high_minus_sham": by_arm["high"]["mean_focal_role"] - by_arm["sham"]["mean_focal_role"],
                     "high_minus_base": by_arm["high"]["mean_focal_role"] - by_arm["base"]["mean_focal_role"],
                     "sham_minus_base": by_arm["sham"]["mean_focal_role"] - by_arm["base"]["mean_focal_role"]})
    differences = [row["high_minus_sham"] for row in rows]
    mean = float(np.mean(differences))
    half = float(t.ppf(.975, FAMILIES - 1) * np.std(differences, ddof=1) / math.sqrt(FAMILIES))
    familiar = [start["familiar_four_choice"] for row in rows for arm in ARMS
                for start in row["arms"][arm]["starts"]]
    return {"complete_models": FAMILIES * len(SEEDS) * len(ARMS),
            "primary_mean_high_minus_sham": mean,
            "primary_p_exact_two_sided": signflip(differences),
            "primary_95pct_t_interval": [mean-half, mean+half],
            "min_familiar_four_choice": min(familiar),
            "prespecified_success": mean <= -.10 and signflip(differences) <= .05 and min(familiar) >= .95,
            "family_rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "audit", "train", "summarize"))
    parser.add_argument("--family", type=int)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.action == "prepare": prepare()
    elif args.action == "audit": print(json.dumps(audit(), indent=2))
    elif args.action == "train": train(args.family, args.arm, args.seed)
    else: print(json.dumps(summarize(), indent=2))


if __name__ == "__main__": main()
