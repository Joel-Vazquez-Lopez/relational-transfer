"""Prospective RQ1 comparison of sparse and clustered training networks.

The manifest and protocol are written before training. Graphs are selected by
topology alone, excluding every previously trained graph isomorphism class.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import networkx as nx
import numpy as np

import fresh_graph_manifest_v1 as old_selector
import generator_gate_v1 as gate

HERE = Path(__file__).resolve().parent
ROOT = HERE / "rq1_structure_v1"
MANIFEST_PATH = ROOT / "GRAPH_MANIFEST.json"
PROTOCOL_PATH = ROOT / "PROTOCOL.json"
SEEDS = (281001, 281002, 281003)
PAIR_COUNT = 8


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def triangles(graph: nx.Graph) -> int:
    return sum(nx.triangles(graph).values()) // 3


def choose_graphs() -> dict:
    previous = [old_selector.graph(item["edges"]) for item in old_selector.PREVIOUS]
    previous += [old_selector.graph(item["canonical_edges"])
                 for item in json.loads(old_selector.DESTINATION.read_text())["worlds"].values()]
    selected: dict[str, list] = {"sparse": [], "clustered": []}
    for candidate_seed in range(20000, 200000):
        candidate = nx.random_regular_graph(4, 10, seed=candidate_seed)
        count = triangles(candidate)
        group = "sparse" if count <= 4 else "clustered" if count >= 8 else None
        if group is None or len(selected[group]) == PAIR_COUNT:
            continue
        if any(nx.is_isomorphic(candidate, old) for old in previous):
            continue
        if any(nx.is_isomorphic(candidate, row["graph"])
               for group_rows in selected.values() for row in group_rows):
            continue
        selected[group].append({"graph": candidate, "generator_seed": candidate_seed,
                                "triangles": count})
        if all(len(rows) == PAIR_COUNT for rows in selected.values()):
            break
    if not all(len(rows) == PAIR_COUNT for rows in selected.values()):
        raise RuntimeError("Could not find eight new graphs per topology group")
    pairs = []
    for index in range(PAIR_COUNT):
        mapping = np.random.default_rng(np.random.SeedSequence(
            [280000, index])).permutation(10).tolist()
        arms = {}
        for group in ("sparse", "clustered"):
            entry = selected[group][index]
            edges = sorted([sorted(edge) for edge in entry["graph"].edges()])
            labeled = sorted([sorted([mapping[a], mapping[b]]) for a, b in edges])
            arms[group] = {"generator_seed": entry["generator_seed"],
                           "triangles": entry["triangles"], "canonical_edges": edges,
                           "labeled_edges": labeled}
        missing = {group: set(itertools.combinations(range(10), 2))
                   - {tuple(edge) for edge in arms[group]["labeled_edges"]}
                   for group in arms}
        shared = sorted([list(edge) for edge in missing["sparse"] & missing["clustered"]])
        if len(shared) < 10:
            raise RuntimeError(f"Only {len(shared)} shared missing pairs in pair {index}")
        pairs.append({"id": f"pair{index}", "label_permutation": mapping,
                      "shared_unseen_pairs": shared, "graphs": arms})
    return {"status": "selected_without_model_outcomes", "selection":
            "first eight new nonisomorphic graph classes with <=4 triangles and first eight with >=8 triangles from networkx random 4-regular 10-node seeds starting at 20000; paired in discovery order", 
            "previous_graph_classes_excluded": len(previous), "training_seeds": list(SEEDS),
            "pairs": pairs, "source_hashes": {
                str(path.relative_to(HERE.parent)): digest(path)
                for path in (HERE / "generator_gate_v1.py", HERE / "fresh_graph_manifest_v1.py",
                             HERE.parent / "d5_indirect_support" / "run.py",
                             HERE.parent / "d5_indirect_support" / "path2_control.py",
                             HERE / "all_pair_audit.py")}}


def protocol() -> dict:
    return {
        "status": "frozen_before_training", "question":
        "Does a less clustered arrangement of the same number of observed actor-patient pairs improve transfer to pairs absent from both training networks?",
        "design": {"paired_networks": PAIR_COUNT, "starts_per_network": len(SEEDS),
                   "models": 2 * PAIR_COUNT * len(SEEDS), "nodes": 10,
                   "degree_each_node": 4, "edges_each_network": 20,
                   "training_updates": 6000, "batch_size": 128,
                   "training_arm": "ordinary balanced four-choice Transformer",
                   "topology_groups": {"sparse": "at most 4 triangles",
                                       "clustered": "at least 8 triangles"}},
        "paired_controls": ["same ten labeled entities and their per-entity training frequency",
                            "same number of observed pairs and meanings",
                            "same label permutation within a pair", "same initialization seed",
                            "same generator, optimizer, training budget and testing renderings"],
        "primary_endpoint": "binary actor/patient accuracy averaged over only the unordered entity pairs absent from BOTH networks of each pair, then over three starts; sparse minus clustered within each network pair",
        "primary_test": "exact two-sided sign-flip test over eight network-pair differences, all 256 assignments",
        "uncertainty": "graph-pair t interval with seven degrees of freedom; report every pair and start",
        "directional_prediction": "sparser triangles distribute observed partners more broadly and may yield higher transfer accuracy",
        "practical_threshold": "mean sparse-minus-clustered gain at least 0.03 and p <= 0.05; this is a pragmatic target, not a power calculation",
        "secondary": ["all 25 unseen pairs per network", "seen four-choice accuracy",
                      "unseen event accuracy", "second scene-rendering seed", "weak shared pairs below 80%"],
        "analysis_unit": "network pair; starts, semantic pairs and scene renderings are nested",
        "no_adaptive_replacement": "All 16 graphs and 48 starts remain in analysis regardless of result; only same-configuration technical reruns are allowed",
        "scope": "An effect concerns this small coded-message, generated-scene Transformer; the triangle count is a selected topology contrast, not an established mechanism. A null does not prove graph structure irrelevant."
    }


def prepare() -> None:
    if MANIFEST_PATH.exists() or PROTOCOL_PATH.exists():
        raise FileExistsError("Prospective files already exist; refusing replacement")
    manifest = choose_graphs()
    ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    PROTOCOL_PATH.write_text(json.dumps(protocol(), indent=2) + "\n")
    print(json.dumps({"manifest": str(MANIFEST_PATH), "protocol": str(PROTOCOL_PATH),
                      "pairs": [(row["id"], len(row["shared_unseen_pairs"]))
                                for row in manifest["pairs"]]}, indent=2))


def load() -> tuple[dict, dict]:
    return json.loads(MANIFEST_PATH.read_text()), json.loads(PROTOCOL_PATH.read_text())


def configure(pair_id: str, group: str) -> None:
    manifest, _ = load()
    item = next(row for row in manifest["pairs"] if row["id"] == pair_id)
    graph = item["graphs"][group]
    world = f"{pair_id}_{group}"
    gate.WORLDS = {world: {"edges": tuple(map(tuple, graph["canonical_edges"]))}}
    gate.SEEDS = SEEDS
    gate.OUTPUT = ROOT / "models"
    def configure_world(_: str) -> tuple[int, ...]:
        gate.base.N_ID = 10
        gate.base.VOCAB = gate.VOCAB
        gate.base.BASE_EDGES = gate.WORLDS[world]["edges"]
        gate.base.MAPPINGS = {"canonical": tuple(range(10))}
        return tuple(item["label_permutation"])
    gate.configure = configure_world


def model_path(pair_id: str, group: str, seed: int) -> Path:
    return ROOT / "models" / "full" / f"{pair_id}_{group}" / "balanced_distractor" / f"seed_{seed}"


def audit() -> dict:
    manifest, plan = load()
    assert manifest == choose_graphs(), "Graph rule or source changed after freeze"
    assert plan == protocol(), "Protocol changed after freeze"
    assert len(manifest["pairs"]) == PAIR_COUNT
    for item in manifest["pairs"]:
        assert len(item["shared_unseen_pairs"]) >= 10
        for group in ("sparse", "clustered"):
            graph = item["graphs"][group]
            assert len(graph["labeled_edges"]) == 20
            assert all(sum(node in edge for edge in graph["labeled_edges"]) == 4
                       for node in range(10))
            assert (triangles(old_selector.graph(graph["canonical_edges"]))
                    == graph["triangles"])
            configure(item["id"], group)
            gate.configure(f"{item['id']}_{group}")
            assert len(gate.base.make_pool("canonical")) == 1280
    return {"valid": True, "network_pairs": PAIR_COUNT, "models": 48,
            "manifest_sha256": digest(MANIFEST_PATH), "protocol_sha256": digest(PROTOCOL_PATH)}


def train(pair_id: str, group: str, seed: int) -> None:
    assert pair_id in {f"pair{i}" for i in range(PAIR_COUNT)}
    assert group in ("sparse", "clustered") and seed in SEEDS
    # Audit was run by the launcher; each worker verifies frozen source hashes.
    manifest, plan = load()
    assert plan == protocol() and manifest["source_hashes"] == choose_graphs()["source_hashes"]
    configure(pair_id, group)
    gate.train(f"{pair_id}_{group}", "balanced_distractor", seed, gate.STEPS, False)


def signflip_p(differences: list[float]) -> float:
    observed = abs(float(np.mean(differences)))
    return sum(abs(float(np.mean(np.asarray(differences) * signs))) >= observed - 1e-12
               for signs in itertools.product((-1, 1), repeat=len(differences))) / 2**len(differences)


def summarize() -> dict:
    from scipy.stats import t
    manifest, _ = load()
    rows = []
    missing = []
    for item in manifest["pairs"]:
        pair_id = item["id"]
        shared = {tuple(pair) for pair in item["shared_unseen_pairs"]}
        arms = {}
        for group in ("sparse", "clustered"):
            starts = []
            for seed in SEEDS:
                directory = model_path(pair_id, group, seed)
                metrics = directory / "metrics.json"
                checkpoint = directory / "model.pt"
                if not metrics.exists() or not checkpoint.exists():
                    missing.append({"pair": pair_id, "group": group, "seed": seed})
                    continue
                data = json.loads(metrics.read_text())
                assert digest(checkpoint) == data["model_sha256"]
                primary = data["evaluation"]["260107"]
                common = [row for row in primary if tuple(row["pair"]) in shared]
                assert len(common) == len(shared) and all(not row["observed"] for row in common)
                starts.append({"seed": seed,
                               "shared_unseen_binary_role": float(np.mean([
                                   row["binary_role_accuracy"] for row in common])),
                               "seen_four_choice": data["summary"]["260107"]["seen"]["four_choice_accuracy"],
                               "all_unseen_binary_role": data["summary"]["260107"]["unseen"]["binary_role_accuracy"],
                               "shared_weak_below_0_8": sum(row["binary_role_accuracy"] < .8 for row in common)})
            arms[group] = starts
        if all(len(arms[group]) == 3 for group in arms):
            low = float(np.mean([r["shared_unseen_binary_role"] for r in arms["sparse"]]))
            high = float(np.mean([r["shared_unseen_binary_role"] for r in arms["clustered"]]))
            rows.append({"pair": pair_id, "shared_unseen_pairs": len(shared),
                         "triangles": {group: item["graphs"][group]["triangles"] for group in arms},
                         "starts": arms, "sparse_mean": low, "clustered_mean": high,
                         "difference": low - high})
    output = {"complete_pairs": len(rows), "planned_pairs": PAIR_COUNT,
              "missing_models": missing, "pair_rows": rows}
    if len(rows) == PAIR_COUNT:
        differences = [row["difference"] for row in rows]
        mean = float(np.mean(differences))
        half = float(t.ppf(.975, 7) * np.std(differences, ddof=1) / np.sqrt(8))
        output.update({"mean_sparse_minus_clustered": mean,
                       "exact_two_sided_signflip_p": signflip_p(differences),
                       "graph_pair_t_interval_95": [mean-half, mean+half],
                       "practical_rule_passed": mean >= .03 and signflip_p(differences) <= .05})
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "audit", "summarize"):
        sub.add_parser(command)
    run = sub.add_parser("train")
    run.add_argument("--pair", required=True, choices=[f"pair{i}" for i in range(PAIR_COUNT)])
    run.add_argument("--group", required=True, choices=("sparse", "clustered"))
    run.add_argument("--seed", required=True, type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "audit":
        print(json.dumps(audit(), indent=2))
    elif args.command == "summarize":
        print(json.dumps(summarize(), indent=2))
    else:
        train(args.pair, args.group, args.seed)


if __name__ == "__main__":
    main()
