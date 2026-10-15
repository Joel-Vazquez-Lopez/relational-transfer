"""Prospective repair test on fresh matched edge-rewiring graph families."""
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
import rq1_overlap_confirmation_v1 as old
import rq1_rewire_v1 as rw

HERE = Path(__file__).resolve().parent
ROOT = HERE / "rq2_rewire_repair_v1"
MANIFEST = ROOT / "GRAPH_MANIFEST.json"
PROTOCOL = ROOT / "PROTOCOL.json"
FAMILIES = 8
SEEDS = (311001, 311002, 311003)
CONDITIONS = ("sham", "high")
ARMS = ("unordered_identities", "ordered_roles")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def excluded_bases() -> list[nx.Graph]:
    prior = old.prior_graphs()
    overlap = json.loads(old.MANIFEST_PATH.read_text())
    prior += [rw.graph_from_edges(row["canonical_edges"])
              for row in overlap["worlds"].values()]
    rewired = json.loads(rw.MANIFEST.read_text())
    prior += [rw.graph_from_edges(row["arms"]["base"]["edges"])
              for row in rewired["families"]]
    return prior


def make_manifest() -> dict:
    excluded = excluded_bases()
    selected = []
    base_graphs = []
    for generator_seed in range(60000, 100000):
        base = nx.random_regular_graph(4, 10, seed=generator_seed)
        if not nx.is_connected(base):
            continue
        if any(nx.is_isomorphic(base, previous) for previous in excluded + base_graphs):
            continue
        pair = next(((a, b) for a, b in itertools.combinations(range(10), 2)
                     if not base.has_edge(a, b) and rw.shared(base, (a, b)) == 1), None)
        if pair is None:
            continue
        found = rw.find_variants(base, pair)
        if found is None:
            continue
        variants, signature = found
        permutation = np.random.default_rng(np.random.SeedSequence(
            [820000, generator_seed])).permutation(10).tolist()
        def mapped(items):
            return sorted([sorted([permutation[a], permutation[b]]) for a, b in items])
        selected.append({"generator_seed": generator_seed,
                         "canonical_focal_pair": list(pair),
                         "focal_pair": sorted([permutation[x] for x in pair]),
                         "label_permutation": permutation,
                         "matched_triangle_signature": list(signature),
                         "base_edges": mapped(rw.edges(base)),
                         "conditions": {
                             condition: {"swap_seed": variants[condition]["swap_seed"],
                                         "edges": mapped(variants[condition]["edges"])}
                             for condition in CONDITIONS}})
        base_graphs.append(base)
        if len(selected) == FAMILIES:
            break
    assert len(selected) == FAMILIES
    sources = [HERE / "rq2_rewire_repair_v1.py", HERE / "rq1_rewire_v1.py",
               HERE / "role_aux_v1.py", HERE / "generator_gate_v1.py",
               HERE / "all_pair_audit.py", HERE.parent / "d5_indirect_support" / "run.py",
               HERE.parent / "d5_indirect_support" / "path2_control.py"]
    return {"status": "selected_from_graphs_only_before_any_repair_model_training",
            "selection": "First eight new nonisomorphic base classes from generator seeds >=60000, excluding 48 previously selected base/graph classes; each has a focal nonedge with one shared partner and matching two-swap sham/high variants. No model outcome used.",
            "historical_graphs_considered": len(excluded),
            "families": selected, "training_seeds": list(SEEDS),
            "source_hashes": {str(p.relative_to(HERE.parent)): digest(p) for p in sources}}


def make_protocol() -> dict:
    return {"status": "frozen_before_training", "date": "2026-10-09",
            "question": "Does ordered-role auxiliary training reduce the focal high-versus-sham transfer penalty relative to a parameter- and compute-matched unordered-identity auxiliary control on fresh matched rewiring graphs?",
            "design": {"families": FAMILIES, "seeds_per_family": len(SEEDS),
                       "conditions": list(CONDITIONS), "training_arms": list(ARMS),
                       "planned_models": FAMILIES * len(SEEDS) * len(CONDITIONS) * len(ARMS),
                       "entities": 10, "observed_pairs": 20, "degree": 4,
                       "generator": "balanced distractors and dedicated padding",
                       "updates": 6000, "batch_size": 128, "auxiliary_weight": 0.5,
                       "same_focal_pair": "within each family and across all four condition/training cells",
                       "evaluation": "all 45 pairs; fixed 512-rendering suites 260107 primary and 260108 robustness"},
            "first_gate": "Unordered-control high-minus-sham focal binary-role accuracy, averaged over paired starts then eight equal-weight families, must be <= -0.10 with two-sided exact family sign-flip p <= 0.05",
            "primary_endpoint": "(ordered high-minus-sham focal binary-role accuracy) minus (unordered high-minus-sham focal binary-role accuracy), paired within seed and averaged within family, then equal-weight across eight families; final checkpoint, scene suite 260107",
            "primary_test": "Only if first gate passes, exact two-sided sign-flip over eight family-level attenuation effects, all 256 assignments",
            "primary_success": "mean attenuation >= +0.10; p <= 0.05; all ordered models familiar four-choice accuracy >=0.95; mean ordered-minus-unordered familiar four-choice >= -0.01 and unseen event accuracy >= -0.02",
            "uncertainty": "report all family-level penalties and attenuation effects with graph-level t(7) 95 percent intervals; seeds, pairs and scene items are nested within graph family",
            "secondary": ["second scene suite attenuation", "high-condition ordered-minus-unordered focal accuracy", "familiar and event accuracy by condition", "all other unseen pairs"],
            "missingness": "no graph, seed or checkpoint replacement based on results; only same-configuration technical reruns",
            "interpretation": "Positive attenuation shows ordered-role training protects against this matched rewiring package. It does not identify overlap as the unique cause or establish transfer to natural-language LLMs."}


def prepare() -> None:
    if MANIFEST.exists() or PROTOCOL.exists():
        raise FileExistsError("Frozen files already exist")
    manifest = make_manifest()
    ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    PROTOCOL.write_text(json.dumps(make_protocol(), indent=2) + "\n")
    print(json.dumps({"families": FAMILIES,
                      "graph_seeds": [row["generator_seed"] for row in manifest["families"]],
                      "planned_models": FAMILIES * len(SEEDS) * len(CONDITIONS) * len(ARMS),
                      "manifest": str(MANIFEST), "protocol": str(PROTOCOL)}, indent=2))


def load() -> tuple[dict, dict]:
    return json.loads(MANIFEST.read_text()), json.loads(PROTOCOL.read_text())


def world_name(index: int, condition: str) -> str:
    return f"family{index}_{condition}"


def configure() -> None:
    manifest, _ = load()
    gate.WORLDS = {world_name(i, condition): {
        "edges": tuple(map(tuple, family["conditions"][condition]["edges"]))}
        for i, family in enumerate(manifest["families"]) for condition in CONDITIONS}
    gate.SEEDS = SEEDS
    auxiliary.ROOT = ROOT
    def configure_world(world: str) -> tuple[int, ...]:
        gate.base.N_ID = 10
        gate.base.VOCAB = gate.VOCAB
        gate.base.BASE_EDGES = gate.WORLDS[world]["edges"]
        gate.base.MAPPINGS = {"canonical": tuple(range(10))}
        return tuple(range(10))
    gate.configure = configure_world


def model_path(index: int, condition: str, arm: str, seed: int) -> Path:
    return ROOT / "full" / world_name(index, condition) / arm / f"seed_{seed}"


def audit() -> dict:
    manifest, protocol = load()
    assert manifest == make_manifest(), "Graph selection or source hashes changed"
    assert protocol == make_protocol(), "Protocol changed"
    configure()
    for i, family in enumerate(manifest["families"]):
        pair = tuple(family["focal_pair"])
        base = rw.graph_from_edges(family["base_edges"])
        graphs = {condition: rw.graph_from_edges(family["conditions"][condition]["edges"])
                  for condition in CONDITIONS}
        assert not base.has_edge(*pair) and rw.shared(base, pair) == 1
        assert [rw.shared(graphs[c], pair) for c in CONDITIONS] == [1, 3]
        for condition in CONDITIONS:
            graph = graphs[condition]
            assert not graph.has_edge(*pair)
            observed = rw.signature(base, graph, pair)
            saved = family["matched_triangle_signature"]
            assert observed is not None and observed[0] == saved[0]
            assert sorted(observed[1:]) == sorted(saved[1:])
            gate.configure(world_name(i, condition))
            assert len(gate.base.make_pool("canonical")) == 1280
    return {"valid": True, "families": FAMILIES,
            "planned_models": FAMILIES * len(SEEDS) * len(CONDITIONS) * len(ARMS),
            "manifest_sha256": digest(MANIFEST), "protocol_sha256": digest(PROTOCOL)}


def train(index: int, condition: str, arm: str, seed: int) -> None:
    assert 0 <= index < FAMILIES and condition in CONDITIONS and arm in ARMS and seed in SEEDS
    manifest, protocol = load()
    assert protocol == make_protocol()
    for relative, expected in manifest["source_hashes"].items():
        assert digest(HERE.parent / relative) == expected, f"Changed source: {relative}"
    configure()
    auxiliary.train(world_name(index, condition), arm, seed)


def signflip(values: list[float]) -> float:
    observed = abs(float(np.mean(values)))
    return sum(abs(float(np.mean(np.asarray(values) * signs))) >= observed - 1e-12
               for signs in itertools.product((-1, 1), repeat=len(values))) / (2 ** len(values))


def interval(values: list[float]) -> list[float]:
    mean = float(np.mean(values))
    half = float(t.ppf(.975, len(values) - 1) * np.std(values, ddof=1) / math.sqrt(len(values)))
    return [mean - half, mean + half]


def summarize() -> dict:
    manifest, _ = load()
    families = []
    for i, family in enumerate(manifest["families"]):
        pair = family["focal_pair"]
        starts = []
        for seed in SEEDS:
            cells = {}
            for condition, arm in itertools.product(CONDITIONS, ARMS):
                path = model_path(i, condition, arm, seed)
                metrics = json.loads((path / "metrics.json").read_text())
                assert digest(path / "model.pt") == metrics["model_sha256"]
                assert metrics["steps"] == gate.STEPS and metrics["seed"] == seed
                suite = metrics["evaluation"]["260107"]
                focal = next(row for row in suite if row["pair"] == pair)
                assert not focal["observed"]
                other = next(row for row in metrics["evaluation"]["260108"] if row["pair"] == pair)
                cells[f"{condition}:{arm}"] = {
                    "focal_role": focal["binary_role_accuracy"],
                    "second_scene_focal_role": other["binary_role_accuracy"],
                    "familiar_four_choice": metrics["summary"]["260107"]["seen"]["four_choice_accuracy"],
                    "unseen_event": metrics["summary"]["260107"]["unseen"]["event_accuracy"]}
            penalties = {arm: cells[f"high:{arm}"]["focal_role"] - cells[f"sham:{arm}"]["focal_role"]
                         for arm in ARMS}
            starts.append({"seed": seed, "cells": cells, "penalties": penalties,
                           "attenuation": penalties["ordered_roles"] - penalties["unordered_identities"]})
        families.append({"family": i, "generator_seed": family["generator_seed"],
                         "focal_pair": pair, "starts": starts,
                         "control_penalty": float(np.mean([s["penalties"]["unordered_identities"] for s in starts])),
                         "ordered_penalty": float(np.mean([s["penalties"]["ordered_roles"] for s in starts])),
                         "attenuation": float(np.mean([s["attenuation"] for s in starts]))})
    control = [f["control_penalty"] for f in families]
    attenuation = [f["attenuation"] for f in families]
    ordered_familiar = [s["cells"][f"{c}:ordered_roles"]["familiar_four_choice"]
                        for f in families for s in f["starts"] for c in CONDITIONS]
    familiar_differences = [s["cells"][f"{c}:ordered_roles"]["familiar_four_choice"]
                            - s["cells"][f"{c}:unordered_identities"]["familiar_four_choice"]
                            for f in families for s in f["starts"] for c in CONDITIONS]
    event_differences = [s["cells"][f"{c}:ordered_roles"]["unseen_event"]
                         - s["cells"][f"{c}:unordered_identities"]["unseen_event"]
                         for f in families for s in f["starts"] for c in CONDITIONS]
    control_gate = float(np.mean(control)) <= -.10 and signflip(control) <= .05
    repair_rule = (float(np.mean(attenuation)) >= .10 and signflip(attenuation) <= .05
                   and min(ordered_familiar) >= .95
                   and float(np.mean(familiar_differences)) >= -.01
                   and float(np.mean(event_differences)) >= -.02)
    second = [float(np.mean([
        (s["cells"]["high:ordered_roles"]["second_scene_focal_role"]
         - s["cells"]["sham:ordered_roles"]["second_scene_focal_role"])
        - (s["cells"]["high:unordered_identities"]["second_scene_focal_role"]
           - s["cells"]["sham:unordered_identities"]["second_scene_focal_role"])
        for s in f["starts"]])) for f in families]
    return {"complete_models": FAMILIES * len(SEEDS) * len(CONDITIONS) * len(ARMS),
            "control_penalty_mean": float(np.mean(control)),
            "control_penalty_p_exact": signflip(control),
            "control_penalty_95pct_t_interval": interval(control),
            "control_gate_pass": control_gate,
            "attenuation_mean": float(np.mean(attenuation)),
            "attenuation_p_exact": signflip(attenuation),
            "attenuation_95pct_t_interval": interval(attenuation),
            "second_scene_attenuation_mean": float(np.mean(second)),
            "minimum_ordered_familiar_four_choice": min(ordered_familiar),
            "mean_ordered_minus_unordered_familiar_four_choice": float(np.mean(familiar_differences)),
            "mean_ordered_minus_unordered_unseen_event": float(np.mean(event_differences)),
            "prespecified_repair_success": control_gate and repair_rule,
            "family_rows": families}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "audit", "train", "summarize"))
    parser.add_argument("--family", type=int)
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.action == "prepare": prepare()
    elif args.action == "audit": print(json.dumps(audit(), indent=2))
    elif args.action == "train": train(args.family, args.condition, args.arm, args.seed)
    else: print(json.dumps(summarize(), indent=2))


if __name__ == "__main__": main()
