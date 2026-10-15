"""Graph-only, outcome-blind selection for a local 2 x 2 role-transfer test.

This script never loads model predictions. Run ``prepare`` once, then ``audit``.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "role_transfer_phase01/rq1_factorial12_v1"
MANIFEST = OUT / "GRAPH_MANIFEST.json"
CELLS = ((1, 3), (1, 4), (2, 3), (2, 4))
FAMILIES = 8
N = 12


def edge_key(graph: nx.Graph) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(tuple(sorted(edge)) for edge in graph.edges()))


def graph_from(edges: tuple[tuple[int, int], ...]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(N))
    graph.add_edges_from(edges)
    return graph


def features(graph: nx.Graph) -> dict:
    neighbors = [set(graph.neighbors(node)) for node in range(N)]
    triangles = nx.triangles(graph)
    return {
        "common_neighbors": len(neighbors[0] & neighbors[1]),
        "three_edge_paths": sum(graph.has_edge(x, y) for x in neighbors[0] for y in neighbors[1]),
        "total_triangles": sum(triangles.values()) // 3,
        "focal_triangles": [triangles[0], triangles[1]],
        "four_cycles": sum(len(neighbors[a] & neighbors[b]) *
                           (len(neighbors[a] & neighbors[b]) - 1) // 2
                           for a, b in itertools.combinations(range(N), 2)) // 2,
    }


def switches(graph: nx.Graph):
    edges = edge_key(graph)
    existing = set(edges)
    for (a, b), (c, d) in itertools.combinations(edges, 2):
        if len({a, b, c, d}) < 4:
            continue
        for first, second in (((a, c), (b, d)), ((a, d), (b, c))):
            x, y = tuple(sorted(first)), tuple(sorted(second))
            if x in existing or y in existing or (0, 1) in (x, y):
                continue
            candidate = graph_from(tuple(sorted((existing - {(a, b), (c, d)}) | {x, y})))
            if nx.is_connected(candidate):
                yield candidate


def choose(seed: int):
    anchor = nx.random_regular_graph(4, N, seed=seed)
    if anchor.has_edge(0, 1):
        return None
    anchor_feat = features(anchor)
    signature = (anchor_feat["total_triangles"], *anchor_feat["focal_triangles"])
    candidates = [anchor, *switches(anchor)]
    buckets = {cell: [] for cell in CELLS}
    seen = set()
    for graph in candidates:
        key = edge_key(graph)
        if key in seen:
            continue
        seen.add(key)
        feat = features(graph)
        cell = feat["common_neighbors"], feat["three_edge_paths"]
        if cell not in buckets or signature != (feat["total_triangles"], *feat["focal_triangles"]):
            continue
        focal_edits = sum(len(set(anchor.neighbors(node)) ^ set(graph.neighbors(node)))
                          for node in (0, 1))
        score = (focal_edits, abs(feat["four_cycles"] - anchor_feat["four_cycles"]), key)
        buckets[cell].append((score, graph, feat))
    if any(not rows for rows in buckets.values()):
        return None
    selected = {f"cn{cn}_p{paths}": {"edges": list(map(list, edge_key(min(buckets[(cn, paths)])[1]))),
                                       "features": min(buckets[(cn, paths)])[2]}
                for cn, paths in CELLS}
    return anchor, {"generator_seed": seed, "focal_pair": [0, 1],
                    "matched_signature": list(signature),
                    "anchor_edges": list(map(list, edge_key(anchor))),
                    "conditions": selected}


def build() -> dict:
    families = []
    anchors = []
    for seed in range(90000, 91000):
        result = choose(seed)
        if result is None:
            continue
        anchor, row = result
        cycle_counts = [item["features"]["four_cycles"] for item in row["conditions"].values()]
        if max(cycle_counts) - min(cycle_counts) > 1:
            continue
        if any(nx.is_isomorphic(anchor, other) for other in anchors):
            continue
        families.append(row)
        anchors.append(anchor)
        if len(families) == FAMILIES:
            break
    assert len(families) == FAMILIES
    return {"status": "selected_without_model_outcomes", "entities": N, "degree": 4,
            "selection": "First eight nonisomorphic anchors from seeds 90000..90999 with all four cells at zero or one degree-preserving switch, equal total and focal triangle counts, and at most one four-cycle of spread across cells. Within-cell graph selection minimizes focal neighbor edits, then four-cycle difference, then edge lexicographic order.",
            "cells": [list(cell) for cell in CELLS], "families": families,
            "selection_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def audit(data: dict) -> dict:
    assert data == build(), "Manifest differs from deterministic graph-only selection"
    anchors = []
    for family in data["families"]:
        anchor = graph_from(tuple(map(tuple, family["anchor_edges"])))
        assert not any(nx.is_isomorphic(anchor, other) for other in anchors)
        anchors.append(anchor)
        for cell in CELLS:
            row = family["conditions"][f"cn{cell[0]}_p{cell[1]}"]
            graph = graph_from(tuple(map(tuple, row["edges"])))
            assert nx.is_connected(graph) and graph.number_of_edges() == 24
            assert all(degree == 4 for _, degree in graph.degree())
            assert not graph.has_edge(0, 1)
            assert features(graph) == row["features"]
            assert (row["features"]["common_neighbors"], row["features"]["three_edge_paths"]) == cell
            assert (row["features"]["total_triangles"], *row["features"]["focal_triangles"]) == tuple(family["matched_signature"])
            assert len(set(edge_key(anchor)) ^ set(edge_key(graph))) in (0, 4)
    return {"valid": True, "families": len(anchors), "cells_per_family": len(CELLS),
            "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest()}


if __name__ == "__main__":
    action = sys.argv[1]
    if action == "prepare":
        OUT.mkdir(parents=True, exist_ok=True)
        if MANIFEST.exists():
            raise FileExistsError(MANIFEST)
        MANIFEST.write_text(json.dumps(build(), indent=2) + "\n")
    elif action == "audit":
        print(json.dumps(audit(json.loads(MANIFEST.read_text())), indent=2))
    else:
        raise ValueError(action)
