"""Outcome-blind fresh graph selection matching CN, P3, P4, triangles, C4.

The historical factorial families are excluded by graph isomorphism. No model
metrics are opened by this script. Run prepare once, then audit.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
PRIOR = ROOT / "role_transfer_phase01/rq1_factorial12_v1/GRAPH_MANIFEST.json"
OUT = ROOT / "role_transfer_phase01/rq1_factorial12_strict_v1"
MANIFEST = OUT / "GRAPH_MANIFEST.json"
N = 12
CELLS = ((1, 3), (1, 4), (2, 3), (2, 4))


def key(graph: nx.Graph):
    return tuple(sorted(tuple(sorted(edge)) for edge in graph.edges()))


def graph_from(edges):
    graph = nx.Graph()
    graph.add_nodes_from(range(N))
    graph.add_edges_from(edges)
    return graph


def switches(edges):
    existing = set(edges)
    for (a, b), (c, d) in itertools.combinations(edges, 2):
        if len({a, b, c, d}) < 4:
            continue
        for u, v in (((a, c), (b, d)), ((a, d), (b, c))):
            x, y = tuple(sorted(u)), tuple(sorted(v))
            if x in existing or y in existing or (0, 1) in (x, y):
                continue
            yield tuple(sorted((existing - {(a, b), (c, d)}) | {x, y}))


def features(edges):
    neighbors = [set() for _ in range(N)]
    for a, b in edges:
        neighbors[a].add(b)
        neighbors[b].add(a)
    if 1 in neighbors[0]:
        return None
    visited = {0}
    stack = [0]
    while stack:
        for node in neighbors[stack.pop()]:
            if node not in visited:
                visited.add(node)
                stack.append(node)
    if len(visited) != N:
        return None
    triangles = [sum(y in neighbors[x] for x, y in itertools.combinations(neighbors[a], 2))
                 for a in range(N)]
    common = len(neighbors[0] & neighbors[1])
    p3 = sum(y in neighbors[x] for x in neighbors[0] for y in neighbors[1])
    p4 = sum(z in neighbors[1] for x in neighbors[0] for y in neighbors[x]
             if y not in (0, 1) for z in neighbors[y] if z not in (0, x, y))
    c4 = sum(len(neighbors[a] & neighbors[b]) *
             (len(neighbors[a] & neighbors[b]) - 1) // 2
             for a, b in itertools.combinations(range(N), 2)) // 2
    return {"common_neighbors": common, "three_edge_paths": p3,
            "four_edge_paths": p4, "total_triangles": sum(triangles) // 3,
            "focal_triangles": [triangles[0], triangles[1]],
            "four_cycles": c4}


def find(seed):
    anchor = nx.random_regular_graph(4, N, seed=seed)
    start = key(anchor)
    original = features(start)
    if original is None or (original["common_neighbors"], original["three_edge_paths"]) not in CELLS:
        return None
    signature = (original["total_triangles"], *original["focal_triangles"],
                 original["four_edge_paths"])
    buckets = defaultdict(dict)
    seen = {start}
    first = [start]

    def consider(edges, depth):
        feat = features(edges)
        if feat is None:
            return
        cell = feat["common_neighbors"], feat["three_edge_paths"]
        if cell not in CELLS:
            return
        if (feat["total_triangles"], *feat["focal_triangles"], feat["four_edge_paths"]) != signature:
            return
        c4 = feat["four_cycles"]
        anchor_n = [set(anchor.neighbors(node)) for node in (0, 1)]
        candidate_n = [set() for _ in range(2)]
        for a, b in edges:
            for focal in (0, 1):
                if a == focal:
                    candidate_n[focal].add(b)
                elif b == focal:
                    candidate_n[focal].add(a)
        focal_edits = sum(len(anchor_n[i] ^ candidate_n[i]) for i in (0, 1))
        rank = (len(set(start) ^ set(edges)) // 2, depth, focal_edits, edges)
        previous = buckets[c4].get(cell)
        if previous is None or rank < previous[0]:
            buckets[c4][cell] = (rank, edges, feat)

    consider(start, 0)
    for edges in switches(start):
        if edges in seen:
            continue
        seen.add(edges)
        first.append(edges)
        consider(edges, 1)
    for nearby in first:
        for edges in switches(nearby):
            if edges in seen:
                continue
            seen.add(edges)
            consider(edges, 2)
    complete = []
    for c4, group in buckets.items():
        if all(cell in group for cell in CELLS):
            ranks = [group[cell][0] for cell in CELLS]
            complete.append(((max(r[0] for r in ranks), sum(r[0] for r in ranks),
                              abs(c4 - original["four_cycles"]), c4), c4, group))
    if not complete:
        return None
    _, c4, group = min(complete)
    row = {"generator_seed": seed, "focal_pair": [0, 1],
           "anchor_edges": list(map(list, start)),
           "matched_signature": [*signature, c4],
           "conditions": {f"cn{cell[0]}_p{cell[1]}": {
               "edges": list(map(list, group[cell][1])), "features": group[cell][2],
               "search_depth": group[cell][0][1]}
               for cell in CELLS}}
    return anchor, row


def build():
    prior = json.loads(PRIOR.read_text())
    old = [graph_from(tuple(map(tuple, item["anchor_edges"]))) for item in prior["families"]]
    anchors = []
    families = []
    for seed in range(91000, 94000):
        found = find(seed)
        if found is None:
            continue
        graph, row = found
        if any(nx.is_isomorphic(graph, previous) for previous in [*old, *anchors]):
            continue
        anchors.append(graph)
        families.append(row)
        if len(families) == 8:
            break
    assert len(families) == 8
    return {"status": "selected_without_model_outcomes", "entities": N, "degree": 4,
            "prior_manifest_sha256": hashlib.sha256(PRIOR.read_bytes()).hexdigest(),
            "selection": "First eight anchor graph classes from seeds 91000..93999, nonisomorphic to each other and the earlier eight. All four cells match total/focal triangles, simple four-edge-path count, and four-cycle count, within two degree-preserving switches of each anchor. Choose the common four-cycle stratum minimizing maximum and total edge differences from anchor; within cell choose minimum edge differences, search depth, focal neighbor edits, then lexicographic edges.",
            "cells": [list(cell) for cell in CELLS], "families": families,
            "selection_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def audit(data):
    assert data == build(), "Frozen strict manifest differs from graph-only selection"
    prior = json.loads(PRIOR.read_text())
    assert data["prior_manifest_sha256"] == hashlib.sha256(PRIOR.read_bytes()).hexdigest()
    previous = [graph_from(tuple(map(tuple, item["anchor_edges"]))) for item in prior["families"]]
    anchors = []
    for family in data["families"]:
        anchor = graph_from(tuple(map(tuple, family["anchor_edges"])))
        assert not any(nx.is_isomorphic(anchor, graph) for graph in [*previous, *anchors])
        anchors.append(anchor)
        signature = family["matched_signature"]
        for cell in CELLS:
            item = family["conditions"][f"cn{cell[0]}_p{cell[1]}"]
            edges = tuple(map(tuple, item["edges"]))
            graph = graph_from(edges)
            assert nx.is_connected(graph) and graph.number_of_edges() == 24
            assert all(degree == 4 for _, degree in graph.degree()) and not graph.has_edge(0, 1)
            feat = features(edges)
            assert feat == item["features"]
            assert (feat["common_neighbors"], feat["three_edge_paths"]) == cell
            assert [feat["total_triangles"], *feat["focal_triangles"],
                    feat["four_edge_paths"], feat["four_cycles"]] == signature
            assert item["search_depth"] <= 2
    return {"valid": True, "families": len(anchors), "cells": 4,
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
