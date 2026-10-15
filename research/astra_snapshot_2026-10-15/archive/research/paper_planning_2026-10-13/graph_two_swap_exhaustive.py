"""Exhaustively check the old two-switch neighborhood for a stricter match.

Reads historical graph manifests only. No model outcomes or training.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "role_transfer_phase01/rq1_rewire_v1/GRAPH_MANIFEST.json"


def features(edges: frozenset[tuple[int, int]], focal: list[int]) -> tuple[int, ...]:
    neighbors = [set() for _ in range(10)]
    for left, right in edges:
        neighbors[left].add(right)
        neighbors[right].add(left)
    actor, patient = focal
    common = len(neighbors[actor] & neighbors[patient])
    three_paths = sum(
        right in neighbors[left]
        for left in neighbors[actor]
        for right in neighbors[patient]
    )
    triangles = sum(
        right in neighbors[left]
        for node in range(10)
        for left, right in itertools.combinations(neighbors[node], 2)
    ) // 3
    focal_triangles = [
        sum(right in neighbors[left] for left, right in itertools.combinations(neighbors[node], 2))
        for node in focal
    ]
    return common, three_paths, triangles, *focal_triangles


def one_switch(edges: frozenset[tuple[int, int]], focal: list[int]):
    fixed_pair = tuple(sorted(focal))
    for (a, b), (c, d) in itertools.combinations(edges, 2):
        for proposed in (((a, c), (b, d)), ((a, d), (b, c))):
            first, second = (tuple(sorted(pair)) for pair in proposed)
            if (
                first[0] == first[1]
                or second[0] == second[1]
                or first == second
                or first in edges
                or second in edges
                or fixed_pair in (first, second)
            ):
                continue
            yield edges - {(a, b), (c, d)} | {first, second}


manifest = json.loads(MANIFEST.read_text())
print("family, valid distinct states within <=2 switches, strict matched high states")
for index, family in enumerate(manifest["families"]):
    focal = family["focal_pair"]
    base = frozenset(tuple(edge) for edge in family["arms"]["base"]["edges"])
    sham = frozenset(tuple(edge) for edge in family["arms"]["sham"]["edges"])
    sham_features = features(sham, focal)
    first = set(one_switch(base, focal))
    reachable = first | {candidate for graph in first for candidate in one_switch(graph, focal)}
    matches = sum(features(graph, focal) == (3, *sham_features[1:]) for graph in reachable)
    print(index, len(reachable), matches)
