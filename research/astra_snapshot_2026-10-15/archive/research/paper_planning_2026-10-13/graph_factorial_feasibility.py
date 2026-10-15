"""Graph-only feasibility screen; no model outputs, training, or sealed tests.

Sample degree-preserving two-edge switches from a fixed historical graph to see
whether common-neighbor and length-3-path counts can be separated while other
graph statistics are matched. This is a design search, not a confirmation study.
"""

from __future__ import annotations

import itertools
import json
import random
from collections import defaultdict
from pathlib import Path


SEED = 131026
ATTEMPTS = 300_000
ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "role_transfer_phase01/rq1_rewire_v1/GRAPH_MANIFEST.json"
family = json.loads(MANIFEST.read_text())["families"][0]
actor, patient = family["focal_pair"]
edges = {tuple(edge) for edge in family["arms"]["base"]["edges"]}
random.seed(SEED)


def features(graph_edges: set[tuple[int, int]]) -> tuple[int, int, int, int, int, bool]:
    neighbors = [set() for _ in range(10)]
    for left, right in graph_edges:
        neighbors[left].add(right)
        neighbors[right].add(left)
    common = len(neighbors[actor] & neighbors[patient])
    three_paths = sum(
        right_neighbor in neighbors[left_neighbor]
        for left_neighbor in neighbors[actor]
        for right_neighbor in neighbors[patient]
    )
    triangles = sum(
        right_neighbor in neighbors[left_neighbor]
        for node in range(10)
        for left_neighbor, right_neighbor in itertools.combinations(neighbors[node], 2)
    ) // 3
    actor_triangles = sum(
        right_neighbor in neighbors[left_neighbor]
        for left_neighbor, right_neighbor in itertools.combinations(neighbors[actor], 2)
    )
    patient_triangles = sum(
        right_neighbor in neighbors[left_neighbor]
        for left_neighbor, right_neighbor in itertools.combinations(neighbors[patient], 2)
    )
    visited = {actor}
    queue = [actor]
    for node in queue:
        for neighbor in neighbors[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return common, three_paths, triangles, actor_triangles, patient_triangles, len(visited) == 10


exact = defaultdict(lambda: defaultdict(set))
total_triangle = defaultdict(lambda: defaultdict(set))
accepted = 0
for _ in range(ATTEMPTS):
    (a, b), (c, d) = random.sample(tuple(edges), 2)
    pairs = [(a, c), (b, d)] if random.random() < 0.5 else [(a, d), (b, c)]
    first, second = (tuple(sorted(pair)) for pair in pairs)
    if (
        first[0] == first[1]
        or second[0] == second[1]
        or first == second
        or first in edges
        or second in edges
        or tuple(sorted((actor, patient))) in (first, second)
    ):
        continue
    edges = edges - {(a, b), (c, d)} | {first, second}
    common, paths, triangles, actor_tri, patient_tri, connected = features(edges)
    if not connected or common not in (1, 3):
        continue
    accepted += 1
    exact[(triangles, actor_tri, patient_tri)][common].add(paths)
    total_triangle[triangles][common].add(paths)


def intersections(table):
    return {
        signature: sorted(counts[1] & counts[3])
        for signature, counts in table.items()
        if counts[1] & counts[3]
    }


print(f"seed={SEED} attempts={ATTEMPTS} accepted_connected_cn1_or3={accepted}")
print("exact triangle-signature overlaps:", intersections(exact))
print("total-triangle-only overlaps:", intersections(total_triangle))
print(
    "exact-signature 2x2 candidates:",
    {signature: paths for signature, paths in intersections(exact).items() if len(paths) >= 2},
)
print(
    "total-triangle-only 2x2 candidates:",
    {signature: paths for signature, paths in intersections(total_triangle).items() if len(paths) >= 2},
)
