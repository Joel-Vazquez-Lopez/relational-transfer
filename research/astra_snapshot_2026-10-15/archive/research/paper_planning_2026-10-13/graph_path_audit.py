"""Read-only audit of focal-pair paths in the two completed rewire manifests.

This script uses only frozen graph metadata. It does not inspect model predictions,
alter a protocol, or train a model. Run from any directory with Python 3.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STUDIES = (
    ROOT / "role_transfer_phase01/rq1_rewire_v1/GRAPH_MANIFEST.json",
    ROOT / "role_transfer_phase01/rq2_rewire_repair_v1/GRAPH_MANIFEST.json",
)


def focal_features(edges: list[list[int]], pair: list[int]) -> tuple[int, int]:
    neighbours = {node: set() for node in range(10)}
    for left, right in edges:
        neighbours[left].add(right)
        neighbours[right].add(left)
    left, right = pair
    assert right not in neighbours[left], "Focal pair must be withheld"
    common = len(neighbours[left] & neighbours[right])
    three_edge_paths = sum(
        next_node in neighbours[previous_node]
        for previous_node in neighbours[left]
        for next_node in neighbours[right]
    )
    return common, three_edge_paths


for study in STUDIES:
    manifest = json.loads(study.read_text())
    print(f"\n{study.parent.name}: family, sham (common, paths3), high (common, paths3), delta")
    for index, family in enumerate(manifest["families"]):
        arms = family.get("arms", family.get("conditions"))
        sham = focal_features(arms["sham"]["edges"], family["focal_pair"])
        high = focal_features(arms["high"]["edges"], family["focal_pair"])
        delta = tuple(right - left for left, right in zip(sham, high))
        print(index, sham, high, delta)
