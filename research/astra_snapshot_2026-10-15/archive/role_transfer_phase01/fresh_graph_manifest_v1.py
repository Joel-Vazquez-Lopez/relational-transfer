"""Select untouched confirmation graphs by graph structure alone.

Run once. Refuses to replace an existing manifest. Does not train or inspect
any model outcomes. The result is input to a separately frozen runner.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parent
STUDY = ROOT.parent / "d5_indirect_support"


def source_worlds(path: Path, target_name: str) -> dict:
    """Read literal historical graph declarations without module import side effects."""
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if target_name == "WORLDS" and isinstance(target, ast.Name) and target.id == "WORLDS":
                return ast.literal_eval(node.value)
            if (target_name == "study.WORLDS" and isinstance(target, ast.Attribute)
                    and target.attr == "WORLDS" and isinstance(target.value, ast.Name)
                    and target.value.id == "study"):
                return ast.literal_eval(node.value)
    raise ValueError(f"Could not find literal {target_name} in {path}")


PREVIOUS = [dict(info) for info in source_worlds(
    STUDY / "prospective_20261006.py", "WORLDS").values()]
PREVIOUS += [dict(info) for info in source_worlds(
    STUDY / "prospective_20261006_confirm.py", "study.WORLDS").values()]
DESTINATION = ROOT / "role_aux_confirmation_v1" / "GRAPH_MANIFEST.json"


def graph(edges: list | tuple) -> nx.Graph:
    result = nx.Graph()
    result.add_nodes_from(range(10))
    result.add_edges_from(edges)
    assert nx.is_connected(result)
    assert result.number_of_edges() == 20
    assert all(degree == 4 for _, degree in result.degree())
    return result


def generate() -> dict:
    old = [graph(info["edges"]) for info in PREVIOUS]
    assert len(old) == 8
    assert all(not nx.is_isomorphic(left, right)
               for i, left in enumerate(old) for right in old[i + 1:])
    selected = []
    for generator_seed in range(10000, 100000):
        candidate = nx.random_regular_graph(4, 10, seed=generator_seed)
        if any(nx.is_isomorphic(candidate, prior)
               for prior in old + [entry["graph"] for entry in selected]):
            continue
        permutation = np.random.default_rng(np.random.SeedSequence(
            [generator_seed, 999])).permutation(10).tolist()
        selected.append({"generator_seed": generator_seed,
                         "graph": candidate, "label_permutation": permutation})
        if len(selected) == 8:
            break
    assert len(selected) == 8
    worlds = {}
    for index, entry in enumerate(selected):
        edges = sorted([list(sorted(edge)) for edge in entry["graph"].edges()])
        permutation = entry["label_permutation"]
        labeled_edges = sorted([sorted([permutation[a], permutation[b]])
                                for a, b in edges])
        assert graph(labeled_edges).number_of_edges() == 20
        worlds[f"fresh{index}"] = {"generator_seed": entry["generator_seed"],
                                   "canonical_edges": edges,
                                   "label_permutation": permutation,
                                   "labeled_edges": labeled_edges}
    return {"status": "selected_without_model_outcomes",
            "algorithm": "first eight connected 10-node degree-4 graph isomorphism classes from networkx.random_regular_graph seeds >=10000, excluding the eight historical classes",
            "labeling": "NumPy SeedSequence([generator_seed, 999]).permutation(10), fixed for all arms and starts",
            "training_seeds": [271001, 271002, 271003],
            "worlds": worlds,
            "source_hashes": {
                str(path.relative_to(ROOT.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (ROOT / "generator_gate_v1.py", ROOT / "role_aux_v1.py",
                             ROOT / "role_aux_v1" / "PROTOCOL.json")}}


def main() -> None:
    if DESTINATION.exists():
        raise FileExistsError(DESTINATION)
    result = generate()
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    DESTINATION.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"written": str(DESTINATION),
                      "worlds": {name: world["generator_seed"]
                                 for name, world in result["worlds"].items()},
                      "training_seeds": result["training_seeds"]}, indent=2))


if __name__ == "__main__":
    main()
