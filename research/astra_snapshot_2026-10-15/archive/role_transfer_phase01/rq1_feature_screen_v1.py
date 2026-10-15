"""Exploratory post-outcome screen of graph descriptors on the 16 RQ1 networks.

This is explicitly development analysis, not a fresh confirmation. Entire
matched network pairs are held out in cross-validation.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE / "rq1_structure_v1"
MANIFEST = json.loads((ROOT / "GRAPH_MANIFEST.json").read_text())
DEST = HERE / "rq1_feature_screen_v1"


def rows() -> list[dict]:
    output = []
    for item in MANIFEST["pairs"]:
        for group, info in item["graphs"].items():
            graph = nx.Graph()
            graph.add_nodes_from(range(10))
            graph.add_edges_from(info["labeled_edges"])
            adjacency = nx.to_numpy_array(graph, nodelist=range(10))
            walk2 = adjacency @ adjacency
            walk3 = walk2 @ adjacency
            walk4 = walk3 @ adjacency
            laplace = nx.laplacian_matrix(graph, nodelist=range(10)).toarray()
            inverse = np.linalg.pinv(laplace)
            saved = []
            for seed in MANIFEST["training_seeds"]:
                path = ROOT / "models" / "full" / f"{item['id']}_{group}" / "balanced_distractor" / f"seed_{seed}" / "metrics.json"
                data = json.loads(path.read_text())
                saved.append({tuple(entry["pair"]): entry["binary_role_accuracy"]
                              for entry in data["evaluation"]["260107"]
                              if not entry["observed"]})
            assert all(set(x) == set(saved[0]) for x in saved)
            for a, b in sorted(saved[0]):
                resistance = inverse[a, a] + inverse[b, b] - 2 * inverse[a, b]
                features = {
                    "common_neighbors": float(walk2[a, b]),
                    "length3_walks": float(walk3[a, b]),
                    "length4_walks": float(walk4[a, b]),
                    "endpoint_triangles": float(nx.triangles(graph, a) + nx.triangles(graph, b)),
                    "effective_resistance": float(resistance),
                    "global_triangles": float(info["triangles"]),
                }
                output.append({"network_pair": item["id"], "group": group,
                               "graph": f"{item['id']}_{group}", "pair": [a, b],
                               "accuracy_by_start": [float(run[(a, b)]) for run in saved],
                               "mean_accuracy": float(np.mean([run[(a, b)] for run in saved])),
                               "features": features})
    return output


def cross_validate(data: list[dict], feature_names: list[str], identity: bool) -> dict:
    y = np.asarray([row["mean_accuracy"] for row in data])
    pairs = [row["network_pair"] for row in data]
    labels = [tuple(row["pair"]) for row in data]
    predicted = np.empty(len(data), dtype=float)
    reference = np.empty(len(data), dtype=float)
    for held in sorted(set(pairs)):
        training = np.asarray([pair != held for pair in pairs])
        testing = ~training
        columns = []
        for name in feature_names:
            columns.append([row["features"][name] for row in data])
        if identity:
            for pair in itertools.combinations(range(10), 2):
                columns.append([float(label == pair) for label in labels])
        x = np.asarray(columns, dtype=float).T
        if feature_names:
            scaler = StandardScaler()
            x_train = x[training].copy()
            x_test = x[testing].copy()
            count = len(feature_names)
            x_train[:, :count] = scaler.fit_transform(x_train[:, :count])
            x_test[:, :count] = scaler.transform(x_test[:, :count])
        else:
            x_train, x_test = x[training], x[testing]
        model = Ridge(alpha=1.0).fit(x_train, y[training])
        predicted[testing] = model.predict(x_test)
        reference[testing] = y[training].mean()
    error = y - predicted
    ref_error = y - reference
    return {"mae_points": float(np.mean(np.abs(error)) * 100),
            "baseline_mae_points": float(np.mean(np.abs(ref_error)) * 100),
            "held_out_pair_r2_vs_train_mean": float(1 - np.sum(error**2) / np.sum(ref_error**2)),
            "fold_mae_points": {held: float(np.mean(np.abs(error[[p == held for p in pairs]])) * 100)
                                for held in sorted(set(pairs))}}


def summarize(data: list[dict]) -> dict:
    graph_groups: dict[str, list[dict]] = {}
    for row in data:
        graph_groups.setdefault(row["graph"], []).append(row)
    start_correlations = []
    for graph_rows in graph_groups.values():
        by_start = np.asarray([row["accuracy_by_start"] for row in graph_rows]).T
        for left, right in itertools.combinations(range(3), 2):
            corr = spearmanr(by_start[left], by_start[right]).statistic
            start_correlations.append(float(corr))
    feature_names = list(data[0]["features"])
    within_graph = {}
    for name in feature_names:
        values = []
        for graph_rows in graph_groups.values():
            feature = [row["features"][name] for row in graph_rows]
            outcome = [row["mean_accuracy"] for row in graph_rows]
            if len(set(feature)) > 1:
                values.append(float(spearmanr(feature, outcome).statistic))
        within_graph[name] = {"graphs_with_variation": len(values),
                              "median_spearman": float(np.median(values)) if values else None,
                              "positive_graphs": sum(value > 0 for value in values),
                              "negative_graphs": sum(value < 0 for value in values)}
    methods = {
        "common_neighbors": (["common_neighbors"], False),
        "walks_and_overlap": (["common_neighbors", "length3_walks", "length4_walks"], False),
        "all_graph_descriptors": (feature_names, False),
        "identity_only": ([], True),
        "identity_plus_graph": (feature_names, True),
    }
    return {"status": "exploratory_after_outcomes_seen", "graphs": len(graph_groups),
            "network_pairs": len(set(row["network_pair"] for row in data)),
            "unseen_pair_rows": len(data),
            "start_pair_spearman": {"mean": float(np.mean(start_correlations)),
                                    "median": float(np.median(start_correlations)),
                                    "min": float(np.min(start_correlations)),
                                    "max": float(np.max(start_correlations))},
            "within_graph_feature_association": within_graph,
            "leave_one_network_pair_out_prediction": {
                name: cross_validate(data, features, identity)
                for name, (features, identity) in methods.items()}}


def main() -> None:
    data = rows()
    result = summarize(data)
    DEST.mkdir(exist_ok=True)
    (DEST / "ROW_DATA.json").write_text(json.dumps(data, indent=2) + "\n")
    (DEST / "SUMMARY.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
