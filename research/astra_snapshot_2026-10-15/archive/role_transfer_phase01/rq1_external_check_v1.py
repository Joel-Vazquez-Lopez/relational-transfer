"""Retrospective external-graph check of the exploratory common-partner lead."""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
DEV = json.loads((HERE / "rq1_feature_screen_v1" / "ROW_DATA.json").read_text())
ROOT = HERE / "role_aux_confirmation_v1"
MANIFEST = json.loads((ROOT / "GRAPH_MANIFEST.json").read_text())
DEST = HERE / "rq1_feature_screen_v1" / "EXTERNAL_CHECK.json"


def external_rows() -> list[dict]:
    rows = []
    for world, info in MANIFEST["worlds"].items():
        graph = nx.Graph()
        graph.add_nodes_from(range(10))
        graph.add_edges_from(info["labeled_edges"])
        adjacency = nx.to_numpy_array(graph, nodelist=range(10))
        walk2 = adjacency @ adjacency
        walk3 = walk2 @ adjacency
        walk4 = walk3 @ adjacency
        scores = []
        for seed in MANIFEST["training_seeds"]:
            path = ROOT / "baseline" / "full" / world / "balanced_distractor" / f"seed_{seed}" / "metrics.json"
            data = json.loads(path.read_text())
            scores.append({tuple(row["pair"]): row["binary_role_accuracy"]
                           for row in data["evaluation"]["260107"] if not row["observed"]})
        assert all(set(run) == set(scores[0]) for run in scores)
        for a, b in sorted(scores[0]):
            rows.append({"graph": world, "pair": [a, b],
                         "common_neighbors": float(walk2[a, b]),
                         "length3_walks": float(walk3[a, b]),
                         "length4_walks": float(walk4[a, b]),
                         "accuracy_by_start": [float(run[(a, b)]) for run in scores],
                         "mean_accuracy": float(np.mean([run[(a, b)] for run in scores]))})
    return rows


def fit_predict(names: list[str], external: list[dict]) -> dict:
    x_train = np.asarray([[row["features"][name] for name in names] for row in DEV])
    y_train = np.asarray([row["mean_accuracy"] for row in DEV])
    x_test = np.asarray([[row[name] for name in names] for row in external])
    y_test = np.asarray([row["mean_accuracy"] for row in external])
    scale = StandardScaler().fit(x_train)
    model = Ridge(alpha=1.0).fit(scale.transform(x_train), y_train)
    predicted = model.predict(scale.transform(x_test))
    baseline = float(np.mean(y_train))
    return {"features": names, "development_mean": baseline,
            "external_mean": float(np.mean(y_test)),
            "external_mae_points": float(np.mean(np.abs(predicted-y_test))*100),
            "mean_baseline_mae_points": float(np.mean(np.abs(baseline-y_test))*100),
            "external_r2_vs_development_mean": float(1 - np.sum((predicted-y_test)**2)
                                                      / np.sum((baseline-y_test)**2)),
            "slope": model.coef_.tolist(), "intercept": float(model.intercept_)}


def main() -> None:
    external = external_rows()
    graph_rows = []
    for world in MANIFEST["worlds"]:
        subset = [row for row in external if row["graph"] == world]
        rho = float(spearmanr([row["common_neighbors"] for row in subset],
                              [row["mean_accuracy"] for row in subset]).statistic)
        graph_rows.append({"graph": world, "common_neighbor_spearman": rho,
                           "mean_accuracy": float(np.mean([row["mean_accuracy"] for row in subset]))})
    result = {"status": "retrospective_external_graphs; feature chosen after new-network results",
              "development_graphs": 16, "external_graphs": 8,
              "external_unseen_pair_rows": len(external),
              "common_neighbor_by_graph": graph_rows,
              "common_neighbor_negative_graphs": sum(row["common_neighbor_spearman"] < 0
                                                     for row in graph_rows),
              "prediction": {
                  "common_neighbors": fit_predict(["common_neighbors"], external),
                  "walks_and_overlap": fit_predict(["common_neighbors", "length3_walks", "length4_walks"], external)}}
    DEST.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
